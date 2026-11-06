// Additively regularized spheroid lubrication + isolated drag + surface contact.
// The legacy periodic mobility backends remain explicit opt-in alternatives.
// Mari gap convention and Singh sliding/rolling law; no positive-gap contact shell.
// This fix owns overdamped integration; no inertial integrator is used.
#include "fix_slurry_bridge.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "slurry_bridge_images.h"
#include "slurry_bridge_solver.h"
#include "slurry_bridge_contact_solver.h"
#include "slurry_bridge_step.h"
#include "slurry_bridge_shape.h"
#include "slurry_compact_bridge.h"
#include "slurry_history_math.h"
#include "slurry_nearfield_hydro.h"
#include "slurry_nearfield_solver.h"
#include "slurry_nearfield_step.h"
#include "slurry_surface_cohesion.h"
#include "slurry_surface_background.h"
#include "slurry_verlet_images.h"
#include "slurry_support_bound.h"
#include "slurry_adaptive_step.h"
#include "slurry_surface_coupled_step.h"
#include "slurry_surface_discrete_gradient.h"
#include "slurry_periodic_mobility.h"
#include "slurry_stiffness_control.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <vector>
using namespace LAMMPS_NS;
using namespace FixConst;
namespace cb = SlurryCompactBridge;
namespace bs = SlurryBridgeSolver;
namespace pm = SlurryPeriodic;
namespace im = SlurryBridgeImages;
namespace gr = slurry::gr_re2::graphite;
namespace step = SlurryBridgeStep;
namespace shape = SlurryBridgeShape;
namespace nh = SlurryNearfieldHydro;
namespace ns = SlurryNearfieldSolver;
namespace ni = SlurryNearfieldImplicitSolver;
namespace nstep = SlurryNearfieldStep;
namespace imex = SlurryNearfieldImexSolver;
namespace {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
struct Record {
  tagint tag;
  double x[3], axes[3], quat[4];
};
using PairKey = std::pair<tagint, tagint>;
struct History {
  V3 predicted, elastic, normal, angle;
  V3 rolling = V3::Zero();
};
gr::Vec3 arr(const V3 &x) { return {x[0], x[1], x[2]}; }
V3 vec(const gr::Vec3 &x) { return {x[0], x[1], x[2]}; }
void store_momentum(Atom *atom, AtomVecEllipsoid *avec, int i, const V3 &omega) {
  auto &b = avec->bonus[atom->ellipsoid[i]];
  double R[3][3], ob[3], lb[3], l[3];
  MathExtra::quat_to_mat(b.quat, R);
  MathExtra::transpose_matvec(R, omega.data(), ob);
  const auto inertia=shape::principal_inertia(atom->rmass[i],b.shape,false);
  for (int k = 0; k < 3; ++k) lb[k] = inertia[k] * ob[k];
  MathExtra::matvec(R, lb, l);
  std::copy(l, l + 3, atom->angmom[i]);
}
std::vector<Record> gather(const std::vector<Record> &local, MPI_Comm world, int nprocs) {
  const int bytes = static_cast<int>(local.size() * sizeof(Record));
  std::vector<int> counts(nprocs), offsets(nprocs);
  MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  int total = 0;
  for (int k = 0; k < nprocs; ++k) {
    offsets[k] = total;
    total += counts[k];
  }
  std::vector<Record> all(total / sizeof(Record));
  MPI_Allgatherv(local.data(), bytes, MPI_BYTE, all.data(), counts.data(), offsets.data(), MPI_BYTE,
                 world);
  std::sort(all.begin(), all.end(), [](const Record &a, const Record &b) { return a.tag < b.tag; });
  return all;
}
} // namespace
struct FixSlurryBridge::Impl {
  AtomVecEllipsoid *avec = nullptr;
  double eta = 0, shear = 0, nominal_diameter = 0, epsilon = 0, kn = 0;
  double reference_radius = 0, regularization_length = 0;
  double mu = 10., kt = 2.4, mu_roll = 10., kr = 2.4; // kn, kt, kr: N/m
  double cohesion_work = 0., cohesion_range = .67e-9; // J/m^2, m; disabled by default
  // Optional overlap-controlled normal stiffness (practical IMEX only). kn is
  // the stiffness of the next solve; kn_load is the stiffness that produced the
  // committed overlaps, from which frozen Coulomb caps take their load.
  bool kn_control = false;
  double kn_load = 0.;
  SlurryStiffnessControl::Settings kn_settings;
  SlurryStiffnessControl::State kn_state;
  SlurrySurfaceBackground::Parameters background;
  double surface_adaptive=0., suggested_surface_step=0.;
  bool surface_roundoff=false, direct_advance=false;
  bool surface_reuse=true,surface_defer=true;
  bool surface_lagged_hydro=false, surface_neighbors=true, surface_sparse=false;
  bool surface_coupled=false, surface_pc=false, surface_dg=false, surface_practical=false, surface_direct=false;
  bool tolerance_specified=false,surface_linear_specified=false;
  int surface_refresh=40;
  double surface_linear=1e-10;
  double spectral = 2, spectral_tol = 1e-8, rtol = 1e-7, avg_start = 5;
  int maxit = 1200, interval = 100, nmax = 0;
  std::ofstream csv;
  std::array<double, 75> diag{};
  std::array<double, 4> stress{};
  std::array<long double, 4> interval_stress{};
  long double selected_integral = 0;
  double selected_duration = 0, interval_duration = 0, interval_start = 0;
  bigint interval_steps = 0, last_output = -1;
  std::map<PairKey, std::vector<History>> history, pending;
  std::vector<tagint> tags;
  // Near-field BE exposes its prepared step velocity through atom v/angmom.
  // Legacy Heun exposes current velocities and stores its private average.
  std::vector<tagint> advance_tags;
  Eigen::VectorXd advance_velocity, advance_endpoint; // xyz + quaternion per particle
  bs::Result warm;
  // Reuse FFT plans/buffers and exact solver caches across physical steps.
  // reset() still refreshes all position, orientation and lattice coefficients.
  pm::Mobility mobility;
  ni::Workspace nearfield_workspace;
  imex::Workspace imex_workspace;
  SlurrySurfaceCoupledStep::Workspace coupled_workspace;
  std::shared_ptr<bs::detail::SolverWorkspace> solver_workspace =
      std::make_shared<bs::detail::SolverWorkspace>();
  bool have_warm = false;
  bool nearfield_mobility = true;
  bool direct_mobility = false;
  bool imex_contacts = true;
  ns::Workspace &hydro_workspace() {
    return imex_contacts?static_cast<ns::Workspace&>(imex_workspace)
                        :static_cast<ns::Workspace&>(nearfield_workspace);
  }
};
FixSlurryBridge::FixSlurryBridge(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), p_(new Impl) {
  if (narg < 8)
    error->all(FLERR, "slurry/bridge requires eta shear_rate nominal_diameter epsilon kn "
                      "[contact mu_s kt] [rolling mu_r kr_N_per_m] [shape spheroid] "
                      "[spectral xi_a] [spectral_tolerance tol] [mobility_backend nearfield|mesh|direct] "
                      "[contact_scheme imex|coupled] "
                      "[cohesion work_J_per_m2 opening_range_m] "
                      "[background AH_J h0_m sigma_m] [surface_adaptive fraction] [surface_roundoff yes|no] "
                      "[kn_adapt target raise lower max_factor relax_strain] "
                      "[surface_reuse yes|no] [surface_defer yes|no] "
                      "[tolerance rtol] [iterations maxit] [history file interval average_start_strain]");
  auto &p = *p_;
  p.background.hamaker=0.; // optional background, exact disabled path
  // Freeze a positive-definite contact Jacobian as the Krylov preconditioner;
  // every linear solve still checks the current exact Jacobian residual.
  p.imex_workspace.use_contact_preconditioner = true;
  // Adapt only intermediate Newton solves; final physical tolerances are unchanged.
  p.imex_workspace.use_adaptive_linear_forcing = true;
  // Keep the tested 40-update primary strategy; allow a generous continuation
  // budget without forcing any extra work after the original residual passes.
  // The batch can expose this independently from its linear Krylov limit.
  int imex_total_iterations = 1000;
  if (const char *setting = std::getenv("SLURRY_IMEX_MAX_ITERATIONS"))
    imex_total_iterations = utils::inumeric(FLERR, setting, false, lmp);
  const int imex_primary_iterations = bs::Options{}.maximum_newton_iterations;
  if (imex_total_iterations < imex_primary_iterations)
    error->all(FLERR, "SLURRY_IMEX_MAX_ITERATIONS must be at least {}", imex_primary_iterations);
  p.imex_workspace.maximum_recovery_iterations = imex_total_iterations - imex_primary_iterations;
  p.eta = utils::numeric(FLERR, arg[3], false, lmp);
  p.shear = utils::numeric(FLERR, arg[4], false, lmp);
  p.nominal_diameter = utils::numeric(FLERR, arg[5], false, lmp);
  p.epsilon = utils::numeric(FLERR, arg[6], false, lmp);
  p.kn = utils::numeric(FLERR, arg[7], false, lmp);
  p.reference_radius = .5 * p.nominal_diameter;
  p.regularization_length = p.epsilon * p.reference_radius;
  std::string filename;
  for (int k = 8; k < narg;) {
    if (!std::strcmp(arg[k], "contact") && k + 2 < narg) {
      p.mu = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.kt = utils::numeric(FLERR, arg[k + 2], false, lmp);
      k += 3;
    } else if (!std::strcmp(arg[k], "rolling") && k + 2 < narg) {
      p.mu_roll = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.kr = utils::numeric(FLERR, arg[k + 2], false, lmp);
      k += 3;
    } else if (!std::strcmp(arg[k], "cohesion") && k + 2 < narg) {
      p.cohesion_work = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.cohesion_range = utils::numeric(FLERR, arg[k + 2], false, lmp);
      k += 3;
    } else if (!std::strcmp(arg[k], "background") && k + 3 < narg) {
      p.background.hamaker=utils::numeric(FLERR,arg[k+1],false,lmp);
      p.background.reference_gap=utils::numeric(FLERR,arg[k+2],false,lmp);
      p.background.sigma=utils::numeric(FLERR,arg[k+3],false,lmp);
      k+=4;
    } else if (!std::strcmp(arg[k], "kn_adapt") && k + 5 < narg) {
      // kn_adapt target raise lower max_factor relax_strain
      p.kn_control = true;
      p.kn_settings.target = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.kn_settings.raise = utils::numeric(FLERR, arg[k + 2], false, lmp);
      p.kn_settings.lower = utils::numeric(FLERR, arg[k + 3], false, lmp);
      p.kn_settings.max_factor = utils::numeric(FLERR, arg[k + 4], false, lmp);
      p.kn_settings.relax_strain = utils::numeric(FLERR, arg[k + 5], false, lmp);
      k += 6;
    } else if (!std::strcmp(arg[k], "surface_adaptive") && k + 1 < narg) {
      p.surface_adaptive=utils::numeric(FLERR,arg[k+1],false,lmp);k+=2;
    } else if (!std::strcmp(arg[k], "surface_integrator") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"variation")&&std::strcmp(arg[k+1],"reuse")&&
          std::strcmp(arg[k+1],"coupled")&&std::strcmp(arg[k+1],"pc")&&std::strcmp(arg[k+1],"dg")&&
          std::strcmp(arg[k+1],"practical"))
        error->all(FLERR,"surface_integrator must be variation, reuse, coupled, pc, dg or practical");
      p.surface_practical=!std::strcmp(arg[k+1],"practical");
      p.surface_dg=!std::strcmp(arg[k+1],"dg");
      p.surface_coupled=p.surface_dg||!std::strcmp(arg[k+1],"coupled");
      p.surface_pc=!std::strcmp(arg[k+1],"pc");
      p.surface_lagged_hydro=p.surface_coupled||p.surface_pc||p.surface_practical||!std::strcmp(arg[k+1],"reuse");k+=2;
    } else if (!std::strcmp(arg[k], "surface_direct") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"yes")&&std::strcmp(arg[k+1],"no"))
        error->all(FLERR,"surface_direct must be yes or no");
      p.surface_direct=!std::strcmp(arg[k+1],"yes");k+=2;
    } else if (!std::strcmp(arg[k], "surface_linear") && k + 1 < narg) {
      p.surface_linear=utils::numeric(FLERR,arg[k+1],false,lmp);
      p.surface_linear_specified=true;
      if(!(p.surface_linear>0.&&p.surface_linear<1.))
        error->all(FLERR,"surface_linear must be in (0,1)");k+=2;
    } else if (!std::strcmp(arg[k], "surface_refresh") && k + 1 < narg) {
      p.surface_refresh=utils::inumeric(FLERR,arg[k+1],false,lmp);
      if(p.surface_refresh<=0)error->all(FLERR,"surface_refresh must be positive");k+=2;
    } else if (!std::strcmp(arg[k], "surface_sparse") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"yes")&&std::strcmp(arg[k+1],"no"))
        error->all(FLERR,"surface_sparse must be yes or no");
      p.surface_sparse=!std::strcmp(arg[k+1],"yes");k+=2;
    } else if (!std::strcmp(arg[k], "surface_neighbors") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"yes")&&std::strcmp(arg[k+1],"no"))
        error->all(FLERR,"surface_neighbors must be yes or no");
      p.surface_neighbors=!std::strcmp(arg[k+1],"yes");k+=2;
    } else if (!std::strcmp(arg[k], "surface_roundoff") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"yes")&&std::strcmp(arg[k+1],"no"))
        error->all(FLERR,"surface_roundoff must be yes or no");
      p.surface_roundoff=!std::strcmp(arg[k+1],"yes");k+=2;
    } else if ((!std::strcmp(arg[k], "surface_reuse") || !std::strcmp(arg[k], "surface_defer")) && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"yes")&&std::strcmp(arg[k+1],"no"))
        error->all(FLERR,"surface_reuse and surface_defer require yes or no");
      const bool enabled=!std::strcmp(arg[k+1],"yes");
      if(!std::strcmp(arg[k],"surface_reuse"))p.surface_reuse=enabled;
      else p.surface_defer=enabled;
      k+=2;
    } else if (!std::strcmp(arg[k], "shape") && k + 1 < narg) {
      if (std::strcmp(arg[k+1],"spheroid"))
        error->all(FLERR,"slurry/bridge surface-contact model requires shape spheroid");
      k += 2;
    } else if (!std::strcmp(arg[k], "spectral") && k + 1 < narg) {
      p.spectral = utils::numeric(FLERR, arg[k + 1], false, lmp);
      k += 2;
    } else if (!std::strcmp(arg[k], "spectral_tolerance") && k + 1 < narg) {
      p.spectral_tol = utils::numeric(FLERR, arg[k + 1], false, lmp);
      k += 2;
    } else if (!std::strcmp(arg[k], "mobility_backend") && k + 1 < narg) {
      if (std::strcmp(arg[k + 1], "nearfield") && std::strcmp(arg[k + 1], "mesh") &&
          std::strcmp(arg[k + 1], "direct"))
        error->all(FLERR, "slurry/bridge mobility_backend must be nearfield, mesh or direct");
      p.nearfield_mobility = !std::strcmp(arg[k + 1], "nearfield");
      p.direct_mobility = !std::strcmp(arg[k + 1], "direct");
      k += 2;
    } else if (!std::strcmp(arg[k], "contact_scheme") && k + 1 < narg) {
      if(std::strcmp(arg[k+1],"imex")&&std::strcmp(arg[k+1],"coupled"))
        error->all(FLERR,"slurry/bridge contact_scheme must be imex or coupled");
      p.imex_contacts=!std::strcmp(arg[k+1],"imex");
      k+=2;
    } else if (!std::strcmp(arg[k], "tolerance") && k + 1 < narg) {
      p.rtol = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.tolerance_specified=true;
      k += 2;
    } else if (!std::strcmp(arg[k], "iterations") && k + 1 < narg) {
      p.maxit = utils::inumeric(FLERR, arg[k + 1], false, lmp);
      k += 2;
    } else if (!std::strcmp(arg[k], "history") && k + 3 < narg) {
      filename = arg[k + 1];
      p.interval = utils::inumeric(FLERR, arg[k + 2], false, lmp);
      p.avg_start = utils::numeric(FLERR, arg[k + 3], false, lmp);
      k += 4;
    } else
      error->all(FLERR, "Unknown or incomplete slurry/bridge option: {}", arg[k]);
  }
  if(p.surface_practical) {
    if(!p.nearfield_mobility||!p.imex_contacts)
      error->all(FLERR,"surface_integrator practical requires nearfield IMEX");
    if(!p.tolerance_specified)p.rtol=1e-4;
    if(!p.surface_linear_specified)p.surface_linear=1e-4;
  }
  if (p.nearfield_mobility && p.imex_contacts && comm->me == 0) {
    std::fprintf(stderr, "slurry/bridge IMEX iteration limits: primary=%d recovery=%d total=%d per_inner_solve\n",
                 imex_primary_iterations, p.imex_workspace.maximum_recovery_iterations, imex_total_iterations);
    std::fflush(stderr);
  }
  if (!(p.eta > 0 && p.nominal_diameter > 0 && p.epsilon > 0 && p.kn > 0 &&
        p.mu >= 0 && p.kt >= 0 && p.mu_roll >= 0 && p.kr >= 0 &&
        p.cohesion_work >= 0 && p.cohesion_range > 0 &&
        p.background.hamaker>=0 && p.surface_adaptive>=0 && p.surface_adaptive<1 &&
        p.spectral > 0 && p.spectral_tol > 0 && p.spectral_tol < 1 && p.rtol > 0 && p.rtol < 1 &&
        p.maxit > 0 && p.interval > 0 && p.avg_start >= 0) ||
      !std::isfinite(p.eta + p.shear + p.nominal_diameter + p.epsilon + p.kn + p.mu + p.kt +
                     p.mu_roll + p.kr + p.spectral + p.spectral_tol + p.rtol + p.avg_start +
                     p.cohesion_work + p.cohesion_range + p.background.hamaker + p.surface_adaptive) ||
      !(p.regularization_length > 0 && std::isfinite(p.regularization_length)) ||
      (p.mu > 0 && p.kt == 0) || (p.mu_roll > 0 && p.kr == 0))
    error->all(FLERR, "Invalid slurry/bridge material or numerical parameter");
  try { SlurrySurfaceBackground::validate(p.background); }
  catch(const std::exception &e) { error->all(FLERR,e.what()); }
  p.kn_load = p.kn;
  p.kn_state = SlurryStiffnessControl::initial_state(p.kn);
  if (p.kn_control) {
    if (!p.surface_practical)
      error->all(FLERR, "kn_adapt requires surface_integrator practical");
    try { SlurryStiffnessControl::validate(p.kn_settings); }
    catch (const std::exception &e) { error->all(FLERR, e.what()); }
  }
  if((p.cohesion_work>0. || p.background.hamaker>0.) && (!p.nearfield_mobility || !p.imex_contacts))
    error->all(FLERR,"Surface cohesion currently requires nearfield frozen-cap IMEX");
  if((p.surface_pc||p.surface_coupled)&&(!p.nearfield_mobility||!p.imex_contacts||
      !(p.surface_adaptive>0.)||!(p.cohesion_work>0.||p.background.hamaker>0.)))
    error->all(FLERR,"PC/coupled surface integration requires nearfield IMEX, surface forces and surface_adaptive > 0");
  if (!filename.empty() && comm->me == 0) {
    p.csv.open(filename, std::ios::trunc);
    if (!p.csv)
      error->one(FLERR, "Cannot open slurry/bridge history {}", filename);
    p.csv << std::setprecision(17)
          << "step,time_s,strain,sigma_solvent_Pa,sigma_hydro_Pa,sigma_contact_Pa,sigma_xz_Pa,eta_"
             "Pa_s,eta_relative,mean_eta_Pa_s,average_duration_s,hydro_dissipation_W,contact_power_"
             "W,stress_power_W,power_balance_error_W,force_residual,mobility_residual,linear_"
             "iterations,newton_iterations,near_pairs,contacts,reciprocal_modes,min_gap_m,max_"
             "overlap_m,profile_geometry_s,profile_middle_s,profile_solve_s,profile_total_s,"
             "interval_steps,interval_start_time_s,interval_end_time_s,interval_start_strain,"
             "interval_end_strain,interval_mean_sigma_solvent_Pa,interval_mean_sigma_hydro_Pa,"
             "interval_mean_sigma_contact_Pa,interval_mean_sigma_xz_Pa,interval_mean_eta_Pa_s,"
             "interval_mean_eta_relative,solver_relative_residual,descent_steps,continuation_stages,"
             "continuation_retries,mobility_applications,breakpoint_steps,geometry_iterations,"
             "contact_gap_consistency_m,actual_endpoint_min_gap_m,actual_endpoint_overlap_m,"
             "normal_elastic_energy_J,tangential_elastic_energy_J,linear_gap_defect_m,"
             "contact_candidates,krylov_dimension,fft_plan_builds,mobility_threads,"
             "preconditioner_symbolic_total,preconditioner_numeric_total,"
             "preconditioner_numeric_reuses_total,monitor_reuses_total,"
             "rolling_yield_contacts,rolling_elastic_energy_J,"
             "rolling_plastic_dissipation_W,rolling_contact_power_W,"
             "next_step_end_time_s,next_step_rolling_projection_work_J,"
             "max_overlap_fraction_min_semiaxis,actual_endpoint_max_overlap_fraction_min_semiaxis,"
             "minimum_semiaxis_m,force_geometry_time_s,force_history_time_s,time_integrator,"
             "friction_cap_time_s,max_normal_cap_lag_N,normal_cap_lag_l1_fraction,"
             "max_sliding_cap_excess_N,max_rolling_cap_excess_N_m,detached_history_release_energy_J,"
             "sigma_cohesion_Pa,cohesion_energy_J,cohesion_power_W,cohesive_pairs,"
             "sigma_background_Pa,background_energy_J,background_power_W,background_pairs,"
             "substeps,substep_rejections,minimum_substep_s,maximum_surface_error,"
             "roundoff_budget,max_roundoff_fraction,solver_acceptance_ratio,"
             "surface_quadrature_evaluations,surface_quadrature_max_relative_work_defect,surface_quadrature_max_panels,"
             "surface_potential_change_power_W,surface_path_work_power_W,"
             "kn_N_per_m,kn_stiffening_retries_total,kn_adjustments_total\n";
  }
  peratom_flag = 1;
  size_peratom_cols = 17;
  peratom_freq = 1;
  time_integrate = 1;
  vector_flag = 1;
  size_vector = 16;
  global_freq = 1;
  extvector = 0;
  nevery = 1;
}
FixSlurryBridge::~FixSlurryBridge() { memory->destroy(array_atom); }
int FixSlurryBridge::setmask() { return INITIAL_INTEGRATE | END_OF_STEP; }
void FixSlurryBridge::init() {
  auto &p = *p_;
  p.avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!p.avec || igroup != 0 || std::strcmp(update->unit_style, "si") || domain->dimension != 3 ||
      !domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "slurry/bridge requires all-group periodic 3D SI ellipsoid atoms");
  if (!force->pair_style || std::strcmp(force->pair_style, "zero"))
    error->all(FLERR, "slurry/bridge owns hydrodynamic/contact forces; use pair_style zero");
  bool before = false, seen = false;
  for (auto *f : modify->get_fix_list()) {
    if (f == this) {
      seen = true;
      continue;
    }
    if (!std::strcmp(f->style, "deform")) {
      auto *deform = dynamic_cast<FixDeform *>(f);
      if (!deform || deform->remapflag != Domain::V_REMAP || f->igroup != 0)
        error->all(FLERR, "slurry/bridge requires fix deform on all atoms with remap v");
      if (!seen && f->nevery == 0)
        before = true;
      else
        error->all(FLERR, "Define fix deform 0 xz before slurry/bridge");
    } else if (f->time_integrate || !std::strncmp(f->style, "slurry/", 7))
      error->all(FLERR,
                 "slurry/bridge cannot be combined with another integrator or slurry force fix");
  }
  if (p.shear != 0 && !before)
    error->all(FLERR, "slurry/bridge requires preceding fix deform 0 xz erate with remap v");
  for (int i = 0; i < atom->nlocal; ++i)
    if (atom->ellipsoid[i] < 0 || !(atom->rmass[i] > 0))
      error->one(FLERR, "slurry/bridge requires finite ellipsoids with positive mass");
}
double FixSlurryBridge::simulation_time() const {
  return update->atime + static_cast<double>(update->ntimestep - update->atimestep) * update->dt;
}
void FixSlurryBridge::setup(int) {
  solve_velocities();
  write_history();
}
void FixSlurryBridge::end_of_step() {
  solve_velocities();
  write_history();
}
double FixSlurryBridge::compute_vector(int k) { return p_->diag[k]; }
void FixSlurryBridge::initial_integrate(int) {
  auto &p = *p_;
  double dt = update->dt, end = simulation_time(), start = end - dt;
  if (p.interval_steps == 0)
    p.interval_start = start;
  ++p.interval_steps;
  p.interval_duration += dt;
  for (int k = 0; k < 4; ++k)
    p.interval_stress[k] += p.stress[k] * dt;
  if (p.shear != 0) {
    double selected = std::max(0., end - std::max(start, p.avg_start / std::abs(p.shear)));
    p.selected_duration += selected;
    p.selected_integral += p.stress[3] * selected;
  }
  // Commit exactly once for the pending advance. Setup/run 0 do not advance springs.
  p.history = p.pending;
  for (int i = 0; i < atom->nlocal; ++i) {
    auto &b = p.avec->bonus[atom->ellipsoid[i]];
    const auto found=std::lower_bound(p.advance_tags.begin(),p.advance_tags.end(),atom->tag[i]);
    if(found==p.advance_tags.end() || *found!=atom->tag[i] ||
       p.advance_velocity.size()!=6*static_cast<Eigen::Index>(p.advance_tags.size()))
      error->one(FLERR,"slurry/bridge missing prepared corrector velocity for atom {}",atom->tag[i]);
    const Eigen::Index j=found-p.advance_tags.begin();
    const V3 w=p.advance_velocity.segment<3>(6*j+3);
    if(p.direct_advance) {
      if(p.advance_endpoint.size()!=7*static_cast<Eigen::Index>(p.advance_tags.size()))
        error->one(FLERR,"slurry/bridge missing prepared adaptive endpoint");
      for(int k=0;k<3;++k)atom->x[i][k]=p.advance_endpoint[7*j+k];
      for(int k=0;k<4;++k)b.quat[k]=p.advance_endpoint[7*j+3+k];
      store_momentum(atom,p.avec,i,w);
      continue;
    }
    for (int k = 0; k < 3; ++k)
      atom->x[i][k] += dt * p.advance_velocity[6*j+k];
    Eigen::Quaterniond q = step::rotate(
        Eigen::Quaterniond(b.quat[0], b.quat[1], b.quat[2], b.quat[3]), w, dt);
    b.quat[0] = q.w();
    b.quat[1] = q.x();
    b.quat[2] = q.y();
    b.quat[3] = q.z();
    store_momentum(atom, p.avec, i, w);
  }
}
void FixSlurryBridge::solve_velocities() {
  auto &p = *p_;
  const double begun = MPI_Wtime();
  const double expected = p.shear * domain->zprd;
  if (std::abs(domain->h_rate[4] - expected) > 1e-10 * std::max(std::abs(expected), 1e-30) ||
      std::abs(domain->h_rate[5]) + std::abs(domain->h_rate[3]) + std::abs(domain->h_rate[0]) +
              std::abs(domain->h_rate[1]) + std::abs(domain->h_rate[2]) >
          1e-20)
    error->all(FLERR, "slurry/bridge requires constant-volume xz shear matching fix deform");
  std::vector<Record> local(atom->nlocal);
  for (int i = 0; i < atom->nlocal; ++i) {
    auto &r = local[i];
    auto &b = p.avec->bonus[atom->ellipsoid[i]];
    r.tag = atom->tag[i];
    std::copy(atom->x[i], atom->x[i] + 3, r.x);
    std::copy(b.shape, b.shape + 3, r.axes);
    std::copy(b.quat, b.quat + 4, r.quat);
  }
  auto records = gather(local, world, comm->nprocs);
  int n = static_cast<int>(records.size());
  if (n == 0 || static_cast<bigint>(n) != atom->natoms)
    error->all(FLERR, "slurry/bridge particle count invalid");
  Eigen::VectorXd physical = Eigen::VectorXd::Zero(6 * n),
                  advance_physical = Eigen::VectorXd::Zero(6 * n),
                  moments = Eigen::VectorXd::Zero(17 * n),
                  direct_endpoint=Eigen::VectorXd::Zero(7*n);
  int direct_advance=0;
  int failed = 0;
  std::string message;
  const char *phase = "initial validation";
  if (comm->me == 0)
    try {
      const double length = p.reference_radius, rate = std::max(std::abs(p.shear), 1.);
      for (const auto &r : records)
        for (double a : r.axes)
          if (!(a > 0))
            throw std::runtime_error("non-positive shape axis");
      double mean_volume_ratio=0.;
      for (const auto &r : records) {
        const double volume_radius = std::cbrt(r.axes[0]*r.axes[1]*r.axes[2]);
        const double aspect = r.axes[0]/r.axes[2];
        if ((!p.nearfield_mobility && std::abs(volume_radius-length) > 1e-8*length) ||
            !std::isfinite(volume_radius) ||
            std::abs(r.axes[0]-r.axes[1]) > 1e-10*r.axes[0] ||
            aspect < 1.-1e-12 || aspect > 5.+1e-12)
          throw std::runtime_error("bridge requires finite oblate spheroids with AR 1..5; mesh/direct require a single volume-equivalent diameter");
        mean_volume_ratio+=std::pow(volume_radius/length,3)/n;
      }
      if(std::abs(mean_volume_ratio-1.)>1e-8)
        throw std::runtime_error("bridge nominal_diameter must match the cubic mean volume-equivalent diameter");
      M3 H;
      H << domain->xprd, domain->xy, domain->xz, 0, domain->yprd, domain->yz, 0, 0, domain->zprd;
      H /= length;
      const double volume = H.determinant(), dt = update->dt * rate;
      M3 kappa = M3::Zero();
      kappa(0, 2) = p.shear / rate;
      M3 E = .5 * (kappa + kappa.transpose());
      V3 omega_inf(0, .5 * p.shear / rate, 0);
      std::vector<gr::Body> bodies(n);
      std::vector<Eigen::Quaterniond> quaternions;
      quaternions.reserve(n);
      for (int i = 0; i < n; ++i) {
        double R[3][3], q[4];
        std::copy(records[i].quat, records[i].quat + 4, q);
        quaternions.emplace_back(q[0], q[1], q[2], q[3]);
        MathExtra::quat_to_mat(q, R);
        for (int k = 0; k < 3; ++k) {
          bodies[i].position[k] = (records[i].x[k] - domain->boxlo[k]) / length;
          bodies[i].axes[k] = records[i].axes[k] / length;
          for (int l = 0; l < 3; ++l)
            bodies[i].rotation[3 * k + l] = R[k][l];
        }
      }
      auto minimum_axis = [](const gr::Body &body) {
        return *std::min_element(body.axes.begin(), body.axes.end());
      };
      double minimum_semiaxis=std::numeric_limits<double>::infinity();
      for(const auto &body:bodies)minimum_semiaxis=std::min(minimum_semiaxis,minimum_axis(body));
      std::vector<double> volume_radii(n);
      double minimum_volume_radius=std::numeric_limits<double>::infinity();
      for(int i=0;i<n;++i) {
        volume_radii[i]=nh::volume_radius(bodies[i]);
        minimum_volume_radius=std::min(minimum_volume_radius,volume_radii[i]);
      }
      // Share normal seeds within this step. Keys use the current lattice
      // basis only; no stale image bookkeeping survives a periodic tilt flip.
      using GapKey=std::tuple<int,int,int,int,int>;
      std::map<GapKey,gr::GapCache> gap_cache;
      auto exact_gap = [&](const gr::Body &first,const gr::Body &second,int i,int j,
                           const im::ImageIndex &image) {
        try {
          auto &cache=gap_cache[GapKey(i,j,image[0],image[1],image[2])];
          return gr::closestEllipsoidGap(first,second,&cache);
        } catch(const std::exception &e) {
          std::ostringstream out;
          out << e.what() << "; pair tags=" << records[i].tag << ',' << records[j].tag
              << "; image=" << image.transpose() << "; displacement/a_ref="
              << (vec(second.position)-vec(first.position)).transpose()
              << "; minimum_semiaxis/a_ref=" << std::min(minimum_axis(first),minimum_axis(second));
          throw std::runtime_error(out.str());
        }
      };
      bool same = p.tags.size() == records.size();
      for (int i = 0; same && i < n; ++i)
        same = p.tags[i] == records[i].tag;
      if (!same) {
        p.have_warm = false;
        p.history.clear();
        p.tags.resize(n);
        for (int i = 0; i < n; ++i)
          p.tags[i] = records[i].tag;
      }
      const bool has_surface=p.cohesion_work>0.||p.background.hamaker>0.;
      p.imex_workspace.use_roundoff_budget=has_surface&&p.surface_roundoff;
      p.imex_workspace.endpoint_inner_tolerance_fraction=p.surface_practical?1.:.1;
      p.imex_workspace.require_endpoint_constitutive_match=!p.surface_practical;
      p.imex_workspace.defer_candidate_validation=(p.surface_practical||(has_surface&&p.surface_adaptive>0.))&&p.surface_defer;
      p.imex_workspace.reuse_contact_preconditioner_across_solves=
          (p.surface_practical||(has_surface&&p.surface_adaptive>0.))&&p.surface_reuse;
      p.imex_workspace.contact_preconditioner_refresh_iterations=p.surface_refresh;
      if(has_surface||p.surface_practical)p.imex_workspace.linear_relative_target=p.surface_linear;
      p.imex_workspace.use_krylov=!(has_surface&&p.surface_direct);
      auto background_parameters=p.background;
      background_parameters.hamaker/=p.eta*std::pow(length,3)*rate;
      background_parameters.sigma/=length;background_parameters.reference_gap/=length;
      background_parameters.curvature_switch/=length;background_parameters.curvature_cutoff/=length;
      background_parameters.far_switch/=length;background_parameters.far_cutoff/=length;
      const SlurrySurfaceCohesion::Parameters cohesion_parameters{
          p.cohesion_work/(p.eta*length*rate),p.cohesion_range/length};
      struct SurfaceSample {
        SlurrySurfaceCohesion::Result cohesion;
        SlurrySurfaceBackground::Result background;
      };
      struct ScreenedPair {int i,j; im::ImageIndex image; gr::Vec3 normal;};
      const double surface_support=std::max(p.cohesion_work>0.?cohesion_parameters.opening_range:0.,
          p.background.hamaker>0.?background_parameters.far_cutoff-background_parameters.reference_gap:0.);
      const bool sparse_contacts=(p.surface_practical||(has_surface&&p.surface_adaptive>0.))&&p.surface_sparse&&!p.surface_dg;
      struct Frame {
        bs::Problem problem;
        step::Geometry geometry;
        std::vector<PairKey> keys;
        std::vector<V3> displacements;
        double min_gap=0., overlap=0., overlap_fraction=0., geometry_time=0., middle_time=0.;
        int active_near=0;
        M3 cohesion_stress=M3::Zero();
        double cohesion_energy=0.;
        int cohesive_pairs=0,background_pairs=0;
        M3 background_stress=M3::Zero();double background_energy=0.;
        bs::Vector background_external;
        std::vector<SurfaceSample> surface_samples;
        std::vector<gr::GapResult> surface_geometry;
        std::vector<ScreenedPair> screened_pairs;
        pm::Settings settings;
      };
      auto candidate_cache=std::make_shared<SlurryVerletImages::Cache>();
      auto center_cutoff=[&](int i,int j) {
        const double ai=shape::bounding_radius(bodies[i],false),aj=shape::bounding_radius(bodies[j],false);
        const auto pair_lengths=nh::pair_lengths(volume_radii[i],volume_radii[j],p.epsilon);
        const double interaction=p.nearfield_mobility?pair_lengths.cutoff:std::min(ai,aj);
        double surface=p.cohesion_work>0.?cohesion_parameters.opening_range:0.;
        if(p.background.hamaker>0.)surface=std::max(surface,background_parameters.far_cutoff);
        return ai+aj+std::max(interaction,surface);
      };
      auto assemble = [&](const std::vector<gr::Body> &stage_bodies, const M3 &stage_H,
                          const std::vector<Eigen::Quaterniond> &stage_quaternions,
                          const std::map<PairKey,std::vector<History>> &stage_history, double stage_dt,
                          const Frame* hydro_frame=nullptr) -> Frame {
      bs::Problem problem;
      problem.count = n;
      problem.dt = stage_dt;
      problem.external = Eigen::VectorXd::Zero(6 * n);
      problem.prescribed = Eigen::VectorXd::Zero(11 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          problem.prescribed[11 * i + 6 + k] = -(E.array() * cb::strain_basis()[k].array()).sum();
      std::vector<Eigen::Triplet<double>> entries;
      std::vector<PairKey> contact_keys;
      std::vector<V3> contact_displacements;
      std::vector<gr::GapResult> surface_geometry;
      std::vector<im::ImageIndex> contact_images;
      int active_near = 0;
      M3 cohesion_stress=M3::Zero();
      double cohesion_energy=0.;
      int cohesive_pairs=0;
      M3 background_stress=M3::Zero();double background_energy=0.;int background_pairs=0;
      bs::Vector background_external=bs::Vector::Zero(6*n);
      std::vector<SurfaceSample> surface_samples;
      std::vector<ScreenedPair> screened_pairs;
      double min_gap = std::numeric_limits<double>::infinity(), overlap = 0, overlap_fraction = 0;
      double section = MPI_Wtime();
      const bool use_candidates=(has_surface||p.surface_practical)&&(p.surface_neighbors||p.surface_dg);
      std::vector<SlurryVerletImages::Candidate> exhaustive;
      if(use_candidates) {
        if(!candidate_cache->valid(stage_bodies,stage_H,p.surface_dg?.5:0.))
          candidate_cache->build(stage_bodies,stage_H,center_cutoff,.1*minimum_volume_radius);
      } else {
        const im::ImageEnumerator enumerate(stage_H);
        for(int i=0;i<n;++i)for(int j=i;j<n;++j) {
          const V3 d=vec(stage_bodies[j].position)-vec(stage_bodies[i].position);
          for(const auto &image:enumerate.within(d,center_cutoff(i,j))) {
            if(i==j&&!im::positive_image(image.image))continue;
            exhaustive.push_back({i,j,image.image});
          }
        }
      }
      const auto &candidates=use_candidates?candidate_cache->candidates():exhaustive;
      for(const auto &candidate:candidates) {
            const int i=candidate.i,j=candidate.j;
            const auto pair_lengths=nh::pair_lengths(volume_radii[i],volume_radii[j],p.epsilon);
            const V3 original=vec(stage_bodies[j].position)-vec(stage_bodies[i].position);
            const V3 shift=stage_H*candidate.image.cast<double>();
            const im::Image image{candidate.image,original+shift};
            const V3 d=image.displacement;
            const double rc=center_cutoff(i,j);
            const double tolerance=64.*std::numeric_limits<double>::epsilon()*
                std::max({rc,original.norm(),shift.norm(),std::numeric_limits<double>::min()});
            if(!p.surface_dg&&d.norm()>rc+tolerance)continue;
            auto bj = stage_bodies[j];
            bj.position = arr(vec(stage_bodies[i].position) + d);
            const auto gap = exact_gap(stage_bodies[i],bj,i,j,image.image);
            SurfaceSample surface;
            if(p.cohesion_work>0. && gap.gap<cohesion_parameters.opening_range) {
              const auto attraction=SlurrySurfaceCohesion::evaluate(stage_bodies[i],bj,gap,cohesion_parameters);
              surface.cohesion=attraction;
              problem.external.segment<3>(6*i)+=vec(attraction.force_i);
              problem.external.segment<3>(6*i+3)+=vec(attraction.torque_i);
              problem.external.segment<3>(6*j)+=vec(attraction.force_j);
              problem.external.segment<3>(6*j+3)+=vec(attraction.torque_j);
              const V3 fi=vec(attraction.force_i);
              cohesion_stress+=.5*(d*fi.transpose()+fi*d.transpose());
              cohesion_energy+=attraction.energy;
              ++cohesive_pairs;
            }
            if(p.background.hamaker>0. && gap.gap<background_parameters.far_cutoff-background_parameters.reference_gap) {
              const auto attraction=SlurrySurfaceBackground::evaluate(stage_bodies[i],bj,gap,background_parameters);
              surface.background=attraction;
              background_external.segment<3>(6*i)+=vec(attraction.force_i);
              background_external.segment<3>(6*i+3)+=vec(attraction.torque_i);
              background_external.segment<3>(6*j)+=vec(attraction.force_j);
              background_external.segment<3>(6*j+3)+=vec(attraction.torque_j);
              const V3 fi=vec(attraction.force_i);
              background_stress+=.5*(d*fi.transpose()+fi*d.transpose());
              background_energy+=attraction.energy;++background_pairs;
            }

            Eigen::Matrix<double,22,22> near_resistance;
            bool near_active=false;
            if(hydro_frame) {
              // Hold only the hydrodynamic operator over this macro interval.
              // Mechanical geometry, cohesion, background and histories stay current.
            } else if(p.nearfield_mobility) {
              // Mari's gap normalization uses the mean pair radius. The nonspherical extension
              // additionally restricts the range by local surface curvature.
              const auto near=nh::evaluate(stage_bodies[i],bj,gap,1.,1.,
                                           pair_lengths.epsilon,pair_lengths.cutoff);
              near_resistance=near.resistance;near_active=near.active;
            } else {
              const auto near=cb::evaluate(stage_bodies[i],bj,gap,1.,1.,p.regularization_length/length);
              near_resistance=near.resistance;near_active=near.active;
            }
            min_gap = std::min(min_gap, gap.gap);
            overlap = std::max(overlap, -gap.gap);
            overlap_fraction=std::max(overlap_fraction,-gap.gap/
                std::min(minimum_axis(stage_bodies[i]),minimum_axis(stage_bodies[j])));
            if (near_active) {
              ++active_near;
              for (int r = 0; r < 22; ++r)
                for (int c = 0; c < 22; ++c) {
                  double value = near_resistance(r, c);
                  if (value != 0)
                    entries.emplace_back(11 * (r < 11 ? i : j) + r % 11,
                                         11 * (c < 11 ? i : j) + c % 11, value);
                }
            }
            if(sparse_contacts&&gap.gap>surface_support+.1*minimum_volume_radius) {
              screened_pairs.push_back({i,j,image.image,gap.normal});
              continue;
            }
            if(has_surface)surface_samples.push_back(surface);
            if(p.surface_dg)surface_geometry.push_back(gap);
            bs::Contact contact;
            contact.i = i;
            contact.j = j;
            contact.normal = vec(gap.normal);
            contact.gap = gap.gap;
            contact.onset = 0.; // actual surface overlap, never a positive-gap shell
            contact.kn = p.kn / (p.eta * length * rate);
            contact.kt = p.kt / (p.eta * length * rate);
            contact.mu = p.mu;
            contact.mu_roll = p.mu_roll;
            // Singh a_ij=2*a_i*a_j/(a_i+a_j), using volume-equivalent radii.
            contact.roll_length = pair_lengths.rolling_radius;
            const double torque_scale = p.eta * std::pow(length,3) * rate;
            const double physical_roll_length=contact.roll_length*length;
            contact.kr = p.kr * physical_roll_length * physical_roll_length / torque_scale;
            V3 li = .5 * (vec(gap.leverI) + d + vec(gap.leverJ)), lj = li - d;
            contact.Bi.leftCols<3>() = -M3::Identity();
            contact.Bi.rightCols<3>() = cb::cross_matrix(li);
            contact.Bj.leftCols<3>() = M3::Identity();
            contact.Bj.rightCols<3>() = -cb::cross_matrix(lj);
            contact.affine = E * d;
            PairKey key(records[i].tag, records[j].tag);
            const auto old = stage_history.find(key);
            if (old != stage_history.end()) {
              std::vector<V3> predicted;
              for (const auto &state : old->second)
                predicted.push_back(state.predicted);
              auto match = im::match_image(predicted, d, 1e-8 * std::max(1., d.norm()), i == j);
              if (match.found) {
                const auto &h = old->second[match.index];
                contact.history = vec(SlurryHistory::transport(arr(match.sign * h.elastic),
                                                               arr(match.sign * h.normal),
                                                               arr(contact.normal), arr(h.angle)));
                contact.rolling_history = vec(SlurryHistory::transport(arr(match.sign * h.rolling),
                                                               arr(match.sign * h.normal),
                                                               arr(contact.normal), arr(h.angle)));
              }
            }
            problem.contacts.push_back(contact);
            contact_keys.push_back(key);
            contact_displacements.push_back(d);
            contact_images.push_back(image.image);
      }
      if(p.background.hamaker>0.)problem.external+=background_external;
      if(hydro_frame) {
        problem.near_resistance=hydro_frame->problem.near_resistance;
        active_near=hydro_frame->active_near;
      } else {
        problem.near_resistance.resize(11 * n, 11 * n);
        problem.near_resistance.setFromTriplets(entries.begin(), entries.end());
        problem.near_resistance.makeCompressed();
      }
      double geometry_time = MPI_Wtime() - section;
      section = MPI_Wtime();
      double amax = 0;
      for (const auto &b : stage_bodies)
        for (double a : b.axes)
          amax = std::max(amax, a);
      pm::Settings settings;
      settings.backend = p.direct_mobility ? pm::Backend::Direct : pm::Backend::Mesh;
      settings.xi = p.spectral / amax;
      settings.spectral_tolerance = p.spectral_tol;
#ifdef _OPENMP
      settings.threads = omp_get_max_threads();
#endif
      auto &mobility = p.mobility;
      const char *trace_setting = std::getenv("SLURRY_BRIDGE_TRACE");
      const bool trace_solver = trace_setting && std::strcmp(trace_setting, "0");
      if (trace_solver) {
        std::fprintf(stderr, "bridge setup step=%lld particles=%d near_pairs=%d contact_candidates=%zu geometry_s=%.6g\n",
                     static_cast<long long>(update->ntimestep), n, active_near,
                     problem.contacts.size(), geometry_time);
        std::fflush(stderr);
      }
      if(p.nearfield_mobility) {
        problem.self_mobility=hydro_frame?hydro_frame->problem.self_mobility:
            ns::build_self_mobility(stage_bodies,p.hydro_workspace());
        settings.threads=1;
      } else {
        std::vector<pm::Body> stage_middle_bodies(n);
        for(int i=0;i<n;++i) {
          stage_middle_bodies[i].x=vec(stage_bodies[i].position);
          stage_middle_bodies[i].shape=cb::shape(stage_bodies[i],1.);
        }
        mobility.reset(stage_middle_bodies, stage_H, settings);
        problem.self_mobility = mobility.self_blocks();
        problem.mobility = [&mobility](const Eigen::VectorXd &x) { return mobility.apply(x); };
      }
      double middle_time = MPI_Wtime() - section;
        Frame out;
        out.problem=std::move(problem);
        out.keys=std::move(contact_keys);
        out.displacements=std::move(contact_displacements);
        out.min_gap=min_gap;out.overlap=overlap;out.overlap_fraction=overlap_fraction;out.active_near=active_near;
        out.geometry_time=geometry_time;out.middle_time=middle_time;out.settings=settings;
        out.cohesion_stress=cohesion_stress;out.cohesion_energy=cohesion_energy;
        out.cohesive_pairs=cohesive_pairs;
        out.background_stress=background_stress;out.background_energy=background_energy;
        out.background_pairs=background_pairs;out.background_external=std::move(background_external);
        out.surface_samples=std::move(surface_samples);
        out.surface_geometry=std::move(surface_geometry);
        out.screened_pairs=std::move(screened_pairs);
        out.geometry.bodies=stage_bodies;out.geometry.quaternions=stage_quaternions;
        out.geometry.lattice=stage_H;out.geometry.kappa=kappa;
        out.geometry.ambient_rotation=omega_inf;out.geometry.contact_onset=0.;
        if(p.surface_coupled||p.surface_pc||p.surface_practical) {
          out.geometry.validation_support=surface_support;
          out.geometry.candidate_cache_support=surface_support;
        }
        out.geometry.flake=false;out.geometry.gap_tolerance=1e-8*p.epsilon*minimum_volume_radius;
        out.geometry.enforce_work_balance=false;out.geometry.images=std::move(contact_images);
        if(use_candidates)out.geometry.candidate_cache=candidate_cache;
        return out;
      };
      phase="current assembly";
      auto current=assemble(bodies,H,quaternions,p.history,dt);
      auto &problem=current.problem;
      auto &geometry=current.geometry;
      auto &contact_keys=current.keys;
      auto &contact_displacements=current.displacements;
      const double min_gap=current.min_gap, overlap=current.overlap;
      const int active_near=current.active_near;
      double geometry_time=current.geometry_time,middle_time=current.middle_time;
      const auto settings=current.settings;
      auto &mobility=p.mobility;
      const char *trace_setting=std::getenv("SLURRY_BRIDGE_TRACE");
      const bool trace_solver=trace_setting && std::strcmp(trace_setting,"0");
      double section=0.;
      bs::Options options;
      options.workspace = p.solver_workspace;
      options.relative_tolerance = p.rtol;
      options.maximum_linear_iterations = p.maxit;
      options.linear_relative_tolerance = p.surface_practical?p.surface_linear:std::min(1e-6, .1 * p.rtol);
      options.max_linear_forcing = .05;
      if (trace_solver) {
        std::fprintf(stderr, "bridge mobility step=%lld backend=%s modes=%zu storage_bytes=%zu setup_s=%.6g\n",
                     static_cast<long long>(update->ntimestep),
                     p.nearfield_mobility?"nearfield":(p.direct_mobility?"direct":"mesh"),
                     p.nearfield_mobility?0:mobility.mode_count(),
                     p.nearfield_mobility?0:mobility.bytes(), middle_time);
        std::fflush(stderr);
        options.trace_callback = [&](const bs::IterationTrace &t) {
          std::fprintf(stderr,
                       "bridge solve step=%lld phase=%s stage=%d stiffness_scale=%.6g friction_scale=%.6g iteration=%d newton_total=%d residual=%.9e merit=%.9e tolerance=%.9e alpha=%.6g "
                       "linear=%d linear_total=%d linear_residual=%.6e linear_target=%.6e linear_defect=%.6e "
                       "active=%d sliding=%d reselections=%d yield_distance=%.6e descent=%d breakpoint=%d elapsed_s=%.6g\n",
                       static_cast<long long>(update->ntimestep), t.phase, t.continuation_stage,
                       t.stiffness_scale, t.friction_scale, t.iteration, t.total_newton_iterations, t.residual,
                       t.merit_residual, t.tolerance, t.step_length, t.linear_iterations, t.total_linear_iterations,
                       t.linear_residual, t.linear_target, t.linear_defect, t.active_contacts,
                       t.sliding_contacts, t.kink_reselections, t.minimum_yield_distance,
                       int(t.descent), int(t.breakpoint), t.elapsed_seconds);
          std::fflush(stderr);
        };
      }
      if(p.surface_practical||(has_surface && p.surface_adaptive>0.)) {
        namespace ad=SlurryAdaptiveStep;
        phase="adaptive coupled surface/contact integration";
        struct Payload {
          std::shared_ptr<Frame> frame;
          step::Result solved;
          imex::FrozenCaps caps;
          std::vector<SurfaceSample> endpoint_samples;
          std::vector<SurfaceSample> applied_samples;
          std::vector<SlurrySurfaceDiscreteGradient::Result> integrated_surface;
          imex::FrozenCaps initial_caps;
          M3 endpoint_cohesion_stress=M3::Zero(),endpoint_background_stress=M3::Zero();
          M3 applied_cohesion_stress=M3::Zero(),applied_background_stress=M3::Zero();
          bs::Vector endpoint_background_external;
          bs::Vector applied_background_external;
          double endpoint_cohesion_energy=0.,endpoint_background_energy=0.;
          double applied_cohesion_energy=0.,applied_background_energy=0.;
          int endpoint_cohesive_pairs=0,endpoint_background_pairs=0;
          double roundoff=0.,roundoff_fraction=0.,acceptance_ratio=0.;
        };
        struct State {
          std::vector<gr::Body> bodies;
          std::vector<Eigen::Quaterniond> quaternions;
          M3 lattice;
          std::map<PairKey,std::vector<History>> history;
          bs::Result warm;
          bool have_warm=false;
          mutable std::shared_ptr<Frame> cached_frame;
          std::shared_ptr<Payload> payload;
          double load_kn=0.; // N/m that produced these overlaps (frozen-cap load)
        };
        State initial{bodies,quaternions,H,p.history,p.warm,p.have_warm,
                      std::make_shared<Frame>(current),{},p.kn_load};
        const double kn_unit=p.eta*length*rate; // N/m per nondimensional stiffness
        const double stress_scale=p.eta*rate/volume;
        const double energy_scale=p.eta*std::pow(length,3)*rate;
        const double power_scale=energy_scale*rate,force_scale=p.eta*length*length*rate;
        const double fraction=p.surface_adaptive,work_fraction=.5*fraction*fraction;
        double solve_seconds=0.;
        p.diag.fill(0);p.stress.fill(0.);
        p.diag[4]=p.diag[27]=std::numeric_limits<double>::infinity();
        p.imex_workspace.maximum_roundoff_tolerance=0.;
        auto attempt=[&](const State &state,double sub_dt)->ad::Trial<State> {
          if(!state.cached_frame) {
            auto frame=assemble(state.bodies,state.lattice,state.quaternions,state.history,sub_dt,
                p.surface_lagged_hydro&&p.nearfield_mobility?&current:nullptr);
            geometry_time+=frame.geometry_time;middle_time+=frame.middle_time;
            state.cached_frame=std::make_shared<Frame>(std::move(frame));
          }
          auto data=std::make_shared<Payload>();
          // Old-force paths only change this numerical cache's trial dt.
          // Their physical geometry/history/external loads remain immutable;
          // endpoint/PC paths retain a private copy because they change loads.
          data->frame=(p.surface_coupled||p.surface_pc)?
              std::make_shared<Frame>(*state.cached_frame):state.cached_frame;
          auto &f=*data->frame;auto &pb=f.problem;pb.dt=sub_dt;
          if(p.kn_control) {
            // The cached frame may predate a stiffness change. Springs use the
            // current kn; the frozen Coulomb load is the PHYSICAL load of the
            // committed overlaps, load_kn*overlap, never kn*overlap.
            for(auto &c:pb.contacts)c.kn=p.kn/kn_unit;
            data->caps=imex::freeze_caps(pb);
            if(state.load_kn!=p.kn)for(auto &cap:data->caps)cap*=state.load_kn/p.kn;
          } else data->caps=imex::freeze_caps(pb);
          auto endpoint_surface=[&](const step::Endpoint&end) {
            SlurrySurfaceCoupledStep::SurfaceEvaluation answer;
            answer.external=bs::Vector::Zero(6*n);
            answer.attraction_gap_derivative.assign(pb.contacts.size(),0.);
            data->endpoint_samples.assign(pb.contacts.size(),{});
            data->endpoint_cohesion_stress.setZero();data->endpoint_background_stress.setZero();
            data->endpoint_background_external=bs::Vector::Zero(6*n);
            data->endpoint_cohesion_energy=data->endpoint_background_energy=0.;
            data->endpoint_cohesive_pairs=data->endpoint_background_pairs=0;
            for(std::size_t k=0;k<pb.contacts.size();++k) {
              const auto&c=pb.contacts[k];auto bj=end.bodies[c.j];
              const V3 d=vec(bj.position)-vec(end.bodies[c.i].position)+
                  end.lattice*f.geometry.images[k].cast<double>();
              bj.position=arr(vec(end.bodies[c.i].position)+d);
              if(end.gaps[k]>=surface_support)continue;
              const auto gap=end.pair_geometry.size()==pb.contacts.size()?end.pair_geometry[k]:
                  exact_gap(end.bodies[c.i],bj,c.i,c.j,f.geometry.images[k]);
              auto&sample=data->endpoint_samples[k];
              if(p.cohesion_work>0.&&gap.gap<cohesion_parameters.opening_range) {
                sample.cohesion=SlurrySurfaceCohesion::evaluate(end.bodies[c.i],bj,gap,cohesion_parameters);
                const auto&a=sample.cohesion;
                answer.external.segment<3>(6*c.i)+=vec(a.force_i);
                answer.external.segment<3>(6*c.i+3)+=vec(a.torque_i);
                answer.external.segment<3>(6*c.j)+=vec(a.force_j);
                answer.external.segment<3>(6*c.j+3)+=vec(a.torque_j);
                data->endpoint_cohesion_stress+=.5*(d*vec(a.force_i).transpose()+vec(a.force_i)*d.transpose());
                data->endpoint_cohesion_energy+=a.energy;++data->endpoint_cohesive_pairs;
                if(gap.gap>0.)answer.attraction_gap_derivative[k]-=
                    a.attraction_at_contact/cohesion_parameters.opening_range;
              }
              if(p.background.hamaker>0.&&gap.gap<background_parameters.far_cutoff-background_parameters.reference_gap) {
                sample.background=SlurrySurfaceBackground::evaluate(end.bodies[c.i],bj,gap,background_parameters);
                const auto&a=sample.background;
                data->endpoint_background_external.segment<3>(6*c.i)+=vec(a.force_i);
                data->endpoint_background_external.segment<3>(6*c.i+3)+=vec(a.torque_i);
                data->endpoint_background_external.segment<3>(6*c.j)+=vec(a.force_j);
                data->endpoint_background_external.segment<3>(6*c.j+3)+=vec(a.torque_j);
                data->endpoint_background_stress+=.5*(d*vec(a.force_i).transpose()+vec(a.force_i)*d.transpose());
                data->endpoint_background_energy+=a.energy;++data->endpoint_background_pairs;
                const double h=gap.gap+background_parameters.reference_gap;
                // Stiff near-gap tangent only; accepted forces retain all
                // curvature and switching derivatives of the original law.
                if(gap.gap>0.&&h<background_parameters.curvature_switch) {
                  const double R=a.attraction_at_contact/(2.*std::acos(-1.)*
                      SlurrySurfaceBackground::work_of_separation(background_parameters));
                  answer.attraction_gap_derivative[k]+=2.*R*background_parameters.hamaker*
                      (-1./(6.*h*h*h)+std::pow(background_parameters.sigma/h,6)/(45.*h*h*h));
                }
              }
            }
            answer.external+=data->endpoint_background_external;
            return answer;
          };
          auto integrated_surface=[&](const step::Endpoint&end) {
            namespace dg=SlurrySurfaceDiscreteGradient;
            auto answer=endpoint_surface(end);
            answer.external.setZero();
            data->applied_samples.assign(pb.contacts.size(),{});
            data->integrated_surface.assign(pb.contacts.size(),{});
            data->applied_cohesion_stress.setZero();data->applied_background_stress.setZero();
            data->applied_background_external=bs::Vector::Zero(6*n);
            for(std::size_t k=0;k<pb.contacts.size();++k) {
              const auto&c=pb.contacts[k];
              dg::Path path;
              path.initial_i=state.bodies[c.i];path.initial_j=state.bodies[c.j];
              path.initial_j.position=arr(vec(path.initial_i.position)+f.displacements[k]);
              path.final_i=end.bodies[c.i];path.final_j=end.bodies[c.j];
              const V3 end_d=vec(path.final_j.position)-vec(path.final_i.position)+
                  end.lattice*f.geometry.images[k].cast<double>();
              path.final_j.position=arr(vec(path.final_i.position)+end_d);
              path.relative_displacement=arr(end_d-f.displacements[k]);
              path.rotation_i=arr(end.rotation_increments[c.i]);
              path.rotation_j=arr(end.rotation_increments[c.j]);
              const dg::Sample initial_sample{f.surface_geometry[k],
                  f.surface_samples[k].cohesion,f.surface_samples[k].background};
              const dg::Sample endpoint_sample{end.pair_geometry[k],
                  data->endpoint_samples[k].cohesion,data->endpoint_samples[k].background};
              auto&r=data->integrated_surface[k];
              r=dg::evaluate(path,cohesion_parameters,background_parameters,{},
                             &initial_sample,&endpoint_sample);
              p.diag[67]+=r.evaluations;
              p.diag[69]=std::max(p.diag[69],double(r.panels));
              for(const auto*a:{&r.cohesion,&r.background})if(a->work_scale>0.)
                p.diag[68]=std::max(p.diag[68],std::abs(a->work_defect)/a->work_scale);
              auto accumulate=[&](const dg::KernelIntegral&a,auto&sample,M3&stress,bs::Vector&external) {
                sample.force_i=a.force_i;sample.force_j=a.force_j;
                sample.torque_i=a.torque_i;sample.torque_j=a.torque_j;
                sample.active=a.average_energy!=0.;
                external.segment<3>(6*c.i)+=vec(a.force_i);
                external.segment<3>(6*c.i+3)+=vec(a.torque_i);
                external.segment<3>(6*c.j)+=vec(a.force_j);
                external.segment<3>(6*c.j+3)+=vec(a.torque_j);
                for(int row=0;row<3;++row)for(int col=0;col<3;++col)
                  stress(row,col)+=a.symmetric_virial[3*row+col];
              };
              accumulate(r.cohesion,data->applied_samples[k].cohesion,
                  data->applied_cohesion_stress,answer.external);
              accumulate(r.background,data->applied_samples[k].background,
                  data->applied_background_stress,data->applied_background_external);
              answer.attraction_gap_derivative[k]=r.cohesion.approximate_normal_tangent+
                  r.background.approximate_normal_tangent;
            }
            answer.external+=data->applied_background_external;
            return answer;
          };
          auto surface_force=[&](const step::Endpoint&end) {
            return p.surface_dg?integrated_surface(end):endpoint_surface(end);
          };
          const double solve_start=MPI_Wtime();
          int predictor_newton=0,predictor_linear=0;
          for(int stiffen=0;;++stiffen) {
          try {
            if(p.surface_coupled) {
              auto coupled=SlurrySurfaceCoupledStep::solve(pb,f.geometry,options,p.imex_workspace,
                  surface_force,state.have_warm?&state.warm:nullptr,&p.coupled_workspace);
              data->solved=std::move(coupled.solved);data->caps=std::move(coupled.caps);
              surface_force(data->solved.endpoint);
              pb.external=std::move(coupled.external);
            } else if(p.surface_pc) {
              const auto prediction=nstep::solve_imex(pb,data->caps,f.geometry,options,p.imex_workspace,
                  state.have_warm?&state.warm:nullptr);
              predictor_newton=prediction.solution.newton_iterations;
              predictor_linear=prediction.solution.linear_iterations;
              const auto predicted_surface=endpoint_surface(prediction.endpoint);
              data->initial_caps=data->caps;
              data->applied_samples=f.surface_samples;
              auto average=[](auto&dst,const auto&src) {
                dst.attraction_at_contact=std::max(dst.attraction_at_contact,src.attraction_at_contact);
                dst.active=dst.active||src.active;
                for(int d=0;d<3;++d) {
                  dst.force_i[d]=.5*(dst.force_i[d]+src.force_i[d]);
                  dst.force_j[d]=.5*(dst.force_j[d]+src.force_j[d]);
                  dst.torque_i[d]=.5*(dst.torque_i[d]+src.torque_i[d]);
                  dst.torque_j[d]=.5*(dst.torque_j[d]+src.torque_j[d]);
                }
              };
              for(std::size_t k=0;k<pb.contacts.size();++k) {
                average(data->applied_samples[k].cohesion,data->endpoint_samples[k].cohesion);
                average(data->applied_samples[k].background,data->endpoint_samples[k].background);
                data->caps[k]=.5*(data->caps[k]+prediction.solution.contacts[k].normal_load);
              }
              data->applied_cohesion_stress=.5*(f.cohesion_stress+data->endpoint_cohesion_stress);
              data->applied_background_stress=.5*(f.background_stress+data->endpoint_background_stress);
              data->applied_background_external=.5*(f.background_external+data->endpoint_background_external);
              data->applied_cohesion_energy=.5*(f.cohesion_energy+data->endpoint_cohesion_energy);
              data->applied_background_energy=.5*(f.background_energy+data->endpoint_background_energy);
              pb.external=.5*(pb.external+predicted_surface.external);
              data->solved=nstep::solve_imex(pb,data->caps,f.geometry,options,p.imex_workspace,&prediction.solution);
              data->solved.solution.newton_iterations+=prediction.solution.newton_iterations;
              data->solved.solution.linear_iterations+=prediction.solution.linear_iterations;
              data->solved.solution.descent_steps+=prediction.solution.descent_steps;
              data->solved.geometry_iterations+=prediction.geometry_iterations;
              endpoint_surface(data->solved.endpoint);
            } else data->solved=nstep::solve_imex(pb,data->caps,f.geometry,options,p.imex_workspace,
                                                state.have_warm?&state.warm:nullptr);
          } catch(const bs::NonlinearFailure &e) {
            solve_seconds+=MPI_Wtime()-solve_start;
            p.diag[8]+=e.linear_iterations+predictor_linear;
            p.diag[9]+=e.newton_iterations+predictor_newton;
            return ad::Trial<State>::retry(e.what());
          } catch(const std::runtime_error &e) {
            solve_seconds+=MPI_Wtime()-solve_start;
            p.diag[8]+=predictor_linear;p.diag[9]+=predictor_newton;
            if(std::string(e.what()).find("surface AVF ")!=std::string::npos ||
               std::string(e.what()).find("endpoint correction did not converge")!=std::string::npos ||
               std::string(e.what()).find("IMEX: convex Newton limit")!=std::string::npos ||
               std::string(e.what()).find("bridge endpoint gained an unlisted contact; reduce timestep")!=std::string::npos)
              return ad::Trial<State>::retry(e.what());
            throw;
          }
          if(!p.kn_control)break;
          // Overlap gate BEFORE any dt reduction: a quasi-static endpoint
          // overlap is set by force balance (N/kn), so only a stiffer spring
          // can remove it. Re-solve the same trial with the same physical
          // frozen caps; a stiffness at its bound falls through to rejection.
          double overlap_ratio=0.;
          for(std::size_t k=0;k<pb.contacts.size();++k) {
            const auto &c=pb.contacts[k];
            overlap_ratio=std::max(overlap_ratio,std::max(0.,-data->solved.endpoint.gaps[k])/
                std::min(minimum_axis(state.bodies[c.i]),minimum_axis(state.bodies[c.j])));
          }
          const double stiffer=SlurryStiffnessControl::retry_stiffness(p.kn_state,p.kn_settings,overlap_ratio);
          if(!(stiffer>p.kn)||stiffen>=p.kn_settings.max_retries)break;
          p.diag[8]+=data->solved.solution.linear_iterations;
          p.diag[9]+=data->solved.solution.newton_iterations;
          std::fprintf(stderr,"slurry/bridge kn_adapt: step=%lld retry substep_dt=%.6g s max_overlap/min_semiaxis=%.6g kn %.6g -> %.6g N/m\n",
                       static_cast<long long>(update->ntimestep),sub_dt/rate,overlap_ratio,p.kn,stiffer);
          std::fflush(stderr);
          p.kn=p.kn_state.current=stiffer;++p.kn_state.retries;
          for(auto &c:pb.contacts)c.kn=p.kn/kn_unit;
          p.imex_workspace.contact_preconditioner_refresh_pending=true;
          }
          solve_seconds+=MPI_Wtime()-solve_start;
          const auto &sol=data->solved.solution;
          const auto &end=data->solved.endpoint;
          p.diag[8]+=sol.linear_iterations;p.diag[9]+=sol.newton_iterations;
          p.diag[20]+=sol.descent_steps;p.diag[21]+=sol.continuation_stages;
          p.diag[22]+=sol.continuation_retries;p.diag[23]+=sol.mobility_applications;
          p.diag[24]+=sol.breakpoint_steps;
          // DG retains every buffered candidate. The cache certificate at
          // both endpoints covers the entire linear center/lattice path, so
          // an omitted pair cannot enter and leave support invisibly.
          if(p.surface_dg&&!candidate_cache->valid(end.bodies,end.lattice))
            return ad::Trial<State>::retry("surface AVF path exceeded its candidate buffer");
          data->roundoff=p.imex_workspace.last_roundoff_tolerance;
          data->acceptance_ratio=p.imex_workspace.last_acceptance_ratio;
          if(data->roundoff>0.)
            data->roundoff_fraction=data->roundoff/std::max(
                p.imex_workspace.last_acceptance_tolerance,std::numeric_limits<double>::min());
          // A screened pair must remain outside ALL surface-force support at
          // the accepted endpoint. Otherwise retry; the buffer includes it in
          // a later assembly before force engagement. HI was never screened.
          const double screened_start=MPI_Wtime();
          for(const auto &pair:f.screened_pairs) {
            auto bj=end.bodies[pair.j];
            bj.position=arr(vec(bj.position)+end.lattice*pair.image.cast<double>());
            if(SlurrySupportBound::separated(end.bodies[pair.i],bj,pair.normal,surface_support))continue;
            const auto gap=exact_gap(end.bodies[pair.i],bj,pair.i,pair.j,pair.image);
            if(gap.gap<surface_support) {
              geometry_time+=MPI_Wtime()-screened_start;
              return ad::Trial<State>::retry("screened pair entered surface support; reduce timestep");
            }
          }
          geometry_time+=MPI_Wtime()-screened_start;
          State next;next.bodies=end.bodies;next.lattice=end.lattice;
          next.quaternions.resize(n);next.have_warm=true;next.warm=sol;next.payload=data;
          next.load_kn=p.kn_control?p.kn:state.load_kn;
          for(int i=0;i<n;++i)next.quaternions[i]=step::rotate(state.quaternions[i],
              sol.velocity.segment<3>(6*i+3)+omega_inf,sub_dt);
          double error_ratio=0.,cap_work_sum=0.,cap_scale_sum=0.;
          const double monitor_start=MPI_Wtime();
          for(std::size_t k=0;k<pb.contacts.size();++k) {
            const auto &c=pb.contacts[k];const auto &response=sol.contacts[k];
            const V3 end_d=vec(end.bodies[c.j].position)-vec(end.bodies[c.i].position)+
                end.lattice*f.geometry.images[k].cast<double>();
            if(response.active) {
              History h;h.elastic=response.tangent_history;h.rolling=response.rolling_history;
              h.normal=c.normal;
              h.angle=.5*sub_dt*(sol.velocity.segment<3>(6*c.i+3)+
                                    sol.velocity.segment<3>(6*c.j+3)+2.*omega_inf);
              h.predicted=end_d;next.history[f.keys[k]].push_back(h);
            }
            if(p.surface_practical) {
              // No pair-force, pair-work or cap-lag accuracy gate. Only a
              // resolved physical overlap above the declared model limit can
              // reject an otherwise converged macro step here.
              error_ratio=std::max(error_ratio,std::max(0.,-end.gaps[k])/
                  (.05*std::min(minimum_axis(state.bodies[c.i]),minimum_axis(state.bodies[c.j]))));
              continue;
            }
            if(p.surface_dg) {
              // Quadrature tests the work of the full potential along this
              // interval. Endpoint rectangular-force work is not an error
              // estimate for an interval-averaged applied force.
              const auto&r=data->integrated_surface[k];
              for(const auto*a:{&r.cohesion,&r.background})if(a->work_scale>0.)
                error_ratio=std::max(error_ratio,std::abs(a->work_defect)/(work_fraction*a->work_scale));
              continue;
            }
            const auto &old=f.surface_samples[k];
            const bool predicted_cohesion=p.surface_pc&&data->applied_samples[k].cohesion.active;
            const bool predicted_background=p.surface_pc&&data->applied_samples[k].background.active;
            const bool possible_cohesion=p.cohesion_work>0. &&
                (predicted_cohesion||std::min(c.gap,end.gaps[k])<cohesion_parameters.opening_range);
            const bool possible_background=p.background.hamaker>0. &&
                (predicted_background||std::min(c.gap,end.gaps[k])<background_parameters.far_cutoff-background_parameters.reference_gap);
            if(!possible_cohesion&&!possible_background)continue;
            auto bj=end.bodies[c.j];bj.position=arr(vec(end.bodies[c.i].position)+end_d);
            const auto end_gap=exact_gap(end.bodies[c.i],bj,c.i,c.j,f.geometry.images[k]);
            SurfaceSample after=(p.surface_coupled||p.surface_pc)?data->endpoint_samples[k]:SurfaceSample{};
            if(possible_cohesion&&!p.surface_coupled&&!p.surface_pc)after.cohesion=SlurrySurfaceCohesion::evaluate(
                end.bodies[c.i],bj,end_gap,cohesion_parameters);
            if(possible_background&&!p.surface_coupled&&!p.surface_pc)after.background=SlurrySurfaceBackground::evaluate(
                end.bodies[c.i],bj,end_gap,background_parameters);
            const double cohesive_scale=std::max({old.cohesion.attraction_at_contact,
                after.cohesion.attraction_at_contact,p.surface_pc?data->applied_samples[k].cohesion.attraction_at_contact:0.});
            const double background_scale=std::max({old.background.attraction_at_contact,
                after.background.attraction_at_contact,p.surface_pc?data->applied_samples[k].background.attraction_at_contact:0.});
            const double normal_scale=cohesive_scale+background_scale;
            const double binding_scale=cohesive_scale*cohesion_parameters.opening_range+
                background_scale*background_parameters.reference_gap;
            if(!(normal_scale>0.)||!(binding_scale>0.))continue;
            const V3 old_f=vec(old.cohesion.force_i)+vec(old.background.force_i);
            const V3 new_f=vec(after.cohesion.force_i)+vec(after.background.force_i);
            const V3 old_ti=vec(old.cohesion.torque_i)+vec(old.background.torque_i);
            const V3 old_tj=vec(old.cohesion.torque_j)+vec(old.background.torque_j);
            const V3 new_ti=vec(after.cohesion.torque_i)+vec(after.background.torque_i);
            const V3 new_tj=vec(after.cohesion.torque_j)+vec(after.background.torque_j);
            const double old_u=old.cohesion.energy+old.background.energy;
            const double new_u=after.cohesion.energy+after.background.energy;
            // Total surface torque includes the normal-force lever arm, whose
            // scale is F0*a. The separate binding-scale work test below still
            // resolves the much smaller curvature-dependent binding energy.
            const double torque_scale=std::max({binding_scale,std::abs(old_u),std::abs(new_u),
                normal_scale*std::max(shape::bounding_radius(state.bodies[c.i],false),
                                      shape::bounding_radius(state.bodies[c.j],false))});
            if(!p.surface_coupled&&!p.surface_pc) {
              error_ratio=std::max(error_ratio,(new_f-old_f).norm()/(fraction*normal_scale));
              error_ratio=std::max(error_ratio,(new_ti-old_ti).norm()/(fraction*torque_scale));
              error_ratio=std::max(error_ratio,(new_tj-old_tj).norm()/(fraction*torque_scale));
            }
            const V3 theta_i=sub_dt*(sol.velocity.segment<3>(6*c.i+3)+omega_inf);
            const V3 theta_j=sub_dt*(sol.velocity.segment<3>(6*c.j+3)+omega_inf);
            V3 work_f=p.surface_coupled?new_f:old_f;
            V3 work_ti=p.surface_coupled?new_ti:old_ti;
            V3 work_tj=p.surface_coupled?new_tj:old_tj;
            if(p.surface_pc) {
              const auto&a=data->applied_samples[k];
              work_f=vec(a.cohesion.force_i)+vec(a.background.force_i);
              work_ti=vec(a.cohesion.torque_i)+vec(a.background.torque_i);
              work_tj=vec(a.cohesion.torque_j)+vec(a.background.torque_j);
              const double corrected_cap=.5*(data->initial_caps[k]+response.normal_load);
              const auto corrected=imex::contact_response(c,corrected_cap,sol.velocity,sub_dt,end.gaps[k]);
              const double cap_work=sub_dt*(std::abs((corrected.tangent_force-response.tangent_force).dot(response.relative))+
                  std::abs((corrected.torque-response.torque).dot(response.relative_rotation)));
              const double cap_scale=binding_scale+response.normal_energy+response.tangent_energy+
                  response.rolling_energy+corrected.tangent_energy+corrected.rolling_energy+
                  sub_dt*(std::abs(response.tangent_force.dot(response.relative))+
                  std::abs(response.torque.dot(response.relative_rotation)));
              // Sum absolute defects over mechanically relevant pairs. Dormant
              // background pairs have hypothetical binding scales, not stored
              // contact energy, and must not dilute the friction-cap error.
              const bool cap_relevant=data->initial_caps[k]>0.||data->caps[k]>0.||
                  response.normal_load>0.||c.history.squaredNorm()>0.||
                  c.rolling_history.squaredNorm()>0.||cap_work>0.;
              if(cap_relevant) {cap_work_sum+=cap_work;cap_scale_sum+=cap_scale;}
            }
            const double work_defect=std::abs(new_u-old_u-work_f.dot(end_d-f.displacements[k])+
                                            work_ti.dot(theta_i)+work_tj.dot(theta_j));
            error_ratio=std::max(error_ratio,work_defect/(work_fraction*binding_scale));
            // Resolve the load which sets the frozen friction cap as well as
            // adhesion itself; a saturated attraction alone misses detachment.
            const double normal_old=data->caps[k],normal_new=response.normal_load;
            if(!p.surface_coupled&&!p.surface_pc)error_ratio=std::max(error_ratio,std::abs(normal_new-normal_old)/
                (fraction*std::max({normal_scale,normal_old,normal_new})));
          }
          if(p.surface_pc&&cap_scale_sum>0.)
            error_ratio=std::max(error_ratio,cap_work_sum/(fraction*cap_scale_sum));
          geometry_time+=MPI_Wtime()-monitor_start;
          return ad::Trial<State>::completed(std::move(next),error_ratio);
        };
        auto accepted=[&](const ad::Trial<State>&trial,double sub_dt) {
          const auto &data=*trial.next->payload;const auto &f=*data.frame;
          const auto &sol=data.solved.solution;const double weight=sub_dt/dt;
          M3 hydro=M3::Zero(),contact_stress=M3::Zero();
          bs::Vector contact_moments=bs::Vector::Zero(6*n);
          double normal_energy=0.,tangent_energy=0.,rolling_energy=0.,rolling_work=0.,rotation_power=0.;
          double lag_max=0.,lag_sum=0.,normal_old_sum=0.,normal_new_sum=0.,detached=0.;
          int active=0,rolling_yield=0;
          for(int i=0;i<n;++i)for(int k=0;k<5;++k)
            hydro-=sol.total_moment[11*i+6+k]*cb::strain_basis()[k];
          for(std::size_t k=0;k<f.problem.contacts.size();++k) {
            const auto &c=f.problem.contacts[k];const auto &r=sol.contacts[k];
            const V3 d=p.surface_coupled?
                vec(data.solved.endpoint.bodies[c.j].position)-vec(data.solved.endpoint.bodies[c.i].position)+
                data.solved.endpoint.lattice*f.geometry.images[k].cast<double>():f.displacements[k];
            contact_stress+=.5*(d*r.traction.transpose()+r.traction*d.transpose());
            contact_moments.segment<6>(6*c.i)-=bs::generalized_contact_map(c,true).transpose()*bs::contact_wrench(r);
            contact_moments.segment<6>(6*c.j)-=bs::generalized_contact_map(c,false).transpose()*bs::contact_wrench(r);
            normal_energy+=r.normal_energy;tangent_energy+=r.tangent_energy;rolling_energy+=r.rolling_energy;
            rolling_work+=r.rolling_work;rotation_power+=r.torque.dot(r.relative_rotation);
            active+=r.active;rolling_yield+=r.rolling;
            const double lag=std::abs(data.caps[k]-r.normal_load);
            lag_max=std::max(lag_max,lag);lag_sum+=lag;
            normal_old_sum+=data.caps[k];normal_new_sum+=r.normal_load;
            p.diag[49]=std::max(p.diag[49],(r.tangent_force.norm()-c.mu*r.normal_load)*force_scale);
            const V3 rolling=r.torque-c.normal*c.normal.dot(r.torque);
            p.diag[50]=std::max(p.diag[50],(rolling.norm()-c.mu_roll*c.roll_length*r.normal_load)*energy_scale);
            if(!r.active)detached+=r.tangent_energy+r.rolling_energy+r.twisting_energy;
            p.diag[27]=std::min(p.diag[27],data.solved.endpoint.gaps[k]*length);
            p.diag[45]=std::max(p.diag[45],-data.solved.endpoint.gaps[k]/
                std::min(minimum_axis(f.geometry.bodies[c.i]),minimum_axis(f.geometry.bodies[c.j])));
          }
          const M3&cohesion_stress=(p.surface_pc||p.surface_dg)?data.applied_cohesion_stress:
              (p.surface_coupled?data.endpoint_cohesion_stress:f.cohesion_stress);
          const M3&background_stress=(p.surface_pc||p.surface_dg)?data.applied_background_stress:
              (p.surface_coupled?data.endpoint_background_stress:f.background_stress);
          const auto&background_external=(p.surface_pc||p.surface_dg)?data.applied_background_external:
              (p.surface_coupled?data.endpoint_background_external:f.background_external);
          const double sigma_c=cohesion_stress(0,2)*stress_scale;
          const double sigma_b=background_stress(0,2)*stress_scale;
          const std::array<double,4> stress{p.eta*p.shear,hydro(0,2)*stress_scale,
            contact_stress(0,2)*stress_scale,
            p.eta*p.shear+(hydro(0,2)+contact_stress(0,2))*stress_scale+sigma_c+sigma_b};
          for(int k=0;k<4;++k)p.stress[k]+=weight*stress[k];
          const double hp=(volume*std::pow(p.shear/rate,2)+sol.hydro_power)*power_scale;
          const double cp=sol.contact_power*power_scale;
          const double sp=stress[3]*p.shear*volume*std::pow(length,3);
          const double bp=(-background_external.dot(sol.velocity)+(background_stress.array()*E.array()).sum())*power_scale;
          const double ap=(-(f.problem.external-background_external).dot(sol.velocity)+
                           (cohesion_stress.array()*E.array()).sum())*power_scale;
          p.diag[2]=std::max(p.diag[2],double(f.active_near));p.diag[3]=std::max(p.diag[3],double(active));
          p.diag[4]=std::min(p.diag[4],f.min_gap*length);p.diag[5]=std::max(p.diag[5],f.overlap*length);
          p.diag[6]=std::max(p.diag[6],sol.force_residual_norm);p.diag[7]=std::max(p.diag[7],sol.mobility_residual_norm);
          p.diag[10]+=weight*hp;p.diag[11]+=weight*cp;p.diag[12]+=weight*sp;p.diag[13]+=weight*(sp-hp-cp-ap-bp);
          p.diag[15]=std::max(p.diag[15],sol.relative_residual);
          p.diag[25]+=data.solved.geometry_iterations;
          p.diag[26]=std::max(p.diag[26],data.solved.maximum_gap_error*length);
          p.diag[29]=normal_energy*energy_scale;p.diag[30]=tangent_energy*energy_scale;
          p.diag[31]=std::max(p.diag[31],data.solved.maximum_linear_gap_defect*length);
          p.diag[32]=std::max(p.diag[32],double(f.problem.contacts.size()));
          p.diag[40]=std::max(p.diag[40],double(rolling_yield));p.diag[41]=rolling_energy*energy_scale;
          p.diag[42]+=rolling_work/dt*power_scale;p.diag[43]+=weight*rotation_power*power_scale;
          p.diag[44]=std::max(p.diag[44],f.overlap_fraction);
          p.diag[47]=std::max(p.diag[47],lag_max*force_scale);
          p.diag[48]=std::max(p.diag[48],lag_sum/std::max({normal_old_sum,normal_new_sum,std::numeric_limits<double>::min()}));
          p.diag[51]+=detached*energy_scale;
          p.diag[52]+=weight*sigma_c;p.diag[53]+=weight*(p.surface_pc?data.applied_cohesion_energy:
              (p.surface_coupled?data.endpoint_cohesion_energy:f.cohesion_energy))*energy_scale;
          p.diag[54]+=weight*ap;p.diag[55]=std::max(p.diag[55],double(p.surface_coupled?data.endpoint_cohesive_pairs:f.cohesive_pairs));
          p.diag[56]+=weight*sigma_b;p.diag[57]+=weight*(p.surface_pc?data.applied_background_energy:
              (p.surface_coupled?data.endpoint_background_energy:f.background_energy))*energy_scale;
          p.diag[58]+=weight*bp;p.diag[59]=std::max(p.diag[59],double(p.surface_coupled?data.endpoint_background_pairs:f.background_pairs));
          p.diag[64]=std::max(p.diag[64],data.roundoff);p.diag[65]=std::max(p.diag[65],data.roundoff_fraction);
          p.diag[66]=std::max(p.diag[66],data.acceptance_ratio);
          if(p.surface_dg)for(const auto&r:data.integrated_surface)
            for(const auto*a:{&r.cohesion,&r.background}) {
              p.diag[70]+=(a->endpoint_energy-a->initial_energy)/dt*power_scale;
              p.diag[71]+=a->work/dt*power_scale;
            }
          // Per-atom values describe the last accepted internal stage; interval
          // stress and powers above are the time-weighted macrostep averages.
          for(int i=0;i<n;++i) {
            physical.segment<3>(6*i)=length*rate*(sol.velocity.segment<3>(6*i)+kappa*vec(f.geometry.bodies[i].position));
            physical.segment<3>(6*i+3)=rate*(sol.velocity.segment<3>(6*i+3)+omega_inf);
            moments.segment<3>(17*i)=-force_scale*sol.total_moment.segment<3>(11*i);
            moments.segment<3>(17*i+3)=-energy_scale*sol.total_moment.segment<3>(11*i+3);
            moments.segment<5>(17*i+6)=energy_scale*sol.total_moment.segment<5>(11*i+6);
            moments.segment<3>(17*i+11)=force_scale*contact_moments.segment<3>(6*i);
            moments.segment<3>(17*i+14)=energy_scale*contact_moments.segment<3>(6*i+3);
          }
        };
        ad::Options adaptive_options;
        adaptive_options.initial_step=p.surface_practical?dt:
            (p.suggested_surface_step>0.?std::min(dt,p.suggested_surface_step):dt);
        adaptive_options.maximum_step=dt;
        adaptive_options.minimum_step=std::ldexp(dt,p.surface_practical?-6:-30);
        adaptive_options.maximum_attempts=p.surface_practical?128:20000;
        auto advanced=ad::advance(std::move(initial),dt,adaptive_options,attempt,accepted);
        p.suggested_surface_step=advanced.suggested_step;
        p.pending=std::move(advanced.state.history);p.warm=std::move(advanced.state.warm);p.have_warm=true;
        p.diag[0]=p.stress[3];p.diag[1]=p.shear?p.stress[3]/p.shear:std::numeric_limits<double>::quiet_NaN();
        p.diag[16]=geometry_time;p.diag[17]=middle_time;p.diag[18]=solve_seconds;
        p.diag[19]=MPI_Wtime()-begun;p.diag[28]=std::max(0.,-p.diag[27]);
        p.diag[33]=6*n;p.diag[35]=1.;p.diag[36]=p.hydro_workspace().symbolic_analyses+
            (p.surface_coupled?p.coupled_workspace.symbolic_analyses:0);
        p.diag[37]=p.hydro_workspace().numeric_factorizations+
            (p.surface_coupled?p.coupled_workspace.numeric_factorizations:0);
        p.diag[38]=p.hydro_workspace().numeric_reuses;
        p.diag[46]=minimum_semiaxis*length;p.diag[60]=advanced.statistics.accepted;
        p.diag[61]=advanced.statistics.rejected;p.diag[62]=advanced.statistics.minimum_accepted_step/rate;
        p.diag[63]=advanced.statistics.maximum_accepted_error;
        // The committed overlaps belong to the stiffness of the last accepted
        // trial; the next step's frozen caps take their load from it.
        p.kn_load=advanced.state.load_kn;
        p.diag[72]=p.kn_load;
        if(p.kn_control) {
          const double before=p.kn;
          if(SlurryStiffnessControl::after_step(p.kn_state,p.kn_settings,p.diag[45],
                                                std::abs(p.shear)*update->dt)) {
            p.kn=p.kn_state.current;
            p.imex_workspace.contact_preconditioner_refresh_pending=true;
            std::fprintf(stderr,"slurry/bridge kn_adapt: step=%lld %s max_overlap/min_semiaxis=%.6g kn %.6g -> %.6g N/m\n",
                         static_cast<long long>(update->ntimestep),p.kn>before?"raise":"relax",
                         p.diag[45],before,p.kn);
            std::fflush(stderr);
          }
        }
        p.diag[73]=double(p.kn_state.retries);
        p.diag[74]=double(p.kn_state.raises+p.kn_state.lowers);
        direct_advance=1;advance_physical=physical;
        for(int i=0;i<n;++i) {
          for(int k=0;k<3;++k)direct_endpoint[7*i+k]=domain->boxlo[k]+length*advanced.state.bodies[i].position[k];
          const auto &q=advanced.state.quaternions[i];
          direct_endpoint[7*i+3]=q.w();direct_endpoint[7*i+4]=q.x();
          direct_endpoint[7*i+5]=q.y();direct_endpoint[7*i+6]=q.z();
        }
      } else {
      section = MPI_Wtime();
      step::Result consistent;
      Eigen::VectorXd integration_velocity;
      bs::Result next_warm;
      std::map<PairKey,std::vector<History>> corrected_history;
      double corrected_rolling_work=0., staged_assembly_time=0.;
      double normal_cap_lag=0.,normal_cap_lag_sum=0.,old_normal_sum=0.,end_normal_sum=0.,
             sliding_cap_excess=0.,rolling_cap_excess=0.,detached_release=0.;
      int corrected_rolling_yield=0;
      double endpoint_min=std::numeric_limits<double>::infinity(), endpoint_overlap_fraction=0.;
      if(p.nearfield_mobility) {
        // One implicit elastic step: the accepted contact responses already
        // contain the next spring histories. Never advance them a second time.
        imex::FrozenCaps caps;
        if(p.imex_contacts) {
          phase="frozen-cap IMEX force/endpoint solve";
          caps=imex::freeze_caps(problem);
          const auto recoveries_before=p.imex_workspace.nonlinear_recoveries;
          const auto recovery_updates_before=p.imex_workspace.nonlinear_recovery_iterations;
          consistent=nstep::solve_imex(problem,caps,geometry,options,p.imex_workspace,
                                      p.have_warm?&p.warm:nullptr);
          if(p.imex_workspace.nonlinear_recoveries>recoveries_before) {
            std::fprintf(stderr,
                "slurry/bridge IMEX recovery accepted: step=%lld solves=%zu additional_updates=%zu relative_residual=%.9e\n",
                static_cast<long long>(update->ntimestep),
                p.imex_workspace.nonlinear_recoveries-recoveries_before,
                p.imex_workspace.nonlinear_recovery_iterations-recovery_updates_before,
                consistent.solution.relative_residual);
            std::fflush(stderr);
          }
        } else {
          phase="coupled backward Euler force/endpoint solve";
          consistent=nstep::solve(problem,geometry,options,p.nearfield_workspace,
                                  p.have_warm?&p.warm:nullptr);
        }
        integration_velocity=consistent.solution.velocity;
        next_warm=consistent.solution;
        for(std::size_t k=0;k<problem.contacts.size();++k) {
          const auto &c=problem.contacts[k];const auto &r=consistent.solution.contacts[k];
          endpoint_min=std::min(endpoint_min,consistent.endpoint.gaps[k]);
          endpoint_overlap_fraction=std::max(endpoint_overlap_fraction,-consistent.endpoint.gaps[k]/
              std::min(minimum_axis(bodies[c.i]),minimum_axis(bodies[c.j])));
          corrected_rolling_work+=r.rolling_work;
          corrected_rolling_yield+=r.rolling;
          if(p.imex_contacts) {
            const double lag=std::abs(caps[k]-r.normal_load);
            normal_cap_lag=std::max(normal_cap_lag,lag);normal_cap_lag_sum+=lag;
            old_normal_sum+=caps[k];end_normal_sum+=r.normal_load;
            sliding_cap_excess=std::max(sliding_cap_excess,r.tangent_force.norm()-c.mu*r.normal_load);
            const Eigen::Vector3d rolling_moment=r.torque-c.normal*c.normal.dot(r.torque);
            rolling_cap_excess=std::max(rolling_cap_excess,rolling_moment.norm()-c.mu_roll*c.roll_length*r.normal_load);
          }
          if(!r.active) {
            // Opening contact histories are released only at commit. Keep
            // their accepted interval forces intact for balance and stress.
            detached_release+=r.tangent_energy+r.rolling_energy+r.twisting_energy;
            continue;
          }
          History h;
          h.elastic=r.tangent_history;h.rolling=r.rolling_history;h.normal=c.normal;
          // The next assembly transports this immutable accepted history
          // through the common rotation and onto the actual endpoint normal.
          h.angle=.5*dt*(integration_velocity.segment<3>(6*c.i+3)+
                         integration_velocity.segment<3>(6*c.j+3)+2.*omega_inf);
          h.predicted=vec(consistent.endpoint.bodies[c.j].position)-
                      vec(consistent.endpoint.bodies[c.i].position)+
                      consistent.endpoint.lattice*geometry.images[k].cast<double>();
          corrected_history[current.keys[k]].push_back(h);
        }
      } else {
      // Explicit Heun predictor/corrector. Both hydrodynamic stages are
      // evaluated at their own configuration and current spring histories.
      phase="current hydrodynamic solve";
      auto initial=SlurryBridgeLinearSolver::solve(problem,options,p.have_warm?&p.warm:nullptr);
      phase="Euler predictor endpoint geometry";
      const auto predictor_endpoint=geometry.advance(problem,initial.velocity,false);
      const auto predictor_contacts=SlurryBridgeLinearSolver::advance_contacts(
          problem,initial,predictor_endpoint.gaps);
      std::map<PairKey,std::vector<History>> predictor_history;
      for(std::size_t k=0;k<problem.contacts.size();++k) {
        const auto &c=problem.contacts[k];const auto &r=predictor_contacts[k];
        if(!r.active)continue;
        History h;h.elastic=r.tangent_history;h.rolling=r.rolling_history;h.normal=c.normal;
        h.angle=.5*dt*(initial.velocity.segment<3>(6*c.i+3)+
                      initial.velocity.segment<3>(6*c.j+3)+2.*omega_inf);
        h.predicted=vec(predictor_endpoint.bodies[c.j].position)-
                    vec(predictor_endpoint.bodies[c.i].position)+
                    predictor_endpoint.lattice*geometry.images[k].cast<double>();
        predictor_history[current.keys[k]].push_back(h);
      }
      std::vector<Eigen::Quaterniond> predictor_quaternions(n);
      for(int i=0;i<n;++i)
        predictor_quaternions[i]=step::rotate(geometry.quaternions[i],
            initial.velocity.segment<3>(6*i+3)+omega_inf,dt);
      phase="predicted assembly";
      auto predicted=assemble(predictor_endpoint.bodies,predictor_endpoint.lattice,
                              predictor_quaternions,predictor_history,dt);
      phase="predicted hydrodynamic solve";
      auto predicted_solution=SlurryBridgeLinearSolver::solve(predicted.problem,options,&initial);
      geometry_time+=predicted.geometry_time;middle_time+=predicted.middle_time;
      staged_assembly_time=predicted.geometry_time+predicted.middle_time;
      integration_velocity=.5*(initial.velocity+predicted_solution.velocity);
      // Average TOTAL translational velocities. The affine stage-1 velocity
      // must be evaluated at the predicted position, not at the old position.
      for(int i=0;i<n;++i)
        integration_velocity.segment<3>(6*i)+=.5*kappa*(
            vec(predictor_endpoint.bodies[i].position)-vec(bodies[i].position));
      phase="Heun corrector endpoint geometry";
      consistent.endpoint=geometry.advance(problem,integration_velocity,true);
      // This explicit step does not enforce equality between CURRENT load
      // and the NEXT endpoint load. An implicit consistency residual is N/A.
      consistent.maximum_gap_error=std::numeric_limits<double>::quiet_NaN();
      for(std::size_t k=0;k<problem.contacts.size();++k) {
        const auto &c=problem.contacts[k];
        endpoint_min=std::min(endpoint_min,consistent.endpoint.gaps[k]);
        endpoint_overlap_fraction=std::max(endpoint_overlap_fraction,-consistent.endpoint.gaps[k]/
            std::min(minimum_axis(bodies[c.i]),minimum_axis(bodies[c.j])));
        const V3 relative=c.Bi*integration_velocity.segment<6>(6*c.i)+
                          c.Bj*integration_velocity.segment<6>(6*c.j)+c.affine;
        const double linear_gap=c.gap+dt*c.normal.dot(relative);
        consistent.maximum_linear_gap_defect=std::max(consistent.maximum_linear_gap_defect,
                            std::abs(consistent.endpoint.gaps[k]-linear_gap));
      }
      using ContactID=std::tuple<int,int,int,int,int>;
      std::map<ContactID,std::size_t> current_index;
      auto id=[](const bs::Contact &c,const im::ImageIndex &image) {
        return ContactID(c.i,c.j,image[0],image[1],image[2]);
      };
      for(std::size_t k=0;k<problem.contacts.size();++k)
        current_index[id(problem.contacts[k],geometry.images[k])]=k;
      phase="corrected contact history geometry";
      for(std::size_t k=0;k<predicted.problem.contacts.size();++k) {
        const auto &c1=predicted.problem.contacts[k];
        const auto &r1=predicted_solution.contacts[k];
        const auto image=predicted.geometry.images[k];
        const V3 d_end=vec(consistent.endpoint.bodies[c1.j].position)-
                       vec(consistent.endpoint.bodies[c1.i].position)+
                       consistent.endpoint.lattice*image.cast<double>();
        auto second=consistent.endpoint.bodies[c1.j];
        second.position=arr(vec(consistent.endpoint.bodies[c1.i].position)+d_end);
        const auto gap_end=exact_gap(consistent.endpoint.bodies[c1.i],second,c1.i,c1.j,image);
        endpoint_min=std::min(endpoint_min,gap_end.gap);
        endpoint_overlap_fraction=std::max(endpoint_overlap_fraction,-gap_end.gap/
            std::min(minimum_axis(bodies[c1.i]),minimum_axis(bodies[c1.j])));
        if(gap_end.gap>=0.)continue;
        const V3 n_end=vec(gap_end.normal);
        const V3 angle_predict=.5*dt*(initial.velocity.segment<3>(6*c1.i+3)+
                              initial.velocity.segment<3>(6*c1.j+3)+2.*omega_inf);
        const V3 angle_correct=.25*dt*(initial.velocity.segment<3>(6*c1.i+3)+
                              initial.velocity.segment<3>(6*c1.j+3)+
                              predicted_solution.velocity.segment<3>(6*c1.i+3)+
                              predicted_solution.velocity.segment<3>(6*c1.j+3)+4.*omega_inf);
        const M3 tangent1=M3::Identity()-c1.normal*c1.normal.transpose();
        const V3 rate1=r1.active?V3(tangent1*r1.relative):V3::Zero();
        const V3 rolling_rate1=r1.active?V3(tangent1*r1.relative_rotation):V3::Zero();
        V3 elastic_end,rolling_end;
        const auto old=current_index.find(id(c1,image));
        if(old!=current_index.end()) {
          const auto &c0=problem.contacts[old->second];const auto &r0=initial.contacts[old->second];
          const M3 tangent0=M3::Identity()-c0.normal*c0.normal.transpose();
          const V3 rate0=r0.active?V3(tangent0*r0.relative):V3::Zero();
          const V3 rolling_rate0=r0.active?V3(tangent0*r0.relative_rotation):V3::Zero();
          const V3 rate1_back=vec(SlurryHistory::transport(arr(rate1),arr(c1.normal),arr(c0.normal),arr(-angle_predict)));
          const V3 rolling_rate1_back=vec(SlurryHistory::transport(arr(rolling_rate1),arr(c1.normal),arr(c0.normal),arr(-angle_predict)));
          const V3 elastic0=r0.active?r0.tangent_history:V3::Zero();
          const V3 rolling0=r0.active?r0.rolling_history:V3::Zero();
          elastic_end=vec(SlurryHistory::transport(arr(elastic0+.5*dt*(rate0+rate1_back)),arr(c0.normal),arr(n_end),arr(angle_correct)));
          rolling_end=vec(SlurryHistory::transport(arr(rolling0+.5*dt*(rolling_rate0+rolling_rate1_back)),arr(c0.normal),arr(n_end),arr(angle_correct)));
        } else {
          elastic_end=vec(SlurryHistory::transport(arr(.5*dt*rate1),arr(c1.normal),arr(n_end),arr(angle_correct-angle_predict)));
          rolling_end=vec(SlurryHistory::transport(arr(.5*dt*rolling_rate1),arr(c1.normal),arr(n_end),arr(angle_correct-angle_predict)));
        }
        auto ce=c1;ce.normal=n_end;ce.gap=gap_end.gap;
        ce.history=elastic_end;ce.rolling_history=rolling_end;
        const auto corrected=bs::contact_response(ce,Eigen::VectorXd::Zero(6*n),0.);
        // The predictor is discarded; only the corrector projection is part
        // of the prepared physical step's rolling plastic work.
        corrected_rolling_work+=corrected.rolling_work;
        corrected_rolling_yield+=corrected.rolling;
        History h;h.elastic=corrected.tangent_history;h.rolling=corrected.rolling_history;
        h.normal=n_end;h.angle.setZero();h.predicted=d_end;
        corrected_history[predicted.keys[k]].push_back(h);
      }
      // Report current-state forces/stress. Advance with the corrected
      // velocity and histories only; predictor states never enter p.history.
      consistent.solution=std::move(initial);
      consistent.solution.linear_iterations+=predicted_solution.linear_iterations;
      consistent.solution.mobility_applications+=predicted_solution.mobility_applications;
      consistent.solution.relative_residual=std::max(consistent.solution.relative_residual,
                                                   predicted_solution.relative_residual);
      consistent.solution.force_residual_norm=std::max(consistent.solution.force_residual_norm,
                                                       predicted_solution.force_residual_norm);
      consistent.solution.mobility_residual_norm=std::max(consistent.solution.mobility_residual_norm,
                                                          predicted_solution.mobility_residual_norm);
      next_warm=std::move(predicted_solution);
      }
      auto result = std::move(consistent.solution);
      // Predictor assembly is already counted in the geometry/middle totals.
      double solve_time = MPI_Wtime() - section - staged_assembly_time;
      M3 hydro = M3::Zero(), contact_stress = M3::Zero();
      Eigen::VectorXd contact_moments = Eigen::VectorXd::Zero(6 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          hydro -= result.total_moment[11 * i + 6 + k] * cb::strain_basis()[k];
      p.pending.clear();
      int contacts = 0;
      double normal_energy = 0., tangent_energy = 0., rolling_energy = 0.;
      // Yield/work refer to the prepared [t_n,t_n+dt] step. Near-field BE uses
      // endpoint spring loads on the current hydrodynamic geometry; legacy
      // Heun reports instantaneous current loads. CSV stage times identify both.
      double rolling_work = corrected_rolling_work, rotation_power = 0.;
      int rolling_yield = corrected_rolling_yield;
      for (size_t k = 0; k < problem.contacts.size(); ++k) {
        const auto &c = problem.contacts[k];
        const auto &response = result.contacts[k];
        const auto &d = contact_displacements[k];
        normal_energy += response.normal_energy;
        tangent_energy += response.tangent_energy;
        rolling_energy += response.rolling_energy;
        rotation_power += response.torque.dot(response.relative_rotation);
        contact_stress +=
            .5 * (d * response.traction.transpose() + response.traction * d.transpose());
        contact_moments.segment<6>(6 * c.i) -=
            bs::generalized_contact_map(c,true).transpose() * bs::contact_wrench(response);
        contact_moments.segment<6>(6 * c.j) -=
            bs::generalized_contact_map(c,false).transpose() * bs::contact_wrench(response);
        if (response.active) ++contacts;
      }
      p.pending=std::move(corrected_history);
      const double stress_scale = p.eta * rate / volume,
                   power_scale = p.eta * std::pow(length, 3) * rate * rate;
      p.stress = {p.eta * p.shear, hydro(0, 2) * stress_scale, contact_stress(0, 2) * stress_scale,
                  0};
      p.stress[3] = p.stress[0] + p.stress[1] + p.stress[2];
      if(p.cohesion_work>0.)p.stress[3]+=current.cohesion_stress(0,2)*stress_scale;
      if(p.background.hamaker>0.)p.stress[3]+=current.background_stress(0,2)*stress_scale;
      const double solvent_power = volume * (p.shear / rate) * (p.shear / rate),
                   hydro_power = (solvent_power + result.hydro_power) * power_scale,
                   contact_power = result.contact_power * power_scale;
      const double stress_power = p.stress[3] * p.shear * volume * std::pow(length, 3);
      // Conservative power is minus pair work at the actual velocities.
      // The affine rotation contribution cancels by pair angular momentum;
      // the remaining affine work is minus the symmetric pair virial:E.
      const double cohesion_power=p.cohesion_work>0. ?
          (-(p.background.hamaker>0.?bs::Vector(problem.external-current.background_external):problem.external).dot(result.velocity)+
           (current.cohesion_stress.array()*E.array()).sum())*power_scale : 0.;
      const double background_power=p.background.hamaker>0.?
          (-current.background_external.dot(result.velocity)+
           (current.background_stress.array()*E.array()).sum())*power_scale:0.;
      p.diag.fill(0);
      p.diag[72] = p.kn; // fixed stiffness outside practical kn_adapt
      p.diag[0] = p.stress[3];
      p.diag[1] = p.shear ? p.stress[3] / p.shear : std::numeric_limits<double>::quiet_NaN();
      p.diag[2] = active_near;
      p.diag[3] = contacts;
      p.diag[4] = min_gap * length;
      p.diag[5] = overlap * length;
      p.diag[6] = result.force_residual_norm;
      p.diag[7] = result.mobility_residual_norm;
      p.diag[8] = result.linear_iterations;
      p.diag[9] = result.newton_iterations;
      p.diag[10] = hydro_power;
      p.diag[11] = contact_power;
      p.diag[12] = stress_power;
      p.diag[13] = stress_power - hydro_power - contact_power;
      if(p.cohesion_work>0.)p.diag[13]-=cohesion_power;
      if(p.background.hamaker>0.)p.diag[13]-=background_power;
      p.diag[14] = p.nearfield_mobility ? 0 : mobility.mode_count();
      p.diag[15] = result.relative_residual;
      p.diag[20] = result.descent_steps;
      p.diag[21] = result.continuation_stages;
      p.diag[22] = result.continuation_retries;
      p.diag[23] = result.mobility_applications;
      p.diag[24] = result.breakpoint_steps;
      p.diag[25] = consistent.geometry_iterations;
      p.diag[26] = consistent.maximum_gap_error * length;
      p.diag[27] = endpoint_min * length;
      p.diag[28] = std::max(0., -endpoint_min) * length;
      const double energy_scale = p.eta * std::pow(length, 3) * rate;
      p.diag[29] = normal_energy * energy_scale;
      p.diag[30] = tangent_energy * energy_scale;
      p.diag[31] = consistent.maximum_linear_gap_defect * length;
      p.diag[32] = problem.contacts.size();
      p.diag[33] = (p.nearfield_mobility ? 6 : 17) * n;
      p.diag[34] = p.nearfield_mobility ? 0 : mobility.mesh_fft_plan_builds();
      p.diag[35] = settings.threads;
      p.diag[36] = p.nearfield_mobility ? p.hydro_workspace().symbolic_analyses
                                      : p.solver_workspace->coupled_symbolic_analyses;
      p.diag[37] = p.nearfield_mobility ? p.hydro_workspace().numeric_factorizations
                                      : p.solver_workspace->coupled_numeric_factorizations;
      p.diag[38] = p.nearfield_mobility ? p.hydro_workspace().numeric_reuses
                                      : p.solver_workspace->coupled_numeric_reuses;
      p.diag[39] = p.nearfield_mobility ? 0 : p.solver_workspace->monitor_reuses;
      p.diag[40] = rolling_yield;
      p.diag[41] = rolling_energy * energy_scale;
      p.diag[42] = rolling_work / dt * power_scale;
      p.diag[43] = rotation_power * power_scale;
      p.diag[44] = current.overlap_fraction;
      p.diag[45] = endpoint_overlap_fraction;
      p.diag[46] = minimum_semiaxis * length;
      const double force_scale=p.eta*length*length*rate;
      p.diag[47] = normal_cap_lag*force_scale;
      p.diag[48] = normal_cap_lag_sum/std::max({old_normal_sum,end_normal_sum,
                                              std::numeric_limits<double>::min()});
      p.diag[49] = sliding_cap_excess*force_scale;
      p.diag[50] = rolling_cap_excess*energy_scale;
      p.diag[51] = detached_release*energy_scale;
      p.diag[52] = current.cohesion_stress(0,2)*stress_scale;
      p.diag[53] = current.cohesion_energy*energy_scale;
      p.diag[54] = cohesion_power;
      p.diag[55] = current.cohesive_pairs;
      p.diag[56]=current.background_stress(0,2)*stress_scale;
      p.diag[57]=current.background_energy*energy_scale;p.diag[58]=background_power;
      p.diag[59]=current.background_pairs;p.diag[60]=1.;p.diag[62]=update->dt;
      p.diag[64]=p.imex_workspace.last_roundoff_tolerance;
      p.diag[66]=p.imex_workspace.last_acceptance_ratio;
      p.diag[16] = geometry_time;
      p.diag[17] = middle_time;
      p.diag[18] = solve_time;
      p.diag[19] = MPI_Wtime() - begun;
      for (int i = 0; i < n; ++i) {
        physical.segment<3>(6 * i) =
            length * rate * (result.velocity.segment<3>(6 * i) + kappa * vec(bodies[i].position));
        physical.segment<3>(6 * i + 3) = rate * (result.velocity.segment<3>(6 * i + 3) + omega_inf);
        advance_physical.segment<3>(6 * i) =
            length * rate * (integration_velocity.segment<3>(6 * i) + kappa * vec(bodies[i].position));
        advance_physical.segment<3>(6 * i + 3) =
            rate * (integration_velocity.segment<3>(6 * i + 3) + omega_inf);
      }
      for (int i = 0; i < n; ++i) {
        double fs = p.eta * length * length * rate, ms = fs * length;
        moments.segment<3>(17 * i) = -fs * result.total_moment.segment<3>(11 * i);
        moments.segment<3>(17 * i + 3) = -ms * result.total_moment.segment<3>(11 * i + 3);
        moments.segment<5>(17 * i + 6) = ms * result.total_moment.segment<5>(11 * i + 6);
        moments.segment<3>(17 * i + 11) = fs * contact_moments.segment<3>(6 * i);
        moments.segment<3>(17 * i + 14) = ms * contact_moments.segment<3>(6 * i + 3);
      }
      p.warm = std::move(next_warm);
      p.have_warm = true;
      } // original single-step path
    } catch (const bs::NonlinearFailure &e) {
      failed = 1;
      std::ostringstream diagnostic;
      diagnostic << e.what() << std::scientific << std::setprecision(9)
                 << "; residual=" << e.residual << "; tolerance=" << e.tolerance
                 << "; Newton iterations=" << e.newton_iterations
                 << "; linear iterations=" << e.linear_iterations
                 << "; phase=" << phase << "; step=" << update->ntimestep
                 << "; time_s=" << simulation_time() << "; dt_s=" << update->dt;
      message = diagnostic.str();
    } catch (const std::exception &e) {
      failed = 1;
      std::ostringstream diagnostic;
      diagnostic << e.what() << "; phase=" << phase << "; step=" << update->ntimestep
                 << "; time_s=" << std::setprecision(17) << simulation_time()
                 << "; dt_s=" << update->dt;
      if (p.kn_control)
        diagnostic << "; kn_N_per_m=" << p.kn << "; kn_max_N_per_m="
                   << SlurryStiffnessControl::maximum(p.kn_state, p.kn_settings);
      message = diagnostic.str();
    }
  MPI_Bcast(&failed, 1, MPI_INT, 0, world);
  if (failed) {
    int m = message.size();
    MPI_Bcast(&m, 1, MPI_INT, 0, world);
    message.resize(m);
    MPI_Bcast(message.data(), m, MPI_CHAR, 0, world);
    error->all(FLERR, "slurry/bridge solve failed: {}", message);
  }
  MPI_Bcast(physical.data(), 6 * n, MPI_DOUBLE, 0, world);
  MPI_Bcast(advance_physical.data(), 6 * n, MPI_DOUBLE, 0, world);
  // Every rank retains the same global tag ordering, so migration does not
  // turn a prepared velocity into an index referring to another particle.
  p.advance_tags.resize(n);
  for(int j=0;j<n;++j)p.advance_tags[j]=records[j].tag;
  p.advance_velocity=std::move(advance_physical);
  MPI_Bcast(&direct_advance,1,MPI_INT,0,world);
  p.direct_advance=direct_advance;
  if(direct_advance)MPI_Bcast(direct_endpoint.data(),7*n,MPI_DOUBLE,0,world);
  p.advance_endpoint=std::move(direct_endpoint);
  MPI_Bcast(moments.data(), 17 * n, MPI_DOUBLE, 0, world);
  if (atom->nmax > p.nmax) {
    p.nmax = atom->nmax;
    memory->grow(array_atom, p.nmax, 17, "slurry/bridge:array");
  }
  MPI_Bcast(p.diag.data(), p.diag.size(), MPI_DOUBLE, 0, world);
  MPI_Bcast(p.stress.data(), 4, MPI_DOUBLE, 0, world);
  for (int i = 0; i < atom->nlocal; ++i) {
    auto it = std::lower_bound(records.begin(), records.end(), atom->tag[i],
                               [](const Record &r, tagint id) { return r.tag < id; });
    int j = it - records.begin();
    for (int k = 0; k < 3; ++k)
      atom->v[i][k] = physical[6 * j + k];
    store_momentum(atom, p.avec, i, physical.segment<3>(6 * j + 3));
    for (int k = 0; k < 17; ++k)
      array_atom[i][k] = moments[17 * j + k];
  }
}
void FixSlurryBridge::write_history() {
  auto &p = *p_;
  if (p.last_output == update->ntimestep ||
      (update->ntimestep % p.interval != 0 && update->ntimestep != update->laststep))
    return;
  const double time = simulation_time(), nan = std::numeric_limits<double>::quiet_NaN();
  auto eta = [&](double s) { return p.shear ? s / p.shear : nan; };
  if (comm->me == 0 && p.csv) {
    auto &o = p.csv;
    o << update->ntimestep << ',' << time << ',' << std::abs(p.shear) * time;
    for (double s : p.stress)
      o << ',' << s;
    o << ',' << eta(p.stress[3]) << ',' << eta(p.stress[3]) / p.eta << ','
      << (p.selected_duration > 0 ? eta(p.selected_integral / p.selected_duration) : nan) << ','
      << p.selected_duration;
    for (int k : {10, 11, 12, 13, 6, 7, 8, 9, 2, 3, 14, 4, 5, 16, 17, 18, 19})
      o << ',' << p.diag[k];
    double begin = p.interval_steps ? p.interval_start : time;
    o << ',' << p.interval_steps << ',' << begin << ',' << time << ',' << std::abs(p.shear) * begin
      << ',' << std::abs(p.shear) * time;
    double mean = nan;
    for (int k = 0; k < 4; ++k) {
      double value = p.interval_duration ? p.interval_stress[k] / p.interval_duration : nan;
      o << ',' << value;
      if (k == 3)
        mean = value;
    }
    o << ',' << eta(mean) << ',' << eta(mean) / p.eta << ',' << p.diag[15] << ',' << p.diag[20]
      << ',' << p.diag[21] << ',' << p.diag[22] << ',' << p.diag[23]
      << ',' << p.diag[24];
    for (int k = 25; k < 44; ++k)
      o << ',' << p.diag[k];
    // These forward-step fields are explicitly time-labelled. Work is the
    // corrected rolling return-map projection, not a cumulative integral.
    o << ',' << time+update->dt << ',' << p.diag[42]*update->dt;
    for(int k=44;k<47;++k)o << ',' << p.diag[k];
    o << ',' << time << ',' << (p.nearfield_mobility?time+update->dt:time)
      << ',' << (p.direct_advance?(p.surface_lagged_hydro?"imex_surface_hydro_reuse":"imex_surface_substeps"):(p.nearfield_mobility?(p.imex_contacts?"imex_frozen_cap":"backward_euler"):"heun"))
      << ',' << (p.nearfield_mobility&&!p.imex_contacts?time+update->dt:time);
    for(int k=47;k<static_cast<int>(p.diag.size());++k)o << ',' << p.diag[k];
    o << '\n';
    o.flush();
    if (!o)
      error->one(FLERR, "Failed writing slurry/bridge history");
  }
  p.last_output = update->ntimestep;
  p.interval_steps = 0;
  p.interval_duration = 0;
  p.interval_stress.fill(0);
}

