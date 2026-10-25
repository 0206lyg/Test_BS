// Compact v4 lubrication + periodic 11-mode mobility + implicit rough contact.
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
#include "slurry_cylinder_pair.h"
#include "slurry_flake_diagnostics.h"
#include "slurry_compact_bridge.h"
#include "slurry_rough_surface.h"
#include "slurry_history_math.h"
#include "slurry_periodic_mobility.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <optional>
#include <iomanip>
#include <limits>
#include <map>
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
namespace cylinder = SlurryCylinderPair;
namespace fd = SlurryFlakeDiagnostics;
namespace rough = SlurryRoughSurface;
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
  double twisting = 0.;
};
gr::Vec3 arr(const V3 &x) { return {x[0], x[1], x[2]}; }
V3 vec(const gr::Vec3 &x) { return {x[0], x[1], x[2]}; }
void store_momentum(Atom *atom, AtomVecEllipsoid *avec, int i, const V3 &omega, bool flake) {
  auto &b = avec->bonus[atom->ellipsoid[i]];
  double R[3][3], ob[3], lb[3], l[3];
  MathExtra::quat_to_mat(b.quat, R);
  MathExtra::transpose_matvec(R, omega.data(), ob);
  const auto inertia=shape::principal_inertia(atom->rmass[i],b.shape,flake);
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
  double eta = 0, shear = 0, floor = 0, kn = 0, hc = 3e-9, mu = .1, kt = 80;
  rough::Options rough_options;
  double mu_roll = 0., mu_twist = 0., roll_length = 0., twist_length = 0., kr = 0., kw = 0.;
  double spectral = 2, spectral_tol = 1e-8, rtol = 1e-7, avg_start = 5;
  int maxit = 1200, interval = 100, nmax = 0;
  std::ofstream csv, flake_channels, flake_bins;
  bool flake_diagnostics = false;
  bigint last_diagnostic = -1;
  std::array<double, 51> diag{};
  std::array<double, 4> stress{};
  std::array<long double, 4> interval_stress{};
  long double selected_integral = 0;
  double selected_duration = 0, interval_duration = 0, interval_start = 0;
  bigint interval_steps = 0, last_output = -1;
  std::map<PairKey, std::vector<History>> history, pending;
  std::vector<tagint> tags;
  bs::Result warm;
  // Reuse FFT plans/buffers and exact solver caches across physical steps.
  // reset() still refreshes all position, orientation and lattice coefficients.
  pm::Mobility mobility;
  std::shared_ptr<bs::detail::SolverWorkspace> solver_workspace =
      std::make_shared<bs::detail::SolverWorkspace>();
  bool have_warm = false;
  bool direct_mobility = false;
  bool flake = false;
};
FixSlurryBridge::FixSlurryBridge(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), p_(new Impl) {
  if (narg < 7)
    error->all(FLERR, "slurry/bridge requires eta shear_rate gap_floor kn [contact hc mu kt] "
                      "[shape spheroid|flake] [spectral xi_a] [spectral_tolerance tol] [mobility_backend mesh|direct] [tolerance rtol] [iterations "
                      "maxit] [history file interval average_start_strain]");
  auto &p = *p_;
  p.eta = utils::numeric(FLERR, arg[3], false, lmp);
  p.shear = utils::numeric(FLERR, arg[4], false, lmp);
  p.floor = utils::numeric(FLERR, arg[5], false, lmp);
  p.kn = utils::numeric(FLERR, arg[6], false, lmp);
  std::string filename;
  for (int k = 7; k < narg;) {
    if (!std::strcmp(arg[k], "contact") && k + 3 < narg) {
      p.hc = utils::numeric(FLERR, arg[k + 1], false, lmp);
      p.mu = utils::numeric(FLERR, arg[k + 2], false, lmp);
      p.kt = utils::numeric(FLERR, arg[k + 3], false, lmp);
      k += 4;
    } else if (!std::strcmp(arg[k], "rough_hydro") && k + 3 < narg) {
      p.rough_options.enabled = true;
      p.rough_options.height = utils::numeric(FLERR, arg[k+1], false, lmp);
      p.rough_options.width = utils::numeric(FLERR, arg[k+2], false, lmp);
      p.rough_options.coverage = utils::numeric(FLERR, arg[k+3], false, lmp);
      p.rough_options.gap_floor = p.floor;
      k += 4;
    } else if (!std::strcmp(arg[k], "rotation_contact") && k + 6 < narg) {
      p.mu_roll = utils::numeric(FLERR, arg[k+1], false, lmp);
      p.mu_twist = utils::numeric(FLERR, arg[k+2], false, lmp);
      p.roll_length = utils::numeric(FLERR, arg[k+3], false, lmp);
      p.twist_length = utils::numeric(FLERR, arg[k+4], false, lmp);
      p.kr = utils::numeric(FLERR, arg[k+5], false, lmp);
      p.kw = utils::numeric(FLERR, arg[k+6], false, lmp);
      k += 7;
    } else if (!std::strcmp(arg[k], "shape") && k + 1 < narg) {
      if (std::strcmp(arg[k+1],"spheroid") && std::strcmp(arg[k+1],"flake"))
        error->all(FLERR,"slurry/bridge shape must be spheroid or flake");
      p.flake=!std::strcmp(arg[k+1],"flake");
      k+=2;
    } else if (!std::strcmp(arg[k], "spectral") && k + 1 < narg) {
      p.spectral = utils::numeric(FLERR, arg[k + 1], false, lmp);
      k += 2;
    } else if (!std::strcmp(arg[k], "spectral_tolerance") && k + 1 < narg) {
      p.spectral_tol = utils::numeric(FLERR, arg[k + 1], false, lmp);
      k += 2;
    } else if (!std::strcmp(arg[k], "mobility_backend") && k + 1 < narg) {
      if (std::strcmp(arg[k + 1], "mesh") && std::strcmp(arg[k + 1], "direct"))
        error->all(FLERR, "slurry/bridge mobility_backend must be mesh or direct");
      p.direct_mobility = !std::strcmp(arg[k + 1], "direct");
      k += 2;
    } else if (!std::strcmp(arg[k], "tolerance") && k + 1 < narg) {
      p.rtol = utils::numeric(FLERR, arg[k + 1], false, lmp);
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
  if (!(p.eta > 0 && p.floor > 0 && p.kn > 0 && p.hc >= 0 && p.mu >= 0 && p.kt >= 0 &&
        p.spectral > 0 && p.spectral_tol > 0 && p.spectral_tol < 1 && p.rtol > 0 && p.rtol < 1 &&
        p.maxit > 0 && p.interval > 0 && p.avg_start >= 0) ||
      !std::isfinite(p.eta + p.shear + p.floor + p.kn + p.hc + p.mu + p.kt + p.spectral +
                     p.spectral_tol + p.rtol + p.avg_start) ||
      (p.mu > 0 && p.kt == 0))
    error->all(FLERR, "Invalid slurry/bridge material or numerical parameter");
  if (!(p.mu_roll >= 0. && p.mu_twist >= 0. && p.roll_length >= 0. &&
        p.twist_length >= 0. && p.kr >= 0. && p.kw >= 0.) ||
      !std::isfinite(p.mu_roll+p.mu_twist+p.roll_length+p.twist_length+p.kr+p.kw) ||
      (p.mu_roll > 0. && !(p.kr > 0. && p.roll_length > 0.)) ||
      (p.mu_twist > 0. && !(p.kw > 0. && p.twist_length > 0.)))
    error->all(FLERR, "Invalid slurry/bridge rotational contact parameter");
  if (p.flake && (p.rough_options.enabled || p.mu_roll > 0. || p.mu_twist > 0.))
    error->all(FLERR, "Rough hydrodynamics and rotational contacts currently require shape spheroid");
  try { rough::validate(p.rough_options); }
  catch (const std::exception &e) { error->all(FLERR, "Invalid rough_hydro parameters: {}", e.what()); }
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
             "rough_hydro_pairs,rough_hydro_dissipation_W,rough_hydro_sigma_xz_Pa,"
             "rolling_yield_contacts,twisting_yield_contacts,rolling_elastic_energy_J,"
             "twisting_elastic_energy_J,rolling_plastic_dissipation_W,"
             "twisting_plastic_dissipation_W,rotational_contact_power_W,rough_geometry_s\n";
  }
  // Read-only diagnostics use the existing history sampling cadence. They
  // never enter the force/velocity solve and do not change the batch settings.
  const char *diagnostics = std::getenv("SLURRY_FLAKE_DIAGNOSTICS");
  p.flake_diagnostics = p.flake && !filename.empty() &&
      !(diagnostics && !std::strcmp(diagnostics, "0"));
  if (p.flake_diagnostics && comm->me == 0) {
    const auto directory = std::filesystem::path(filename).parent_path();
    p.flake_channels.open(directory / "flake_channels.csv", std::ios::trunc);
    p.flake_bins.open(directory / "flake_channel_bins.csv", std::ios::trunc);
    if (!p.flake_channels || !p.flake_bins) {
      std::fprintf(stderr, "WARNING: flake diagnostic files could not be opened; dynamics continue without diagnostic output.\n");
      p.flake_diagnostics = false;
    } else {
      p.flake_channels << std::setprecision(17);
      p.flake_bins << std::setprecision(17);
      p.flake_channels << "step,time_s,strain," << fd::csv_columns()
          << ",middle_power_W,hydro_particle_power_W,solvent_power_W,contact_power_W,"
             "stress_power_W,middle_sigma_xz_Pa,hydro_sigma_xz_Pa,contact_sigma_xz_Pa,"
             "eta_relative,eta_relative_face,eta_relative_rim,eta_relative_middle,"
             "eta_relative_contact,near_moment_closure_relative,power_closure_relative,"
             "hydro_stress_closure_Pa,active_contacts,sliding_contacts,contacted_particles,"
             "normal_load_sum_N,normal_load_max_N,sliding_dissipation_W,gap_floor_m,"
             "diagnostic_pair_storage_bytes,diagnostic_capture_s,diagnostic_reduce_s\n";
      p.flake_bins << "step,time_s,strain,bin_kind,bin_label," << fd::csv_columns() << '\n';
    }
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
    double wr[3];
    auto inertia=shape::principal_inertia(atom->rmass[i],b.shape,p.flake);
    MathExtra::mq_to_omega(atom->angmom[i], b.quat, inertia.data(), wr);
    V3 w(wr[0], wr[1], wr[2]);
    for (int k = 0; k < 3; ++k)
      atom->x[i][k] += dt * atom->v[i][k];
    Eigen::Quaterniond q = step::rotate(
        Eigen::Quaterniond(b.quat[0], b.quat[1], b.quat[2], b.quat[3]), w, dt);
    b.quat[0] = q.w();
    b.quat[1] = q.x();
    b.quat[2] = q.y();
    b.quat[3] = q.z();
    store_momentum(atom, p.avec, i, w, p.flake);
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
  Eigen::VectorXd physical = Eigen::VectorXd::Zero(6 * n), moments = Eigen::VectorXd::Zero(17 * n);
  int failed = 0;
  std::string message;
  if (comm->me == 0)
    try {
      double length = records.front().axes[2], rate = std::max(std::abs(p.shear), 1.);
      for (const auto &r : records)
        for (double a : r.axes)
          if (!(a > 0))
            throw std::runtime_error("non-positive shape axis");
      for (const auto &r : records)
        if (std::abs(r.axes[2] - length) > 1e-10 * length ||
            std::abs(r.axes[0] - r.axes[1]) > 1e-10 * r.axes[0] || r.axes[0] < r.axes[2])
          throw std::runtime_error("bridge supports axisymmetric oblate particles with common polar half-extent c");
      M3 H;
      H << domain->xprd, domain->xy, domain->xz, 0, domain->yprd, domain->yz, 0, 0, domain->zprd;
      H /= length;
      const double volume = H.determinant(), dt = update->dt * rate;
      M3 kappa = M3::Zero();
      kappa(0, 2) = p.shear / rate;
      M3 E = .5 * (kappa + kappa.transpose());
      V3 omega_inf(0, .5 * p.shear / rate, 0);
      std::vector<gr::Body> bodies(n);
      std::vector<pm::Body> middle_bodies(n);
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
        middle_bodies[i].x = vec(bodies[i].position);
        // The common positive periodic mobility uses the existing ellipsoid
        // representation. For flake this is the explicit matched-spheroid middle/self
        // approximation; the cylinder surface enters every near/contact term.
        middle_bodies[i].shape = cb::shape(bodies[i], 1.);
      }
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
      bs::Problem problem;
      problem.count = n;
      problem.dt = dt;
      problem.external = Eigen::VectorXd::Zero(6 * n);
      problem.prescribed = Eigen::VectorXd::Zero(11 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          problem.prescribed[11 * i + 6 + k] = -(E.array() * cb::strain_basis()[k].array()).sum();
      const bool diagnose = p.flake_diagnostics && p.last_diagnostic != update->ntimestep &&
          (update->ntimestep % p.interval == 0 || update->ntimestep == update->laststep);
      std::vector<fd::PairRecord> diagnostic_pairs;
      double diagnostic_capture_s = 0.;
      struct RoughPair { int i, j; rough::Mat22 resistance; };
      std::vector<RoughPair> rough_pairs;
      double rough_geometry_time = 0.;
      rough::Options rough_options = p.rough_options;
      rough_options.height /= length;
      rough_options.width /= length;
      rough_options.gap_floor = p.floor / length;
      std::vector<Eigen::Triplet<double>> entries;
      std::vector<PairKey> contact_keys;
      std::vector<V3> contact_displacements;
      std::vector<im::ImageIndex> contact_images;
      int active_near = 0;
      double min_gap = std::numeric_limits<double>::infinity(), overlap = 0;
      double section = MPI_Wtime();
      for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
          const double ai = shape::bounding_radius(bodies[i],p.flake),
                       aj = shape::bounding_radius(bodies[j],p.flake);
          const double near_range=p.flake ? std::min(bodies[i].axes[0],bodies[j].axes[0])
                                          : std::min(ai,aj);
          const double rough_range = rough_options.enabled ? 3.*rough_options.height : 0.;
          const double rc = ai + aj + std::max({near_range, p.hc / length, rough_range});
          const V3 original = vec(bodies[j].position) - vec(bodies[i].position);
          for (const auto &image : im::images_within(original, H, rc)) {
            if (i == j && !im::positive_image(image.image))
              continue;
            const V3 d = image.displacement;
            auto bj = bodies[j];
            bj.position = arr(vec(bodies[i].position) + d);
            gr::GapResult gap;
            Eigen::Matrix<double,22,22> near_resistance;
            bool near_active=false;
            if (p.flake) {
              const auto ci=shape::cylinder(bodies[i]), cj=shape::cylinder(bj);
              const auto g=shape::cylinder_gap(ci,cj,1e-8*p.floor/length);
              gap=shape::gap_result(ci,cj,g);
              cylinder::Options model;
              model.film.eta=1.; model.film.gap_floor=p.floor/length;
              std::optional<cylinder::Diagnostics> diagnostic;
              if (diagnose) diagnostic.emplace();
              const auto near=cylinder::evaluate(ci,cj,g,model,diagnose ? &*diagnostic : nullptr);
              if (diagnose && near.cutoff_weight > 0.) {
                const double t = MPI_Wtime();
                diagnostic_pairs.push_back(fd::make_pair(i,j,ci,cj,g,*diagnostic,model.film.gap_floor));
                diagnostic_capture_s += MPI_Wtime() - t;
              }
              near_resistance=near.resistance; near_active=near.active;
            } else {
              gap=gr::closestEllipsoidGap(bodies[i],bj);
              const auto near=cb::evaluate(bodies[i],bj,gap,1.,1.,p.floor/length);
              near_resistance=near.resistance; near_active=near.active;
              if (rough_options.enabled && gap.gap < 3.*rough_options.height) {
                const double t = MPI_Wtime();
                const auto extra = rough::evaluate(bodies[i],bj,gap,near,1.,rough_options);
                rough_geometry_time += MPI_Wtime() - t;
                if (extra.active) {
                  near_resistance += extra.resistance;
                  near_active = true;
                  rough_pairs.push_back({i,j,extra.resistance});
                }
              }
            }
            min_gap = std::min(min_gap, gap.gap);
            overlap = std::max(overlap, -gap.gap);
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
            bs::Contact contact;
            contact.i = i;
            contact.j = j;
            contact.normal = vec(gap.normal);
            contact.gap = gap.gap;
            contact.onset = p.hc / length;
            contact.kn = p.kn / (p.eta * length * rate);
            contact.kt = p.kt / (p.eta * length * rate);
            contact.mu = p.mu;
            contact.mu_roll = p.mu_roll;
            contact.mu_twist = p.mu_twist;
            contact.roll_length = p.roll_length / length;
            contact.twist_length = p.twist_length / length;
            const double torque_scale = p.eta * std::pow(length,3) * rate;
            contact.kr = p.kr / torque_scale;
            contact.kw = p.kw / torque_scale;
            V3 li = .5 * (vec(gap.leverI) + d + vec(gap.leverJ)), lj = li - d;
            contact.Bi.leftCols<3>() = -M3::Identity();
            contact.Bi.rightCols<3>() = cb::cross_matrix(li);
            contact.Bj.leftCols<3>() = M3::Identity();
            contact.Bj.rightCols<3>() = -cb::cross_matrix(lj);
            contact.affine = E * d;
            PairKey key(records[i].tag, records[j].tag);
            const auto old = p.history.find(key);
            if (old != p.history.end()) {
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
                contact.twisting_history = h.twisting;
              }
            }
            problem.contacts.push_back(contact);
            contact_keys.push_back(key);
            contact_displacements.push_back(d);
            contact_images.push_back(image.image);
          }
        }
      problem.near_resistance.resize(11 * n, 11 * n);
      problem.near_resistance.setFromTriplets(entries.begin(), entries.end());
      problem.near_resistance.makeCompressed();
      double geometry_time = MPI_Wtime() - section;
      section = MPI_Wtime();
      double amax = 0;
      for (const auto &b : bodies)
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
      mobility.reset(middle_bodies, H, settings);
      problem.self_mobility = mobility.self_blocks();
      problem.mobility = [&mobility](const Eigen::VectorXd &x) { return mobility.apply(x); };
      double middle_time = MPI_Wtime() - section;
      bs::Options options;
      options.workspace = p.solver_workspace;
      options.relative_tolerance = p.rtol;
      options.maximum_linear_iterations = p.maxit;
      options.linear_relative_tolerance = std::min(1e-6, .1 * p.rtol);
      options.max_linear_forcing = .05;
      if (trace_solver) {
        std::fprintf(stderr, "bridge mobility step=%lld modes=%zu storage_bytes=%zu setup_s=%.6g\n",
                     static_cast<long long>(update->ntimestep), mobility.mode_count(), mobility.bytes(), middle_time);
        std::fflush(stderr);
        options.trace_callback = [&](const bs::IterationTrace &t) {
          std::fprintf(stderr,
                       "bridge solve step=%lld phase=%s stage=%d stiffness_scale=%.6g iteration=%d newton_total=%d residual=%.9e merit=%.9e tolerance=%.9e alpha=%.6g "
                       "linear=%d linear_total=%d linear_residual=%.6e linear_target=%.6e linear_defect=%.6e "
                       "active=%d sliding=%d reselections=%d yield_distance=%.6e descent=%d breakpoint=%d elapsed_s=%.6g\n",
                       static_cast<long long>(update->ntimestep), t.phase, t.continuation_stage,
                       t.stiffness_scale, t.iteration, t.total_newton_iterations, t.residual,
                       t.merit_residual, t.tolerance, t.step_length, t.linear_iterations, t.total_linear_iterations,
                       t.linear_residual, t.linear_target, t.linear_defect, t.active_contacts,
                       t.sliding_contacts, t.kink_reselections, t.minimum_yield_distance,
                       int(t.descent), int(t.breakpoint), t.elapsed_seconds);
          std::fflush(stderr);
        };
      }
      section = MPI_Wtime();
      step::Geometry geometry;
      geometry.bodies = bodies;
      geometry.quaternions = std::move(quaternions);
      geometry.lattice = H;
      geometry.kappa = kappa;
      geometry.ambient_rotation = omega_inf;
      geometry.contact_onset = p.hc / length;
      geometry.flake = p.flake;
      geometry.gap_tolerance = 1e-8 * p.floor / length;
      geometry.enforce_work_balance = p.flake;
      geometry.images = std::move(contact_images);
      auto consistent = step::solve(problem, geometry, options, p.have_warm ? &p.warm : nullptr);
      auto result = std::move(consistent.solution);
      double solve_time = MPI_Wtime() - section;
      M3 hydro = M3::Zero(), contact_stress = M3::Zero();
      Eigen::VectorXd contact_moments = Eigen::VectorXd::Zero(6 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          hydro -= result.total_moment[11 * i + 6 + k] * cb::strain_basis()[k];
      p.pending.clear();
      int contacts = 0;
      double normal_energy = 0., tangent_energy = 0., rolling_energy = 0., twisting_energy = 0.;
      double rolling_work = 0., twisting_work = 0., rotation_power = 0.;
      int rolling_yield = 0, twisting_yield = 0;
      int diagnostic_sliding = 0;
      double diagnostic_load_sum = 0., diagnostic_load_max = 0., diagnostic_sliding_work = 0.;
      std::vector<bool> diagnostic_contacted(diagnose ? n : 0, false);
      for (size_t k = 0; k < problem.contacts.size(); ++k) {
        const auto &c = problem.contacts[k];
        const auto &response = result.contacts[k];
        const auto &d = contact_displacements[k];
        normal_energy += response.normal_energy;
        tangent_energy += response.tangent_energy;
        rolling_energy += response.rolling_energy;
        twisting_energy += response.twisting_energy;
        rolling_work += response.rolling_work;
        twisting_work += response.twisting_work;
        rotation_power += response.torque.dot(response.relative_rotation);
        rolling_yield += response.rolling;
        twisting_yield += response.twisting;
        contact_stress +=
            .5 * (d * response.traction.transpose() + response.traction * d.transpose());
        contact_moments.segment<6>(6 * c.i) -=
            bs::generalized_contact_map(c,true).transpose() * bs::contact_wrench(response);
        contact_moments.segment<6>(6 * c.j) -=
            bs::generalized_contact_map(c,false).transpose() * bs::contact_wrench(response);
        if (response.active) {
          ++contacts;
          if (diagnose) {
            diagnostic_contacted[c.i] = diagnostic_contacted[c.j] = true;
            diagnostic_sliding += response.sliding;
            diagnostic_sliding_work += response.sliding_work;
            diagnostic_load_sum += response.normal_load;
            diagnostic_load_max = std::max(diagnostic_load_max, response.normal_load);
          }
          History h;
          h.elastic = response.tangent_history;
          h.rolling = response.rolling_history;
          h.twisting = response.twisting_history;
          h.normal = c.normal;
          h.angle = .5 * dt *
                    (result.velocity.segment<3>(6 * c.i + 3) +
                     result.velocity.segment<3>(6 * c.j + 3) + 2 * omega_inf);
          h.predicted = d + dt * (result.velocity.segment<3>(6 * c.j) -
                                  result.velocity.segment<3>(6 * c.i) + kappa * d);
          p.pending[contact_keys[k]].push_back(h);
        }
      }
      const double stress_scale = p.eta * rate / volume,
                   power_scale = p.eta * std::pow(length, 3) * rate * rate;
      p.stress = {p.eta * p.shear, hydro(0, 2) * stress_scale, contact_stress(0, 2) * stress_scale,
                  0};
      p.stress[3] = p.stress[0] + p.stress[1] + p.stress[2];
      const double solvent_power = volume * (p.shear / rate) * (p.shear / rate),
                   hydro_power = (solvent_power + result.hydro_power) * power_scale,
                   contact_power = result.contact_power * power_scale;
      const double stress_power = p.stress[3] * p.shear * volume * std::pow(length, 3);
      double rough_power = 0.;
      M3 rough_stress = M3::Zero();
      for (const auto &pair : rough_pairs) {
        Eigen::Matrix<double,22,1> qpair;
        qpair << result.q.segment<11>(11*pair.i), result.q.segment<11>(11*pair.j);
        const Eigen::Matrix<double,22,1> moment = pair.resistance * qpair;
        rough_power += qpair.dot(moment);
        for (int k=0;k<5;++k)
          rough_stress -= (moment[6+k]+moment[17+k])*cb::strain_basis()[k];
      }
      if (diagnose) {
        // Diagnostics never feed back into the accepted solution. A failed
        // diagnostic write disables only diagnostics, not the simulation.
        try {
          const double diagnostic_start = MPI_Wtime();
          fd::Accumulator accumulated(n);
          for (const auto &pair : diagnostic_pairs) accumulated.add(pair, result.q);
          const auto &observed = accumulated.total;
          const Eigen::VectorXd near_moment = problem.near_resistance * result.q;
          const double middle_power = result.q.dot(result.mid_moment);
          const double middle_stress = fd::global_shear_moment(result.mid_moment) * stress_scale;
          const double near_closure =
              (accumulated.face_moment + accumulated.rim_moment - near_moment).norm() /
              std::max(near_moment.norm(), 1e-100);
          const double power_closure =
              (observed.face_power + observed.rim_power + middle_power - result.hydro_power) /
              std::max(std::abs(result.hydro_power), 1e-100);
          const double stress_closure =
              (observed.face_shear_moment + observed.rim_shear_moment) * stress_scale +
              middle_stress - p.stress[1];
          const double time = simulation_time(), strain = std::abs(p.shear) * time;
          const double eta_denominator = p.eta * p.shear;
          auto relative_eta = [&](double stress) {
            return eta_denominator ? stress / eta_denominator : std::numeric_limits<double>::quiet_NaN();
          };
          const double force_scale = p.eta * length * length * rate;
          auto &o = p.flake_channels;
          o << update->ntimestep << ',' << time << ',' << strain << ',';
          fd::write_csv_values(o,observed,power_scale,stress_scale,length*rate);
          o << ',' << middle_power*power_scale << ',' << result.hydro_power*power_scale
            << ',' << solvent_power*power_scale << ',' << contact_power << ',' << stress_power
            << ',' << middle_stress << ',' << p.stress[1] << ',' << p.stress[2]
            << ',' << relative_eta(p.stress[3])
            << ',' << relative_eta(observed.face_shear_moment*stress_scale)
            << ',' << relative_eta(observed.rim_shear_moment*stress_scale)
            << ',' << relative_eta(middle_stress) << ',' << relative_eta(p.stress[2])
            << ',' << near_closure << ',' << power_closure << ',' << stress_closure
            << ',' << contacts << ',' << diagnostic_sliding << ','
            << std::count(diagnostic_contacted.begin(),diagnostic_contacted.end(),true)
            << ',' << diagnostic_load_sum*force_scale << ',' << diagnostic_load_max*force_scale
            << ',' << diagnostic_sliding_work/dt*power_scale
            << ',' << p.floor << ',' << diagnostic_pairs.capacity()*sizeof(fd::PairRecord)
            << ',' << diagnostic_capture_s << ',' << MPI_Wtime()-diagnostic_start << '\n';
          auto write_bin = [&](const char *kind,const char *label,const fd::Stats &stats) {
            p.flake_bins << update->ntimestep << ',' << time << ',' << strain << ',' << kind << ',' << label << ',';
            fd::write_csv_values(p.flake_bins,stats,power_scale,stress_scale,length*rate);
            p.flake_bins << '\n';
          };
          for (size_t k=0;k<accumulated.by_angle.size();++k)
            write_bin("angle",fd::angle_labels()[k],accumulated.by_angle[k]);
          for (size_t k=0;k<accumulated.by_gap.size();++k)
            write_bin("gap_over_floor",fd::gap_labels()[k],accumulated.by_gap[k]);
          o.flush(); p.flake_bins.flush();
          if (!o || !p.flake_bins) throw std::runtime_error("diagnostic CSV write failed");
          p.last_diagnostic = update->ntimestep;
        } catch (const std::exception &diagnostic_error) {
          std::fprintf(stderr,"WARNING: flake diagnostics disabled: %s; dynamics continue.\n",diagnostic_error.what());
          p.flake_diagnostics = false;
        }
      }
      p.diag.fill(0);
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
      p.diag[14] = mobility.mode_count();
      p.diag[15] = result.relative_residual;
      p.diag[20] = result.descent_steps;
      p.diag[21] = result.continuation_stages;
      p.diag[22] = result.continuation_retries;
      p.diag[23] = result.mobility_applications;
      p.diag[24] = result.breakpoint_steps;
      p.diag[25] = consistent.geometry_iterations;
      p.diag[26] = consistent.maximum_gap_error * length;
      const double endpoint_min = consistent.endpoint.gaps.empty()
          ? std::numeric_limits<double>::infinity()
          : *std::min_element(consistent.endpoint.gaps.begin(), consistent.endpoint.gaps.end());
      p.diag[27] = endpoint_min * length;
      p.diag[28] = std::max(0., -endpoint_min) * length;
      const double energy_scale = p.eta * std::pow(length, 3) * rate;
      p.diag[29] = normal_energy * energy_scale;
      p.diag[30] = tangent_energy * energy_scale;
      p.diag[31] = consistent.maximum_linear_gap_defect * length;
      p.diag[32] = problem.contacts.size();
      p.diag[33] = 17 * n;
      p.diag[34] = mobility.mesh_fft_plan_builds();
      p.diag[35] = settings.threads;
      p.diag[36] = p.solver_workspace->coupled_symbolic_analyses;
      p.diag[37] = p.solver_workspace->coupled_numeric_factorizations;
      p.diag[38] = p.solver_workspace->coupled_numeric_reuses;
      p.diag[39] = p.solver_workspace->monitor_reuses;
      p.diag[40] = rough_pairs.size();
      p.diag[41] = rough_power * power_scale;
      p.diag[42] = rough_stress(0,2) * stress_scale;
      p.diag[43] = rolling_yield;
      p.diag[44] = twisting_yield;
      p.diag[45] = rolling_energy * energy_scale;
      p.diag[46] = twisting_energy * energy_scale;
      p.diag[47] = rolling_work / dt * power_scale;
      p.diag[48] = twisting_work / dt * power_scale;
      p.diag[49] = rotation_power * power_scale;
      p.diag[50] = rough_geometry_time;
      p.diag[16] = geometry_time;
      p.diag[17] = middle_time;
      p.diag[18] = solve_time;
      p.diag[19] = MPI_Wtime() - begun;
      for (int i = 0; i < n; ++i) {
        physical.segment<3>(6 * i) =
            length * rate * (result.velocity.segment<3>(6 * i) + kappa * vec(bodies[i].position));
        physical.segment<3>(6 * i + 3) = rate * (result.velocity.segment<3>(6 * i + 3) + omega_inf);
      }
      for (int i = 0; i < n; ++i) {
        double fs = p.eta * length * length * rate, ms = fs * length;
        moments.segment<3>(17 * i) = -fs * result.total_moment.segment<3>(11 * i);
        moments.segment<3>(17 * i + 3) = -ms * result.total_moment.segment<3>(11 * i + 3);
        moments.segment<5>(17 * i + 6) = ms * result.total_moment.segment<5>(11 * i + 6);
        moments.segment<3>(17 * i + 11) = fs * contact_moments.segment<3>(6 * i);
        moments.segment<3>(17 * i + 14) = ms * contact_moments.segment<3>(6 * i + 3);
      }
      p.warm = std::move(result);
      p.have_warm = true;
    } catch (const std::exception &e) {
      failed = 1;
      message = e.what();
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
    store_momentum(atom, p.avec, i, physical.segment<3>(6 * j + 3), p.flake);
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
    for (int k = 25; k < static_cast<int>(p.diag.size()); ++k)
      o << ',' << p.diag[k];
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

