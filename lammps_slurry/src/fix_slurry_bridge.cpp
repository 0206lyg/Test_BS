// Additively regularized spheroid lubrication + periodic mobility + surface contact.
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
  double mu = 10., kt = .24, mu_roll = 10., kr = .24; // kn, kt, kr: N/m
  double spectral = 2, spectral_tol = 1e-8, rtol = 1e-7, avg_start = 5;
  int maxit = 1200, interval = 100, nmax = 0;
  std::ofstream csv;
  std::array<double, 44> diag{};
  std::array<double, 4> stress{};
  std::array<long double, 4> interval_stress{};
  long double selected_integral = 0;
  double selected_duration = 0, interval_duration = 0, interval_start = 0;
  bigint interval_steps = 0, last_output = -1;
  std::map<PairKey, std::vector<History>> history, pending;
  std::vector<tagint> tags;
  // Current physical velocities are exposed through atom v/angmom. The
  // corrector's average velocities are private data for the next advance.
  std::vector<tagint> advance_tags;
  Eigen::VectorXd advance_velocity;
  bs::Result warm;
  // Reuse FFT plans/buffers and exact solver caches across physical steps.
  // reset() still refreshes all position, orientation and lattice coefficients.
  pm::Mobility mobility;
  std::shared_ptr<bs::detail::SolverWorkspace> solver_workspace =
      std::make_shared<bs::detail::SolverWorkspace>();
  bool have_warm = false;
  bool direct_mobility = false;
};
FixSlurryBridge::FixSlurryBridge(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), p_(new Impl) {
  if (narg < 8)
    error->all(FLERR, "slurry/bridge requires eta shear_rate nominal_diameter epsilon kn "
                      "[contact mu_s kt] [rolling mu_r kr_N_per_m] [shape spheroid] "
                      "[spectral xi_a] [spectral_tolerance tol] [mobility_backend mesh|direct] "
                      "[tolerance rtol] [iterations maxit] [history file interval average_start_strain]");
  auto &p = *p_;
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
  if (!(p.eta > 0 && p.nominal_diameter > 0 && p.epsilon > 0 && p.kn > 0 &&
        p.mu >= 0 && p.kt >= 0 && p.mu_roll >= 0 && p.kr >= 0 &&
        p.spectral > 0 && p.spectral_tol > 0 && p.spectral_tol < 1 && p.rtol > 0 && p.rtol < 1 &&
        p.maxit > 0 && p.interval > 0 && p.avg_start >= 0) ||
      !std::isfinite(p.eta + p.shear + p.nominal_diameter + p.epsilon + p.kn + p.mu + p.kt +
                     p.mu_roll + p.kr + p.spectral + p.spectral_tol + p.rtol + p.avg_start) ||
      !(p.regularization_length > 0 && std::isfinite(p.regularization_length)) ||
      (p.mu > 0 && p.kt == 0) || (p.mu_roll > 0 && p.kr == 0))
    error->all(FLERR, "Invalid slurry/bridge material or numerical parameter");
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
             "next_step_end_time_s,next_step_rolling_projection_work_J\n";
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
                  moments = Eigen::VectorXd::Zero(17 * n);
  int failed = 0;
  std::string message;
  if (comm->me == 0)
    try {
      const double length = p.reference_radius, rate = std::max(std::abs(p.shear), 1.);
      for (const auto &r : records)
        for (double a : r.axes)
          if (!(a > 0))
            throw std::runtime_error("non-positive shape axis");
      for (const auto &r : records) {
        const double volume_radius = std::cbrt(r.axes[0]*r.axes[1]*r.axes[2]);
        const double aspect = r.axes[0]/r.axes[2];
        if (std::abs(volume_radius-length) > 1e-8*length ||
            std::abs(r.axes[0]-r.axes[1]) > 1e-10*r.axes[0] ||
            aspect < 1.-1e-12 || aspect > 5.+1e-12)
          throw std::runtime_error("bridge requires oblate spheroids with AR 1..5 and volume-equivalent diameter matching nominal_diameter");
      }
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
        // Retain the existing positive periodic ellipsoid mobility approximation.
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
      struct Frame {
        bs::Problem problem;
        step::Geometry geometry;
        std::vector<PairKey> keys;
        std::vector<V3> displacements;
        double min_gap=0., overlap=0., geometry_time=0., middle_time=0.;
        int active_near=0;
        pm::Settings settings;
      };
      auto assemble = [&](const std::vector<gr::Body> &stage_bodies, const M3 &stage_H,
                          const std::vector<Eigen::Quaterniond> &stage_quaternions,
                          const std::map<PairKey,std::vector<History>> &stage_history) -> Frame {
        std::vector<pm::Body> stage_middle_bodies(n);
        for(int i=0;i<n;++i) {
          stage_middle_bodies[i].x=vec(stage_bodies[i].position);
          stage_middle_bodies[i].shape=cb::shape(stage_bodies[i],1.);
        }
      bs::Problem problem;
      problem.count = n;
      problem.dt = dt;
      problem.external = Eigen::VectorXd::Zero(6 * n);
      problem.prescribed = Eigen::VectorXd::Zero(11 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          problem.prescribed[11 * i + 6 + k] = -(E.array() * cb::strain_basis()[k].array()).sum();
      std::vector<Eigen::Triplet<double>> entries;
      std::vector<PairKey> contact_keys;
      std::vector<V3> contact_displacements;
      std::vector<im::ImageIndex> contact_images;
      int active_near = 0;
      double min_gap = std::numeric_limits<double>::infinity(), overlap = 0;
      double section = MPI_Wtime();
      for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
          const double ai = shape::bounding_radius(stage_bodies[i],false),
                       aj = shape::bounding_radius(stage_bodies[j],false);
          const double rc = ai + aj + std::min(ai,aj);
          const V3 original = vec(stage_bodies[j].position) - vec(stage_bodies[i].position);
          for (const auto &image : im::images_within(original, stage_H, rc)) {
            if (i == j && !im::positive_image(image.image))
              continue;
            const V3 d = image.displacement;
            auto bj = stage_bodies[j];
            bj.position = arr(vec(stage_bodies[i].position) + d);
            const auto gap = gr::closestEllipsoidGap(stage_bodies[i],bj);
            const auto near = cb::evaluate(stage_bodies[i],bj,gap,1.,1.,p.regularization_length/length);
            const auto &near_resistance = near.resistance;
            const bool near_active = near.active;
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
            contact.onset = 0.; // actual surface overlap, never a positive-gap shell
            contact.kn = p.kn / (p.eta * length * rate);
            contact.kt = p.kt / (p.eta * length * rate);
            contact.mu = p.mu;
            contact.mu_roll = p.mu_roll;
            // Singh a_ij=2*a_i*a_j/(a_i+a_j); monodisperse nominal radii give a_ij=a_ref.
            contact.roll_length = p.reference_radius / length;
            const double torque_scale = p.eta * std::pow(length,3) * rate;
            contact.kr = p.kr * p.reference_radius * p.reference_radius / torque_scale;
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
        }
      problem.near_resistance.resize(11 * n, 11 * n);
      problem.near_resistance.setFromTriplets(entries.begin(), entries.end());
      problem.near_resistance.makeCompressed();
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
      mobility.reset(stage_middle_bodies, stage_H, settings);
      problem.self_mobility = mobility.self_blocks();
      problem.mobility = [&mobility](const Eigen::VectorXd &x) { return mobility.apply(x); };
      double middle_time = MPI_Wtime() - section;
        Frame out;
        out.problem=std::move(problem);
        out.keys=std::move(contact_keys);
        out.displacements=std::move(contact_displacements);
        out.min_gap=min_gap;out.overlap=overlap;out.active_near=active_near;
        out.geometry_time=geometry_time;out.middle_time=middle_time;out.settings=settings;
        out.geometry.bodies=stage_bodies;out.geometry.quaternions=stage_quaternions;
        out.geometry.lattice=stage_H;out.geometry.kappa=kappa;
        out.geometry.ambient_rotation=omega_inf;out.geometry.contact_onset=0.;
        out.geometry.flake=false;out.geometry.gap_tolerance=1e-8*p.regularization_length/length;
        out.geometry.enforce_work_balance=false;out.geometry.images=std::move(contact_images);
        return out;
      };
      auto current=assemble(bodies,H,quaternions,p.history);
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
      // Explicit Heun predictor/corrector. Both hydrodynamic stages are
      // evaluated at their own configuration and current spring histories.
      auto initial=SlurryBridgeLinearSolver::solve(problem,options,p.have_warm?&p.warm:nullptr);
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
      auto predicted=assemble(predictor_endpoint.bodies,predictor_endpoint.lattice,
                              predictor_quaternions,predictor_history);
      auto predicted_solution=SlurryBridgeLinearSolver::solve(predicted.problem,options,&initial);
      geometry_time+=predicted.geometry_time;middle_time+=predicted.middle_time;
      Eigen::VectorXd integration_velocity=.5*(initial.velocity+predicted_solution.velocity);
      // Average TOTAL translational velocities. The affine stage-1 velocity
      // must be evaluated at the predicted position, not at the old position.
      for(int i=0;i<n;++i)
        integration_velocity.segment<3>(6*i)+=.5*kappa*(
            vec(predictor_endpoint.bodies[i].position)-vec(bodies[i].position));
      step::Result consistent;
      consistent.endpoint=geometry.advance(problem,integration_velocity,true);
      // This explicit step does not enforce equality between CURRENT load
      // and the NEXT endpoint load. An implicit consistency residual is N/A.
      consistent.maximum_gap_error=std::numeric_limits<double>::quiet_NaN();
      for(std::size_t k=0;k<problem.contacts.size();++k) {
        const auto &c=problem.contacts[k];
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
      std::map<PairKey,std::vector<History>> corrected_history;
      double corrected_rolling_work=0.;
      int corrected_rolling_yield=0;
      for(std::size_t k=0;k<predicted.problem.contacts.size();++k) {
        const auto &c1=predicted.problem.contacts[k];
        const auto &r1=predicted_solution.contacts[k];
        const auto image=predicted.geometry.images[k];
        const V3 d_end=vec(consistent.endpoint.bodies[c1.j].position)-
                       vec(consistent.endpoint.bodies[c1.i].position)+
                       consistent.endpoint.lattice*image.cast<double>();
        auto second=consistent.endpoint.bodies[c1.j];
        second.position=arr(vec(consistent.endpoint.bodies[c1.i].position)+d_end);
        const auto gap_end=gr::closestEllipsoidGap(consistent.endpoint.bodies[c1.i],second);
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
      auto result = std::move(consistent.solution);
      // Predictor assembly is already counted in the geometry/middle totals.
      double solve_time = MPI_Wtime() - section - predicted.geometry_time - predicted.middle_time;
      M3 hydro = M3::Zero(), contact_stress = M3::Zero();
      Eigen::VectorXd contact_moments = Eigen::VectorXd::Zero(6 * n);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 5; ++k)
          hydro -= result.total_moment[11 * i + 6 + k] * cb::strain_basis()[k];
      p.pending.clear();
      int contacts = 0;
      double normal_energy = 0., tangent_energy = 0., rolling_energy = 0.;
      // Legacy CSV yield/work columns refer to the prepared [t_n,t_n+dt]
      // corrector; energy and contact power remain CURRENT-state quantities.
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
      const double solvent_power = volume * (p.shear / rate) * (p.shear / rate),
                   hydro_power = (solvent_power + result.hydro_power) * power_scale,
                   contact_power = result.contact_power * power_scale;
      const double stress_power = p.stress[3] * p.shear * volume * std::pow(length, 3);
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
      p.diag[40] = rolling_yield;
      p.diag[41] = rolling_energy * energy_scale;
      p.diag[42] = rolling_work / dt * power_scale;
      p.diag[43] = rotation_power * power_scale;
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
      p.warm = std::move(predicted_solution);
      p.have_warm = true;
    } catch (const bs::NonlinearFailure &e) {
      failed = 1;
      std::ostringstream diagnostic;
      diagnostic << e.what() << std::scientific << std::setprecision(9)
                 << "; residual=" << e.residual << "; tolerance=" << e.tolerance
                 << "; Newton iterations=" << e.newton_iterations
                 << "; linear iterations=" << e.linear_iterations;
      message = diagnostic.str();
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
  MPI_Bcast(advance_physical.data(), 6 * n, MPI_DOUBLE, 0, world);
  // Every rank retains the same global tag ordering, so migration does not
  // turn a prepared velocity into an index referring to another particle.
  p.advance_tags.resize(n);
  for(int j=0;j<n;++j)p.advance_tags[j]=records[j].tag;
  p.advance_velocity=std::move(advance_physical);
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
    for (int k = 25; k < static_cast<int>(p.diag.size()); ++k)
      o << ',' << p.diag[k];
    // These forward-step fields are explicitly time-labelled. Work is the
    // corrected rolling return-map projection, not a cumulative integral.
    o << ',' << time+update->dt << ',' << p.diag[42]*update->dt << '\n';
    o.flush();
    if (!o)
      error->one(FLERR, "Failed writing slurry/bridge history");
  }
  p.last_output = update->ntimestep;
  p.interval_steps = 0;
  p.interval_duration = 0;
  p.interval_stress.fill(0);
}

