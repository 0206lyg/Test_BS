#ifdef FIX_CLASS
FixStyle(slurry/implicit,FixSlurryImplicit);
#else
#ifndef LMP_FIX_SLURRY_IMPLICIT_H
#define LMP_FIX_SLURRY_IMPLICIT_H
#include "fix.h"
#include "slurry_hydro_math.h"
#include "slurry_stress_math.h"
#include "ellipsoidGap.h"
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace SlurryImplicit { struct SolverWorkspace; }
namespace SlurryFiniteGap { struct Catalogue; }
namespace SlurryScattering { class Provider; }
namespace SlurrySPDTable { class Provider; }
namespace LAMMPS_NS {
class FixSlurryImplicit : public Fix {
 public:
  FixSlurryImplicit(class LAMMPS *, int, char **);
  ~FixSlurryImplicit() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void end_of_step() override;
  double compute_vector(int) override;
 private:
  class AtomVecEllipsoid *avec_ = nullptr;
  double viscosity_, shear_, cutoff_, roughness_, stiffness_;
  double contact_gap_=0., friction_=0., tangential_stiffness_=0.;
  bool mari_lubrication_=false;
  bool spheroid_lubrication_=false, finite_gap_lubrication_=false, root_solver_=true;
  bool finite_gap_fast_=false;
  double finite_gap_range_=0.5, finite_gap_inner_ratio_=0.8;
  std::unique_ptr<SlurryFiniteGap::Catalogue> finite_gap_catalogue_;
#ifdef SLURRY_HAVE_EIGEN
  std::unique_ptr<SlurryScattering::Provider> finite_gap_provider_;
  std::unique_ptr<SlurrySPDTable::Provider> finite_gap_fast_provider_;
#endif
  int preconditioner_backend_=0;
  bool adaptive_forcing_=false;
  bool solver_cache_=false;
  std::unique_ptr<SlurryImplicit::SolverWorkspace> solver_workspace_;
  std::vector<tagint> solver_tags_;
  double solver_cache_dt_=0.,solver_cache_length_=0.;
  double ilu_drop_tolerance_=1e-4;
  int ilu_fill_factor_=20,ilu_refresh_iterations_=40;
  double lubrication_delta_=-1., lubrication_range_=-1.;
  std::array<double,16> diagnostics_{};
  bool have_solution_ = false;
  std::map<std::array<double,3>,SlurryHydro::Resistance> resistance_cache_;
  std::map<std::array<double,3>,SlurryStress::Coefficients> stress_cache_;
  struct ContactHistory {
    std::array<double,3> elastic{},normal{},common_angle{};
  };
  using ContactKey=std::pair<tagint,tagint>;
  std::map<ContactKey,ContactHistory> history_,pending_history_;
  std::map<ContactKey,slurry::gr_re2::graphite::GapCache> gap_cache_;
  struct Snapshot {
    int kink_reselections=0;
    std::array<double,7> solver_counts{};
    std::array<double,4> lu_counts{};
    std::array<double,8> profile{};
    double maximum_gap_curvature_ratio=0.,completion_added_power=0.;
    double finite_gap_stress=0.,finite_gap_power=0.,finite_gap_bound_min=0.;
    double force_residual_N=0.,force_tolerance_N=0.;
    std::array<double,9> stress{};
    std::array<double,16> diagnostic{};
    double normal_energy=0.,tangent_energy=0.,sliding_power=0.,hydro_power=0.;
    double stress_power=0.,elastic_rate=0.,discrete_dissipation=0.,power_error=0.;
    double normal_diff1=0.,normal_diff2=0.;
    std::array<double,3> lubrication_power{};
  } snapshot_;
  std::ofstream history_file_;
  int output_interval_=0;
  double average_start_=0.;
  bigint last_output_step_=-1,interval_steps_=0;
  double interval_duration_=0.,interval_start_strain_=0.,interval_end_strain_=0.;
  std::array<long double,9> interval_stress_{};
  long double interval_finite_gap_stress_=0.;
  std::array<long double,8> interval_profile_{};
  double interval_max_overlap_=0.,interval_max_compression_=0.,interval_max_residual_=0.;
  long double selected_stress_integral_=0.;
  double selected_duration_=0.;
  void solve_velocities();
  double simulation_time() const;
  void accumulate_step();
  void write_history();
};
}
#endif
#endif
