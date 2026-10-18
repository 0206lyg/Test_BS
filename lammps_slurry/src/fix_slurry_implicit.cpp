// Athermal zero-inertia integration. The finite regularized lubrication law
// is solved simultaneously with one-body drag and backward-Euler contact.
// Geometry is frozen in each solve; configuration updates are first order.
#include "fix_slurry_implicit.h"
#include "slurry_implicit_math.h"
#include "slurry_history_math.h"
#include "slurry_pair_hydro_math.h"
#include "slurry_spheroid_hydro_math.h"
#include "slurry_finite_gap_math.h"
#include "slurry_grand_system_check.h"
#ifdef SLURRY_HAVE_EIGEN
#include "slurry_scattering_provider.h"
#include "slurry_spd_table.h"
#endif
#include "ellipsoidGap.h"
#include "nearFieldResistance.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "modify.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
namespace gr = slurry::gr_re2::graphite;
namespace si = SlurryImplicit;
namespace ph = SlurryPairHydro;
namespace sh = SlurrySpheroidHydro;
namespace fg = SlurryFiniteGap;
namespace gp = SlurryGrandPair;

namespace {
const char *profile_names[]={"geometry","communication","solver","diagnostics","total",
                             "preconditioner_setup","preconditioner_solve","linear_apply"};
struct SolverReport {
  double residual_norm=0.,relative_residual=0.,residual_scale=0.;
  int cg=0,gmres=0,newton=0,contacts=0,kinks=0;
  int lu_counts[4]{};
  double counts[4]{},factor_nonzeros=0.,setup_seconds=0.,precondition_seconds=0.,apply_seconds=0.;
};
struct Record {
  tagint tag;
  double position[3], axes[3], quat[4], mass, velocity[3], momentum[3];
};
template<class T> std::vector<T> gather_records(const std::vector<T>& local, MPI_Comm world,
                                              int nprocs)
{
  static_assert(std::is_trivially_copyable<T>::value,"MPI records must be POD");
  const int bytes = static_cast<int>(local.size()*sizeof(T));
  std::vector<int> sizes(nprocs), offsets(nprocs);
  MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,world);
  int total=0;
  for(int p=0;p<nprocs;++p) { offsets[p]=total; total+=sizes[p]; }
  std::vector<T> all(total/sizeof(T));
  MPI_Allgatherv(local.data(),bytes,MPI_BYTE,all.data(),sizes.data(),offsets.data(),MPI_BYTE,world);
  return all;
}
gr::Body body_from_record(const Record &r)
{
  gr::Body b;
  double q[4],rot[3][3];
  std::copy(r.quat,r.quat+4,q);
  MathExtra::quat_to_mat(q,rot);
  for(int k=0;k<3;++k) {
    b.position[k]=r.position[k]; b.axes[k]=r.axes[k];
    b.velocity[k]=r.velocity[k];
    for(int l=0;l<3;++l) b.rotation[3*k+l]=rot[k][l];
    b.inertiaBody[k]=r.mass/5*(r.axes[(k+1)%3]*r.axes[(k+1)%3]+
                              r.axes[(k+2)%3]*r.axes[(k+2)%3]);
  }
  b.mass=r.mass;
  const gr::Vec3 momentum{r.momentum[0],r.momentum[1],r.momentum[2]};
  auto wb=gr::mul(gr::transpose(b.rotation),momentum);
  for(int k=0;k<3;++k)wb[k]/=b.inertiaBody[k];
  b.omega=gr::mul(b.rotation,wb);
  return b;
}
void store_momentum(Atom *atom, AtomVecEllipsoid *avec, int i,const gr::Vec3& omega)
{
  auto &b=avec->bonus[atom->ellipsoid[i]];
  double rotation[3][3], obody[3], lb[3], l[3];
  MathExtra::quat_to_mat(b.quat,rotation);
  MathExtra::transpose_matvec(rotation,omega.data(),obody);
  for(int k=0;k<3;++k)
    lb[k]=atom->rmass[i]/5*(b.shape[(k+1)%3]*b.shape[(k+1)%3]+
                           b.shape[(k+2)%3]*b.shape[(k+2)%3])*obody[k];
  MathExtra::matvec(rotation,lb,l);
  std::copy(l,l+3,atom->angmom[i]);
}
}

FixSlurryImplicit::FixSlurryImplicit(LAMMPS *lmp,int narg,char **arg):Fix(lmp,narg,arg)
{
  if(narg<8) error->all(FLERR,"Fix slurry/implicit requires eta shear_rate hcut hmin kn [lubrication legacy|mari|spheroid|finite_gap|finite_gap_fast] [finite_gap_table path] [finite_gap_fast_table path] [finite_gap_range lambda] [solver_preconditioner auto|sparse|nested|lu|ilu] [solver_layout root|replicated] [contact_gap hc] [friction mu kt] [history file interval average_start_strain]");
  viscosity_=utils::numeric(FLERR,arg[3],false,lmp);
  shear_=utils::numeric(FLERR,arg[4],false,lmp);
  cutoff_=utils::numeric(FLERR,arg[5],false,lmp);
  roughness_=utils::numeric(FLERR,arg[6],false,lmp);
  stiffness_=utils::numeric(FLERR,arg[7],false,lmp);
  std::string history_path,finite_gap_table_path,finite_gap_basis_path,finite_gap_fast_table_path;
  int finite_gap_film_grid=161;
  for(int i=8;i<narg;) {
    if(std::strcmp(arg[i],"lubrication")==0 && i+1<narg) {
      mari_lubrication_=spheroid_lubrication_=finite_gap_lubrication_=finite_gap_fast_=false;
      if(std::strcmp(arg[i+1],"mari")==0) mari_lubrication_=true;
      else if(std::strcmp(arg[i+1],"spheroid")==0) spheroid_lubrication_=true;
      else if(std::strcmp(arg[i+1],"finite_gap")==0) finite_gap_lubrication_=true;
      else if(std::strcmp(arg[i+1],"finite_gap_fast")==0) finite_gap_lubrication_=finite_gap_fast_=true;
      else if(std::strcmp(arg[i+1],"legacy")!=0)
        error->all(FLERR,"slurry/implicit lubrication must be legacy, mari, spheroid, finite_gap or finite_gap_fast");
      i+=2;
    } else if(std::strcmp(arg[i],"finite_gap_table")==0 && i+1<narg) {
      finite_gap_table_path=arg[i+1];i+=2;
    } else if(std::strcmp(arg[i],"finite_gap_basis")==0 && i+1<narg) {
      finite_gap_basis_path=arg[i+1];i+=2;
    } else if(std::strcmp(arg[i],"finite_gap_fast_table")==0 && i+1<narg) {
      finite_gap_fast_table_path=arg[i+1];i+=2;
    } else if(std::strcmp(arg[i],"finite_gap_film_grid")==0 && i+1<narg) {
      finite_gap_film_grid=utils::inumeric(FLERR,arg[i+1],false,lmp);i+=2;
    } else if(std::strcmp(arg[i],"finite_gap_range")==0 && i+1<narg) {
      finite_gap_range_=utils::numeric(FLERR,arg[i+1],false,lmp);i+=2;
    } else if(std::strcmp(arg[i],"solver_preconditioner")==0 && i+1<narg) {
      if(std::strcmp(arg[i+1],"auto")==0) preconditioner_backend_=0;
      else if(std::strcmp(arg[i+1],"sparse")==0) preconditioner_backend_=1;
      else if(std::strcmp(arg[i+1],"nested")==0) preconditioner_backend_=2;
      else if(std::strcmp(arg[i+1],"lu")==0) preconditioner_backend_=3;
      else if(std::strcmp(arg[i+1],"ilu")==0) preconditioner_backend_=4;
      else error->all(FLERR,"slurry/implicit solver_preconditioner must be auto, sparse, nested, lu or ilu");
      i+=2;
    } else if(std::strcmp(arg[i],"solver_forcing")==0 && i+1<narg) {
      if(std::strcmp(arg[i+1],"adaptive")==0) adaptive_forcing_=true;
      else if(std::strcmp(arg[i+1],"fixed")==0) adaptive_forcing_=false;
      else error->all(FLERR,"slurry/implicit solver_forcing must be fixed or adaptive");
      i+=2;
    } else if(std::strcmp(arg[i],"solver_cache")==0 && i+1<narg) {
      if(std::strcmp(arg[i+1],"on")==0) solver_cache_=true;
      else if(std::strcmp(arg[i+1],"off")==0) solver_cache_=false;
      else error->all(FLERR,"slurry/implicit solver_cache must be on or off");
      i+=2;
    } else if(std::strcmp(arg[i],"solver_ilu")==0 && i+3<narg) {
      ilu_drop_tolerance_=utils::numeric(FLERR,arg[i+1],false,lmp);
      ilu_fill_factor_=utils::inumeric(FLERR,arg[i+2],false,lmp);
      ilu_refresh_iterations_=utils::inumeric(FLERR,arg[i+3],false,lmp);
      i+=4;
    } else if(std::strcmp(arg[i],"solver_layout")==0 && i+1<narg) {
      if(std::strcmp(arg[i+1],"root")==0) root_solver_=true;
      else if(std::strcmp(arg[i+1],"replicated")==0) root_solver_=false;
      else error->all(FLERR,"slurry/implicit solver_layout must be root or replicated");
      i+=2;
    } else if(std::strcmp(arg[i],"lubrication_delta")==0 && i+1<narg) {
      lubrication_delta_=utils::numeric(FLERR,arg[i+1],false,lmp);
      if(!(lubrication_delta_>0.) || !std::isfinite(lubrication_delta_))
        error->all(FLERR,"slurry/implicit lubrication_delta must be positive and finite");
      i+=2;
    } else if(std::strcmp(arg[i],"lubrication_range")==0 && i+1<narg) {
      lubrication_range_=utils::numeric(FLERR,arg[i+1],false,lmp);
      if(!(lubrication_range_>0.) || !std::isfinite(lubrication_range_))
        error->all(FLERR,"slurry/implicit lubrication_range must be positive and finite");
      i+=2;
    } else if(std::strcmp(arg[i],"contact_gap")==0 && i+1<narg) {
      contact_gap_=utils::numeric(FLERR,arg[i+1],false,lmp);i+=2;
    } else if(std::strcmp(arg[i],"friction")==0 && i+2<narg) {
      friction_=utils::numeric(FLERR,arg[i+1],false,lmp);
      tangential_stiffness_=utils::numeric(FLERR,arg[i+2],false,lmp);i+=3;
    } else if(std::strcmp(arg[i],"history")==0 && i+3<narg) {
      history_path=arg[i+1];output_interval_=utils::inumeric(FLERR,arg[i+2],false,lmp);
      average_start_=utils::numeric(FLERR,arg[i+3],false,lmp);i+=4;
    } else error->all(FLERR,"Invalid slurry/implicit optional arguments");
  }
  if(!(viscosity_>0 && cutoff_>roughness_ && roughness_>0 && stiffness_>0) ||
     !std::isfinite(viscosity_+shear_+cutoff_+roughness_+stiffness_))
    error->all(FLERR,"Invalid slurry/implicit material parameters");
  if(!(contact_gap_>=0. && contact_gap_<cutoff_ && friction_>=0. && tangential_stiffness_>=0.) ||
     !std::isfinite(contact_gap_+friction_+tangential_stiffness_+average_start_) ||
     (friction_>0. && tangential_stiffness_<=0.) || average_start_<0. ||
     (!history_path.empty() && output_interval_<=0))
    error->all(FLERR,"Invalid slurry/implicit contact/history parameters");
  if(!mari_lubrication_ && (lubrication_delta_>0. || lubrication_range_>0.))
    error->all(FLERR,"Relative lubrication parameters require lubrication mari");
  if(finite_gap_fast_) {
    if(finite_gap_fast_table_path.empty() || !finite_gap_table_path.empty() || !finite_gap_basis_path.empty() ||
       finite_gap_film_grid!=161 || !(finite_gap_range_>0.) || !std::isfinite(finite_gap_range_))
      error->all(FLERR,"finite_gap_fast requires only finite_gap_fast_table and positive finite_gap_range; no runtime film grid or fallback");
#ifdef SLURRY_HAVE_EIGEN
    try {
      finite_gap_fast_provider_=std::make_unique<SlurrySPDTable::Provider>();
      finite_gap_fast_provider_->load(finite_gap_fast_table_path);
      finite_gap_fast_provider_->epsilon_m=roughness_;
    } catch(const std::exception&e){error->all(FLERR,"Cannot load precomputed finite-gap trial: {}",e.what());}
#else
    error->all(FLERR,"finite_gap_fast requires bundled Eigen enabled");
#endif
  } else if(!finite_gap_fast_table_path.empty()) {
    error->all(FLERR,"finite_gap_fast_table requires lubrication finite_gap_fast; no implicit model change");
  } else if(finite_gap_lubrication_) {
    if(finite_gap_film_grid<21 || !(finite_gap_range_>0.)||!std::isfinite(finite_gap_range_)||
       (finite_gap_table_path.empty()==finite_gap_basis_path.empty()))
      error->all(FLERR,"Experimental finite_gap requires exactly one finite_gap_table or finite_gap_basis and positive finite_gap_range");
    try {
      if(!finite_gap_table_path.empty())finite_gap_catalogue_=std::make_unique<fg::Catalogue>(fg::read_catalogue(finite_gap_table_path));
      else {
#ifdef SLURRY_HAVE_EIGEN
        finite_gap_provider_=std::make_unique<SlurryScattering::Provider>();finite_gap_provider_->load(finite_gap_basis_path);
        finite_gap_provider_->epsilon_m=roughness_;finite_gap_provider_->film_grid=finite_gap_film_grid;
#else
        error->all(FLERR,"Experimental finite_gap_basis requires bundled Eigen enabled");
#endif
      }
    } catch(const std::exception&e){error->all(FLERR,"Cannot load experimental finite-gap model: {}",e.what());}
  } else if(!finite_gap_table_path.empty()||!finite_gap_basis_path.empty()) error->all(FLERR,"finite_gap_table/basis requires lubrication finite_gap");
  if(!history_path.empty() && comm->me==0) {
    history_file_.open(history_path,std::ios::out|std::ios::trunc);
    if(!history_file_) error->one(FLERR,"Cannot open slurry rheology history {}",history_path);
    history_file_<<std::setprecision(17);
    history_file_<<"step,time_s,strain";
    const char *name[]={"solvent","single_free","single_rotation","lub_normal","lub_tangent","contact_normal","contact_tangent","lub_pump","xy"};
    for(auto s:name)history_file_<<",sigma_"<<s<<"_Pa";
    history_file_<<",eta_Pa_s,eta_relative";
    for(int k=0;k<8;++k)history_file_<<",eta_"<<name[k]<<"_Pa_s";
    history_file_<<",contacts,sticking,sliding,lubrication_pairs,actual_overlaps,max_contact_compression_m,max_overlap_m,min_gap_m,force_residual,linear_iterations,newton_iterations,kink_reselections,max_speed_m_s,coulomb_violation_N";
    history_file_<<",normal_elastic_energy_J,tangential_elastic_energy_J,sliding_power_W,hydro_dissipation_W,stress_power_W,elastic_energy_rate_W,contact_discrete_dissipation_W,power_balance_error_W,N1_Pa,N2_Pa,lub_squeeze_dissipation_W,lub_shear_dissipation_W,lub_pump_dissipation_W";
    history_file_<<",interval_steps,interval_start_strain,interval_end_strain";
    for(auto s:name)history_file_<<",interval_mean_sigma_"<<s<<"_Pa";
    history_file_<<",interval_mean_eta_Pa_s,interval_mean_eta_relative";
    for(int k=0;k<8;++k)history_file_<<",interval_mean_eta_"<<name[k]<<"_Pa_s";
    history_file_<<",interval_max_contact_compression_m,interval_max_overlap_m,interval_max_force_residual,mean_eta_Pa_s,average_duration_s";
    history_file_<<",solver_cg_iterations,solver_gmres_iterations,sparse_factorizations,sparse_fallbacks,sparse_preconditioner_solves,sparse_factor_nonzeros,sparse_preconditioner_reuses";
    for(auto name:profile_names)history_file_<<",profile_"<<name<<"_s";
    for(auto name:profile_names)history_file_<<",interval_mean_profile_"<<name<<"_s";
    history_file_<<",spheroid_max_gap_curvature_ratio,spheroid_completion_added_power_W,force_residual_N,force_tolerance_N,lu_factorizations,lu_solves,lu_fallbacks,forcing_retries,sigma_finite_gap_Pa,eta_finite_gap_Pa_s,finite_gap_correction_power_W,interval_mean_sigma_finite_gap_Pa,finite_gap_strain_bound_min\n";
  }
  time_integrate=1;
  vector_flag=1;size_vector=16;global_freq=1;extvector=0;nevery=1;
}
FixSlurryImplicit::~FixSlurryImplicit() = default;
int FixSlurryImplicit::setmask() { return INITIAL_INTEGRATE | END_OF_STEP; }
void FixSlurryImplicit::init()
{
  avec_=dynamic_cast<AtomVecEllipsoid*>(atom->style_match("ellipsoid"));
  if(!avec_ || atom->superellipsoid_flag || igroup!=0 ||
     std::strcmp(update->unit_style,"si")!=0 || domain->dimension!=3 ||
     !domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR,"slurry/implicit requires group all, SI, and periodic 3D ellipsoid atoms");
  if(!force->pair_style || std::strcmp(force->pair_style,"zero")!=0)
    error->all(FLERR,"slurry/implicit owns the force law: use pair_style zero for communication");
  if(domain->xz!=0 || domain->yz!=0)
    error->all(FLERR,"slurry/implicit supports xy shear only");
  bool preceding_shear=false;
  for(auto *f:modify->get_fix_list()) {
    if(f==this) break;
    if(std::strcmp(f->style,"deform")==0) {
      if(f->nevery!=0)
        error->all(FLERR,"slurry/implicit requires fix deform 0 so shear geometry updates before periodic wrapping");
      preceding_shear=true;
    }
  }
  if(shear_!=0 && !preceding_shear)
    error->all(FLERR,"Define fix deform 0 xy before slurry/implicit");
  for(auto *f:modify->get_fix_list()) {
    if(f!=this && (f->time_integrate || std::strcmp(f->style,"slurry/drag")==0))
      error->all(FLERR,"Do not combine slurry/implicit with another integrator or slurry/drag");
  }
  for(int i=0;i<atom->nlocal;++i)
    if(atom->ellipsoid[i]<0 || !(atom->rmass[i]>0))
      error->one(FLERR,"slurry/implicit requires finite-size ellipsoids with positive mass");
}
void FixSlurryImplicit::setup(int) { solve_velocities();write_history(); }
void FixSlurryImplicit::end_of_step() { solve_velocities();write_history(); }
double FixSlurryImplicit::compute_vector(int i) { return diagnostics_[i]; }

double FixSlurryImplicit::simulation_time() const
{
  return update->atime+static_cast<double>(update->ntimestep-update->atimestep)*update->dt;
}

void FixSlurryImplicit::accumulate_step()
{
  if(output_interval_==0)return;
  const double dt=update->dt, end=simulation_time(), start=end-dt;
  if(interval_steps_==0)interval_start_strain_=std::abs(shear_)*start;
  interval_end_strain_=std::abs(shear_)*end;
  ++interval_steps_;interval_duration_+=dt;
  for(int c=0;c<9;++c)interval_stress_[c]+=static_cast<long double>(snapshot_.stress[c])*dt;
  interval_finite_gap_stress_+=static_cast<long double>(snapshot_.finite_gap_stress)*dt;
  for(int c=0;c<8;++c)interval_profile_[c]+=snapshot_.profile[c];
  interval_max_overlap_=std::max(interval_max_overlap_,snapshot_.diagnostic[2]);
  interval_max_compression_=std::max(interval_max_compression_,snapshot_.diagnostic[9]);
  interval_max_residual_=std::max(interval_max_residual_,snapshot_.diagnostic[3]);
  if(shear_!=0.) {
    const double selected=std::max(0.,end-std::max(start,average_start_/std::abs(shear_)));
    selected_duration_+=selected;
    selected_stress_integral_+=static_cast<long double>(snapshot_.stress[8])*selected;
  }
}

void FixSlurryImplicit::write_history()
{
  if(output_interval_==0 || last_output_step_==update->ntimestep ||
     (update->ntimestep%output_interval_!=0 && update->ntimestep!=update->laststep))return;
  const double nan=std::numeric_limits<double>::quiet_NaN();
  const auto eta=[&](double stress){return shear_!=0.?stress/shear_:nan;};
  const auto &d=snapshot_.diagnostic;
  interval_max_overlap_=std::max(interval_max_overlap_,d[2]);
  interval_max_compression_=std::max(interval_max_compression_,d[9]);
  interval_max_residual_=std::max(interval_max_residual_,d[3]);
  if(comm->me==0) {
    auto &out=history_file_;
    out<<update->ntimestep<<','<<simulation_time()<<','<<std::abs(shear_)*simulation_time();
    for(double value:snapshot_.stress)out<<','<<value;
    out<<','<<eta(snapshot_.stress[8])<<','<<eta(snapshot_.stress[8])/viscosity_;
    for(int c=0;c<8;++c)out<<','<<eta(snapshot_.stress[c]);
    out<<','<<d[6]<<','<<d[10]<<','<<d[11]<<','<<d[0]<<','<<d[1]
       <<','<<d[9]<<','<<d[2]<<','<<d[7]<<','<<d[3]<<','<<d[4]<<','<<d[5]<<','<<snapshot_.kink_reselections<<','<<d[14]<<','<<d[15];
    out<<','<<snapshot_.normal_energy<<','<<snapshot_.tangent_energy
       <<','<<snapshot_.sliding_power<<','<<snapshot_.hydro_power<<','<<snapshot_.stress_power
       <<','<<snapshot_.elastic_rate<<','<<snapshot_.discrete_dissipation<<','<<snapshot_.power_error
       <<','<<snapshot_.normal_diff1<<','<<snapshot_.normal_diff2;
    for(double power:snapshot_.lubrication_power)out<<','<<power;
    out<<','<<interval_steps_<<','<<interval_start_strain_<<','<<interval_end_strain_;
    std::array<double,9> mean{};
    for(int c=0;c<9;++c) {
      mean[c]=interval_duration_>0.?static_cast<double>(interval_stress_[c]/interval_duration_):nan;
      out<<','<<mean[c];
    }
    out<<','<<eta(mean[8])<<','<<eta(mean[8])/viscosity_;
    for(int c=0;c<8;++c)out<<','<<eta(mean[c]);
    const double selected_mean=selected_duration_>0.?eta(static_cast<double>(selected_stress_integral_/selected_duration_)):nan;
    out<<','<<interval_max_compression_<<','<<interval_max_overlap_<<','<<interval_max_residual_
       <<','<<selected_mean<<','<<selected_duration_;
    for(double count:snapshot_.solver_counts)out<<','<<count;
    for(double seconds:snapshot_.profile)out<<','<<seconds;
    for(auto seconds:interval_profile_)out<<','<<(interval_steps_>0?static_cast<double>(seconds/interval_steps_):nan);
    out<<','<<snapshot_.maximum_gap_curvature_ratio<<','<<snapshot_.completion_added_power
       <<','<<snapshot_.force_residual_N<<','<<snapshot_.force_tolerance_N;
    for(double count:snapshot_.lu_counts)out<<','<<count;
    out<<','<<snapshot_.finite_gap_stress<<','<<eta(snapshot_.finite_gap_stress)
       <<','<<snapshot_.finite_gap_power<<','<<(interval_duration_>0.?static_cast<double>(interval_finite_gap_stress_/interval_duration_):nan)
       <<','<<snapshot_.finite_gap_bound_min;
    out<<'\n';
    out.flush();
    if(!out)error->one(FLERR,"Failed writing slurry rheology history");
  }
  last_output_step_=update->ntimestep;
  interval_steps_=0;interval_duration_=0.;interval_stress_.fill(0.);interval_finite_gap_stress_=0.;
  interval_profile_.fill(0.);
  interval_start_strain_=interval_end_strain_=std::abs(shear_)*simulation_time();
  interval_max_overlap_=interval_max_compression_=interval_max_residual_=0.;
}

void FixSlurryImplicit::initial_integrate(int)
{
  const double dt=update->dt;
  // The pending solution belongs to this advance. Commit once, never during
  // setup, run 0, residual evaluations, or output queries.
  accumulate_step();
  history_=pending_history_;
  for(int i=0;i<atom->nlocal;++i) {
    auto &b=avec_->bonus[atom->ellipsoid[i]];
    double inertia[3],omega_raw[3];
    for(int k=0;k<3;++k)
      inertia[k]=atom->rmass[i]/5*(b.shape[(k+1)%3]*b.shape[(k+1)%3]+
                                  b.shape[(k+2)%3]*b.shape[(k+2)%3]);
    MathExtra::mq_to_omega(atom->angmom[i],b.quat,inertia,omega_raw);
    const gr::Vec3 omega{omega_raw[0],omega_raw[1],omega_raw[2]};
    for(int k=0;k<3;++k) atom->x[i][k]+=dt*atom->v[i][k];
    const double magnitude=gr::norm(omega), half=0.5*dt*magnitude;
    const double fac=magnitude>0?std::sin(half)/magnitude:0.5*dt;
    const double w=std::cos(half),x=fac*omega[0],y=fac*omega[1],z=fac*omega[2];
    const double q0=b.quat[0],q1=b.quat[1],q2=b.quat[2],q3=b.quat[3];
    b.quat[0]=w*q0-x*q1-y*q2-z*q3;
    b.quat[1]=w*q1+x*q0+y*q3-z*q2;
    b.quat[2]=w*q2-x*q3+y*q0+z*q1;
    b.quat[3]=w*q3+x*q2-y*q1+z*q0;
    double norm=0;for(int k=0;k<4;++k)norm+=b.quat[k]*b.quat[k];
    norm=std::sqrt(norm);for(int k=0;k<4;++k)b.quat[k]/=norm;
    store_momentum(atom,avec_,i,omega);
  }
}

void FixSlurryImplicit::solve_velocities()
{
  const double profile_start=MPI_Wtime();
  std::array<double,8> profile{};
  if(std::abs(domain->h_rate[5]-shear_*domain->yprd)>
       1e-10*std::max(std::abs(shear_*domain->yprd),1e-30))
    error->all(FLERR,"slurry/implicit shear rate disagrees with fix deform xy erate");
  std::vector<Record> local(atom->nlocal);
  for(int i=0;i<atom->nlocal;++i) {
    auto &r=local[i]; const auto &b=avec_->bonus[atom->ellipsoid[i]];
    r.tag=atom->tag[i];r.mass=atom->rmass[i];
    std::copy(atom->x[i],atom->x[i]+3,r.position);
    std::copy(b.shape,b.shape+3,r.axes);std::copy(b.quat,b.quat+4,r.quat);
    std::copy(atom->v[i],atom->v[i]+3,r.velocity);
    std::copy(atom->angmom[i],atom->angmom[i]+3,r.momentum);
  }
  double section_start=MPI_Wtime();
  auto records=gather_records(local,world,comm->nprocs);
  profile[1]+=MPI_Wtime()-section_start;
  section_start=MPI_Wtime();
  std::sort(records.begin(),records.end(),[](const Record&a,const Record&b){return a.tag<b.tag;});
  if(static_cast<bigint>(records.size())!=atom->natoms)
    error->all(FLERR,"slurry/implicit particle count changed during integration");
  const size_t n=records.size();
  const double volume=domain->xprd*domain->yprd*domain->zprd;
  std::vector<gr::Body> bodies; bodies.reserve(n);
  double length=0;
  for(const auto &r:records) {
    bodies.push_back(body_from_record(r));
    length=std::max(length,*std::max_element(r.axes,r.axes+3));
  }
  if(mari_lubrication_)for(const auto &b:bodies)
    if(std::abs(b.axes[1]-b.axes[0])>1e-12*b.axes[0] ||
       std::abs(b.axes[2]-b.axes[0])>1e-12*b.axes[0])
      error->all(FLERR,"Mari lubrication requires spherical particles (AR=1); select spheroid or legacy for nonspherical particles");
#ifdef SLURRY_HAVE_EIGEN
  if(finite_gap_fast_provider_ || finite_gap_provider_)for(const auto &b:bodies) {
    const double expected_ar=finite_gap_fast_provider_?finite_gap_fast_provider_->aspect_ratio():finite_gap_provider_->basis.aspect_ratio;
    if(!(b.axes[2]>0.) || expected_ar<=1. ||
       std::abs(b.axes[0]/b.axes[2]-expected_ar)>1e-10*expected_ar ||
       std::abs(b.axes[1]/b.axes[2]-expected_ar)>1e-10*expected_ar)
      error->all(FLERR,"finite-gap model requires oblate spheroids matching its basis/table AR; use Mari for AR=1");
    for(int k=0;k<3;++k)if(std::abs(b.axes[k]-bodies.front().axes[k])>1e-10*length)
      error->all(FLERR,"finite-gap basis/table requires equal-sized particles throughout the simulation; no fallback");
  }
  if(finite_gap_fast_provider_)for(const auto &b:bodies) {
    const auto &model=*finite_gap_fast_provider_;
    const double ar=b.axes[0]/b.axes[2],eps=roughness_/b.axes[2];
    if(std::abs(ar-model.aspect_ratio())>1e-10*model.aspect_ratio() ||
       std::abs(b.axes[1]/b.axes[2]-ar)>1e-10*ar || ar<=1.)
      error->all(FLERR,"finite_gap_fast table does not match particle spheroid AR; no fallback");
    if(eps<model.min_gap()*(1.-1e-10) || finite_gap_range_*ar+eps>model.max_gap()*(1.+1e-10))
      error->all(FLERR,"finite_gap_fast table does not cover regularization and outer range; regenerate it");
  }
#endif
  const double maximum_cutoff=mari_lubrication_ && lubrication_range_>0. ?
                              lubrication_range_*length : finite_gap_lubrication_?std::max(cutoff_,finite_gap_range_*length):cutoff_;
  const ph::Mat3 strain{0.,.5*shear_,0.,.5*shear_,0.,0.,0.,0.,0.};
  auto background=[&](size_t i,size_t j) {
    ph::Vec12 q{};
    q[0]=shear_*(bodies[i].position[1]-domain->boxlo[1]);
    q[6]=shear_*(bodies[j].position[1]-domain->boxlo[1]);
    q[5]=q[11]=-.5*length*shear_;
    return q;
  };
  auto sphere_hydro=[&](const si::Pair &p) {
    const double ai=bodies[p.i].axes[0],aj=bodies[p.j].axes[0],mean=.5*(ai+aj);
    return ph::make_sphere_pair(ai,aj,p.normal,p.gap,viscosity_,
             lubrication_delta_>0.?lubrication_delta_*mean:roughness_,
             lubrication_range_>0.?lubrication_range_*mean:cutoff_);
  };
  auto is_sphere=[](const gr::Body &b) {
    return std::abs(b.axes[1]-b.axes[0])<=1e-12*b.axes[0] &&
           std::abs(b.axes[2]-b.axes[0])<=1e-12*b.axes[0];
  };
  auto spheroid_hydro=[&](const si::Pair &p) {
    auto bj=bodies[p.j];
    bj.position=gr::add(bodies[p.i].position,gr::sub(p.lever_i,p.lever_j));
    gr::GapResult gap;gap.gap=p.gap;gap.normal=p.normal;gap.converged=true;
    gap.leverI=p.hydro_lever_i;gap.leverJ=p.hydro_lever_j;
    return sh::make_pair(bodies[p.i],bj,gap,viscosity_,roughness_,cutoff_);
  };
#ifdef SLURRY_HAVE_EIGEN
  if(finite_gap_provider_) {
    finite_gap_provider_->anchor_cache.begin_step(static_cast<std::uint64_t>(update->ntimestep));
    finite_gap_provider_->anchor_cache.prune(64);
  }
#endif
  auto make_finite_gap=[&](const gr::Body&bi,const gr::Body&bj,const gr::GapResult&gap,size_t i,size_t j) {
#ifdef SLURRY_HAVE_EIGEN
    if(finite_gap_fast_provider_)return finite_gap_fast_provider_->make_pair(bi,bj,gap,viscosity_,length,finite_gap_range_,finite_gap_inner_ratio_,static_cast<std::uint64_t>(records[i].tag),static_cast<std::uint64_t>(records[j].tag));
    if(finite_gap_provider_)return finite_gap_provider_->make_pair(bi,bj,gap,viscosity_,length,finite_gap_range_,finite_gap_inner_ratio_,static_cast<std::uint64_t>(records[i].tag),static_cast<std::uint64_t>(records[j].tag));
#endif
    return fg::make_pair(*finite_gap_catalogue_,bi,bj,gap,viscosity_,length,finite_gap_range_,finite_gap_inner_ratio_);
  };
  std::vector<si::Particle> particles(n);
  for(size_t i=0;i<n;++i) {
    const auto &b=bodies[i];auto &p=particles[i];
    auto found=resistance_cache_.find(b.axes);
    if(found==resistance_cache_.end())
      found=resistance_cache_.emplace(b.axes,SlurryHydro::unit_resistance(b.axes)).first;
    const auto &r=found->second;
    const auto trans=gr::rotatedDiagonal(b.rotation,r.translation);
    const auto rot=gr::rotatedDiagonal(b.rotation,r.rotation);
    for(int k=0;k<3;++k)for(int l=0;l<3;++l) {
      p.resistance[k*6+l]=viscosity_*trans[k*3+l];
      p.resistance[(k+3)*6+l+3]=viscosity_*rot[k*3+l]/(length*length);
    }
    const gr::Vec3 axis{b.rotation[2],b.rotation[5],b.rotation[8]};
    const auto w=SlurryHydro::jeffery_omega(axis,b.axes[0],b.axes[2],shear_);
    p.target[0]=shear_*(b.position[1]-domain->boxlo[1]);
    for(int k=0;k<3;++k)p.target[k+3]=length*w[k];
  }
  std::vector<si::Pair> local_pairs;
  std::vector<SlurryGrandSystem::PairBlock> local_grand_blocks;
  gr::NearFieldSettings settings;
  settings.viscosity=viscosity_;settings.matchingGap=cutoff_;
  // The candidate padding also admits contacts approached within this step.
  const double padding=std::max({maximum_cutoff,contact_gap_,8*std::abs(shear_)*length*update->dt});
  if(2*length+padding>=0.5*std::min({domain->xprd,domain->yprd,domain->zprd}))
    error->all(FLERR,"slurry/implicit interaction range exceeds the single-image box limit; increase particle count/box size");
  int failed=0;std::string failure;
  std::map<ContactKey,gr::GapCache> next_gap_cache;
  try {
    for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j) {
      const size_t pair_index=i*(2*n-i-1)/2+j-i-1;
      if(pair_index%comm->nprocs!=static_cast<size_t>(comm->me))continue;
      const auto &bi=bodies[i];auto bj=bodies[j];
      const auto raw=gr::sub(bj.position,bi.position);
      const double ny=std::nearbyint(raw[1]/domain->yprd);
      gr::Vec3 dr=raw;
      dr[1]-=ny*domain->yprd;dr[0]-=ny*domain->xy;
      dr[0]-=std::nearbyint(dr[0]/domain->xprd)*domain->xprd;
      dr[2]-=std::nearbyint(dr[2]/domain->zprd)*domain->zprd;
      const double reach=*std::max_element(bi.axes.begin(),bi.axes.end())+
                         *std::max_element(bj.axes.begin(),bj.axes.end())+padding;
      if(gr::dot(dr,dr)>reach*reach)continue;
      bj.position=gr::add(bi.position,dr);
      gr::GapResult gap;
      if(mari_lubrication_ || ((spheroid_lubrication_||finite_gap_lubrication_) && is_sphere(bi) && is_sphere(bj))) {
        const double distance=gr::norm(dr);
        if(!(distance>0.))throw std::runtime_error("Coincident sphere centers in Mari lubrication");
        gap.normal=gr::scale(dr,1./distance);
        gap.gap=distance-bi.axes[0]-bj.axes[0];
        gap.leverI=gr::scale(gap.normal,bi.axes[0]);
        gap.leverJ=gr::scale(gap.normal,-bj.axes[0]);
        gap.converged=true;
      } else {
        const ContactKey key{records[i].tag,records[j].tag};
        auto cached=gap_cache_.find(key);
        gr::GapCache cache=cached!=gap_cache_.end()?cached->second:gr::GapCache{};
        try { gap=gr::closestEllipsoidGap(bi,bj,&cache); }
        catch(const std::exception &) {
          // An old normal is only an acceleration. Retry the same exact
          // geometry problem from its uncached initial guess if needed.
          cache=gr::GapCache{};gap=gr::closestEllipsoidGap(bi,bj,&cache);
        }
        next_gap_cache.emplace(key,cache);
      }
      if(gap.gap>=padding)continue;
      const auto point=gr::scale(gr::add(gr::add(bi.position,gap.leverI),
                                       gr::add(bj.position,gap.leverJ)),0.5);
      si::Pair p{};p.i=i;p.j=j;p.normal=gap.normal;
      p.lever_i=gr::sub(point,bi.position);p.lever_j=gr::sub(point,bj.position);
      p.hydro_lever_i=gap.leverI;p.hydro_lever_j=gap.leverJ;
      p.image_velocity={-shear_*ny*domain->yprd,0,0};
      p.gap=gap.gap;p.stiffness=stiffness_;
      if(mari_lubrication_) {
        const auto hydro=sphere_hydro(p);
        // Uj_image and Uinf_j_image carry the SAME Lees--Edwards shift.
        // It cancels in nonaffine velocities; contact still uses image_velocity.
        const auto matrix=ph::assemble(hydro,background(i,j),ph::Vec3{},strain,length);
        p.generalized_hydro=true;
        p.hydro_resistance=matrix.resistance;p.hydro_affine=matrix.affine;
        p.hydro_energy_constant=matrix.energy_constant;
      } else if(finite_gap_lubrication_ && !(is_sphere(bi)&&is_sphere(bj))) {
        const auto hydro=make_finite_gap(bi,bj,gap,i,j);
        gp::Inputs input;input.background=background(i,j);input.strain=strain;
        const auto matrix=gp::assemble(hydro.model,hydro.frame,input);
        p.generalized_hydro=true;p.hydro_resistance=matrix.resistance;p.hydro_affine=matrix.affine;
        p.hydro_energy_constant=matrix.energy_constant;
        const auto blocks=gp::strain_blocks(hydro.model,hydro.frame);
        SlurryGrandSystem::PairBlock block;block.i=i;block.j=j;block.coupling=blocks.coupling;block.strain=blocks.strain;
        block.has_finite_model=true;block.active=hydro.active;block.cutoff=hydro.cutoff;
        block.frame=hydro.frame;block.resistance=hydro.model.resistance;
        local_grand_blocks.push_back(block);
      } else if(spheroid_lubrication_||finite_gap_lubrication_) {
        const auto hydro=sh::make_pair(bi,bj,gap,viscosity_,roughness_,cutoff_);
        const auto matrix=sh::assemble(hydro,background(i,j),ph::Vec3{},strain,length);
        p.generalized_hydro=true;
        p.hydro_resistance=matrix.resistance;p.hydro_affine=matrix.affine;
        p.hydro_energy_constant=matrix.energy_constant;
        if(finite_gap_lubrication_) {
          SlurryGrandSystem::PairBlock block;block.i=i;block.j=j;
          for(int c=0;c<5;++c){
            const auto assembly=sh::assemble(hydro,ph::Vec12{},ph::Vec3{},gp::strain_basis()[c],length);
            const auto evaluated=sh::evaluate(hydro,ph::Vec12{},ph::Vec12{},ph::Vec3{},gp::strain_basis()[c],length);
            const auto column=gp::strain_coordinates(evaluated.stress);
            for(int r=0;r<12;++r)block.coupling[5*r+c]=assembly.affine[r];
            for(int r=0;r<5;++r)block.strain[5*r+c]=column[r];
          }
          local_grand_blocks.push_back(block);
        }
      } else {
        const auto hydro=gr::nearFieldResistance(bi,bj,std::max(gap.gap,roughness_),
                              gap.normal,gap.leverI,gap.leverJ,settings);
        p.normal_resistance=hydro.normalResistance;p.tangent_resistance=hydro.tangentialResistance;
      }
      p.contact_onset=contact_gap_;p.friction_coefficient=friction_;
      p.tangent_stiffness=tangential_stiffness_;
      const ContactKey key{records[i].tag,records[j].tag};
      const auto old=history_.find(key);
      if(old!=history_.end() && gap.gap<contact_gap_)
        p.tangent_history=SlurryHistory::transport(old->second.elastic,old->second.normal,
                                                  p.normal,old->second.common_angle);
      local_pairs.push_back(p);
    }
  } catch(const std::exception &e) { failed=1;failure=e.what(); }
  int any_failed=0;MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,world);
  if(any_failed)error->all(FLERR,"slurry/implicit geometry failed: {}",failure);
  gap_cache_.swap(next_gap_cache);
  profile[0]=MPI_Wtime()-section_start;
  section_start=MPI_Wtime();
  auto pairs=gather_records(local_pairs,world,comm->nprocs);
  profile[1]+=MPI_Wtime()-section_start;
  std::sort(pairs.begin(),pairs.end(),[](const si::Pair&a,const si::Pair&b){
    return a.i<b.i || (a.i==b.i && a.j<b.j);});
  std::map<std::pair<std::size_t,std::size_t>,fg::Pair> finite_gap_cache;
  double finite_gap_margin=std::numeric_limits<double>::quiet_NaN();
  if(finite_gap_lubrication_) {
    const auto grand_blocks=gather_records(local_grand_blocks,world,comm->nprocs);
    for(const auto&block:grand_blocks)if(block.has_finite_model){fg::Pair cached;cached.active=block.active;cached.cutoff=block.cutoff;
      cached.frame=block.frame;cached.model.resistance=block.resistance;finite_gap_cache.emplace(std::make_pair(block.i,block.j),cached);}
    fg::Mat5 strain_resistance{};
    for(int k=0;k<5;++k)strain_resistance[6*k]=2*viscosity_*volume;
    std::vector<SlurryGrandSystem::Coupling6> body_coupling(n);
    for(size_t i=0;i<n;++i) {
      auto found=stress_cache_.find(bodies[i].axes);
      if(found==stress_cache_.end())found=stress_cache_.emplace(bodies[i].axes,SlurryStress::unit_stresslet(bodies[i].axes)).first;
      const auto& coefficients=found->second;
      std::array<si::Vec6,5> targets{},weighted{};
      for(int c=0;c<5;++c) {
        const auto one=SlurryStress::stresslet(coefficients,bodies[i].rotation,viscosity_,gp::strain_basis()[c],gr::Vec3{},gr::Vec3{});
        const auto free_column=gp::strain_coordinates(one.free);
        for(int d=0;d<3;++d)targets[c][3+d]=length*one.jeffery_omega[d];
        for(int r=0;r<6;++r)for(int d=0;d<6;++d)weighted[c][r]+=particles[i].resistance[6*r+d]*targets[c][d];
        for(int r=0;r<6;++r)body_coupling[i][5*r+c]=-weighted[c][r];
        for(int r=0;r<5;++r)strain_resistance[5*r+c]+=free_column[r];
      }
      for(int r=0;r<5;++r)for(int c=0;c<5;++c)for(int d=0;d<6;++d)strain_resistance[5*r+c]+=targets[r][d]*weighted[c][d];
    }
    failed=0;failure.clear();
    if(comm->me==0){try {
      const auto check=SlurryGrandSystem::verify(particles,pairs,body_coupling,strain_resistance,grand_blocks,viscosity_,length,volume);
      finite_gap_margin=check.strain_margin;
    }catch(const std::exception&e){failed=1;failure=e.what();}}
    MPI_Bcast(&failed,1,MPI_INT,0,world);
    if(failed){int count=static_cast<int>(failure.size());MPI_Bcast(&count,1,MPI_INT,0,world);failure.resize(count);MPI_Bcast(failure.data(),count,MPI_CHAR,0,world);
      error->all(FLERR,"Finite-gap global passivity check failed: {}",failure);}
    MPI_Bcast(&finite_gap_margin,1,MPI_DOUBLE,0,world);
    // Only this successful global check authorizes an indefinite pair block.
    for(auto&p:pairs)if(p.generalized_hydro)p.globally_verified_hydro=true;
  }
  si::Result result;
  std::vector<si::Vec6> initial;
  if(have_solution_) {
    initial.resize(n);
    for(size_t i=0;i<n;++i)for(int k=0;k<3;++k) {
      initial[i][k]=bodies[i].velocity[k];initial[i][k+3]=length*bodies[i].omega[k];
    }
  }
  si::Options options;
  // Inexact Newton: avoid oversolving each correction, while the outer solve
  // still enforces the 1e-10 force-balance tolerance on the complete system.
  options.linear_relative_tolerance=friction_>0.?1e-3:1e-8;
  options.preconditioner_backend=preconditioner_backend_==1?si::PreconditionerBackend::Sparse:
    preconditioner_backend_==2?si::PreconditionerBackend::Nested:
    preconditioner_backend_==3?si::PreconditionerBackend::FullLU:
    preconditioner_backend_==4?si::PreconditionerBackend::IncompleteLU:si::PreconditionerBackend::Auto;
  options.ilu_drop_tolerance=ilu_drop_tolerance_;
  options.ilu_fill_factor=ilu_fill_factor_;
  options.ilu_refresh_iterations=ilu_refresh_iterations_;
  options.adaptive_forcing=adaptive_forcing_;
  section_start=MPI_Wtime();
  failed=0;failure.clear();
  if(!root_solver_ || comm->me==0) {
    try {
      si::SolverWorkspace *workspace=nullptr;
      if(solver_cache_ && options.preconditioner_backend==si::PreconditionerBackend::IncompleteLU) {
        bool same_tags=solver_tags_.size()==records.size();
        if(same_tags)for(size_t i=0;i<records.size();++i)
          if(solver_tags_[i]!=records[i].tag) { same_tags=false;break; }
        if(!solver_workspace_)solver_workspace_=std::make_unique<si::SolverWorkspace>();
        if(!same_tags || solver_cache_dt_!=update->dt || solver_cache_length_!=length) {
          solver_workspace_->reset();
          solver_tags_.resize(records.size());
          for(size_t i=0;i<records.size();++i)solver_tags_[i]=records[i].tag;
          solver_cache_dt_=update->dt;solver_cache_length_=length;
        }
        workspace=solver_workspace_.get();
      }
      result=si::solve(particles,pairs,update->dt,length,options,initial,workspace);
    }
    catch(const std::exception &e) { failed=1;failure=e.what(); }
  }
  profile[2]=MPI_Wtime()-section_start;
  if(root_solver_) {
    section_start=MPI_Wtime();
    MPI_Bcast(&failed,1,MPI_INT,0,world);
    if(failed) {
      int message_size=static_cast<int>(failure.size());
      MPI_Bcast(&message_size,1,MPI_INT,0,world);failure.resize(message_size);
      MPI_Bcast(failure.data(),message_size,MPI_CHAR,0,world);
      error->all(FLERR,"slurry/implicit solve failed: {}",failure);
    }
    SolverReport report;
    if(comm->me==0) {
      report.residual_norm=result.residual_norm;report.relative_residual=result.relative_residual;
      report.residual_scale=result.residual_scale;
      report.cg=result.cg_iterations;report.gmres=result.gmres_iterations;
      report.newton=result.newton_iterations;report.contacts=result.contacts;report.kinks=result.kink_reselections;
      report.counts[0]=result.sparse_factorizations;report.counts[1]=result.sparse_fallbacks;
      report.counts[2]=result.sparse_preconditioner_solves;report.factor_nonzeros=result.max_sparse_factor_nonzeros;
      report.counts[3]=result.sparse_preconditioner_reuses;
      report.lu_counts[0]=result.lu_factorizations;report.lu_counts[1]=result.lu_solves;
      report.lu_counts[2]=result.lu_fallbacks;
      report.lu_counts[3]=result.forcing_retries;
      report.setup_seconds=result.preconditioner_setup_seconds;
      report.precondition_seconds=result.preconditioner_solve_seconds;report.apply_seconds=result.linear_apply_seconds;
    }
    MPI_Bcast(&report,sizeof(report),MPI_BYTE,0,world);
    if(comm->me!=0)result.velocity.resize(n);
    MPI_Bcast(result.velocity.data(),static_cast<int>(n*sizeof(si::Vec6)),MPI_BYTE,0,world);
    MPI_Bcast(&profile[2],1,MPI_DOUBLE,0,world);
    profile[1]+=MPI_Wtime()-section_start;
    if(comm->me!=0) {
      si::detail::fill_result(pairs,update->dt,length,result);
      result.residual_norm=report.residual_norm;result.relative_residual=report.relative_residual;
      result.residual_scale=report.residual_scale;
      result.cg_iterations=report.cg;result.gmres_iterations=report.gmres;result.newton_iterations=report.newton;
      result.contacts=report.contacts;result.kink_reselections=report.kinks;
      result.sparse_factorizations=report.counts[0];result.sparse_fallbacks=report.counts[1];
      result.sparse_preconditioner_solves=report.counts[2];result.max_sparse_factor_nonzeros=report.factor_nonzeros;
      result.sparse_preconditioner_reuses=report.counts[3];
      result.lu_factorizations=report.lu_counts[0];result.lu_solves=report.lu_counts[1];
      result.lu_fallbacks=report.lu_counts[2];
      result.forcing_retries=report.lu_counts[3];
      result.preconditioner_setup_seconds=report.setup_seconds;
      result.preconditioner_solve_seconds=report.precondition_seconds;result.linear_apply_seconds=report.apply_seconds;
    }
  } else {
    MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,world);
    if(any_failed)error->all(FLERR,"slurry/implicit solve failed: {}",failure);
  }
  section_start=MPI_Wtime();
  have_solution_=true;
  diagnostics_.fill(0);diagnostics_[7]=maximum_cutoff;
  diagnostics_[3]=result.relative_residual;diagnostics_[4]=result.cg_iterations+result.gmres_iterations;
  diagnostics_[5]=result.newton_iterations;diagnostics_[6]=result.contacts;
  snapshot_=Snapshot{};
  snapshot_.finite_gap_bound_min=finite_gap_margin;
  snapshot_.force_residual_N=result.residual_norm;
  snapshot_.force_tolerance_N=options.absolute_tolerance+options.relative_tolerance*result.residual_scale;
  snapshot_.solver_counts={static_cast<double>(result.cg_iterations),static_cast<double>(result.gmres_iterations),
    static_cast<double>(result.sparse_factorizations),static_cast<double>(result.sparse_fallbacks),
    static_cast<double>(result.sparse_preconditioner_solves),static_cast<double>(result.max_sparse_factor_nonzeros),
    static_cast<double>(result.sparse_preconditioner_reuses)};
  snapshot_.kink_reselections=result.kink_reselections;
  snapshot_.lu_counts={static_cast<double>(result.lu_factorizations),static_cast<double>(result.lu_solves),
                      static_cast<double>(result.lu_fallbacks),static_cast<double>(result.forcing_retries)};
  pending_history_.clear();
  std::array<gr::Mat3,9> stress{};
  gr::Mat3 finite_gap_stress{};
  stress[0][1]=stress[0][3]=viscosity_*shear_;
  double drag_power=0.,lubrication_power=0.,contact_power=0.;
  double old_normal_energy=0.,old_tangent_energy=0.;
  std::vector<si::Vec6> net(n);
  for(size_t i=0;i<n;++i) {
    for(int k=0;k<6;++k)for(int l=0;l<6;++l)
      net[i][k]-=particles[i].resistance[k*6+l]*(result.velocity[i][l]-particles[i].target[l]);
    for(int k=0;k<6;++k)drag_power-=(result.velocity[i][k]-particles[i].target[k])*net[i][k];
    auto found=stress_cache_.find(bodies[i].axes);
    if(found==stress_cache_.end())
      found=stress_cache_.emplace(bodies[i].axes,SlurryStress::unit_stresslet(bodies[i].axes)).first;
    const gr::Vec3 omega{result.velocity[i][3]/length,result.velocity[i][4]/length,result.velocity[i][5]/length};
    const auto s=SlurryStress::stresslet(found->second,bodies[i].rotation,viscosity_,shear_,omega);
    for(int k=0;k<9;++k) {stress[1][k]+=s.free[k]/volume;stress[2][k]+=s.rotational[k]/volume;}
    const gr::Vec3 velocity{result.velocity[i][0],result.velocity[i][1],result.velocity[i][2]};
    diagnostics_[14]=std::max(diagnostics_[14],gr::norm(velocity));
  }
  auto moment=[&](int component,const gr::Vec3 &dr,const gr::Vec3 &f) {
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)
      stress[component][3*a+b]+=.5*(dr[a]*f[b]+dr[b]*f[a])/volume;
  };
  for(size_t k=0;k<pairs.size();++k) {
    const auto &p=pairs[k];
    sh::Pair nonsphere_hydro;fg::Pair finite_hydro;
    const bool use_finite=finite_gap_lubrication_&&!(is_sphere(bodies[p.i])&&is_sphere(bodies[p.j]));
    if(use_finite)finite_hydro=finite_gap_cache.at(std::make_pair(p.i,p.j));
    else if(spheroid_lubrication_||finite_gap_lubrication_)nonsphere_hydro=spheroid_hydro(p);
    const double pair_cutoff=use_finite?finite_hydro.cutoff:(spheroid_lubrication_||finite_gap_lubrication_)?nonsphere_hydro.cutoff:
      mari_lubrication_ && lubrication_range_>0. ?
      lubrication_range_*.5*(bodies[p.i].axes[0]+bodies[p.j].axes[0]) : cutoff_;
    if(p.gap<pair_cutoff)++diagnostics_[0];
    if(p.gap<0)++diagnostics_[1];
    diagnostics_[2]=std::max(diagnostics_[2],-p.gap);
    diagnostics_[7]=std::min(diagnostics_[7],p.gap);
    if(p.gap<contact_gap_)++diagnostics_[8];
    diagnostics_[9]=std::max(diagnostics_[9],contact_gap_-p.gap);
    const auto slip=si::pair_velocity(p,result.velocity,length);
    const double sn=gr::dot(p.normal,slip);
    const auto tangential_slip=gr::sub(slip,gr::scale(p.normal,sn));
    const auto fn_lub=gr::scale(p.normal,p.normal_resistance*sn);
    const auto ft_lub=gr::scale(tangential_slip,p.tangent_resistance);
    const auto fn_contact=gr::scale(p.normal,-result.contact_load[k]);
    const auto ft_contact=result.tangent_force[k];
    const auto f_contact=gr::add(fn_contact,ft_contact);
    const auto f=gr::add(gr::add(fn_lub,ft_lub),f_contact);
    const auto dr=gr::sub(p.lever_i,p.lever_j);
    if(use_finite) {
      gp::Inputs input;input.background=background(p.i,p.j);input.strain=strain;
      for(int d=0;d<6;++d){input.velocity[d]=result.velocity[p.i][d];input.velocity[6+d]=result.velocity[p.j][d];}
      const auto h=gp::evaluate(finite_hydro.model,finite_hydro.frame,input);
      for(int d=0;d<9;++d)finite_gap_stress[d]+=h.stress[d]/volume;
      snapshot_.finite_gap_power+=h.power;lubrication_power+=h.power;
      for(int d=0;d<3;++d){net[p.i][d]+=h.force_i[d];net[p.j][d]+=h.force_j[d];
        net[p.i][d+3]+=h.torque_i[d]/length;net[p.j][d+3]+=h.torque_j[d]/length;}
    } else if(mari_lubrication_ || spheroid_lubrication_ || finite_gap_lubrication_) {
      ph::Vec12 q{};
      for(int d=0;d<6;++d) {q[d]=result.velocity[p.i][d];q[d+6]=result.velocity[p.j][d];}
      ph::Evaluation h;
      if(mari_lubrication_)h=ph::evaluate(sphere_hydro(p),q,background(p.i,p.j),ph::Vec3{},strain,length);
      else {
        const auto evaluated=sh::evaluate(nonsphere_hydro,q,background(p.i,p.j),ph::Vec3{},strain,length);
        h=evaluated;
        snapshot_.maximum_gap_curvature_ratio=std::max(snapshot_.maximum_gap_curvature_ratio,
          nonsphere_hydro.active?nonsphere_hydro.regularized_gap/nonsphere_hydro.minimum_curvature_radius:0.);
        snapshot_.completion_added_power+=evaluated.power_completion;
      }
      for(int d=0;d<9;++d) {
        stress[3][d]+=h.stress_squeeze[d]/volume;
        stress[4][d]+=h.stress_shear[d]/volume;
        stress[7][d]+=h.stress_pump[d]/volume;
      }
      snapshot_.lubrication_power[0]+=h.power_squeeze;
      snapshot_.lubrication_power[1]+=h.power_shear;
      snapshot_.lubrication_power[2]+=h.power_pump;
      lubrication_power+=h.power_squeeze+h.power_shear+h.power_pump;
      for(int d=0;d<3;++d) {
        net[p.i][d]+=h.force_i[d];net[p.j][d]+=h.force_j[d];
        net[p.i][d+3]+=h.torque_i[d]/length;net[p.j][d+3]+=h.torque_j[d]/length;
      }
    } else {
      moment(3,dr,fn_lub);moment(4,dr,ft_lub);
      const double pn=p.normal_resistance*sn*sn;
      const double pt=p.tangent_resistance*gr::dot(tangential_slip,tangential_slip);
      snapshot_.lubrication_power[0]+=pn;snapshot_.lubrication_power[1]+=pt;
      lubrication_power+=pn+pt;
    }
    moment(5,dr,fn_contact);moment(6,dr,ft_contact);
    contact_power+=gr::dot(f_contact,slip);
    snapshot_.normal_energy+=result.normal_elastic_energy[k];
    snapshot_.tangent_energy+=result.tangential_elastic_energy[k];
    snapshot_.sliding_power+=result.sliding_dissipation[k]/update->dt;
    const double compression=std::max(contact_gap_-p.gap,0.);
    old_normal_energy+=.5*stiffness_*compression*compression;
    old_tangent_energy+=.5*tangential_stiffness_*gr::dot(p.tangent_history,p.tangent_history);
    diagnostics_[15]=std::max(diagnostics_[15],gr::norm(ft_contact)-friction_*result.contact_load[k]);
    if(result.contact_load[k]>0.) {
      if(result.sliding[k])++diagnostics_[11];else ++diagnostics_[10];
      if(friction_>0.) {
        ContactHistory state;
        state.elastic=result.tangent_history[k];state.normal=p.normal;
        for(int d=0;d<3;++d)
          state.common_angle[d]=.5*update->dt*(result.velocity[p.i][d+3]+result.velocity[p.j][d+3])/length;
        pending_history_[ContactKey{records[p.i].tag,records[p.j].tag}]=state;
      }
    }
    const auto ti=gr::cross(p.lever_i,f), tj=gr::scale(gr::cross(p.lever_j,f),-1);
    for(int l=0;l<3;++l) {
      net[p.i][l]+=f[l];net[p.j][l]-=f[l];
      net[p.i][l+3]+=ti[l]/length;net[p.j][l+3]+=tj[l]/length;
    }
  }
  for(int c=0;c<8;++c)for(int k=0;k<9;++k)stress[8][k]+=stress[c][k];
  for(int k=0;k<9;++k)stress[8][k]+=finite_gap_stress[k];
  snapshot_.finite_gap_stress=finite_gap_stress[1];
  for(int c=0;c<9;++c)snapshot_.stress[c]=stress[c][1];
  diagnostics_[12]=snapshot_.stress[8];
  diagnostics_[13]=shear_!=0.?snapshot_.stress[8]/shear_:std::numeric_limits<double>::quiet_NaN();
  snapshot_.diagnostic=diagnostics_;
  snapshot_.normal_diff1=stress[8][0]-stress[8][4];snapshot_.normal_diff2=stress[8][4]-stress[8][8];
  snapshot_.hydro_power=volume*shear_*(stress[0][1]+stress[1][1])+drag_power+lubrication_power;
  snapshot_.stress_power=volume*shear_*stress[8][1];
  snapshot_.elastic_rate=(snapshot_.normal_energy+snapshot_.tangent_energy-old_normal_energy-old_tangent_energy)/update->dt;
  snapshot_.discrete_dissipation=contact_power-snapshot_.elastic_rate-snapshot_.sliding_power;
  snapshot_.power_error=snapshot_.stress_power-snapshot_.hydro_power-contact_power;
  // State energies are at the recorded geometry. Rates/dissipation refer to
  // the pending backward-Euler increment, like the recorded contact forces.
  snapshot_.normal_energy=old_normal_energy;
  snapshot_.tangent_energy=old_tangent_energy;
  for(int i=0;i<atom->nlocal;++i) {
    const auto it=std::lower_bound(records.begin(),records.end(),atom->tag[i],
                             [](const Record&r,tagint t){return r.tag<t;});
    if(it==records.end() || it->tag!=atom->tag[i])error->one(FLERR,"slurry/implicit missing atom tag");
    const size_t j=it-records.begin();gr::Vec3 omega{};
    for(int k=0;k<3;++k) {
      atom->v[i][k]=result.velocity[j][k];omega[k]=result.velocity[j][k+3]/length;
      atom->f[i][k]=net[j][k];atom->torque[i][k]=length*net[j][k+3];
    }
    store_momentum(atom,avec_,i,omega);
  }
  profile[3]=MPI_Wtime()-section_start;
  profile[4]=MPI_Wtime()-profile_start;
  profile[5]=result.preconditioner_setup_seconds;
  profile[6]=result.preconditioner_solve_seconds;
  profile[7]=result.linear_apply_seconds;
  snapshot_.profile=profile;
}
