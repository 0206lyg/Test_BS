#include "pair_slurry_spheroid.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include "ellipsoidGap.h"
#include "nearFieldResistance.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
namespace gr = slurry::gr_re2::graphite;

namespace {

gr::Body read_body(Atom *atom, AtomVecEllipsoid *avec, int i)
{
  const auto &bonus = avec->bonus[atom->ellipsoid[i]];
  gr::Body body;
  double rotation[3][3];
  MathExtra::quat_to_mat(bonus.quat, rotation);
  for (int k = 0; k < 3; ++k) {
    body.position[k] = atom->x[i][k];
    body.velocity[k] = atom->v[i][k];
    // AtomVecEllipsoid stores semi-axes, whereas set shape/read_data take
    // full axis lengths.  Do not divide bonus.shape by two here.
    body.axes[k] = bonus.shape[k];
    for (int l = 0; l < 3; ++l) body.rotation[3*k+l] = rotation[k][l];
  }
  body.mass = atom->rmass[i];
  for (int k = 0; k < 3; ++k)
    body.inertiaBody[k] = body.mass / 5.0 *
        (body.axes[(k+1)%3]*body.axes[(k+1)%3] +
         body.axes[(k+2)%3]*body.axes[(k+2)%3]);
  const gr::Vec3 angular_momentum{atom->angmom[i][0], atom->angmom[i][1],
                                 atom->angmom[i][2]};
  gr::Vec3 omega_body = gr::mul(gr::transpose(body.rotation), angular_momentum);
  for (int k = 0; k < 3; ++k) omega_body[k] /= body.inertiaBody[k];
  body.omega = gr::mul(body.rotation, omega_body);
  return body;
}

}    // namespace

PairSlurrySpheroid::PairSlurrySpheroid(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 1;
  // Forces depend on velocities; request their forward communication.
  // comm_modify vel yes is also checked explicitly in init_style().
  ghostneigh = 0;
}

PairSlurrySpheroid::~PairSlurrySpheroid()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSlurrySpheroid::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n+1, n+1, "slurry/spheroid:setflag");
  memory->create(cutsq, n+1, n+1, "slurry/spheroid:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) setflag[i][j] = 0;
}

void PairSlurrySpheroid::settings(int narg, char **arg)
{
  if (narg != 4)
    error->all(FLERR, "Pair slurry/spheroid requires viscosity hcut hmin kn");
  viscosity_ = utils::numeric(FLERR, arg[0], false, lmp);
  cutoff_gap_ = utils::numeric(FLERR, arg[1], false, lmp);
  minimum_gap_ = utils::numeric(FLERR, arg[2], false, lmp);
  normal_stiffness_ = utils::numeric(FLERR, arg[3], false, lmp);
  if (!(std::isfinite(viscosity_) && viscosity_ > 0.0 &&
        std::isfinite(cutoff_gap_) && cutoff_gap_ > 0.0 &&
        std::isfinite(minimum_gap_) && minimum_gap_ > 0.0 &&
        minimum_gap_ < cutoff_gap_ && std::isfinite(normal_stiffness_) &&
        normal_stiffness_ > 0.0))
    error->all(FLERR, "Invalid slurry/spheroid parameters: require eta>0, "
                      "0<hmin<hcut, and kn>0");
}

void PairSlurrySpheroid::coeff(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR, "Pair slurry/spheroid coefficients require two type ranges only");
  if (!allocated) allocate();
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  int count = 0;
  for (int i = ilo; i <= ihi; ++i)
    for (int j = std::max(jlo, i); j <= jhi; ++j) {
      setflag[i][j] = 1;
      ++count;
    }
  if (count == 0) error->all(FLERR, "Incorrect slurry/spheroid pair coefficients");
}

void PairSlurrySpheroid::init_style()
{
  if (std::strcmp(update->unit_style, "si") != 0)
    error->all(FLERR, "Pair slurry/spheroid requires units si");
  if (domain->dimension != 3)
    error->all(FLERR, "Pair slurry/spheroid requires three dimensions");
  avec_ = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec_ || !atom->ellipsoid_flag || atom->superellipsoid_flag ||
      !atom->angmom_flag || !atom->torque_flag)
    error->all(FLERR, "Pair slurry/spheroid requires atom_style ellipsoid");
  if (!comm->ghost_velocity)
    error->all(FLERR, "Pair slurry/spheroid requires comm_modify vel yes");
  if (!force->newton_pair)
    error->all(FLERR, "Pair slurry/spheroid requires newton pair on");

  const int n = atom->ntypes + 1;
  std::vector<double> local_maximum(n, 0.0);
  int invalid = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    if (atom->ellipsoid[i] < 0 || !(atom->rmass[i] > 0.0) ||
        !std::isfinite(atom->rmass[i])) {
      invalid = 1;
      continue;
    }
    const auto &bonus = avec_->bonus[atom->ellipsoid[i]];
    for (int k = 0; k < 3; ++k) {
      if (!(bonus.shape[k] > 0.0) || !std::isfinite(bonus.shape[k])) invalid = 1;
      local_maximum[atom->type[i]] =
          std::max(local_maximum[atom->type[i]], bonus.shape[k]);
    }
  }
  int any_invalid;
  MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, world);
  if (any_invalid)
    error->all(FLERR, "Pair slurry/spheroid requires positive finite ellipsoid axes and mass");
  maximum_axis_.resize(n);
  MPI_Allreduce(local_maximum.data(), maximum_axis_.data(), n, MPI_DOUBLE, MPI_MAX, world);
  neighbor->add_request(this);
}

double PairSlurrySpheroid::init_one(int i, int j)
{
  if (!setflag[i][j])
    error->all(FLERR, "All slurry/spheroid pair coefficients must be specified");
  return maximum_axis_[i] + maximum_axis_[j] + cutoff_gap_;
}

void PairSlurrySpheroid::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  const int nlocal = atom->nlocal;
  gr::NearFieldSettings settings;
  settings.viscosity = viscosity_;
  // This is a physical finite interaction range, with no LB dx dependence.
  settings.matchingGap = cutoff_gap_;
  settings.enabled = true;
  settings.tangential = true;

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    const gr::Body bi = read_body(atom, avec_, i);
    int *jlist = list->firstneigh[i];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int entry = jlist[jj];
      const double factor = force->special_lj[sbmask(entry)];
      const int j = entry & NEIGHMASK;
      const gr::Vec3 rij{atom->x[i][0]-atom->x[j][0], atom->x[i][1]-atom->x[j][1],
                         atom->x[i][2]-atom->x[j][2]};
      if (factor == 0.0 || gr::dot(rij, rij) >= cutsq[atom->type[i]][atom->type[j]])
        continue;
      try {
        const gr::Body bj = read_body(atom, avec_, j);
        // Neighbor-list ghosts already contain the correct periodic/shear image.
        // LAMMPS remaps their velocities for fix deform remap v.  Subtracting
        // another affine velocity here would count that correction twice.
        const gr::GapResult gap = gr::closestEllipsoidGap(bi, bj);
        if (gap.gap >= cutoff_gap_) continue;
        const gr::NearFieldResult lubrication = gr::nearFieldResistance(
            bi, bj, std::max(gap.gap, minimum_gap_), gap.normal, gap.leverI,
            gap.leverJ, settings);
        const double overlap = std::max(-gap.gap, 0.0);
        const gr::Vec3 repulsion = gr::scale(gap.normal, -normal_stiffness_*overlap);
        const gr::Vec3 point = gr::scale(gr::add(gr::add(bi.position, gap.leverI),
                                                gr::add(bj.position, gap.leverJ)), 0.5);
        const gr::Vec3 force_i = gr::scale(gr::add(lubrication.forceI, repulsion), factor);
        const gr::Vec3 torque_i = gr::scale(gr::add(lubrication.torqueI,
            gr::cross(gr::sub(point, bi.position), repulsion)), factor);
        const gr::Vec3 torque_j = gr::scale(gr::sub(lubrication.torqueJ,
            gr::cross(gr::sub(point, bj.position), repulsion)), factor);
        if (!gr::finite(force_i) || !gr::finite(torque_i) || !gr::finite(torque_j))
          throw std::runtime_error("nonfinite near-field force/torque");
        for (int k = 0; k < 3; ++k) {
          atom->f[i][k] += force_i[k];
          atom->f[j][k] -= force_i[k];
          atom->torque[i][k] += torque_i[k];
          atom->torque[j][k] += torque_j[k];
        }
        const double energy = factor*0.5*normal_stiffness_*overlap*overlap;
        if (evflag)
          ev_tally_xyz(i, j, nlocal, force->newton_pair, energy, 0.0,
                       force_i[0], force_i[1], force_i[2], rij[0], rij[1], rij[2]);
      } catch (const std::exception &exception) {
        error->one(FLERR, "slurry/spheroid pair {} {} failed: {}", atom->tag[i],
                   atom->tag[j], exception.what());
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

void PairSlurrySpheroid::write_restart_settings(FILE *fp)
{
  const double parameters[4] = {viscosity_, cutoff_gap_, minimum_gap_, normal_stiffness_};
  fwrite(parameters, sizeof(double), 4, fp);
}

void PairSlurrySpheroid::read_restart_settings(FILE *fp)
{
  double parameters[4];
  if (comm->me == 0) utils::sfread(FLERR, parameters, sizeof(double), 4, fp, nullptr, error);
  MPI_Bcast(parameters, 4, MPI_DOUBLE, 0, world);
  viscosity_ = parameters[0];
  cutoff_gap_ = parameters[1];
  minimum_gap_ = parameters[2];
  normal_stiffness_ = parameters[3];
}

void PairSlurrySpheroid::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) fwrite(&setflag[i][j], sizeof(int), 1, fp);
}

void PairSlurrySpheroid::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  if (!allocated) allocate();
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) {
      if (comm->me == 0)
        utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
    }
}

double PairSlurrySpheroid::memory_usage()
{
  return (atom->ntypes+1.0)*(atom->ntypes+1.0)*(sizeof(int)+sizeof(double)) +
      maximum_axis_.capacity()*sizeof(double);
}
