#include "fix_slurry_drag.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;

FixSlurryDrag::FixSlurryDrag(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), avec(nullptr), viscosity(0), shear_rate(0)
{
  if (narg != 5)
    error->all(FLERR, "Fix slurry/drag requires: viscosity shear_rate (SI units)");
  viscosity = utils::numeric(FLERR, arg[3], false, lmp);
  shear_rate = utils::numeric(FLERR, arg[4], false, lmp);
  if (!(viscosity > 0) || !std::isfinite(viscosity))
    error->all(FLERR, "Fix slurry/drag viscosity must be finite and positive");
  if (!std::isfinite(shear_rate))
    error->all(FLERR, "Fix slurry/drag shear rate must be finite");
}

int FixSlurryDrag::setmask()
{
  return POST_FORCE;
}

void FixSlurryDrag::init()
{
  if (strcmp(update->unit_style, "si") != 0)
    error->all(FLERR, "Fix slurry/drag requires units si");
  if (!utils::strmatch(update->integrate_style, "^verlet"))
    error->all(FLERR, "Fix slurry/drag requires run_style verlet");
  if (domain->dimension != 3)
    error->all(FLERR, "Fix slurry/drag requires three-dimensional particles");
  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec || atom->superellipsoid_flag)
    error->all(FLERR, "Fix slurry/drag requires regular atom_style ellipsoid");
  if (!atom->rmass || !atom->angmom || !atom->torque)
    error->all(FLERR, "Fix slurry/drag requires particle masses and angular momenta");
}

const SlurryHydro::Resistance &FixSlurryDrag::resistance(const double *shape)
{
  const std::array<double, 3> key{shape[0], shape[1], shape[2]};
  if (!(shape[0] > 0) || !(shape[1] > 0) || !(shape[2] > 0) ||
      !std::isfinite(shape[0]) || !std::isfinite(shape[1]) || !std::isfinite(shape[2]))
    error->one(FLERR, "Fix slurry/drag requires finite, positive semi-axes");
  if (std::abs(shape[0] - shape[1]) > 1.0e-10 * std::max(shape[0], shape[1]))
    error->one(FLERR, "Fix slurry/drag requires spheroid semi-axes (a,a,c)");
  auto found = resistance_cache.find(key);
  if (found != resistance_cache.end()) return found->second;
  try {
    return resistance_cache.emplace(key, SlurryHydro::unit_resistance(key)).first->second;
  } catch (const std::exception &exception) {
    error->one(FLERR, "Fix slurry/drag: {}", exception.what());
  }
}

void FixSlurryDrag::setup(int vflag)
{
  post_force(vflag);
}

void FixSlurryDrag::post_force(int /*vflag*/)
{
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  for (int i = 0; i < nlocal; ++i) {
    if (!(atom->mask[i] & groupbit)) continue;
    const int j = atom->ellipsoid[i];
    if (j < 0) error->one(FLERR, "Fix slurry/drag requires finite-size ellipsoid particles");
    double *shape = avec->bonus[j].shape;
    double *quat = avec->bonus[j].quat;
    const auto &r = resistance(shape);
    if (!(atom->rmass[i] > 0) || !std::isfinite(atom->rmass[i]))
      error->one(FLERR, "Fix slurry/drag particle masses must be finite and positive");

    double rotation[3][3];
    MathExtra::quat_to_mat(quat, rotation);
    const std::array<double, 3> axis{rotation[0][2], rotation[1][2], rotation[2][2]};
    const auto omega_flow = SlurryHydro::jeffery_omega(axis, shape[0], shape[2], shear_rate);
    double inertia[3];
    for (int k = 0; k < 3; ++k)
      inertia[k] = 0.2 * atom->rmass[i] *
          (shape[(k + 1) % 3] * shape[(k + 1) % 3] +
           shape[(k + 2) % 3] * shape[(k + 2) % 3]);
    double omega[3];
    MathExtra::mq_to_omega(atom->angmom[i], quat, inertia, omega);
    double velocity_slip[3]{atom->v[i][0] - shear_rate * (atom->x[i][1] - domain->boxlo[1]),
                            atom->v[i][1], atom->v[i][2]};
    double rotation_slip[3]{omega[0] - omega_flow[0], omega[1] - omega_flow[1],
                            omega[2] - omega_flow[2]};
    double velocity_body[3], rotation_body[3];
    MathExtra::transpose_matvec(rotation, velocity_slip, velocity_body);
    MathExtra::transpose_matvec(rotation, rotation_slip, rotation_body);
    for (int k = 0; k < 3; ++k) {
      velocity_body[k] *= -viscosity * r.translation[k];
      rotation_body[k] *= -viscosity * r.rotation[k];
    }
    double force[3], torque[3];
    MathExtra::matvec(rotation, velocity_body, force);
    MathExtra::matvec(rotation, rotation_body, torque);
    for (int k = 0; k < 3; ++k) {
      atom->f[i][k] += force[k];
      atom->torque[i][k] += torque[k];
    }
  }
}
