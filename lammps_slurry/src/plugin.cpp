// Project-local LAMMPS plugin; upstream LAMMPS sources remain unchanged.
#include "lammpsplugin.h"
#include "version.h"

#include "fix_slurry_drag.h"
#include "fix_slurry_implicit.h"
#ifdef SLURRY_HAVE_EIGEN
#include "fix_slurry_bridge.h"
#endif
#include "pair_slurry_spheroid.h"

using namespace LAMMPS_NS;

namespace {
void *create_pair(void *lmp)
{
  return new PairSlurrySpheroid(static_cast<LAMMPS *>(lmp));
}

void *create_drag(void *lmp, int argc, char **argv)
{
  return new FixSlurryDrag(static_cast<LAMMPS *>(lmp), argc, argv);
}
void *create_implicit(void *lmp, int argc, char **argv)
{
  return new FixSlurryImplicit(static_cast<LAMMPS *>(lmp), argc, argv);
}
#ifdef SLURRY_HAVE_EIGEN
void *create_bridge(void *lmp, int argc, char **argv)
{
  return new FixSlurryBridge(static_cast<LAMMPS *>(lmp), argc, argv);
}
#endif
}    // namespace

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  const auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.author = "Test_BS contributors";
  plugin.handle = handle;

  plugin.style = "pair";
  plugin.name = "slurry/spheroid";
  plugin.info = "Spheroid lubrication and repulsive contact";
  plugin.creator.v1 = &create_pair;
  register_plugin(&plugin, lmp);

  plugin.style = "fix";
  plugin.name = "slurry/drag";
  plugin.info = "Spheroid drag and Jeffery rotation in simple shear";
  plugin.creator.v2 = &create_drag;
  register_plugin(&plugin, lmp);
  plugin.name = "slurry/implicit";
  plugin.info = "Coupled overdamped spheroid resistance and implicit elastic contact";
  plugin.creator.v2 = &create_implicit;
  register_plugin(&plugin, lmp);
#ifdef SLURRY_HAVE_EIGEN
  plugin.name = "slurry/bridge";
  plugin.info = "Compact spheroid lubrication, periodic mobility and implicit contact";
  plugin.creator.v2 = &create_bridge;
  register_plugin(&plugin, lmp);
#endif
}
