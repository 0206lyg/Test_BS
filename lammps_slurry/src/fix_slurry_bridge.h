#ifdef FIX_CLASS
FixStyle(slurry / bridge, FixSlurryBridge);
#else
#ifndef LMP_FIX_SLURRY_BRIDGE_H
#define LMP_FIX_SLURRY_BRIDGE_H
#include "fix.h"
#include <memory>
namespace LAMMPS_NS {
class FixSlurryBridge : public Fix {
public:
  FixSlurryBridge(class LAMMPS *, int, char **);
  ~FixSlurryBridge() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void end_of_step() override;
  double compute_vector(int) override;

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  void solve_velocities();
  void write_history();
  double simulation_time() const;
};
} // namespace LAMMPS_NS
#endif
#endif
