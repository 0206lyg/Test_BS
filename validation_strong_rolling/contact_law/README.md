# Strong sliding and rolling contact check

`check.cpp` independently evaluates the new physical force and torque caps at a
prescribed 1 nN normal load, then solves three controlled two-body problems with
the production contact solver: all elastic, all yielding, and mixed branches.
The reference solution follows directly from force/torque balance with diagonal
identity mobility; it is not a comparison against another implementation of the
same iterative solver.

Parameters: sliding coefficient 0.3, rolling coefficient 0.3, rolling length
3 micrometres (AR20, full thickness 300 nm), normal stiffness 100 N/m,
tangential stiffness 80 N/m, rolling stiffness 9e-10 N m/rad. Twisting retains
coefficient 0.1 and length 45 nm. The timestep is 1 microsecond. The solve uses
the same dimensional scaling as the bridge (150 nm, 0.01 s, 0.01 Pa s).

The test checks Coulomb force/torque saturation, elastic yield displacement,
nonnegative plastic work, analytic force/torque balance, mechanical power
balance, and reuse of a converged solution without extra Krylov iterations.
`result.json` contains the measured values and source SHA256 hashes.

Compile from the repository root with a suitable Eigen include path:

```
g++ -std=c++17 -O1 -I /path/to/eigen3 -I lammps_slurry/src validation_strong_rolling/contact_law/check.cpp -o /tmp/slurry_strong_contact_check
/tmp/slurry_strong_contact_check
```

This is a constitutive and solver check, not a bulk viscosity measurement, and
does not establish a jamming volume fraction.
