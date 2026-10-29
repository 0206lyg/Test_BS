# Build test correction — 2026-10-28

The submitted HPC log completed compilation and passed 53 of 54 CTests. The
remaining test, `slurry_bridge_tools`, compared the selected normal stiffness
against the retired value 3 N/m in two assertions. Both expectations now correctly
require 30 N/m. Solver code and numerical settings are unchanged.

The earlier local run skipped the two tests requiring a compiled packer. The
corrected CTest was rerun with `SLURRY_TEST_PACKER` pointing to the compiled
`slurry_pack`: all 24 Python tests passed, with no skips. The complete rerun log is
`validation_nearfield_20261028/checks/buildfix_bridge_tools_ctest.log`.

The ZIP now also stores UTC file modification times. Extract with `-DD` to use
the extraction time and remove the previous bundle's future timestamps:

```bash
unzip -DD -o slurry_nearfield_overlay_20261028.zip
sbatch build_slurry_cpu.sbatch
```

The current run script is excluded from the ZIP. Its existing settings remain
intact, and the settings migration helper need not be rerun for this correction.
