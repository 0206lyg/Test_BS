# Finalization procedure for the independent-reference candidate

These are preparation instructions, not a release or qualification record.
Do not refresh hashes while implementation or result files are still being
edited. No archive is created by this document's commands.

## Keep the deliverables distinct

- Keep the production full17 hard-floor implementation and its existing
  candidate status in the complete repository source bundle.
- Include this directory's driver, geometry, backend, matrix-free operator,
  comparator, exporter, dependency build script, runner, tests, and README.
- Keep same-pose scientific reports with their `.solution.npz`, native logs,
  comparison JSON, source provenance, and dependency manifest. Preserve
  failed and unqualified attempts. Empty native logs are meaningful files.
- Retain synthetic routing/storage reports as implementation evidence; they
  do not qualify a Stokes reference or a production table.
- Build pinned external dependencies outside the repository. Distribute
  their reproducible build instructions and build manifest; do not insert an
  active dependency checkout or workstation-specific library into the source
  bundle.

The reference currently has no fixed matching anchor. A future status update
must distinguish isolated-body checks, pair boundary accuracy, pair
refinement, held-out matching accuracy, and table encoding. A passing process
or a small linear residual cannot replace those checks.

## What the existing release guards actually cover

`release.json` stores two separate production identities:

| Field | Definition |
| --- | --- |
| `prepare_case_sha256` | SHA-256 of `lammps_slurry/tools/prepare_case.py` |
| `numerical_source_fingerprint` | `prepare_case.fast_source_fingerprint()` over selected numerical headers and generator sources |

`bundle_guard.check_source` checks the expected release ID, its required file
list, and the `prepare_case.py` hash. It does not compare the numerical
fingerprint and checks only the required presence of selected independent-workflow files. Their contents are covered by the separate source snapshot/manifest.

`bundle_guard.source_snapshot` hashes a larger production set, including
`release.json`, sources, selected tests, and vendor files. Store that snapshot
outside its own input set; adding its aggregate hash to `release.json` would
create a self-reference. Existing verified build manifests bind that snapshot,
absolute source directory, tool paths, and binary hashes. Do not rewrite an
old build manifest to make it match changed sources.

The release ID is also embedded in `bundle_guard.py`, `build_slurry_cpu.sbatch`,
and `lammps_slurry/run_slurry_cpu.sbatch`. If the ID is intentionally changed,
change all four places together before refreshing hashes. Changing only the
release ID does not grant physical qualification.

## Commands after all source edits are final

Run from the repository root. First update only the two production identity
fields; retain the factual status and qualification fields:

```bash
python3 -B - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path.cwd().resolve()
prepare_path = root / 'lammps_slurry/tools/prepare_case.py'
spec = importlib.util.spec_from_file_location('slurry_prepare_final', prepare_path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
path = root / 'release.json'
release = json.loads(path.read_text())
release['prepare_case_sha256'] = hashlib.sha256(prepare_path.read_bytes()).hexdigest()
release['numerical_source_fingerprint'] = prepare.fast_source_fingerprint()
pending = path.with_suffix('.json.tmp')
pending.write_text(json.dumps(release, indent=2) + '\n')
pending.replace(path)
print(json.dumps({key: release[key] for key in (
    'release_id', 'prepare_case_sha256', 'numerical_source_fingerprint')}, indent=2))
PY
```

Check the ID agreement and source guard, then capture a separate final source
manifest for the independent workflow and the production bundle:

```bash
python3 -B - <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

root = Path.cwd().resolve()
guard_path = root / 'lammps_slurry/tools/bundle_guard.py'
spec = importlib.util.spec_from_file_location('slurry_guard_final', guard_path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
release = guard.check_source(root, guard.RELEASE_ID)
for name in ('build_slurry_cpu.sbatch', 'lammps_slurry/run_slurry_cpu.sbatch'):
    match = re.search(r'^readonly SLURRY_LAUNCHER_RELEASE=(\S+)',
                      (root / name).read_text(), re.MULTILINE)
    if not match or match.group(1) != release['release_id']:
        raise RuntimeError('Launcher release mismatch: ' + name)
directory = root / 'validation_basis/independent_stokes'
suffixes = {'.py', '.f', '.f90', '.c', '.cpp', '.h', '.md', '.sh', '.sbatch', '.patch'}
paths = list(directory.iterdir())
if (directory / 'patches').is_dir():
    paths.extend((directory / 'patches').rglob('*'))
files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(paths)
         if path.is_file() and path.suffix in suffixes}
encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
manifest = {
    'schema': 'independent_reference_source_manifest_v1',
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'physical_qualification_claim': False,
    'release_id': release['release_id'],
    'independent_source_sha256': hashlib.sha256(encoded).hexdigest(),
    'independent_source_files': files,
    'production_source_snapshot': guard.source_snapshot(root),
    'scope': 'Source identity only; scientific reports retain their original hashes and status',
}
path = directory / 'source_manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
print(path)
print('Source identities verified; no build or physical qualification asserted.')
PY

git diff --check
```

The source manifest deliberately excludes itself, result JSON, densities,
and logs. A delivery inventory should separately hash the exact result files
selected for delivery after running jobs have stopped. Keep each historical
report's recorded source hashes unchanged even if current sources differ.

After extraction on the target machine, the source guard can be checked with:

```bash
python3 lammps_slurry/tools/bundle_guard.py check-source \
  --root "$PWD" \
  --expected-release 20261019-multi-ar-reference-campaign
```

Use the actual final release ID if it changed. A source-only delivery does
not contain a valid target-machine build identity. `check-build` additionally
requires a successful build, CTest/plugin load checks, all four offline
tools, and matching binary/source paths and hashes. Generate that identity
through the existing build workflow on the target machine; do not fabricate
it from source checks. The independent reference has its own external-library
build/run manifests and does not require a LAMMPS build.
