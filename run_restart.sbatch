#!/bin/bash
#SBATCH --job-name=restart
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=restart_%j.out

# Edit these five settings, then submit: sbatch run_restart.sbatch
# One entry point for bl, pG, and cmc. The original in.bridge supplies all
# remaining force, solver, output, and averaging settings without regeneration.
SOURCE_RUN="/scratch/negishi/lee6456/pG/pG_cw0.0212_n500_sr100_45690406"
NEW_SHEAR_RATE=10
NEW_TIMESTEP=2.5e-6
NEW_STEPS=1000000
RESULTS_ROOT="/scratch/negishi/lee6456/pG"

# final.data retains positions, orientations, particle properties and box tilt.
# This is a configuration restart: contact springs, adaptive-kn state and solver
# warm starts initialize anew. Configured kn and its control settings are kept.
# No history.csv, release.json, source hash, or history-column check is used.
# Prefer the source run's runtime to keep exactly the same solver binary.
# History columns follow that binary; old history files are never inspected.
# If that runtime was removed, the existing build/lammps/env.sh is a fallback.
# No compilation or packing is performed by this script.

set -Eeuo pipefail
if (( $# != 0 )); then
    printf 'Edit the five settings in run_restart.sbatch, then submit it without arguments.\n' >&2
    exit 2
fi
if type module >/dev/null 2>&1; then
    module purge
    module load "${SLURRY_PYTHON_MODULE:-anaconda/2024.02-py311}"
    module load "${SLURRY_GCC_MODULE:-gcc/12.2.0}"
    module load "${SLURRY_MPI_MODULE:-openmpi/4.1.4}"
fi
export SLURRY_RESTART_SOURCE_ROOT="${SLURRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
export PYTHONUNBUFFERED=1
python3 - "$SOURCE_RUN" "$NEW_SHEAR_RATE" "$NEW_TIMESTEP" "$NEW_STEPS" "$RESULTS_ROOT" <<'PY_RESTART'
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def optional_json(path):
    """Ancillary records are helpful, never a version/schema gate."""
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def command_tokens(line):
    # Retain spans so all unmodified input text, including comments, survives.
    return list(re.finditer(r'''"[^"\n]*"|'[^'\n]*'|[^\s#]+''', line.splitlines()[0] if line else ''))


def rewrite_input(text, shear_rate, timestep, steps):
    if not math.isfinite(shear_rate) or not math.isfinite(timestep) or timestep <= 0 or steps < 0:
        raise ValueError('shear rate must be finite, timestep positive, and steps nonnegative')
    replacements = {'read_data': 0, 'timestep': 0, 'deform': 0, 'bridge': 0, 'run': 0}
    rewritten = []
    for line in text.splitlines(keepends=True):
        matches = command_tokens(line)
        # Stop at the first unquoted comment marker. The generated inputs have
        # one command per line; quoted file names are kept as single tokens.
        comment = None
        quote = None
        for i, char in enumerate(line):
            if quote:
                if char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == '#':
                comment = i
                break
        if comment is not None:
            matches = [m for m in matches if m.start() < comment]
        tokens = [m.group() for m in matches]
        change = None
        if tokens and tokens[0] == 'read_data' and len(tokens) >= 2:
            change = (1, 'particles.data', 'read_data')
        elif tokens and tokens[0] == 'timestep' and len(tokens) >= 2:
            change = (1, format(timestep, '.17g'), 'timestep')
        elif len(tokens) >= 8 and tokens[0] == 'fix' and tokens[3] == 'deform':
            for i in range(4, len(tokens)-2):
                if tokens[i:i+2] == ['xz', 'erate']:
                    change = (i+2, format(shear_rate, '.17g'), 'deform')
                    break
        elif len(tokens) >= 6 and tokens[0] == 'fix' and tokens[3] == 'slurry/bridge':
            change = (5, format(shear_rate, '.17g'), 'bridge')
        elif tokens and tokens[0] == 'run' and len(tokens) >= 2:
            change = (1, str(steps), 'run')
        if change:
            index, value, kind = change
            m = matches[index]
            line = line[:m.start()]+value+line[m.end():]
            replacements[kind] += 1
        rewritten.append(line)
    missing = [key for key, count in replacements.items() if count != 1]
    if missing:
        raise ValueError('Cannot identify a single '+', '.join(missing)+
                         ' command in the source in.bridge; no input was guessed')
    return ''.join(rewritten)


def output_name(source_name, shear_rate, job_id):
    # Preserve all existing labels, including CMC concentration and rounded cw.
    match = re.fullmatch(r'(.+)_sr[^_]+(?:_.*)?', source_name)
    prefix = match.group(1) if match else source_name
    if not job_id or any(c in job_id for c in ('/', '\\', '\n', '\r')):
        raise ValueError('Invalid job identifier for the output directory')
    return '{}_sr{}_{}'.format(prefix, format(shear_rate, '.12g'), job_id)


def iteration_limit(source, launch):
    saved = launch.get('solver_environment', {})
    try:
        value = int(saved.get('SLURRY_IMEX_MAX_ITERATIONS', 0))
        if value > 0:
            return value, 'source launch.json'
    except (TypeError, ValueError, AttributeError):
        pass
    pattern = re.compile(r'IMEX iteration limits:.*\btotal=(\d+)\b')
    for name in ('console.log', 'console.txt', 'log.lammps'):
        try:
            with (source/name).open(errors='replace') as stream:
                for index, line in enumerate(stream):
                    match = pattern.search(line)
                    if match:
                        return int(match.group(1)), 'source '+name
                    if index >= 5000:
                        break
        except OSError:
            pass
    # All three current batch launchers explicitly use 100; the plugin's
    # internal default is different, so do not accidentally fall back to it.
    return 100, 'common bl/pG/cmc launcher default (original log unavailable)'


def environment_file(path):
    values = {}
    try:
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if len(parts) == 2 and parts[0] == 'export' and '=' in parts[1]:
                key, value = parts[1].split('=', 1)
                values[key] = value
    except OSError:
        pass
    return values


def choose_runtime(source, environ):
    runtime = source/'runtime'
    if (runtime/'bin/lmp').is_file() and (runtime/'bin/libslurry.so').is_file():
        identity = optional_json(runtime/'runtime_identity.json')
        return {'kind': 'source_private_runtime', 'directory': runtime,
                'executable': runtime/'bin/lmp', 'plugin': runtime/'bin/libslurry.so',
                'identity': identity}
    # No compatibility gate: use an already built runtime if the old private
    # copy was deliberately removed. Nothing in the original history is read.
    roots = [Path(environ.get('SLURRY_RESTART_SOURCE_ROOT', Path.cwd()))]
    identity = optional_json(runtime/'runtime_identity.json')
    if identity.get('source_root'):
        roots.append(Path(identity['source_root']))
    for root in roots:
        if root.name == 'lammps_slurry':
            root = root.parent
        values = environment_file(root/'build/lammps/env.sh')
        executable = Path(values.get('SLURRY_LAMMPS_EXECUTABLE', '/nonexistent/lmp'))
        plugin = Path(values.get('SLURRY_LAMMPS_PLUGIN', '/nonexistent/libslurry.so'))
        if executable.is_file() and plugin.is_file():
            return {'kind': 'existing_build_fallback', 'executable': executable,
                    'plugin': plugin, 'identity': {}}
    raise FileNotFoundError('The source runtime/bin/lmp and libslurry.so are missing, '
                            'and no existing build/lammps/env.sh points to usable binaries. '
                            'Submit from the built slurry repository or restore the source runtime folder.')


def uses_mpi(runtime, launch, environment):
    approved = runtime['identity'].get('approved_build', {})
    if isinstance(approved, dict) and isinstance(approved.get('mpi'), bool):
        return approved['mpi']
    # A restarted launch records this directly, including serial runs.
    if runtime['kind'] == 'source_private_runtime' and isinstance(launch.get('mpi'), bool):
        return launch['mpi']
    if runtime['kind'] == 'source_private_runtime':
        command = launch.get('command', [])
        if command and Path(str(command[0])).name in ('mpirun', 'mpiexec'):
            return True
    try:
        linked = subprocess.run(['ldd', str(runtime['executable'])], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, check=False)
        return bool(re.search(r'\blib(?:mpi|mpich)[\w.\-]*\.so', linked.stdout))
    except OSError:
        return False


def quoted_plugin(path):
    value = str(path)
    if any(c in value for c in ('"', '\n', '\r')):
        raise ValueError('LAMMPS plugin path cannot contain a double quote or a line break')
    return '"'+value+'"'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def relocate_paths(value, old, new):
    if isinstance(value, dict):
        return {key: relocate_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_paths(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old+os.sep)):
        return new+value[len(old):]
    return value


def prepare_restart(source, results_root, shear_rate, timestep, steps, job_id, environ):
    source = Path(source).expanduser().resolve()
    results_root = Path(results_root).expanduser()
    if not results_root.is_absolute():
        raise ValueError('RESULTS_ROOT must be an absolute path')
    results_root = results_root.resolve()
    final_data = source/'final.data'
    if not final_data.is_file():
        raise FileNotFoundError('Missing completed configuration: '+str(final_data))
    original = (source/'in.bridge').read_text()
    new_input = rewrite_input(original, shear_rate, timestep, steps)
    launch = optional_json(source/'launch.json')
    parameters = optional_json(source/'parameters.json')
    runtime = choose_runtime(source, environ)
    output = results_root/output_name(source.name, shear_rate, job_id)
    if output == source or source in output.parents:
        raise ValueError('The new run must be outside the source run directory')
    environment = dict(environ)
    saved_threads = launch.get('threads', 1)
    threads = int(environment.get('SLURM_CPUS_PER_TASK', saved_threads))
    if threads < 1:
        raise ValueError('CPU allocation must be positive')
    environment.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                       OMP_PLACES='cores', OMP_PROC_BIND='close', OMP_MAX_ACTIVE_LEVELS='1',
                       MKL_NUM_THREADS=str(threads), MKL_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1')
    limit, limit_origin = iteration_limit(source, launch)
    environment['SLURRY_IMEX_MAX_ITERATIONS'] = str(limit)
    environment.setdefault('SLURRY_BRIDGE_TRACE', '0')
    results_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    # A complete byte copy retains the triclinic box, image flags, shapes and
    # quaternions. Never call the packing/preparation pipeline on this data.
    shutil.copy2(final_data, output/'particles.data')
    (output/'source.in.bridge').write_text(original)
    (output/'in.bridge').write_text(new_input)
    if runtime['kind'] == 'source_private_runtime':
        old_runtime = str(runtime['directory'])
        shutil.copytree(runtime['directory'], output/'runtime')
        runtime['directory'] = output/'runtime'
        runtime['executable'] = output/'runtime/bin/lmp'
        runtime['plugin'] = output/'runtime/bin/libslurry.so'
        runtime['identity'] = relocate_paths(runtime['identity'], old_runtime, str(runtime['directory']))
        if runtime['identity']:
            write_json(output/'runtime/runtime_identity.json', runtime['identity'])
        parameters = relocate_paths(parameters, old_runtime, str(runtime['directory']))
        private_lib = str(output/'runtime/lib')
        environment['LD_LIBRARY_PATH'] = private_lib+(':'+environment['LD_LIBRARY_PATH']
                                                       if environment.get('LD_LIBRARY_PATH') else '')
    mpi = uses_mpi(runtime, launch, environment)
    command = [str(runtime['executable']), '-var', 'slurry_plugin',
               quoted_plugin(runtime['plugin']), '-in', 'in.bridge']
    if mpi:
        launcher = shutil.which('mpirun', path=environment.get('PATH')) or shutil.which('mpiexec', path=environment.get('PATH'))
        if not launcher:
            raise FileNotFoundError('The inherited MPI executable requires mpirun/mpiexec; load its MPI module')
        command = [launcher, '-np', '1', '--map-by', 'slot:PE={}'.format(threads),
                   '--bind-to', 'core', '--report-bindings']+command
    changes = {'shear_rate': shear_rate, 'timestep': timestep, 'steps': steps}
    restart = {'source_run': str(source), 'source_configuration': str(final_data),
               'configuration_file': 'particles.data', 'changed_simulation_settings': changes,
               'input_settings': 'All other in.bridge commands and options retained verbatim',
               'initialization_used': 'read_data from source final.data; no repacking or box reset',
               'time_origin': 'New run starts at step=0, time=0, strain=0; inherited box tilt is retained',
               'state_not_saved_in_data': ['contact spring histories', 'solver warm start',
                                            'current adaptive kn and controller state'],
               'runtime_selection': runtime['kind'], 'imex_iteration_limit_origin': limit_origin}
    if (source/'parameters.json').is_file():
        shutil.copy2(source/'parameters.json', output/'source.parameters.json')
    parameters.update(changes)
    parameters.update(output=str(output), accumulated_strain=abs(shear_rate)*timestep*steps,
                      configuration_restart=restart)
    write_json(output/'parameters.json', parameters)
    if (source/'run_metadata.json').is_file():
        metadata = optional_json(source/'run_metadata.json')
        metadata.update(changes)
        metadata.update(output=str(output), configuration_restart=restart)
        write_json(output/'run_metadata.json', metadata)
    thread_keys = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PLACES', 'OMP_PROC_BIND',
                   'OMP_MAX_ACTIVE_LEVELS', 'MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS')
    record = {'configuration_restart': restart, 'command': command, 'mpi': mpi,
              'mpi_ranks': 1, 'threads': threads,
              'thread_environment': {k: environment[k] for k in thread_keys},
              'solver_environment': {'SLURRY_IMEX_MAX_ITERATIONS': str(limit)},
              'runtime_identity': 'runtime/runtime_identity.json' if (output/'runtime/runtime_identity.json').is_file() else None}
    write_json(output/'launch.json', record)
    write_json(output/'restart_info.json', dict(restart, status='prepared'))
    return output, command, environment, record


def main():
    if len(sys.argv) != 6:
        raise ValueError('Expected source run, shear rate, timestep, steps, results root')
    source, rate, dt, steps, results = sys.argv[1:]
    job_id = os.environ.get('SLURM_JOB_ID') or 'local_{}_{}'.format(
        datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'), os.getpid())
    output, command, environment, record = prepare_restart(
        Path(source), Path(results), float(rate), float(dt), int(steps), job_id, os.environ)
    print('Source run: {}'.format(Path(source).resolve()), flush=True)
    print('Run directory: {}'.format(output), flush=True)
    print('New shear rate={} /s; timestep={} s; steps={}'.format(rate, dt, steps), flush=True)
    print('Runtime: {}; IMEX total iterations={}'.format(
        record['configuration_restart']['runtime_selection'],
        record['solver_environment']['SLURRY_IMEX_MAX_ITERATIONS']), flush=True)
    print('Command: {}'.format(' '.join(shlex.quote(s) for s in command)), flush=True)
    with (output/'console.log').open('w') as log:
        process = subprocess.Popen(command, cwd=str(output), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    write_json(output/'restart_info.json', dict(record['configuration_restart'],
                                               status='completed' if code == 0 else 'failed',
                                               exit_code=code))
    if code:
        raise RuntimeError('LAMMPS exited {}; see {}'.format(code, output/'console.log'))
    print('Run completed: {}'.format(output), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print('run_restart.sbatch: {}'.format(error), file=sys.stderr)
        sys.exit(2)
PY_RESTART
