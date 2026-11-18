#!/bin/bash
#SBATCH --job-name=restart
#SBATCH --account=ardekani
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=14-00:00:00
#SBATCH --output=restart_%j.out

# Edit these settings, then submit from the built repository: sbatch run_restart.sbatch
# One entry point for bl, pG, and cmc. The original in.bridge supplies all
# remaining force, solver, output, and averaging settings without regeneration.
SOURCE_RUN="/scratch/negishi/lee6456/cmc_c12/cmc_c12_cw0.0000_h7.5_n500_sr100_45929568"
NEW_SHEAR_RATE=10
NEW_TIMESTEP=2.5e-6
NEW_STEPS=1000000
RESULTS_ROOT="/scratch/negishi/lee6456/cmc_c12"

# N/m. Empty means inherit the value in the source run's in.bridge.
# Example: for a 100 -> 10 /s test with old kn=kt=kr=100, set all three to 10.
# No automatic shear-rate scaling is applied. The existing kn_adapt settings
# remain unchanged; NEW_NORMAL_STIFFNESS sets its initial/minimum stiffness.
NEW_NORMAL_STIFFNESS="450"
NEW_TANGENTIAL_STIFFNESS="150"
NEW_ROLLING_STIFFNESS="150"

# final.data retains positions, orientations, particle properties and box tilt.
# This is a configuration restart: contact springs, adaptive-kn state and solver
# warm starts initialize anew. Only the explicitly supplied settings change.
# No history.csv, release.json, source hash, or history-column check is used.
# Always use the CURRENT repository's build/lammps/env.sh, just as new runs do.
# Its plugin records N1, N2 and non-rattler Z regardless of the old run's columns.
# After updating the diagnostic source, build it once before submitting this job.
# The source run's old runtime is never reused; old history is never inspected.
# No compilation or packing is performed by this script.

set -Eeuo pipefail
if (( $# != 0 )); then
    printf 'Edit the settings in run_restart.sbatch, then submit it without arguments.\n' >&2
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
python3 - "$SOURCE_RUN" "$NEW_SHEAR_RATE" "$NEW_TIMESTEP" "$NEW_STEPS" "$RESULTS_ROOT" \
    "$NEW_NORMAL_STIFFNESS" "$NEW_TANGENTIAL_STIFFNESS" "$NEW_ROLLING_STIFFNESS" <<'PY_RESTART'
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


def stiffness_overrides(normal_stiffness=None, tangential_stiffness=None, rolling_stiffness=None):
    overrides = {}
    for name, value in (('normal_stiffness', normal_stiffness),
                        ('tangential_stiffness', tangential_stiffness),
                        ('rolling_stiffness', rolling_stiffness)):
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0 or (name == 'normal_stiffness' and value == 0):
            raise ValueError(name+' must be finite and '+
                             ('positive' if name == 'normal_stiffness' else 'nonnegative'))
        overrides[name] = value
    return overrides


def stiffness_positions(tokens):
    positions = {'normal_stiffness': 8} if len(tokens) > 8 else {}
    for name, keyword in (('tangential_stiffness', 'contact'), ('rolling_stiffness', 'rolling')):
        found = [i+2 for i in range(9, len(tokens)-2) if tokens[i] == keyword]
        if len(found) == 1:
            positions[name] = found[0]
    return positions


def effective_stiffness(text):
    for line in text.splitlines():
        tokens = shlex.split(line, comments=True)
        if len(tokens) >= 4 and tokens[0] == 'fix' and tokens[3] == 'slurry/bridge':
            return {name: float(tokens[index]) for name, index in stiffness_positions(tokens).items()}
    return {}


def rewrite_input(text, shear_rate, timestep, steps, normal_stiffness=None,
                  tangential_stiffness=None, rolling_stiffness=None):
    if not math.isfinite(shear_rate) or not math.isfinite(timestep) or timestep <= 0 or steps < 0:
        raise ValueError('shear rate must be finite, timestep positive, and steps nonnegative')
    overrides = stiffness_overrides(normal_stiffness, tangential_stiffness, rolling_stiffness)
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
        changes = []
        if tokens and tokens[0] == 'read_data' and len(tokens) >= 2:
            changes.append((1, 'particles.data', 'read_data'))
        elif tokens and tokens[0] == 'timestep' and len(tokens) >= 2:
            changes.append((1, format(timestep, '.17g'), 'timestep'))
        elif len(tokens) >= 8 and tokens[0] == 'fix' and tokens[3] == 'deform':
            for i in range(4, len(tokens)-2):
                if tokens[i:i+2] == ['xz', 'erate']:
                    changes.append((i+2, format(shear_rate, '.17g'), 'deform'))
                    break
        elif len(tokens) >= 6 and tokens[0] == 'fix' and tokens[3] == 'slurry/bridge':
            changes.append((5, format(shear_rate, '.17g'), 'bridge'))
            positions = stiffness_positions(tokens)
            for name, value in overrides.items():
                if name not in positions:
                    raise ValueError('Cannot locate '+name+' in source slurry/bridge command')
                changes.append((positions[name], format(value, '.17g'), None))
        elif tokens and tokens[0] == 'run' and len(tokens) >= 2:
            changes.append((1, str(steps), 'run'))
        # Replace right to left so multiple edits keep all original spans valid.
        for index, value, kind in sorted(changes, reverse=True):
            m = matches[index]
            line = line[:m.start()]+value+line[m.end():]
            if kind:
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
    # The old private plugin may predate N1/N2/Z. Always select the current
    # repository build; final.data and physical inputs are independent of it.
    root = Path(environ.get('SLURRY_RESTART_SOURCE_ROOT', Path.cwd())).expanduser().resolve()
    if root.name == 'lammps_slurry':
        root = root.parent
    envfile = root/'build/lammps/env.sh'
    values = environment_file(envfile)
    executable = Path(values.get('SLURRY_LAMMPS_EXECUTABLE', '/nonexistent/lmp')).expanduser()
    plugin = Path(values.get('SLURRY_LAMMPS_PLUGIN', '/nonexistent/libslurry.so')).expanduser()
    if not executable.is_absolute():
        executable = root/executable
    if not plugin.is_absolute():
        plugin = root/plugin
    if executable.is_file() and plugin.is_file():
        return {'kind': 'current_repository_build', 'source_root': root,
                'environment_file': envfile, 'environment': values,
                'executable': executable.resolve(), 'plugin': plugin.resolve(), 'identity': {}}
    raise FileNotFoundError('Current repository build is unavailable: '+str(envfile)+
                            '. Run sbatch build_slurry_cpu.sbatch in the updated repository, '
                            'wait for completion, then submit run_restart.sbatch there.')


def uses_mpi(runtime, launch, environment):
    # Detect the selected executable, never the old run's MPI metadata.
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


def prepare_restart(source, results_root, shear_rate, timestep, steps, job_id, environ,
                    normal_stiffness=None, tangential_stiffness=None, rolling_stiffness=None):
    source = Path(source).expanduser().resolve()
    results_root = Path(results_root).expanduser()
    if not results_root.is_absolute():
        raise ValueError('RESULTS_ROOT must be an absolute path')
    results_root = results_root.resolve()
    final_data = source/'final.data'
    if not final_data.is_file():
        raise FileNotFoundError('Missing completed configuration: '+str(final_data))
    original = (source/'in.bridge').read_text()
    overrides = stiffness_overrides(normal_stiffness, tangential_stiffness, rolling_stiffness)
    new_input = rewrite_input(original, shear_rate, timestep, steps, **overrides)
    stiffness = effective_stiffness(new_input)
    launch = optional_json(source/'launch.json')
    parameters = optional_json(source/'parameters.json')
    runtime = choose_runtime(source, environ)
    output = results_root/output_name(source.name, shear_rate, job_id)
    if output == source or source in output.parents:
        raise ValueError('The new run must be outside the source run directory')
    environment = dict(environ)
    environment.update(runtime['environment'])
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
    mpi = uses_mpi(runtime, launch, environment)
    command = [str(runtime['executable']), '-var', 'slurry_plugin',
               quoted_plugin(runtime['plugin']), '-in', 'in.bridge']
    if mpi:
        launcher = shutil.which('mpirun', path=environment.get('PATH')) or shutil.which('mpiexec', path=environment.get('PATH'))
        if not launcher:
            raise FileNotFoundError('The current MPI executable requires mpirun/mpiexec; load its MPI module')
        command = [launcher, '-np', '1', '--map-by', 'slot:PE={}'.format(threads),
                   '--bind-to', 'core', '--report-bindings']+command
    changes = {'shear_rate': shear_rate, 'timestep': timestep, 'steps': steps}
    changes.update(overrides)
    runtime_build = {'source_root': str(runtime['source_root']),
                     'environment_file': str(runtime['environment_file']),
                     'executable': str(runtime['executable']), 'plugin': str(runtime['plugin'])}
    restart = {'source_run': str(source), 'source_configuration': str(final_data),
               'configuration_file': 'particles.data', 'changed_simulation_settings': changes,
               'effective_stiffness': stiffness,
               'input_settings': 'All other in.bridge commands and options retained verbatim',
               'initialization_used': 'read_data from source final.data; no repacking or box reset',
               'time_origin': 'New run starts at step=0, time=0, strain=0; inherited box tilt is retained',
               'state_not_saved_in_data': ['contact spring histories', 'solver warm start',
                                            'current adaptive kn and controller state'],
               'runtime_selection': runtime['kind'], 'runtime_build': runtime_build,
               'imex_iteration_limit_origin': limit_origin}
    if (source/'parameters.json').is_file():
        shutil.copy2(source/'parameters.json', output/'source.parameters.json')
    parameters.update(changes)
    parameters.update(stiffness)
    parameters.pop('runtime_identity', None)
    parameters.update(output=str(output), accumulated_strain=abs(shear_rate)*timestep*steps,
                      configuration_restart=restart, runtime_build=runtime_build)
    write_json(output/'parameters.json', parameters)
    if (source/'run_metadata.json').is_file():
        metadata = optional_json(source/'run_metadata.json')
        metadata.update(changes)
        metadata.update(stiffness)
        metadata.pop('runtime_identity', None)
        metadata.update(output=str(output), configuration_restart=restart, runtime_build=runtime_build)
        write_json(output/'run_metadata.json', metadata)
    thread_keys = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PLACES', 'OMP_PROC_BIND',
                   'OMP_MAX_ACTIVE_LEVELS', 'MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS')
    record = {'configuration_restart': restart, 'command': command, 'mpi': mpi,
              'mpi_ranks': 1, 'threads': threads,
              'thread_environment': {k: environment[k] for k in thread_keys},
              'solver_environment': {'SLURRY_IMEX_MAX_ITERATIONS': str(limit)},
              'runtime_identity': None, 'runtime_build': runtime_build}
    write_json(output/'launch.json', record)
    write_json(output/'restart_info.json', dict(restart, status='prepared'))
    return output, command, environment, record


def main():
    if len(sys.argv) != 9:
        raise ValueError('Expected source run, shear rate, timestep, steps, results root, kn, kt, kr')
    source, rate, dt, steps, results, kn, kt, kr = sys.argv[1:]
    job_id = os.environ.get('SLURM_JOB_ID') or 'local_{}_{}'.format(
        datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'), os.getpid())
    output, command, environment, record = prepare_restart(
        Path(source), Path(results), float(rate), float(dt), int(steps), job_id, os.environ,
        normal_stiffness=kn, tangential_stiffness=kt, rolling_stiffness=kr)
    print('Source run: {}'.format(Path(source).resolve()), flush=True)
    print('Run directory: {}'.format(output), flush=True)
    print('New shear rate={} /s; timestep={} s; steps={}'.format(rate, dt, steps), flush=True)
    print('Contact stiffness (N/m): {}'.format(', '.join(
        '{}={:.12g}'.format(key, value)
        for key, value in record['configuration_restart']['effective_stiffness'].items())), flush=True)
    print('Runtime: {}; IMEX total iterations={}'.format(
        record['configuration_restart']['runtime_selection'],
        record['solver_environment']['SLURRY_IMEX_MAX_ITERATIONS']), flush=True)
    print('Plugin: {}'.format(record['runtime_build']['plugin']), flush=True)
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
