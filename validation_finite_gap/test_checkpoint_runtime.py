#!/usr/bin/env python3
"""Exercise the real table driver's OpenMP, publication and restart code.

Only the expensive physical provider is replaced by a deterministic SPD fixture.
This is NOT a Stokes, source-accuracy, interpolation, or rheology test. The actual
precompute_fast.cpp translation unit supplies all scheduling, grid expansion,
checkpoint, table serialization and exception handling code. Python 3.6 stdlib.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

SHIM = r'''
#ifndef SLURRY_TEST_SPD_FIXTURE_H
#define SLURRY_TEST_SPD_FIXTURE_H
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace SlurryScattering {
using Mat11=Eigen::Matrix<double,11,11>;
using Mat3=Eigen::Matrix3d;
using Vec3=Eigen::Vector3d;
inline Eigen::Matrix<double,5,5> strain_rotation(const Mat3&){return Eigen::Matrix<double,5,5>::Identity();}
struct Provider {
  struct {double aspect_ratio=0.;} basis;
  struct {void clear(){}} anchor_cache;
  int hybrid_version=3,anchor_count=2,film_grid=161;
  std::array<double,3> anchor_gaps{{.5,1.,0.}};
  void load(const std::string& p){std::ifstream f(p);if(!(f>>basis.aspect_ratio))throw std::runtime_error("fixture input");}
  Mat11 correction(const std::array<Vec3,2>& a,double h,double,unsigned,unsigned) const {
    Mat11 r=Mat11::Zero();
    for(int k=0;k<11;++k)r(k,k)=.1+.01*k+.005*(a[0].x()*a[0].x()+a[1].y()*a[1].y())+.001*h;
    return r;
  }
};
}
namespace SlurrySPDCoordinates {
using SlurryScattering::Mat11;
struct Shape {
  explicit Shape(double){}
  Mat11 self(const std::array<SlurryScattering::Vec3,2>&,double)const{return Mat11::Identity();}
  Mat11 contact_map(const std::array<SlurryScattering::Vec3,2>&,double)const{return Mat11::Identity();}
  Mat11 preconditioner(const std::array<SlurryScattering::Vec3,2>&,double,double)const{return Mat11::Identity();}
};
inline Mat11 inverse_contact_map(const Mat11& m){return m;}
}
namespace SlurrySPDTable {
inline SlurryScattering::Mat3 z_rotation(double){return SlurryScattering::Mat3::Identity();}
inline SlurryScattering::Mat11 rotation11(const SlurryScattering::Mat3&){return SlurryScattering::Mat11::Identity();}
}
#endif
'''


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    if result.returncode:
        raise RuntimeError('Driver failed ({}):\n{}'.format(result.returncode, result.stdout))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4,
                        help='Parallel worker count to compare with serial (default: 4)')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be at least 1')
    project = Path(__file__).resolve().parents[1]
    source = project / 'lammps_slurry/tools/precompute_fast.cpp'
    eigen = project / 'lammps_slurry/vendor/eigen3'
    with tempfile.TemporaryDirectory(prefix='slurry_checkpoint_runtime_') as temporary:
        root = Path(temporary)
        (root / 'slurry_spd_table.h').write_text(SHIM)
        binary = root / 'driver'
        subprocess.run([args.cxx, '-std=c++17', '-O1', '-DEIGEN_DONT_PARALLELIZE',
                        '-fopenmp', '-I' + str(root), '-I' + str(eigen),
                        str(source), '-o', str(binary)], check=True)
        basis = root / 'fixture.ssb'
        basis.write_text('2\n')
        def command(out, checkpoints, threads):
            return [str(binary), '--basis', str(basis), '--output', str(out), '--ar', '2',
                    '--epsilon-over-c', '.1', '--range', '.25', '--phi-nodes', '16',
                    '--angle-intervals', '8', '--threads', str(threads),
                    '--checkpoint-dir', str(checkpoints)]
        single = root / 'single.sft'
        single_log = run(command(single, root / 'single-checkpoints', 1))
        reference = json.loads(Path(str(single) + '.json').read_text())
        assert reference['chunks'] > 25, reference
        concurrent = root / 'concurrent.sft'
        concurrent_cache = root / 'concurrent-checkpoints'
        process = subprocess.Popen(command(concurrent, concurrent_cache, args.threads), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        observed = []
        try:
            for line in process.stdout:
                observed.append(line)
                if line.startswith('CHUNK_COMPLETED chunk=24/'):
                    process.kill()
                    break
            remainder, _ = process.communicate()
            if not any(x.startswith('CHUNK_COMPLETED chunk=24/') for x in observed):
                raise RuntimeError('Did not reach interruption point:\n' + ''.join(observed) + remainder)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        assert process.returncode < 0, process.returncode
        assert not concurrent.exists(), 'Interrupted calculation published a final table'
        directories = list(concurrent_cache.glob('spd_*'))
        assert len(directories) == 1, directories
        checkpoint_root = directories[0]
        completed = list(checkpoint_root.glob('chunk_*.bin'))
        assert 24 <= len(completed) < reference['chunks'], len(completed)
        # A killed writer may leave a partial pending file. Only atomically
        # published .bin shards may be read on restart.
        (checkpoint_root / 'chunk_24.bin.tmp.999999').write_bytes(b'interrupted write')
        resume_log = run(command(concurrent, concurrent_cache, args.threads))
        resumed = json.loads(Path(str(concurrent) + '.json').read_text())
        assert resumed['resumed_chunks'] == len(completed), (resumed, len(completed))
        assert digest(single) == digest(concurrent), 'Serial/parallel or restart results differ'
        original_root = next((root / 'single-checkpoints').glob('spd_*'))
        shards = list(original_root.glob('chunk_*.bin'))
        for shard in shards:
            assert digest(shard) == digest(checkpoint_root / shard.name), shard.name
        # Check the exact previously interrupted region: corrupted completed
        # chunk 24 must fail explicitly, with no final output publication.
        bad = checkpoint_root / 'chunk_23.bin'
        with bad.open('r+b') as stream:
            stream.seek(-16, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(bytes([byte[0] ^ 1]))
        rejected = root / 'must-not-publish.sft'
        rejection = subprocess.run(command(rejected, concurrent_cache, args.threads), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        assert rejection.returncode == 2, rejection.stdout
        assert 'Checkpoint checksum/length mismatch' in rejection.stdout, rejection.stdout
        assert not rejected.exists(), 'Corrupt checkpoint led to final publication'
        result = {'passed': True, 'scope': 'Actual table driver with synthetic SPD provider; no physical validation',
                  'source_sha256': digest(source), 'source_poses': reference['source_poses'],
                  'chunks': reference['chunks'], 'payload_bytes': reference['payload_bytes'],
                  'interrupted_after_chunk': 24, 'resumed_chunks': resumed['resumed_chunks'],
                  'threads_compared': [1, args.threads], 'byte_identical_table': True,
                  'byte_identical_checkpoint_shards': len(shards),
                  'incomplete_pending_shard_ignored': True, 'corrupt_completed_chunk_rejected': True}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
