#!/usr/bin/env python3
"""Compile the actual generator checkpoint helpers; test integrity without a physics solve.

Requires the same C++17 compiler as the executable; no Python third-party modules.
Run from any directory. Optional --output writes the compact JSON result.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

HEADERS = r"""#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
"""
TEST = r"""
int main(){
  const std::string path="slurry_checkpoint_roundtrip.bin",identity="test_identity";
  constexpr std::size_t stride=66;
  const std::vector<std::uint64_t> rows{2,0,4};
  std::vector<double> original(5*stride),loaded(5*stride,-1.);
  for(std::size_t k=0;k<original.size();++k)original[k]=.125*double(k)-1.;
  checkpoint::write(path,identity,7,rows,stride,original,{.017,.0002});
  auto bounds=checkpoint::read(path,identity,7,rows,stride,loaded);
  if(bounds.minimum!=.017||bounds.pole_change!=.0002)return 1;
  for(auto row:rows)for(std::size_t k=0;k<stride;++k)if(loaded[row*stride+k]!=original[row*stride+k])return 2;
  for(std::uint64_t row:{1,3})for(std::size_t k=0;k<stride;++k)if(loaded[row*stride+k]!=-1.)return 3;
  int expected_failures=0;
  auto reject=[&](auto operation){try{operation();}catch(const std::exception&){++expected_failures;}};
  reject([&]{checkpoint::read(path,"wrong",7,rows,stride,loaded);});
  reject([&]{checkpoint::read(path,identity,8,rows,stride,loaded);});
  reject([&]{checkpoint::read(path,identity,7,{0,2,4},stride,loaded);});
  reject([&]{checkpoint::read(path,identity,7,rows,stride-1,loaded);});
  std::ifstream in(path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(in)),{});in.close();
  auto install_bytes=[&](std::string content){std::ofstream f(path,std::ios::binary);f.write(content.data(),content.size());};
  auto damaged=bytes;damaged[damaged.size()-16]^=1;install_bytes(damaged);
  reject([&]{checkpoint::read(path,identity,7,rows,stride,loaded);});
  install_bytes(bytes.substr(0,bytes.size()-4));
  reject([&]{checkpoint::read(path,identity,7,rows,stride,loaded);});
  install_bytes(bytes+"x");
  reject([&]{checkpoint::read(path,identity,7,rows,stride,loaded);});
  if(expected_failures!=7)return 4;
  std::remove(path.c_str());
  std::cout<<"PASS: checkpoint roundtrip; exact row coverage; wrong identity/chunk/order/stride, corrupt payload, truncation and trailing bytes rejected\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent / "lammps_slurry/tools/precompute_fast.cpp"
    contents = source.read_text()
    begin = contents.index("namespace checkpoint {")
    end = contents.index("\nint main(", begin)
    with tempfile.TemporaryDirectory(prefix="slurry_checkpoint_test_") as working:
        working = Path(working)
        cpp = working / "checkpoint_test.cpp"
        binary = working / "checkpoint_test"
        cpp.write_text(HEADERS + contents[begin:end] + TEST)
        subprocess.run([args.cxx, "-O1", "-std=c++17", str(cpp), "-o", str(binary)], check=True)
        run = subprocess.run([str(binary)], cwd=str(working), check=True, stdout=subprocess.PIPE, universal_newlines=True)
    result = {
        "schema_version": 1,
        "scope": "Checkpoint helper integrity only; no source-physics or interpolation certification",
        "source": "lammps_slurry/tools/precompute_fast.cpp",
        "passed": True,
        "checks": ["binary roundtrip", "exact selected row preservation", "unselected rows untouched",
                   "wrong identity rejected", "wrong chunk rejected", "wrong row order rejected",
                   "wrong stride rejected", "payload corruption rejected", "truncation rejected", "trailing bytes rejected"],
        "result": run.stdout.strip(),
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
