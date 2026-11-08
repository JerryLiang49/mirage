"""Compile the actual host/device helper and check its partition invariants."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_adaptive_boundary_host_invariants(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler required for the actual boundary helper")
    source = tmp_path / "boundary.cc"
    source.write_text(r'''
#include <algorithm>
#include <climits>
#include <cstdio>
#include "mirage/static_megakernel/tasks/sm100_gqa/gqa_adaptive_boundary.cuh"
static_assert(gqa_adaptive_boundary(0, 0) == 0);
static_assert(gqa_adaptive_boundary(127, 64) == 128);
static_assert(gqa_adaptive_boundary(129, 128) == 128);
static_assert(gqa_adaptive_boundary(4161, 0) == 2112);
static_assert(gqa_adaptive_boundary(8256, 0) == 4160);
static_assert(gqa_adaptive_boundary(8256, 1) == 8256);
int main() {
  unsigned cases = 0;
  for (int n = 0; n <= 8256; ++n) {
    for (int w : {0, 1, 2, 63, 64, 65, 127, 128, 129, 4160, 4161,
                  8255, 8256, INT_MAX, n, n + 1, std::max(0, n - 1)}) {
      int lo = w > 0 && w < n ? n - w : 0;
      int b = gqa_adaptive_boundary(n, w);
      // Independently find the first page boundary at/above the midpoint.
      int reference = 0;
      while (2 * reference < lo + n) reference += 64;
      int end0 = std::min(n, b), begin1 = std::max(lo, b);
      int covered = std::max(0, end0 - lo) + std::max(0, n - begin1);
      int writers = 0;
      if (n) {
        int tile = (n - 1) / 64;
        writers += end0 > lo && tile >= lo / 64 && tile < (end0 + 63) / 64;
        writers += n > begin1 && tile >= begin1 / 64 && tile < (n + 63) / 64;
      }
      if (b != reference || b < 0 || b > 8256 || b % 64 ||
          covered != n - lo || writers != (n > 0)) {
        std::fprintf(stderr, "bad n=%d w=%d b=%d writers=%d\n", n,w,b,writers);
        return 1;
      }
      ++cases;
    }
  }
  std::printf("%u valid boundary/coverage/append-owner cases\n", cases);
}
''')
    executable = tmp_path / "boundary"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([compiler, "-O2", "-std=c++17", "-I", str(root / "include"),
                    str(source), "-o", str(executable)], check=True, timeout=60)
    result = subprocess.run([str(executable)], check=True, capture_output=True,
                            text=True, timeout=30)
    assert result.stdout.strip() == "140369 valid boundary/coverage/append-owner cases"
