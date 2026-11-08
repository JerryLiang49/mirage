#pragma once
// The caller has validated 0 <= n <= 8256 and a nonnegative int32 window.
// Round the visible interval midpoint up to the next 64-token page boundary.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int gqa_adaptive_boundary(int n, int window) {
  int const visible_start = window > 0 && window < n ? n - window : 0;
  return ((visible_start + n + 127) / 128) * 64;
}
