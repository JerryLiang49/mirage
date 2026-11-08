// Experimental BF16 GQA merge body. Layouts: LSE[T,G,K,H],
// partial[T,G,K,H,D], output[T,G,H,D], optional natural-log sinks[G,H].
// Phase1 only: inputs are ready on entry. No producer publication contract yet.
#pragma once
#include "mirage/static_megakernel/core.cuh"
#include <math_constants.h>
namespace static_mk {
template <int G, int H, int D, int K, bool SINK>
__device__ __forceinline__ void
    gqa_merge_one_request_task(float const *lse,
                               __nv_bfloat16 const *partial,
                               __nv_bfloat16 *out,
                               __nv_bfloat16 const *sink,
                               int first,
                               int group,
                               int request) {
  static_assert(D % 16 == 0 && D <= 256 && H > 0 && K > 0);
  constexpr int V = D / 16;
  for (int head = threadIdx.x / 16; head < H; head += 16) {
    int t = request, h = head % H, lane = threadIdx.x % 16;
    float maximum = -CUDART_INF_F, denominator = 0.f, acc[V] = {};
    for (int k = first; k < K; ++k) {
      size_t index = ((size_t(t) * G + group) * K + k) * H + h;
      float other = lse[index];
      if (other == -CUDART_INF_F) {
        continue;
      }
      if (denominator == 0.f) {
        maximum = other;
        denominator = 1.f;
#pragma unroll
        for (int v = 0; v < V; ++v) {
          acc[v] = __bfloat162float(partial[index * D + lane * V + v]);
        }
      } else {
        float next = fmaxf(maximum, other);
        float previous_scale = exp2f(maximum - next),
              other_scale = exp2f(other - next);
        denominator = denominator * previous_scale + other_scale;
#pragma unroll
        for (int v = 0; v < V; ++v) {
          acc[v] =
              acc[v] * previous_scale +
              __bfloat162float(partial[index * D + lane * V + v]) * other_scale;
        }
        maximum = next;
      }
    }
    if constexpr (SINK) {
      if (denominator > 0.f) {
        float other =
            __bfloat162float(sink[group * H + h]) * 1.4426950408889634f;
        float next = fmaxf(maximum, other), scale = exp2f(maximum - next);
        denominator = denominator * scale + exp2f(other - next);
#pragma unroll
        for (int v = 0; v < V; ++v) {
          acc[v] *= scale;
        }
      }
    }
#pragma unroll
    for (int v = 0; v < V; ++v) {
      out[((size_t(t) * G + group) * H + h) * D + lane * V + v] =
          __float2bfloat16_rn(
              denominator > 0.f ? __fdividef(acc[v], denominator) : 0.f);
    }
  }
  __syncthreads();
}
} // namespace static_mk
