#pragma once
#ifdef STATIC_NO_START_BARRIER
#error "STATIC_NO_START_BARRIER is not validated for GQA"
#endif
#include "merge_one_request.cuh"
namespace static_mk {
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class PRODUCER,
          class... IN>
__device__ __forceinline__ void run_gqa_merge_ready(Maps const &,
                                                    G const &g,
                                                    KernelLocals &,
                                                    StaticTask const &tk) {
  constexpr int NG = PARAMS::v[0], H = PARAMS::v[1], D = PARAMS::v[2],
                K = PARAMS::v[3];
  constexpr bool SINK = PARAMS::v[4] != 0;
  static_assert(PRODUCER::counter >= 0 && PRODUCER::y == K &&
                PRODUCER::buf >= 0 && SELF::buf >= 0 && SELF::counter >= 0);
  if (threadIdx.x == 0) {
    cnt_wait(g.cnt + PRODUCER::counter + tk.x, PRODUCER::y * PRODUCER::z);
  }
  __syncthreads();
  gqa_merge_one_request_task<NG, H, D, K, SINK>(
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, PRODUCER::buf),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      SINK ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()) : nullptr,
      0,
      tk.x,
      tk.z);
  asm volatile("fence.proxy.async;" ::: "memory");
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter, 1);
  }
  __syncthreads();
}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gqa_merge_ready() {
  return 0;
}
} // namespace static_mk
