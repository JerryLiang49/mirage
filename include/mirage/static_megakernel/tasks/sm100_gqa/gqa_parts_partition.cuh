#pragma once
#ifdef STATIC_NO_START_BARRIER
#error "STATIC_NO_START_BARRIER is not validated for GQA"
#endif
#include "attention_separate_partition.cuh"
#include "gqa_adaptive_boundary.cuh"
#include "mirage/static_megakernel/core.cuh"
#include "tasks/blackwell/task_header.cuh"
namespace static_mk {
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_gqa_paged_parts(Maps const &,
                                                    G const &g,
                                                    KernelLocals &,
                                                    StaticTask const &tk) {
  constexpr int NG = PARAMS::v[0], H = PARAMS::v[1], D = PARAMS::v[2],
                K = PARAMS::v[3], C = PARAMS::v[4], P = PARAMS::v[5],
                NP = PARAMS::v[6];
  static_assert(H == 8 && D == 128 && C >= 64 && C % 64 == 0 && P == 64 &&
                K == (NP * P + C - 1) / C && NP == 129 && SELF::z == T);
  using BF = type::bfloat16_t;
  auto query = buf_at<BF const>(g, slot_at<BUF_SLOTS, 1>());
  auto keys = buf_at<BF>(g, slot_at<BUF_SLOTS, 2>());
  auto values = buf_at<BF>(g, slot_at<BUF_SLOTS, 3>());
  auto pages = buf_at<int const>(g, slot_at<BUF_SLOTS, 4>());
  auto lengths = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
  auto windows = buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
  auto nk = buf_at<BF const>(g, slot_at<BUF_SLOTS, 7>());
  auto nv = buf_at<BF const>(g, slot_at<BUF_SLOTS, 8>());
  auto part = buf_at<BF>(g, SELF::buf);
  auto lse = buf_at<float>(g, slot_at<BUF_SLOTS, 0>());
  int group = tk.x, chunk = tk.y;
  {
    int t = tk.z;
    int begin = chunk * C;
    int end = (chunk + 1) * C;
    if constexpr (K == 2 && C == 4160) {
      // Keep partition boundaries page aligned so one producer owns the append.
      int const boundary = gqa_adaptive_boundary(lengths[t], windows[t]);
      begin = chunk == 0 ? 0 : boundary;
      end = chunk == 0 ? boundary : NP * P;
    }
    kernel::gqa_separate_partition_sm100_task_impl<BF,
                                                   H,
                                                   1,
                                                   NG * D,
                                                   NG * H * D,
                                                   NG * K * H * D,
                                                   D,
                                                   NP * P,
                                                   P,
                                                   0,
                                                   0,
                                                   1,
                                                   D,
                                                   0,
                                                   0>(
        query + group * H * D,
        keys + group * D,
        values + group * D,
        part + (group * K + chunk) * H * D,
        lengths,
        pages,
        t,
        false,
        false,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.f,
        0.f,
        nullptr,
        lse + (group * K + chunk) * H,
        NG * K * H,
        begin,
        end,
        windows[t],
        nk + group * D,
        nv + group * D);
    __syncthreads(); // join the inactive upper128 threads before reusing smem
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter + group, 1);
  }
  __syncthreads();
}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gqa_paged_parts() {
  constexpr int H = PARAMS::v[1], D = PARAMS::v[2];
  // Same native MAX_TOKENS1 layout: zero, Q, four KV tiles, two norm
  // arrays, per-warp max/denominator arrays, and output fragments.
  constexpr int end_kv = 16 + 2 * H * D + 4 * 2 * 64 * D;
  return ((end_kv + 3) / 4) * 4 + 2 * 4 * 4 + 2 * 4 * 128 * 2 + 4 * 128 * 64;
}
} // namespace static_mk
