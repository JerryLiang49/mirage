#pragma once
#ifdef STATIC_NO_START_BARRIER
#error "STATIC_NO_START_BARRIER is not validated for GQA"
#endif
#ifndef MIRAGE_GQA_SM100_REQUEST_PARALLEL
#error "partitioned GQA requires MIRAGE_GQA_SM100_REQUEST_PARALLEL"
#endif
#include "sm100_gqa/gqa_merge_ready_request.cuh"
