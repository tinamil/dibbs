#pragma once
#include <cstdint>
#include "Constants.h"

//void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint8_t* __restrict__ mult_results, uint8_t* __restrict__ d_batch_answers);
void bitwise_set_intersection(cudaStream_t stream,
                              int rows_a,
                              int rows_b,
                              const uint32_t* __restrict__ hash_a,
                              const uint32_t* __restrict__ g_vals,
                              const uint32_t* __restrict__ hash_b,
                              uint32_t* __restrict__ mult_results,
                              uint32_t* __restrict__ answers,
                              const Coordinate* __restrict__ coordinates);

void empty(cudaStream_t stream, int rows_a, int rows_b);
//void transpose_cuda(cudaStream_t stream,
//                    const int rows,
//                    const int cols,
//                    const uint32_t* __restrict__ input,
//                    uint32_t* __restrict__ output);