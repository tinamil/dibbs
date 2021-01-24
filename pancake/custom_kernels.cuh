#pragma once
#include <cstdint>

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint8_t* __restrict__ mult_results, uint8_t* __restrict__ d_batch_answers);
void bitwise_set_intersection(cudaStream_t stream, 
                              int rows_a, 
                              int rows_b, 
                              const uint32_t* __restrict__ hash_a, 
                              const uint8_t* __restrict__ g_vals, 
                              const uint32_t* __restrict__ hash_b, 
                              uint8_t* __restrict__ mult_results);