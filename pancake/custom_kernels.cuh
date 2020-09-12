#pragma once

void vector_add(int num_batch, int num_frontier, const float* __restrict__ g_vals, float* __restrict__ mult_results);
void reduce_min(int num_batch, int num_frontier, const float* __restrict__ mult_results, float* __restrict__ d_batch_answers);