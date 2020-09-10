#pragma once

__global__
void extract_min_kernel(int n, float* x, float* y);

void extract_min(int n, float* x, float* y);

template <class T>
void min_reduce(int size, int threads, int blocks, T* d_idata, T* d_odata);

__global__
void cuda_heuristic_kernel(int num_batch, int num_hash, int num_frontier, float* hash_matrix, float* frontier, float* g_vals, float* mult_results, float* d_batch_answers);

void cuda_heuristic(int num_batch, int num_hash, int num_frontier, float* hash_matrix, float* frontier, float* g_vals, float* mult_results, float* d_batch_answers);