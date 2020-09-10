#include "mycuda.h"
#include <vector>
#include <thrust/fill.h>
#include "Pancake.h"

mycuda::mycuda() : d_a(nullptr), d_hash_vals(nullptr), d_batch_hash_vals(nullptr),
d_mult_results(nullptr), d_g_vals(nullptr), num_pancakes_constant(nullptr), d_batch_answers(nullptr),
min_idx(nullptr), compare_answer(nullptr), batch_answers(nullptr)
{
  CUBLAS_CHECK_RESULT(cublasCreate(&handle));
  CUBLAS_CHECK_RESULT(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK_RESULT(cublasSetStream(handle, cudaStreamPerThread));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_idx, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&one, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&neg_one, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&zero, sizeof(float)));
  static constexpr float a = 1, b = -1, c = 0;
  CUDA_CHECK_RESULT(cudaMemcpy(one, &a, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMemcpy(neg_one, &b, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMemcpy(zero, &c, sizeof(float), cudaMemcpyHostToDevice));
}

mycuda::~mycuda()
{
  if(d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
  if(d_hash_vals) CUDA_CHECK_RESULT(cudaFree(d_hash_vals));
  if(d_batch_hash_vals) CUDA_CHECK_RESULT(cudaFree(d_batch_hash_vals));
  if(d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
  if(d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
  if(d_batch_answers)CUDA_CHECK_RESULT(cudaFree(d_batch_answers));
  if(num_pancakes_constant) CUDA_CHECK_RESULT(cudaFree(num_pancakes_constant));

  if(d_idx)CUDA_CHECK_RESULT(cudaFree(d_idx));
  if(one)CUDA_CHECK_RESULT(cudaFree(one));
  if(neg_one)CUDA_CHECK_RESULT(cudaFree(neg_one));
  if(zero)CUDA_CHECK_RESULT(cudaFree(zero));

  cublasDestroy(handle);

  if(min_idx) CUDA_CHECK_RESULT(cudaFreeHost(min_idx));
  if(compare_answer) CUDA_CHECK_RESULT(cudaFreeHost(compare_answer));
  if(batch_answers) CUDA_CHECK_RESULT(cudaFreeHost(batch_answers));
}

// A must be an m by n matrix
// x must be an n-vector
// y must be an m-vector

void mycuda::set_matrix(size_t num_pancakes, size_t num_hash, const float* A, const float* g_vals)
{
  if(min_idx == nullptr)
    CUDA_CHECK_RESULT(cudaHostAlloc(&min_idx, sizeof(int), cudaHostAllocDefault));
  if(compare_answer == nullptr)
    CUDA_CHECK_RESULT(cudaHostAlloc(&compare_answer, sizeof(float), cudaHostAllocDefault));
  if(batch_answers == nullptr)
    CUDA_CHECK_RESULT(cudaHostAlloc(&batch_answers, sizeof(float) * MAX_BATCH, cudaHostAllocDefault));

  if(!d_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_hash_vals, num_hash * sizeof(float)));
  if(!d_batch_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_hash_vals, MAX_BATCH * num_hash * sizeof(float)));
  if(!d_batch_answers) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_answers, MAX_BATCH * num_hash * sizeof(float)));

  if(d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_a, num_hash * num_pancakes * sizeof(*A)));
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_a, A, num_hash * num_pancakes * sizeof(*A), cudaMemcpyHostToDevice));

  if(d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_g_vals, num_pancakes * sizeof(float)));
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_g_vals, g_vals, num_pancakes * sizeof(*g_vals), cudaMemcpyHostToDevice));

  if(d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, MAX_BATCH * num_pancakes * sizeof(float)));

  if(num_pancakes_constant) CUDA_CHECK_RESULT(cudaFree(num_pancakes_constant));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&num_pancakes_constant, num_pancakes * sizeof(float)));

  float p = NUM_PANCAKES;
  for(int i = 0; i < num_pancakes; ++i)
  {
    CUDA_CHECK_RESULT(cudaMemcpy(num_pancakes_constant + i, &p, sizeof(float), cudaMemcpyHostToDevice));
  }
}

//TODO: Collect batches of up to 1000? nodes before performing copy 
__declspec(noinline) float mycuda::min_vector_matrix(size_t num_pancakes, size_t num_hash, const float* hash_vals)
{
  int min_idx;
  float compare_answer;

  CUDA_CHECK_RESULT(cudaMemcpy(d_hash_vals, hash_vals, sizeof(float) * num_hash, cudaMemcpyHostToDevice));

  CUBLAS_CHECK_RESULT(cublasSgemv(handle, CUBLAS_OP_T, num_hash, num_pancakes, neg_one, d_a, num_hash, d_hash_vals, 1, zero, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, d_g_vals, 1, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, num_pancakes_constant, 1, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasIsamin(handle, num_pancakes, d_mult_results, 1, d_idx));
  CUDA_CHECK_RESULT(cudaMemcpy(&min_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RESULT(cudaMemcpy(&compare_answer, d_mult_results + (min_idx - 1), sizeof(float), cudaMemcpyDeviceToHost));
  return compare_answer;
}

__declspec(noinline) float* mycuda::batch_vector_matrix(size_t num_pancakes, size_t num_hash, size_t num_vals, const float* hash_vals)
{
  assert(num_vals <= MAX_BATCH);

  CUDA_CHECK_RESULT(cudaMemcpy(d_batch_hash_vals, hash_vals, sizeof(float) * num_vals * num_hash, cudaMemcpyHostToDevice));

  cuda_heuristic(num_vals, num_hash, num_pancakes, d_batch_hash_vals, d_a, d_g_vals, d_mult_results, d_batch_answers);
  //for(int i = 0; i < num_vals; ++i)
  //{
  //  CUBLAS_CHECK_RESULT(cublasSgemv(handle, CUBLAS_OP_T, num_hash, num_pancakes, neg_one, d_a, num_hash, d_batch_hash_vals + i * num_hash, 1, zero, d_mult_results, 1));
  //  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, d_g_vals, 1, d_mult_results, 1));
  //  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, num_pancakes_constant, 1, d_mult_results, 1));
  //  //min_reduce(num_pancakes, 128, 256, d_mult_results, d_batch_answers + i);
  //  extract_min(num_pancakes, d_mult_results, d_batch_answers + i);
  //}

  CUDA_CHECK_RESULT(cudaMemcpy(batch_answers, d_batch_answers, sizeof(float) * num_vals, cudaMemcpyDeviceToHost));
  return batch_answers;
}

//A = m x k, B = k x n, C = m x n
// A * B + C
__declspec(noinline) float mycuda::matrix_matrix(size_t num_pancakes_A, size_t num_pancakes_B, size_t num_hash, const float* A, const float* B)
{
  float* d_a, * d_b, * d_c;

  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_a, num_hash * num_pancakes_A * sizeof(*A)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_a, A, num_hash * num_pancakes_A * sizeof(*A), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_b, num_hash * num_pancakes_B * sizeof(*B)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_b, B, num_hash * num_pancakes_B * sizeof(*B), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_c, num_pancakes_A * num_pancakes_B * sizeof(float)));

  CUBLAS_CHECK_RESULT(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_pancakes_A, num_pancakes_B, num_hash, one, d_a, num_pancakes_A, d_b, num_pancakes_B, zero, d_c, num_pancakes_A));
  return 0;
}