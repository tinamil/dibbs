#pragma once
#include "cuda_helper.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<typename T>
class cuda_vector
{
  T* d_ptr = nullptr;
  size_t size_val = 0;
  size_t capacity = 0;

  void resize(size_t new_capacity);

public:
  T* push_back(const T& data);
  void erase(size_t index);

  inline void clear()
  {
    if(d_ptr) CUDA_CHECK_RESULT(cudaFree(d_ptr));
    d_ptr = nullptr;
    capacity = 0;
    size_val = 0;
  }

  inline size_t size() { return size_val; }
  inline T* data() { return d_ptr; }
};

template <typename T>
void cuda_vector<T>::resize(size_t new_capacity)
{
  if(new_capacity < capacity)
  {
    throw std::exception("Tried to resize down, not supported");
  }
  if(new_capacity == capacity) return;

  T* tmp_ptr;
  CUDA_CHECK_RESULT(cudaMalloc((void**)&tmp_ptr, new_capacity * sizeof(T)));
  capacity = new_capacity;

  if(d_ptr)
  {
    CUDA_CHECK_RESULT(cudaMemcpyAsync(tmp_ptr, d_ptr, size_val * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_RESULT(cudaFree(d_ptr));
  }

  d_ptr = tmp_ptr;
}

template <typename T>
T* cuda_vector<T>::push_back(const T& data)
{
  if(d_ptr == nullptr || size_val == capacity)
  {
    resize(std::max(64ui64, static_cast<uint64_t>(2 * capacity)));
  }
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_ptr + size_val, &data, sizeof(T), cudaMemcpyHostToDevice));
  size_val += 1;
  return d_ptr + size_val - 1;
}

template <typename T>
void cuda_vector<T>::erase(size_t index)
{
  return;
  #ifndef NDEBUG
  if(index < 0 || index >= size_val)
  {
    throw std::exception("Invalid ptr to erase");
  }
  #endif

  size_val -= 1;
  if(index != size_val)
  {
    CUDA_CHECK_RESULT(cudaMemcpyAsync(d_ptr + index, d_ptr + size_val, sizeof(T), cudaMemcpyDeviceToDevice));
  }
}