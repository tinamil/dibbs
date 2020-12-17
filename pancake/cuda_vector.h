#pragma once
#include "cuda_helper.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

template<typename T>
class cuda_vector
{
  static constexpr double GROWTH_FACTOR = 2;
  T* d_ptr = nullptr;
  size_t size_val = 0;
  size_t capacity = 0;

  void resize(size_t new_capacity);


public:
  cudaStream_t stream;
  cuda_vector()
  {
    cudaStreamCreate(&stream);
  }
  ~cuda_vector()
  {
    clear();
    cudaStreamDestroy(stream);
  }
  T* push_back(const T& data);
  void insert(typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end);
  void insert(const T* begin, const T* end);
  void erase(size_t index);

  inline void clear()
  {
    if(d_ptr) CUDA_CHECK_RESULT(cudaFree(d_ptr));
    d_ptr = nullptr;
    capacity = 0;
    size_val = 0;
  }
  inline const T* begin()
  {
    return d_ptr;
  }
  inline const T* end()
  {
    return d_ptr + size_val;
  }

  inline size_t size() { return size_val; }
  inline T* data() { return d_ptr; }
};

template <typename T>
void cuda_vector<T>::resize(size_t new_capacity)
{
  if(new_capacity <= capacity)
  {
    throw std::exception("Tried to resize down, not supported");
  }

  T* tmp_ptr;
  CUDA_CHECK_RESULT(cudaMalloc(&tmp_ptr, new_capacity * sizeof(T)));

  if(d_ptr)
  {
    CUDA_CHECK_RESULT(cudaMemcpyAsync(tmp_ptr, d_ptr, size_val * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);
    CUDA_CHECK_RESULT(cudaFree(d_ptr));
  }

  capacity = new_capacity;
  d_ptr = tmp_ptr;
}

template <typename T>
T* cuda_vector<T>::push_back(const T& data)
{
  if(d_ptr == nullptr || size_val == capacity)
  {
    resize(std::max(64ui64, static_cast<uint64_t>(GROWTH_FACTOR * capacity)));
  }
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_ptr + size_val, &data, sizeof(T), cudaMemcpyHostToDevice, stream));
  size_val += 1;
  return d_ptr + size_val - 1;
}

template <typename T>
void cuda_vector<T>::insert(typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end)
{
  insert((T*)begin, (T*)end);
}

template <typename T>
void cuda_vector<T>::insert(const T* begin, const T* end)
{
  int size_diff = end - begin;
  if(size_diff == 0) return;
  if(d_ptr == nullptr || size_val + size_diff >= capacity)
  {
    resize(std::max(size_val + size_diff, static_cast<uint64_t>(GROWTH_FACTOR * capacity)));
  }
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_ptr + size_val, begin, sizeof(T) * size_diff, cudaMemcpyHostToDevice, stream));
  size_val += size_diff;
}


template <typename T>
void cuda_vector<T>::erase(size_t index)
{
  #ifndef NDEBUG
  if(index >= size_val || size_val == 0)
  {
    throw std::exception("Invalid ptr to erase");
  }
  #endif

  size_val -= 1;
  if(index != size_val)
  {
    CUDA_CHECK_RESULT(cudaMemcpyAsync(d_ptr + index, d_ptr + size_val, sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }
}