#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cassert>
#include <vector>
#include "custom_kernels.cuh"
#include "cuda_helper.h"

class mycuda
{
  size_t my_num_nodes = 0;
  size_t max_other_nodes = 0;
  size_t d_mult_results_size = 0;
  uint32_t* d_nodes = nullptr;
  uint32_t* d_mult_results = nullptr;
  uint32_t* d_answers = nullptr;
  uint32_t* h_answers = nullptr;
  uint32_t* d_a = nullptr;
  uint32_t* d_g_vals = nullptr;
  static inline fCoordinate* d_coordinates = nullptr;

public:
  cudaStream_t stream = nullptr;
  size_t other_num_nodes = 0;
  uint32_t* h_nodes = nullptr;

  mycuda()
  {
    std::cout << "Allocating CUDA variables and stream\n";
    initialize();
    CUDA_CHECK_RESULT(cudaStreamCreate(&stream));
  }

  ~mycuda()
  {
    cudaDeviceSynchronize();
    //Do NOT free d_a or d_g_vals, they are from cuda_vectors that cleanup during destruction
    if(stream) { cudaStreamDestroy(stream); stream = nullptr; }
    if(h_answers) { cudaFreeHost(h_answers); h_answers = nullptr; }
    if(d_answers) { cudaFree(d_answers); d_answers = nullptr; }
    if(h_nodes) { cudaFreeHost(h_nodes); h_nodes = nullptr; }
    if(d_nodes) { cudaFree(d_nodes); d_nodes = nullptr; }
    if(d_mult_results) { cudaFree(d_mult_results); d_mult_results = nullptr; }
  }

  void set_d_hash_vals(uint32_t* d_vals)
  {
    if(d_nodes) {
      cudaFree(d_nodes);
    }
    d_nodes = d_vals;
  }

  void clear_d_hash_vals()
  {
    d_nodes = nullptr;
  }

  uint32_t* get_answers()
  {
    cudaStreamSynchronize(stream);
    return h_answers;
  }

  static void initialize()
  {

  }
  void set_ptrs(size_t m_rows, size_t n_cols, uint32_t* A, uint32_t* g_vals);

  void batch_vector_matrix();
  void load_then_batch_vector_matrix();

  static inline void LoadCoordinates(const std::vector<Coordinate>& coordinates)
  {
    cudaDeviceSynchronize();
    std::vector<fCoordinate> fcoordinates;
    for(const auto& c : coordinates) {
      fcoordinates.push_back(fCoordinate{static_cast<float>(c.lng), static_cast<float>(c.lat)});
    }
    if(d_coordinates) { cudaFree(d_coordinates); d_coordinates = nullptr; }
    CUDA_CHECK_RESULT(cudaMalloc(&d_coordinates, fcoordinates.size() * sizeof(fCoordinate)));
    cudaMemcpy(d_coordinates, fcoordinates.data(), fcoordinates.size() * sizeof(fCoordinate), cudaMemcpyHostToDevice);
  }
};

