#pragma once
#include "ftf_node.h"
#include "cuda_vector.h"
#include "mycuda.h"
#include <cstdint>
#include <vector>


template <typename Type, typename Hash, typename Equal>
class ftf_cudastructure
{
public:
  cuda_vector<uint32_t> device_nodes;
  cuda_vector<uint32_t> device_g_values;
  std::unordered_map <const Type*, size_t, Hash, Equal> index_map;
  std::vector<const Type*> nodes;
  bool valid_device_cache = false;

  uint32_t* tmp_g_vals = nullptr;
  uint32_t* tmp_nodes = nullptr;
  size_t capacity = 0;

  void synchronize()
  {
    CUDA_CHECK_RESULT(cudaStreamSynchronize(device_g_values.stream));
    CUDA_CHECK_RESULT(cudaStreamSynchronize(device_nodes.stream));
  }

public:
  ftf_cudastructure() {}
  ~ftf_cudastructure()
  {
    device_g_values.clear();
    device_nodes.clear();
    //No CUDA_CHECK_RESULT because can't throw exceptions during destructor
    if(tmp_nodes) cudaFreeHost(tmp_nodes);
    if(tmp_g_vals) cudaFreeHost(tmp_g_vals);
  }

  size_t size() const
  {
    return nodes.size();
  }

  void resize_tmp_arrays(size_t min_size)
  {
    if(min_size > capacity) {
      capacity = std::max(min_size, 2 * capacity);
      if(tmp_nodes) CUDA_CHECK_RESULT(cudaFreeHost(tmp_nodes));
      if(tmp_g_vals) CUDA_CHECK_RESULT(cudaFreeHost(tmp_g_vals));
      CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_g_vals, sizeof(uint32_t) * capacity, cudaHostAllocWriteCombined));
      CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_nodes, sizeof(uint32_t) * capacity, cudaHostAllocWriteCombined));
    }
  }

  void load_device()
  {
    if(valid_device_cache) return;

    resize_tmp_arrays(nodes.size());

    for(int i = 0; i < nodes.size(); ++i) {
      tmp_g_vals[i] = nodes[i]->g;
      tmp_nodes[i] = nodes[i]->vertex_index;
    }

    device_g_values.set_empty();
    device_g_values.insert(tmp_g_vals, tmp_g_vals + nodes.size());
    device_nodes.set_empty();
    device_nodes.insert(tmp_nodes, tmp_nodes + nodes.size());
    assert(device_g_values.size() == device_nodes.size());
    valid_device_cache = true;
  }

  void insert(const Type* val)
  {
    assert(index_map.count(val) == 0);
    index_map[val] = device_nodes.size();
    device_nodes.size_val += 1;
    nodes.push_back(val);
    valid_device_cache = false;
  }

  void insert(const std::vector<Type*>& vector)
  {
    if(vector.size() == 0) return;

    for(int i = 0; i < vector.size(); ++i) {
      assert(index_map.count(vector[i]) == 0);
      index_map[vector[i]] = device_nodes.size() + i;
    }

    device_nodes.size_val += vector.size();
    nodes.insert(nodes.end(), vector.begin(), vector.end());
    valid_device_cache = false;
  }

  bool contains(const Type* val)
  {
    return index_map.count(val) == 1;
  }

  void erase(const Type* val)
  {
    assert(index_map.count(val) == 1);
    assert(index_map.size() == nodes.size());
    if(val != nodes.back()) {
      size_t index = index_map[val];
      index_map[nodes.back()] = index;
      nodes[index] = nodes.back();
    }
    device_nodes.size_val -= 1;
    nodes.resize(nodes.size() - 1);
    valid_device_cache = false;
    size_t num_erased = index_map.erase(val);
    assert(num_erased == 1);
    assert(index_map.size() == nodes.size());
  }

  uint32_t match_one(mycuda& cuda, const Type* val);
  void match(mycuda& cuda, std::vector<Type*>& val);
};


template <typename Type, typename Hash, typename Equal>
uint32_t ftf_cudastructure<Type, Hash, Equal>::match_one(mycuda& cuda, const Type* val)
{
  load_device();
  assert(device_nodes.size() == device_g_values.size());
  cuda.set_ptrs(device_nodes.size(), 1, device_nodes.data(), device_g_values.data());
  cuda.h_nodes[0] = val->vertex_index;
  cuda.load_then_batch_vector_matrix();
  cudaStreamSynchronize(cuda.stream);
  uint32_t* ptr = cuda.get_answers();
  return ptr[0];
}

//Must call get_answers() to retrieve the answers, because this is calculated asynchronously and get_answers() will wait until its done
template <typename Type, typename Hash, typename Equal>
void ftf_cudastructure<Type, Hash, Equal>::match(mycuda& cuda, std::vector<Type*>& val)
{
  load_device();
  assert(device_nodes.size() == device_g_values.size());
  cuda.set_ptrs(device_nodes.size(), val.size(), device_nodes.data(), device_g_values.data());
  for(size_t i = 0; i < val.size(); ++i)
  {
    cuda.h_nodes[i] = val[i]->vertex_index;
  }
  cuda.load_then_batch_vector_matrix();
}