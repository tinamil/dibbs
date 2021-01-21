#pragma once
#include "ftf-pancake.h"
#include "cuda_vector.h"
#include "mycuda.h"
#include "hash_array.h"


template <typename PancakeType>
struct FTF_Less
{
  bool operator()(const PancakeType* lhs, const PancakeType* rhs) const;
};


template <typename PancakeType, typename Hash, typename Equal>
class ftf_cudastructure
{
public:
  #define CUDA_VECTOR
  #ifdef CUDA_VECTOR
  cuda_vector<hash_array> device_hash_values;
  cuda_vector<uint32_t> device_g_values;
  #else
  std::vector<hash_array> opposite_hash_values;
  std::vector<uint32_t> g_values;
  #endif
  std::unordered_map <const PancakeType*, size_t, Hash, Equal > index_map;
  std::vector<const PancakeType*> pancakes;
  bool valid_device_cache = false;

  uint32_t* tmp_g_vals = nullptr;
  hash_array* tmp_hash_arrays = nullptr;
  size_t capacity = 0;

  static inline void to_hash_array(const PancakeType* val, hash_array* p_hash_array)
  {
    memcpy(p_hash_array, &val->hash_ints, sizeof(hash_array));
  }

  void synchronize()
  {
    CUDA_CHECK_RESULT(cudaStreamSynchronize(device_g_values.stream));
    CUDA_CHECK_RESULT(cudaStreamSynchronize(device_hash_values.stream));
  }

public:
  ftf_cudastructure() {}

  size_t size() const
  {
    return pancakes.size();
  }

  void resize_tmp_arrays(size_t min_size)
  {
    if(min_size > capacity) {
      capacity = std::max(min_size, 2 * capacity);
      if(tmp_hash_arrays) CUDA_CHECK_RESULT(cudaFreeHost(tmp_hash_arrays));
      if(tmp_g_vals) CUDA_CHECK_RESULT(cudaFreeHost(tmp_g_vals));
      CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_g_vals, sizeof(uint32_t) * capacity, cudaHostAllocWriteCombined));
      CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_hash_arrays, sizeof(hash_array) * capacity, cudaHostAllocWriteCombined));
    }
  }

  void load_device()
  {
    if(valid_device_cache) return;

    resize_tmp_arrays(pancakes.size());

    for(int i = 0; i < pancakes.size(); ++i) {
      tmp_g_vals[i] = pancakes[i]->g;
      to_hash_array(pancakes[i], &tmp_hash_arrays[i]);
    }

    device_g_values.set_empty();
    device_g_values.insert(tmp_g_vals, tmp_g_vals + pancakes.size());
    device_hash_values.set_empty();
    device_hash_values.insert(tmp_hash_arrays, tmp_hash_arrays + pancakes.size());

    valid_device_cache = true;
  }

  void insert(const PancakeType* val)
  {
    assert(index_map.count(val) == 0);
    index_map[val] = device_hash_values.size();
    device_hash_values.size_val += 1;
    pancakes.push_back(val);
    valid_device_cache = false;
  }

  void insert(const std::vector<PancakeType*>& vector)
  {
    if(vector.size() == 0) return;

    for(int i = 0; i < vector.size(); ++i) {
      assert(index_map.count(vector[i]) == 0);
      index_map[vector[i]] = device_hash_values.size() + i;
    }

    device_hash_values.size_val += vector.size();
    pancakes.insert(pancakes.end(), vector.begin(), vector.end());
    valid_device_cache = false;
  }

  bool contains(const PancakeType* val)
  {
    return index_map.count(val) == 1;
  }

  void erase(const PancakeType* val)
  {
    assert(index_map.count(val) == 1);
    assert(index_map.size() == pancakes.size());
    if(val != pancakes.back()) {
      size_t index = index_map[val];
      index_map[pancakes.back()] = index;
      pancakes[index] = pancakes.back();
      valid_device_cache = false;
    }
    device_hash_values.size_val -= 1;
    pancakes.resize(pancakes.size() - 1);
    size_t num_erased = index_map.erase(val);
    assert(num_erased == 1);
    assert(index_map.size() == pancakes.size());
  }

  uint32_t match_one(const PancakeType* val);
  void match(mycuda& cuda, std::vector<PancakeType*>& val);
  void match_all(ftf_cudastructure<PancakeType, Hash, Equal>& other);
  //void match_all_batched(ftf_cudastructure<PancakeType>& other);
};

template <typename PancakeType>
bool FTF_Less<PancakeType>::operator()(const PancakeType* lhs, const PancakeType* rhs) const
{
  int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
  if(cmp == 0)
  {
    return false;
  }
  else if(lhs->g == rhs->g)
  {
    return cmp < 0;
  }
  else
  {
    return lhs->g < rhs->g;
  }
}

template <typename PancakeType, typename Hash, typename Equal>
uint32_t ftf_cudastructure<PancakeType, Hash, Equal>::match_one(const PancakeType* val)
{
  load_device();
  mycuda cuda;
  #ifndef CUDA_VECTOR
  cuda.set_matrix(opposite_hash_values.size(), (float*)opposite_hash_values.data(), g_values.data());
  #else
  assert(device_hash_values.size() == device_g_values.size());
  cuda.set_ptrs(device_hash_values.size(), 1, reinterpret_cast<uint32_t*>(device_hash_values.data()), device_g_values.data());
  #endif

  to_hash_array(val, cuda.h_hash_vals);
  cuda.load_then_batch_vector_matrix();
  cudaStreamSynchronize(cuda.stream);
  uint32_t* ptr = cuda.get_answers();
  return ptr[0];
}

//Must call get_answers() to retrieve the answers, because this is calculated asynchronously and get_answers() will wait until its done
template <typename PancakeType, typename Hash, typename Equal>
void ftf_cudastructure<PancakeType, Hash, Equal>::match(mycuda& cuda, std::vector<PancakeType*>& val)
{
  load_device();
  //if(!valid_device_cache)
  //{
  #ifndef CUDA_VECTOR
  cuda.set_matrix(opposite_hash_values.size(), (float*)opposite_hash_values.data(), g_values.data());
  #else
  assert(device_hash_values.size() == device_g_values.size());
  cuda.set_ptrs(device_hash_values.size(), val.size(), reinterpret_cast<uint32_t*>(device_hash_values.data()), device_g_values.data());
  #endif
  //valid_device_cache = true;
  //}
  //size_t num_vals = std::min(val.size() - batch * mycuda::MAX_BATCH, mycuda::MAX_BATCH);
  for(size_t i = 0; i < val.size(); ++i)
  {
    to_hash_array(val[i], cuda.h_hash_vals + i);
  }
  cuda.load_then_batch_vector_matrix();
}

//Must call get_answers() to retrieve the answers, because this is calculated asynchronously and get_answers() will wait until its done
template <typename PancakeType, typename Hash, typename Equal>
void ftf_cudastructure<PancakeType, Hash, Equal>::match_all(ftf_cudastructure<PancakeType, Hash, Equal>& other)
{
  load_device();
  other.load_device();
  mycuda cuda;
  size_t completed = 0;
  while(completed < other.pancakes.size()) {
    size_t to_do = std::min(BATCH_SIZE, other.pancakes.size() - completed);

    cuda.set_ptrs(device_hash_values.size(), to_do, reinterpret_cast<uint32_t*>(device_hash_values.data()), device_g_values.data());
    cuda.set_d_hash_vals((uint32_t*)(other.device_hash_values.data() + completed));
    other.synchronize();
    synchronize();
    cuda.batch_vector_matrix();
    uint32_t* answers = cuda.get_answers();
    cuda.clear_d_hash_vals();
    for(int i = 0; i < to_do; ++i) {
      FTF_Pancake* ptr = const_cast<FTF_Pancake*>(other.pancakes[i + completed]);
      ptr->ftf_h = answers[i];
      ptr->f = answers[i] + other.pancakes[i + completed]->g;
    }
    completed += to_do;
  }
}


#ifdef FTF_HASH
class ftf_matchstructure
{
  #define SORTED_DATASET true

  #if SORTED_DATASET
  typedef std::set<const FTF_Pancake*, FTF_Less<FTF_Pancake>> dataset_t;
  #else
  typedef std::vector<const FTF_Pancake*> dataset_t;
  std::array<std::unordered_map<const FTF_Pancake*, size_t>, MAX_VAL> index_maps;
  #endif

  std::array<dataset_t, MAX_PANCAKES> dataset;
public:
  void insert(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      #if SORTED_DATASET
      if(val->hash_values[i] >= GAP_COUNT)
        dataset[val->hash_values[i]].insert(val);
      #else
      index_maps[val->hash_values[i]][val] = dataset[val->hash_values[i]].size();
      dataset[val->hash_values[i]].push_back(val);
      #endif
    }
    for(int i = 0; i < GAP_COUNT; ++i) {
      dataset[i].insert(val);
    }
  }

  void erase(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      #if SORTED_DATASET
      if(val->hash_values[i] >= GAP_COUNT)
        dataset[val->hash_values[i]].erase(val);
      #else
      hash_t hash = val->hash_values[i];
      size_t index = index_maps[hash][val];
      assert(memcmp(dataset[hash][index]->source, val->source, NUM_PANCAKES) == 0);
      dataset[hash][index] = dataset[hash].back();
      dataset[hash].resize(dataset[hash].size() - 1);
      index_maps[hash].erase(val);
      #endif
    }
    for(int i = 0; i < GAP_COUNT; ++i) {
      dataset[i].erase(val);
    }
  }

  uint32_t match(const FTF_Pancake* val);
  void match(std::vector<FTF_Pancake*> values);
};



void ftf_matchstructure::match(std::vector<FTF_Pancake*> values)
{
  for(auto ptr : values)
  {
    ptr->ftf_h = match(ptr);
    ptr->f = ptr->g + ptr->ftf_h;
  }
}

uint32_t ftf_matchstructure::match(const FTF_Pancake* val)
{
  static size_t counter = 0;
  uint8_t match = NUM_PANCAKES;
  const FTF_Pancake* ptr = nullptr;

  std::array<dataset_t*, NUM_PANCAKES> set;
  for(size_t i = 1; i <= NUM_PANCAKES; ++i)
  {
    set[i - 1] = &dataset[val->hash_values[i]];
  }
  std::sort(set.begin(), set.end(), [](dataset_t* a, dataset_t* b) {
    return a->size() < b->size();
  });

  //The largest value of match possible is NUM_PANCAKES - i
  for(size_t i = 0; i < NUM_PANCAKES; ++i)
  {
    if(match <= i) break;
    for(auto begin = set[i]->begin(), end = set[i]->end(); begin != end; ++begin)
    {
      const FTF_Pancake* other_pancake = (*begin);
      if constexpr(SORTED_DATASET)
      {
        if(match <= i + other_pancake->g) break;
      }

      /*uint8_t pop_match = __popcnt64(val->hash_64 & other_pancake->hash_64);
      if(pop_match > NUM_PANCAKES - i)
      {
        continue;
      }*/

      uint8_t hash_index1 = 1;
      uint8_t hash_index2 = 1;

      uint8_t tmp_match_b = 0;
      while(other_pancake->hash_values[hash_index2] < GAP_COUNT) {
        hash_index2 += 1;
        tmp_match_b += 1;
      }
      while(hash_index1 <= NUM_PANCAKES && hash_index2 <= NUM_PANCAKES)
      {
        if(val->hash_values[hash_index1] < other_pancake->hash_values[hash_index2])
        {
          hash_index1 += 1;
        }
        else if(val->hash_values[hash_index1] > other_pancake->hash_values[hash_index2])
        {
          hash_index2 += 1;
        }
        else
        {
          tmp_match_b += 1;
          hash_index1 += 1;
          hash_index2 += 1;
        }
      }

      uint8_t tmp_match_f = 0;
      hash_index1 = 1;
      hash_index2 = 1;
      while(val->hash_values[hash_index1] < GAP_COUNT) {
        hash_index1 += 1;
        tmp_match_f += 1;
      }
      while(hash_index1 <= NUM_PANCAKES && hash_index2 <= NUM_PANCAKES)
      {
        if(val->hash_values[hash_index1] < other_pancake->hash_values[hash_index2])
        {
          hash_index1 += 1;
        }
        else if(val->hash_values[hash_index1] > other_pancake->hash_values[hash_index2])
        {
          hash_index2 += 1;
        }
        else
        {
          tmp_match_f += 1;
          hash_index1 += 1;
          hash_index2 += 1;
        }
      }

     /* uint8_t miss = 0;
      for(int hash_index = 1; hash_index <= NUM_PANCAKES; ++hash_index)
      {
        if(other_pancake->g + miss >= match)
        {
          break;
        }
        for(int other_hash_index = 1; other_hash_index <= NUM_PANCAKES; ++other_hash_index)
        {
          if(val->hash_values[hash_index] == other_pancake->hash_values[other_hash_index])
          {
            tmp_match += 1;
            break;
          }
          else if(other_hash_index == NUM_PANCAKES) miss += 1;
        }
      }*/
      //uint64_t combined = val->hash_64 & other_pancake->hash_64;
      //tmp_match = (uint8_t)__popcnt64(combined);
      counter += 1;
      tmp_match_f = NUM_PANCAKES + other_pancake->g - MIN(tmp_match_f, tmp_match_b);
      if(tmp_match_f < match)
      {
        ptr = other_pancake;
        match = tmp_match_f;
      }
    }
  }
  //std::cout << counter << "\n";
  return match;
}
#endif