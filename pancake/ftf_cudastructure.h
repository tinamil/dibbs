#pragma once
#include "ftf-pancake.h"
#include "cuda_vector.h"
#include "mycuda.h"

struct FTF_Less
{
  bool operator()(const FTF_Pancake *lhs, const FTF_Pancake *rhs) const;
};

struct hash_array
{
  float hash[MAX_PANCAKES];
};

class ftf_cudastructure
{
  #define CUDA_VECTOR
  #ifdef CUDA_VECTOR
  cuda_vector<hash_array> opposite_hash_values;
  cuda_vector<float> g_values;
  #else
  std::vector<hash_array> opposite_hash_values;
  std::vector<float> g_values;
  #endif
  std::unordered_map<const FTF_Pancake *, size_t> index_map;
  std::unordered_map<size_t, const FTF_Pancake *> reverse_map;
  //bool valid_device_cache = false;


  inline void to_hash_array(const FTF_Pancake *val, float *hash_array)
  {
    memset(hash_array, 0, MAX_PANCAKES * sizeof(float));
    for(size_t i = 1; i < NUM_PANCAKES; ++i)
    {
      hash_array[hash_table::hash(val->source[i], val->source[i + 1])] = 1;
    }
    hash_array[hash_table::hash(val->source[NUM_PANCAKES], NUM_PANCAKES + 1)] = 1;
  }

  inline void to_hash_array(const FTF_Pancake *val, hash_array *hash_array)
  {
    to_hash_array(val, hash_array->hash);
  }

public:
  ftf_cudastructure() {}

  void insert(const FTF_Pancake *val)
  {
    index_map[val] = opposite_hash_values.size();
    reverse_map[opposite_hash_values.size()] = val;
    assert(opposite_hash_values.size() == g_values.size());

    g_values.push_back(val->g);

    #ifndef CUDA_VECTOR
    opposite_hash_values.resize(opposite_hash_values.size() + 1);
    to_hash_array(val, opposite_hash_values.back().hash);
    #else
    hash_array tmp;
    to_hash_array(val, tmp.hash);
    opposite_hash_values.push_back(tmp);
    #endif

    //valid_device_cache = false;
  }

  void insert(const std::vector<FTF_Pancake *> &vector)
  {
    if(vector.size() == 0) return;
    float *tmp_g_vals;
    hash_array *tmp_hash_arrays;
    CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_g_vals, sizeof(float) * vector.size(), cudaHostAllocDefault));
    CUDA_CHECK_RESULT(cudaHostAlloc(&tmp_hash_arrays, sizeof(hash_array) * vector.size(), cudaHostAllocDefault));
    //std::vector<float> tmp_g_vals;
    //std::vector<hash_array> tmp_hash_arrays;
    //tmp_hash_arrays.resize(vector.size());
    //tmp_g_vals.reserve(vector.size());
    for(int i = 0; i < vector.size(); ++i) {
      index_map[vector[i]] = opposite_hash_values.size() + i;
      reverse_map[opposite_hash_values.size() + i] = vector[i];
      tmp_g_vals[i] = vector[i]->g;
      //tmp_g_vals.push_back(vector[i]->g);
      to_hash_array(vector[i], &tmp_hash_arrays[i]);
    }

    g_values.insert(tmp_g_vals, tmp_g_vals + vector.size());
    opposite_hash_values.insert(tmp_hash_arrays, tmp_hash_arrays + vector.size());
    cudaStreamSynchronize(g_values.stream);
    CUDA_CHECK_RESULT(cudaFreeHost(tmp_g_vals));
    cudaStreamSynchronize(opposite_hash_values.stream);
    CUDA_CHECK_RESULT(cudaFreeHost(tmp_hash_arrays));

    //valid_device_cache = false;
  }

  __declspec(noinline) void erase(const FTF_Pancake *val)
  {
    size_t index = index_map[val];
    index_map.erase(val);

    const FTF_Pancake *back_ptr = reverse_map[g_values.size() - 1];
    reverse_map[index] = back_ptr;
    reverse_map.erase(g_values.size() - 1);
    if(back_ptr != val)
    {
      index_map[back_ptr] = index;
    }
    #ifndef CUDA_VECTOR
    g_values[index] = g_values.back();
    g_values.resize(g_values.size() - 1);
    memcpy(opposite_hash_values[index].hash, opposite_hash_values.back().hash, sizeof(float) * MAX_PANCAKES);
    opposite_hash_values.resize(opposite_hash_values.size() - 1);
    #else
    g_values.erase(index);
    opposite_hash_values.erase(index);
    #endif

    //valid_device_cache = false;
  }

  uint32_t match_one(const FTF_Pancake *val);
  void match(mycuda &cuda, std::vector<FTF_Pancake *> &val);
};

bool FTF_Less::operator()(const FTF_Pancake *lhs, const FTF_Pancake *rhs) const
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

uint32_t ftf_cudastructure::match_one(const FTF_Pancake *val)
{
  mycuda cuda;

  #ifndef CUDA_VECTOR
  cuda.set_matrix(opposite_hash_values.size(), (float *)opposite_hash_values.data(), g_values.data());
  #else
  cuda.set_ptrs(opposite_hash_values.size(), reinterpret_cast<float *>(opposite_hash_values.data()), g_values.data());
  #endif

  to_hash_array(val, cuda.h_hash_vals);
  cuda.batch_vector_matrix();
  return round(cuda.get_answers()[0]);
}

//Must call get_answers() to retrieve the answers, because this is calculated asynchronously and get_answers() will wait until its done
void ftf_cudastructure::match(mycuda &cuda, std::vector<FTF_Pancake *> &val)
{
  //if(!valid_device_cache)
  //{
  #ifndef CUDA_VECTOR
  cuda.set_matrix(opposite_hash_values.size(), (float *)opposite_hash_values.data(), g_values.data());
  #else
  cuda.set_ptrs(opposite_hash_values.size(), reinterpret_cast<float *>(opposite_hash_values.data()), g_values.data());
  #endif
  //valid_device_cache = true;
  //}
  assert(val.size() <= mycuda::MAX_BATCH);
  //size_t num_vals = std::min(val.size() - batch * mycuda::MAX_BATCH, mycuda::MAX_BATCH);
  for(size_t i = 0; i < val.size(); ++i)
  {
    to_hash_array(val[i], cuda.h_hash_vals + i * MAX_PANCAKES);
  }
  cuda.num_vals = val.size();
  cuda.batch_vector_matrix();
}

//
//class ftf_matchstructure
//{
//  #define SORTED_DATASET true
//
//  #if SORTED_DATASET
//  typedef std::set<const FTF_Pancake*, FTF_Less> dataset_t;
//  #else
//  typedef std::vector<const FTF_Pancake*> dataset_t;
//  std::array<std::unordered_map<const FTF_Pancake*, size_t>, MAX_VAL> index_maps;
//  #endif
//
//  std::array<dataset_t, MAX_PANCAKES> dataset;
//public:
//  void insert(const FTF_Pancake* val)
//  {
//    for(int i = 1; i <= NUM_PANCAKES; ++i)
//    {
//      #if SORTED_DATASET
//      dataset[val->hash_values[i]].insert(val);
//      #else
//      index_maps[val->hash_values[i]][val] = dataset[val->hash_values[i]].size();
//      dataset[val->hash_values[i]].push_back(val);
//      #endif
//    }
//  }
//
//  void erase(const FTF_Pancake* val)
//  {
//    for(int i = 1; i <= NUM_PANCAKES; ++i)
//    {
//      #if SORTED_DATASET
//      dataset[val->hash_values[i]].erase(val);
//      #else
//      hash_t hash = val->hash_values[i];
//      size_t index = index_maps[hash][val];
//      assert(memcmp(dataset[hash][index]->source, val->source, NUM_PANCAKES) == 0);
//      dataset[hash][index] = dataset[hash].back();
//      dataset[hash].resize(dataset[hash].size() - 1);
//      index_maps[hash].erase(val);
//      #endif
//    }
//  }
//
//  uint32_t match(const FTF_Pancake* val);
//  void match(std::vector<FTF_Pancake*> values);
//};
//

//
//void ftf_matchstructure::match(std::vector<FTF_Pancake*> values)
//{
//  for(auto ptr : values)
//  {
//    ptr->h = match(ptr);
//    ptr->f = ptr->g + ptr->h;
//  }
//}
//
//uint32_t ftf_matchstructure::match(const FTF_Pancake* val)
//{
//  static size_t counter = 0;
//  uint8_t match = NUM_PANCAKES;
//  const FTF_Pancake* ptr = nullptr;
//
//  std::array<dataset_t*, NUM_PANCAKES> set;
//  for(size_t i = 1; i <= NUM_PANCAKES; ++i)
//  {
//    set[i - 1] = &dataset[val->hash_values[i]];
//  }
//  std::sort(set.begin(), set.end(), [](dataset_t* a, dataset_t* b) {
//    return a->size() < b->size();
//  });
//
//  //The largest value of match possible is NUM_PANCAKES - i
//  for(size_t i = 0; i < NUM_PANCAKES; ++i)
//  {
//    if(match <= i) break;
//    for(auto begin = set[i]->begin(), end = set[i]->end(); begin != end; ++begin)
//    {
//      const FTF_Pancake* other_pancake = (*begin);
//      uint8_t tmp_match;
//      if constexpr(SORTED_DATASET)
//      {
//        if(match <= i + other_pancake->g) break;
//      }
//
//      /*uint8_t pop_match = __popcnt64(val->hash_64 & other_pancake->hash_64);
//      if(pop_match > NUM_PANCAKES - i)
//      {
//        continue;
//      }*/
//
//      uint8_t hash_index1 = 1;
//      uint8_t hash_index2 = 1;
//
//      tmp_match = 0;
//      while(hash_index1 <= NUM_PANCAKES && hash_index2 <= NUM_PANCAKES)
//      {
//        if(val->hash_values[hash_index1] < other_pancake->hash_values[hash_index2])
//        {
//          hash_index1 += 1;
//        }
//        else if(val->hash_values[hash_index1] > other_pancake->hash_values[hash_index2])
//        {
//          hash_index2 += 1;
//        }
//        else
//        {
//          tmp_match += 1;
//          hash_index1 += 1;
//          hash_index2 += 1;
//        }
//      }
//
//     /* uint8_t miss = 0;
//      for(int hash_index = 1; hash_index <= NUM_PANCAKES; ++hash_index)
//      {
//        if(other_pancake->g + miss >= match)
//        {
//          break;
//        }
//        for(int other_hash_index = 1; other_hash_index <= NUM_PANCAKES; ++other_hash_index)
//        {
//          if(val->hash_values[hash_index] == other_pancake->hash_values[other_hash_index])
//          {
//            tmp_match += 1;
//            break;
//          }
//          else if(other_hash_index == NUM_PANCAKES) miss += 1;
//        }
//      }*/
//      //uint64_t combined = val->hash_64 & other_pancake->hash_64;
//      //tmp_match = (uint8_t)__popcnt64(combined);
//      counter += 1;
//      tmp_match = NUM_PANCAKES - tmp_match + other_pancake->g;
//      if(tmp_match < match)
//      {
//        ptr = other_pancake;
//        match = tmp_match;
//      }
//    }
//  }
//  //std::cout << counter << "\n";
//  return match;
//}
