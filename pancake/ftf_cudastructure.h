#pragma once
#include "ftf-pancake.h"

struct FTF_Less
{
  bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const;
};

struct hash_array
{
  float hash[MAX_PANCAKES];
};

class ftf_cudastructure
{
  static inline float* batch_hash_vals = nullptr;
  mycuda cuda;
  std::vector<hash_array> opposite_hash_values;
  std::vector<float> g_values;
  std::unordered_map<const FTF_Pancake*, size_t> index_map;
  std::unordered_map<size_t, const FTF_Pancake*> reverse_map;
  bool valid_device_cache = false;

  inline void to_hash_array(const FTF_Pancake* val, float* hash_array)
  {
    memset(hash_array, 0, MAX_PANCAKES * sizeof(float));
    for(size_t i = 1; i < NUM_PANCAKES; ++i)
    {
      hash_array[hash_table::hash(val->source[i], val->source[i + 1])] = 1;
    }
    hash_array[hash_table::hash(val->source[NUM_PANCAKES], NUM_PANCAKES + 1)] = 1;
  }

public:
  ftf_cudastructure() {}

  void insert(const FTF_Pancake* val)
  {
    index_map[val] = opposite_hash_values.size();
    reverse_map[opposite_hash_values.size()] = val;
    assert(opposite_hash_values.size() == g_values.size());
    g_values.push_back(val->g);
    opposite_hash_values.resize(opposite_hash_values.size() + 1);
    to_hash_array(val, opposite_hash_values.back().hash);
    valid_device_cache = false;
  }

  void erase(const FTF_Pancake* val)
  {
    size_t index = index_map[val];
    index_map.erase(val);

    const FTF_Pancake* back_ptr = reverse_map[g_values.size() - 1];
    reverse_map[index] = back_ptr;
    reverse_map.erase(g_values.size() - 1);
    index_map[back_ptr] = index;

    g_values[index] = g_values.back();
    g_values.resize(g_values.size() - 1);

    memcpy(opposite_hash_values[index].hash, opposite_hash_values.back().hash, sizeof(float) * MAX_PANCAKES);
    opposite_hash_values.resize(opposite_hash_values.size() - 1);
    valid_device_cache = false;
  }

  uint32_t match(const FTF_Pancake* val);
  void match(std::vector<FTF_Pancake*>& val);
};

bool FTF_Less::operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const
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

uint32_t ftf_cudastructure::match(const FTF_Pancake* val)
{
  //size_t m_rows, size_t n_cols, float alpha, float* A, float* x, float beta, float* y
  if(!valid_device_cache)
  {
    cuda.set_matrix(opposite_hash_values.size(), (float*)opposite_hash_values.data(), g_values.data());
    valid_device_cache = true;
  }
  if(batch_hash_vals == nullptr)
  {
    cudaError_t result = cudaHostAlloc(&batch_hash_vals, mycuda::MAX_BATCH * MAX_PANCAKES * sizeof(float), cudaHostAllocWriteCombined);
  }
  to_hash_array(val, batch_hash_vals);
  return round(*cuda.batch_vector_matrix(opposite_hash_values.size(), 1, batch_hash_vals));
}

void ftf_cudastructure::match(std::vector<FTF_Pancake*>& val)
{
  //size_t m_rows, size_t n_cols, float alpha, float* A, float* x, float beta, float* y
  if(!valid_device_cache)
  {
    cuda.set_matrix(opposite_hash_values.size(), (float*)opposite_hash_values.data(), g_values.data());
    valid_device_cache = true;
  }
  if(batch_hash_vals == nullptr)
  {
    cudaError_t result = cudaHostAlloc(&batch_hash_vals, mycuda::MAX_BATCH * MAX_PANCAKES * sizeof(float), cudaHostAllocWriteCombined);
  }
  assert(val.size() <= mycuda::MAX_BATCH);
  //size_t num_vals = std::min(val.size() - batch * mycuda::MAX_BATCH, mycuda::MAX_BATCH);
  for(size_t i = 0; i < val.size(); ++i)
  {
    to_hash_array(val[i], batch_hash_vals + i * MAX_PANCAKES);
  }
  float* answers = cuda.batch_vector_matrix(opposite_hash_values.size(), val.size(), batch_hash_vals);
  for(int i = 0; i < val.size(); ++i)
  {
    val[i]->h = round(answers[i]);
    val[i]->f = val[i]->g + val[i]->h;
  }
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
