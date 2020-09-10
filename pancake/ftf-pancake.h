#pragma once
#include "hash_table.h"
#include "Direction.h"
#include "Pancake.h"
#include <cstdint>
#include <unordered_map>
#include <set>
#include <tsl\hopscotch_map.h>
#include "mycuda.h"

class ftf_matchstructure;
class ftf_cudastructure;

class FTF_Pancake
{
public:
  Direction dir;
  #ifdef HISTORY
  std::vector<uint8_t> actions;
  const Pancake* parent = nullptr;
  #endif
  uint8_t source[NUM_PANCAKES + 1];                // source sequence of Pancakes
  uint8_t g;
  uint8_t h;
  uint8_t f;
  hash_t hash_values[NUM_PANCAKES + 1];
  uint64_t hash_64;

  FTF_Pancake() {}
  FTF_Pancake(const uint8_t* data, Direction dir) : dir(dir), g(0), h(0), f(0)
  {
    assert(NUM_PANCAKES > 0);
    memcpy(source, data, NUM_PANCAKES + 1);
    //h = gap_lb(dir);
    for(int i = 1; i < NUM_PANCAKES; ++i)
    {
      hash_values[i] = hash_table::hash(source[i], source[i + 1]);
    }
    hash_values[NUM_PANCAKES] = hash_table::hash(source[NUM_PANCAKES], NUM_PANCAKES + 1);
    std::sort(std::begin(hash_values) + 1, std::end(hash_values));
    hash_64 = hash_table::compress(hash_values);
  }

  FTF_Pancake(const FTF_Pancake& copy) : dir(copy.dir), g(copy.g), h(copy.h), f(copy.f), hash_64(copy.hash_64)
    #ifdef HISTORY
    , actions(copy.actions), parent(copy.parent)
    #endif
  {
    memcpy(source, copy.source, NUM_PANCAKES + 1);
    memcpy(hash_values, copy.hash_values, (NUM_PANCAKES + 1) * sizeof(hash_t));
  }

  inline bool operator==(const FTF_Pancake& right) const
  {
    return memcmp(source, right.source, NUM_PANCAKES + 1) == 0;
  }

  //Reverses the pancakes between 1 and i
  inline void apply_flip(int i)
  {
    assert(i >= 1 && i <= NUM_PANCAKES);
    std::reverse(source + 1, source + i + 1);
    //std::reverse(hash_values + 1, hash_values + i);
  }

  uint8_t gap_lb(Direction dir) const;
  uint8_t update_gap_lb(Direction dir, int i, uint8_t LB) const;
  //Copies pancake, applies a flip, and updates g/h/f values
  template<typename T>
  FTF_Pancake apply_action(const int i, bool match, T& structure) const;
};

  //Copies pancake, applies a flip, and updates g/h/f values
template<typename T>
__declspec(noinline) FTF_Pancake FTF_Pancake::apply_action(const int i, bool match, T& structure) const
{
  FTF_Pancake new_node(*this);
  #ifdef HISTORY
  new_node.actions.push_back(i);
  new_node.parent = this;
  #endif
  assert(i > 1 && i <= NUM_PANCAKES);
  new_node.apply_flip(i);
  //new_node.h = new_node.update_gap_lb(dir, i, new_node.h);
  /*if(i < NUM_PANCAKES)
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], new_node.source[i + 1]);
  else
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], NUM_PANCAKES + 1);*/
  for(int i = 1; i < NUM_PANCAKES; ++i)
  {
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], new_node.source[i + 1]);
  }
  new_node.hash_values[NUM_PANCAKES] = hash_table::hash(new_node.source[NUM_PANCAKES], NUM_PANCAKES + 1);
  std::sort(std::begin(new_node.hash_values) + 1, std::end(new_node.hash_values));
  new_node.hash_64 = hash_table::compress(new_node.hash_values);
  new_node.g = g + 1;
  if(match)
  {
    new_node.h = structure.match(&new_node);
    new_node.f = new_node.g + new_node.h;
    //assert(new_node.f >= f); //Consistency check
  }
  return new_node;
}


struct FTF_Less
{
  bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const;
};

class ftf_cudastructure
{
  constexpr inline static uint32_t MAX_VAL = NUM_PANCAKES * (NUM_PANCAKES + 1) / 2;
  struct hash_array
  {
    float hash[ftf_cudastructure::MAX_VAL];
  };

  float* hash_vals = nullptr;
  float* batch_hash_vals = nullptr;
  mycuda cuda;
  std::vector<hash_array> opposite_hash_values;
  std::vector<float> g_values;
  std::unordered_map<const FTF_Pancake*, size_t> index_map;
  std::unordered_map<size_t, const FTF_Pancake*> reverse_map;
  bool valid_device_cache = false;

  inline void to_hash_array(const FTF_Pancake* val, float* hash_array)
  {
    memset(hash_array, 0, MAX_VAL * sizeof(float));
    for(size_t i = 1; i <= NUM_PANCAKES; ++i)
    {
      hash_array[val->hash_values[i]] = 1;
    }
  }

public:
  ftf_cudastructure() {}
  ~ftf_cudastructure()
  {
    if(batch_hash_vals) cudaFreeHost(batch_hash_vals);
  }
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

    memcpy(opposite_hash_values[index].hash, opposite_hash_values.back().hash, sizeof(float) * MAX_VAL);
    opposite_hash_values.resize(opposite_hash_values.size() - 1);
    valid_device_cache = false;
  }

  uint32_t match(const FTF_Pancake* val); 
  void match(std::vector<FTF_Pancake*>& val);
};

class ftf_matchstructure
{
  constexpr inline static uint32_t MAX_VAL = NUM_PANCAKES * (NUM_PANCAKES + 1) / 2;
  #define SORTED_DATASET true

  #if SORTED_DATASET
  typedef std::set<const FTF_Pancake*, FTF_Less> dataset_t;
  #else
  typedef std::vector<const FTF_Pancake*> dataset_t;
  std::array<std::unordered_map<const FTF_Pancake*, size_t>, MAX_VAL> index_maps;
  #endif

  std::array<dataset_t, MAX_VAL> dataset;
public:
  void insert(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      #if SORTED_DATASET
      dataset[val->hash_values[i]].insert(val);
      #else
      index_maps[val->hash_values[i]][val] = dataset[val->hash_values[i]].size();
      dataset[val->hash_values[i]].push_back(val);
      #endif
    }
  }

  void erase(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      #if SORTED_DATASET
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
  }

  uint32_t match(const FTF_Pancake* val);
  void match(std::vector<FTF_Pancake*> values);
};

