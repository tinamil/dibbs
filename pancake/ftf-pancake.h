#pragma once
#include "hash_table.h"
#include "Direction.h"
#include "Pancake.h"
#include <cstdint>
#include <unordered_map>
#include <set>
#include <tsl\hopscotch_map.h>

class ftf_matchstructure;


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
    std::reverse(hash_values + 1, hash_values + i);
  }

  uint8_t gap_lb(Direction dir) const;
  uint8_t update_gap_lb(Direction dir, int i, uint8_t LB) const;
  //Copies pancake, applies a flip, and updates g/h/f values
  FTF_Pancake apply_action(const int i, ftf_matchstructure& structure) const;
};


struct FTF_Less
{
  bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const;
};

class ftf_matchstructure
{
  constexpr inline static uint32_t MAX_VAL = (NUM_PANCAKES + 1) * (NUM_PANCAKES + 2) / 2;

  //std::array<std::set<const FTF_Pancake*, FTF_Less>, MAX_VAL> dataset;
  std::array<std::vector<const FTF_Pancake*>, MAX_VAL> dataset;
public:
  void insert(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      //dataset[val->hash_values[i]].insert(val);
      dataset[val->hash_values[i]].push_back(val);
    }
  }

  void erase(const FTF_Pancake* val)
  {
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      //dataset[val->hash_values[i]].erase(val);
      hash_t hash = val->hash_values[i];
      for(int j = 0; j < dataset[hash].size(); ++j)
      {
        if(memcmp(dataset[hash][j]->source, val->source, NUM_PANCAKES) == 0)
        {
          dataset[hash][j] = dataset[hash].back();
          dataset[hash].resize(dataset[hash].size() - 1);
        }
      }
    }
  }

  uint32_t match(const FTF_Pancake* val);
};

