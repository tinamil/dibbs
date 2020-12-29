#pragma once
#include "hash_table.h"
#include "Direction.h"
#include "Pancake.h"
#include <cstdint>
#include <unordered_map>
#include <set>
#include <tsl\hopscotch_map.h>
#include "hash_array.h"

static constexpr size_t BATCH_SIZE = 1024;

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
  uint8_t h2;
  uint8_t ftf_h;
  uint8_t f;
  uint8_t f_bar;
  #ifdef FTF_HASH
  hash_t hash_values[NUM_PANCAKES + 1];
  uint64_t hash_64;
  #endif
  hash_array hash_ints;

  void load_hash_ints()
  {
    hash_ints.clear_hash();
    for(int i = 1; i < NUM_PANCAKES; ++i)
    {
      if(source[i] > GAPX && source[i + 1] > GAPX) {
        hash_ints.set_hash(hash_table::hash(source[i], source[i + 1]));
      }
    }
    if(source[NUM_PANCAKES] > GAPX) {
      hash_ints.set_hash(hash_table::hash(source[NUM_PANCAKES], NUM_PANCAKES + 1));
    }
  }

  FTF_Pancake() : dir(Direction::forward), g(0), h(0), h2(0), ftf_h(0), f(0), f_bar(0)
    #ifdef FTF_HASH  
    , hash_64(0)
    #endif 
  {}
  FTF_Pancake(const uint8_t* data, Direction dir) : dir(dir), g(0), h2(0), ftf_h(0), f(0)
  {
    assert(NUM_PANCAKES > 0);
    memcpy(source, data, NUM_PANCAKES + 1);
    h = gap_lb(dir);
    f_bar = h;
    load_hash_ints();
    #ifdef FTF_HASH
    for(int i = 1; i < NUM_PANCAKES; ++i)
    {
      hash_values[i] = hash_table::hash(source[i], source[i + 1]);
    }
    hash_values[NUM_PANCAKES] = hash_table::hash(source[NUM_PANCAKES], NUM_PANCAKES + 1);
    std::sort(std::begin(hash_values) + 1, std::end(hash_values));
    hash_64 = hash_table::compress(hash_values);
    #endif
  }

  FTF_Pancake(const FTF_Pancake& copy) : dir(copy.dir), g(copy.g), ftf_h(copy.ftf_h), h(copy.h), h2(copy.h2), f(copy.f), f_bar(copy.f_bar) /*,hash_64(copy.hash_64)*/
    #ifdef HISTORY
    , actions(copy.actions), parent(copy.parent)
    #endif
  {
    memcpy(source, copy.source, NUM_PANCAKES + 1);
    memcpy(&hash_ints, &copy.hash_ints, sizeof(hash_array));
    #ifdef FTF_HASH
    memcpy(hash_values, copy.hash_values, (NUM_PANCAKES + 1) * sizeof(hash_t));
    #endif
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
    #ifdef FTF_HASH
    std::reverse(hash_values + 1, hash_values + i);
    #endif
  }

  uint8_t gap_lb(Direction dir) const;
  uint8_t update_gap_lb(Direction dir, int i, uint8_t LB) const;
  //Copies pancake, applies a flip, and updates g/h/f values
  FTF_Pancake apply_action(const int i) const;
};

struct FTFPancakeFSortHighG
{
  bool operator()(const FTF_Pancake& lhs, const FTF_Pancake& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const
  {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if(cmp == 0)
    {
      return false;
    }
    else if(lhs->f == rhs->f)
    {
      if(lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g > rhs->g;
    }
    else
    {
      return lhs->f < rhs->f;
    }
  }
};

struct FTFPancakeF_barSortHighG
{
  bool operator()(const FTF_Pancake& lhs, const FTF_Pancake& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const
  {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if(cmp == 0)
    {
      return false;
    }
    else if(lhs->f_bar == rhs->f_bar)
    {
      if(lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g > rhs->g;
    }
    else
    {
      return lhs->f_bar < rhs->f_bar;
    }
  }
};

struct FTFPancakeHash
{
  inline std::size_t operator() (const FTF_Pancake& x) const
  {
    return operator()(&x);
  }
  inline std::size_t operator() (const FTF_Pancake* x) const
  {
    return SuperFastHash(x->source + 1, NUM_PANCAKES);
  }
};

struct FTFPancakeEqual
{
  inline bool operator() (const FTF_Pancake* x, const FTF_Pancake* y) const
  {
    return memcmp(x->source, y->source, NUM_PANCAKES + 1) == 0;
  }
  inline bool operator() (const FTF_Pancake x, const FTF_Pancake y) const
  {
    return x == y;
  }
};
