#pragma once
#include "hash_table.h"
#include "Direction.h"
#include "Pancake.h"
#include <cstdint>
#include <unordered_map>
#include <set>
#include <tsl\hopscotch_map.h>
#include "mycuda.h"

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
  //hash_t hash_values[NUM_PANCAKES + 1];
  //uint64_t hash_64;

  FTF_Pancake() : dir(Direction::forward), g(0), h(0), f(0) /*,hash_64(0) */{}
  FTF_Pancake(const uint8_t* data, Direction dir) : dir(dir), g(0), h(0), f(0)
  {
    assert(NUM_PANCAKES > 0);
    memcpy(source, data, NUM_PANCAKES + 1);
    //h = gap_lb(dir);
    //for(int i = 1; i < NUM_PANCAKES; ++i)
    //{
    //  hash_values[i] = hash_table::hash(source[i], source[i + 1]);
    //}
    //hash_values[NUM_PANCAKES] = hash_table::hash(source[NUM_PANCAKES], NUM_PANCAKES + 1);
    //std::sort(std::begin(hash_values) + 1, std::end(hash_values));
    //hash_64 = hash_table::compress(hash_values);
  }

  FTF_Pancake(const FTF_Pancake& copy) : dir(copy.dir), g(copy.g), h(copy.h), f(copy.f) /*,hash_64(copy.hash_64)*/
    #ifdef HISTORY
    , actions(copy.actions), parent(copy.parent)
    #endif
  {
    memcpy(source, copy.source, NUM_PANCAKES + 1);
    //memcpy(hash_values, copy.hash_values, (NUM_PANCAKES + 1) * sizeof(hash_t));
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
FTF_Pancake FTF_Pancake::apply_action(const int i, bool match, T& structure) const
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
  //for(int i = 1; i < NUM_PANCAKES; ++i)
  //{
  //  new_node.hash_values[i] = hash_table::hash(new_node.source[i], new_node.source[i + 1]);
  //}
  //new_node.hash_values[NUM_PANCAKES] = hash_table::hash(new_node.source[NUM_PANCAKES], NUM_PANCAKES + 1);
  //std::sort(std::begin(new_node.hash_values) + 1, std::end(new_node.hash_values));
  //new_node.hash_64 = hash_table::compress(new_node.hash_values);
  new_node.g = g + 1;
  if(match)
  {
    new_node.h = structure.match(&new_node);
    new_node.f = new_node.g + new_node.h;
    //assert(new_node.f >= f); //Consistency check
  }
  return new_node;
}

