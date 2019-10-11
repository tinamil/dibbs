#pragma once
#include <iostream>
#include <cassert>
#include <algorithm>
#include "Direction.h"
#include "hash.hpp"

constexpr int NUM_PANCAKES = 10;
constexpr int GAPX = 0;
class Pancake {

private:
  uint8_t inv_source[NUM_PANCAKES + 1];            // inverse of sequence of pancakes
  static uint8_t*& goal() { static uint8_t* I; return I; }  // static goal sequence of Pancakes

  Direction dir;

public:

  uint8_t source[NUM_PANCAKES + 1];                // source sequence of Pancakes
  uint8_t g;
  uint8_t f;
  uint8_t h;

  uint8_t gap_lb() const;
  uint8_t update_gap_lb(int i, uint8_t LB) const;
  int check_inputs() const;

  Pancake(const uint8_t* data, Direction dir) : dir(dir), g(0), h(0), f(0)
  {
    assert(NUM_PANCAKES > 0);
    memcpy(source, data, NUM_PANCAKES + 1);
    f = h = gap_lb();
    std::reverse_copy(source + 1, source + NUM_PANCAKES + 1, inv_source + 1);
    inv_source[0] = source[0];
  }

  Pancake(const Pancake& copy) : dir(copy.dir), g(copy.g), h(copy.h), f(copy.f) {
    memcpy(source, copy.source, NUM_PANCAKES + 1);
    memcpy(inv_source, copy.inv_source, NUM_PANCAKES + 1);
  }

  static inline void initialize_goal(int n) {
    goal() = new uint8_t[n + 1];
    goal()[0] = n;
    for (int i = 1; i <= n; i++) goal()[i] = i;
  }

  inline bool is_solution() const {
    return memcmp(source, goal(), NUM_PANCAKES + 1) == 0;
  }

  inline bool operator==(const Pancake& right) const {
    return memcmp(source, right.source, NUM_PANCAKES + 1) == 0;
  }

  void apply_flip(int i) {
    assert(i >= 1 && i <= NUM_PANCAKES);
    std::reverse(source + 1, source + i + 1);
  }

  Pancake apply_action(int i) const {
    Pancake new_node(*this);
    new_node.h = new_node.update_gap_lb(i, new_node.h);
    new_node.g = g + 1;
    new_node.f = new_node.g + new_node.h;
    new_node.apply_flip(i);
    assert(new_node.f >= f); //Consistency check
    return new_node;
  }
};

struct PancakeSort {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return lhs.f > rhs.f;
  }
};

struct PancakeHash
{
  inline std::size_t operator() (const Pancake& x) const
  {
    return SuperFastHash(x.source, NUM_PANCAKES + 1);
  }
};

