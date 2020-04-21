#pragma once
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include "Direction.h"
#include "hash.hpp"

//#define HISTORY

constexpr int NUM_PANCAKES = 30;
constexpr int GAPX = 0;
constexpr size_t MEM_LIMIT = 100ui64 * 1024 * 1024 * 1024; //100GB

class Pancake {

private:     // inverse of sequence of pancakes
  static uint8_t*& DUAL_SOURCE() { static uint8_t* I = nullptr; return I; };  // static goal sequence of Pancakes

public:
  Direction dir;
#ifdef HISTORY
  std::vector<uint8_t> actions;
#endif
  uint8_t source[NUM_PANCAKES + 1];                // source sequence of Pancakes
  uint8_t g;
  uint8_t h;
  uint8_t h2;
  uint8_t f;
  uint8_t f_bar;
  int32_t hdiff;
  uint8_t delta;
  bool threshold;

  uint8_t gap_lb(Direction dir) const;
  uint8_t update_gap_lb(Direction dir, int i, uint8_t LB) const;
  int check_inputs() const;

  Pancake() {}
  Pancake(const uint8_t* data, Direction dir) : dir(dir), g(0), h(0), h2(0), f(0), f_bar(0)
  {
    assert(NUM_PANCAKES > 0);
    memcpy(source, data, NUM_PANCAKES + 1);
    h = gap_lb(dir);
    f = h;
    f_bar = f;
    hdiff = h;
    delta = 0;
    threshold = h == 0;
  }

  Pancake(const Pancake& copy) : dir(copy.dir), g(copy.g), h(copy.h), h2(copy.h2), f(copy.f),
    f_bar(copy.f_bar), hdiff(copy.hdiff), delta(copy.delta), threshold(copy.threshold)
#ifdef HISTORY
    , actions(copy.actions)
#endif
  {
    memcpy(source, copy.source, NUM_PANCAKES + 1);
  }

  //Required to calculate reverse heuristics, not needed for forward only search
  static void Initialize_Dual(uint8_t src[]) {
    if (DUAL_SOURCE() == nullptr) DUAL_SOURCE() = new uint8_t[NUM_PANCAKES + 1];
    DUAL_SOURCE()[0] = NUM_PANCAKES;
    for (int i = 1; i <= NUM_PANCAKES; i++) DUAL_SOURCE()[src[i]] = i;
  }

  inline static Pancake GetSortedStack(Direction dir) {
    uint8_t pancakes[NUM_PANCAKES + 1];
    pancakes[0] = NUM_PANCAKES;
    for (int i = 1; i <= NUM_PANCAKES; ++i) { pancakes[i] = i; }

    return Pancake(pancakes, dir);
  }

  inline bool operator==(const Pancake& right) const {
    return memcmp(source, right.source, NUM_PANCAKES + 1) == 0;
  }

  //Reverses the pancakes between 1 and i
  inline void apply_flip(int i) {
    assert(i >= 1 && i <= NUM_PANCAKES);
    std::reverse(source + 1, source + i + 1);
  }

  //Copies pancake, applies a flip, and updates g/h/f values
  Pancake apply_action(int i) const {
    Pancake new_node(*this);
#ifdef HISTORY
    new_node.actions.push_back(i);
#endif
    new_node.h = new_node.update_gap_lb(dir, i, new_node.h);
    new_node.h2 = new_node.update_gap_lb(OppositeDirection(dir), i, new_node.h2);
    new_node.g = g + 1;
    new_node.f = new_node.g + new_node.h;

    new_node.f_bar = 2 * new_node.g + new_node.h - new_node.h2;
    new_node.hdiff = new_node.h - new_node.h2;
    new_node.threshold = threshold || new_node.h <= new_node.h2;
    new_node.delta = new_node.g - new_node.h2;
    new_node.apply_flip(i);
    assert(new_node.f >= f); //Consistency check
    return new_node;
  }
};

//Returns smallest f value with largest g value
struct PancakeFSort {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    if (lhs->f == rhs->f) {
      return lhs->g < rhs->g;
    }
    return lhs->f > rhs->f;
  }
};

struct PancakeFSortLowGSetComparer {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if (cmp == 0) {
      return false;
    }
    else if (lhs->f == rhs->f) {
      if (lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g > rhs->g;
    }
    else {
      return lhs->f < rhs->f;
    }
  }
};

struct FSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    if (lhs->f == rhs->f) {
      return lhs->g > rhs->g;
    }
    return lhs->f > rhs->f;
  }
};

//Returns smallest f value with smallest g value
struct PancakeFSortLowG {

  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if (cmp == 0) {
      return false;
    }
    else if (lhs->f == rhs->f) {
      if (lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else {
      return lhs->f < rhs->f;
    }
  }
};

//Returns smallest g value
struct PancakeGSortLow {

  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if (cmp == 0) {
      return false;
    }
    if (lhs->g == rhs->g)
      return cmp < 0;
    else
      return lhs->g < rhs->g;
  }
};

struct PancakeGSortHigh {

  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if (cmp == 0) {
      return false;
    }
    if (lhs->g == rhs->g)
      return cmp < 0;
    else
      return lhs->g > rhs->g;
  }
};

//Returns smallest fbar with smallest g value
struct PancakeFBarSortLowG {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
    if (cmp == 0) {
      return false;
    }
    else if (lhs->f_bar == rhs->f_bar) {
      if (lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else {
      return lhs->f_bar < rhs->f_bar;
    }
  }
};

struct GSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    if (lhs->g == rhs->g) {
      return lhs->h < rhs->h;
    }
    return lhs->g > rhs->g;
  }
};

struct DeltaSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    uint8_t ld = lhs->g - lhs->h2;
    uint8_t rd = rhs->g - rhs->h2;

    if (ld == rd) {
      return lhs->g < rhs->g;
    }
    return ld > rd;
  }
};

struct FBarSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    if (lhs->f_bar == rhs->f_bar) {
      return lhs->g > rhs->g;
    }
    else {
      return lhs->f_bar > rhs->f_bar;
    }
  }
};

struct HfSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    return lhs->h > rhs->h;
  }
};

struct HbSortHighDuplicate {
  bool operator()(const Pancake& lhs, const Pancake& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Pancake* lhs, const Pancake* rhs) const {
    return lhs->h2 > rhs->h2;
  }
};

struct PancakeHash
{
  inline std::size_t operator() (const Pancake& x) const
  {
    return operator()(&x);
  }
  inline std::size_t operator() (const Pancake* x) const
  {
    return SuperFastHash(x->source + 1, NUM_PANCAKES);
  }
};

struct PancakeEqual
{
  inline bool operator() (const Pancake* x, const Pancake* y) const
  {
    return memcmp(x->source, y->source, NUM_PANCAKES + 1) == 0;
  }
  inline bool operator() (const Pancake x, const Pancake y) const
  {
    return x == y;
  }
};
