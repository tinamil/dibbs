#pragma once
#include <iostream>
#include <cassert>
#include <algorithm>
#include "Node.h"

class Pancake {

  uint8_t* inv_source;            // invserse of sequence of pancakes
  static uint8_t* goal;           // goal sequence of Pancakes

public:
  uint8_t* source;                // source sequence of Pancakes
  uint8_t n;                      // n = number of Pancakes.

  Pancake(uint8_t* data, uint8_t n) : n(n)
  {
    source = new uint8_t[n];
    inv_source = new uint8_t[n];
    memcpy(source, data, n);

    for (int i = n - 1; i >= 0; --i) {

    }
    std::reverse_copy(source, source + n, inv_source);
  }

  ~Pancake() {
    delete[] inv_source;
    delete[] source;
  }

  static inline void initialize_goal(int n) {
    goal = new uint8_t[n + 1];
    for (int i = 1; i <= n; i++) goal[i] = i;
    goal[0] = n;
  }

  unsigned char gap_lb(int direction, int x);
  uint8_t update_gap_lb(int direction, int i, uint8_t LB, int x);
  int check_inputs();

  inline bool is_solution() {
    return memcmp(source, goal, n) == 0;
  }
};
