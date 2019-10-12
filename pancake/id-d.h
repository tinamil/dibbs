#pragma once
#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>

class ID_D
{
  std::unordered_set<Pancake, PancakeHash> open_f, open_b;
  //std::unordered_set<Pancake, PancakeHash> closed;
  std::stack<Pancake> stack;
  
  size_t UB;

  ID_D() : open_f(), open_b() {}

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    size_t expansions = 0;
    stack.push(start);
    UB = std::numeric_limits<size_t>().max();

    double c_star = std::numeric_limits<double>::infinity();
    while (stack.size() > 0) {
      Pancake next_val = stack.top();
      stack.pop();
      ++expansions;

      //TODO: Set termination condition correctly
      if (next_val.f < UB) {
        c_star = next_val.g;
        assert(next_val.h == 0);
        break;
      }

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val.apply_action(i);
        auto it = closed.find(new_action);
        if (it == closed.end()) {
          open.push(new_action);
        }
      }


      auto in_closed = closed.find(next_val);
      if (in_closed != closed.end())
      {
        continue;
      }

      closed.insert(next_val);
    }

    return std::make_pair(c_star, expansions);
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    ID_D instance;
    return instance.run_search(start, goal);
  }
};

