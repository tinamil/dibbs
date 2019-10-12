#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <string>


class Astar
{
  std::priority_queue<Pancake, std::vector<Pancake>, PancakeFSort> open;
  std::unordered_set<Pancake, PancakeHash> closed;

  Astar() : open() {}

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    size_t expansions = 0;
    open.push(start);

    double UB = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      Pancake next_val = open.top();
      open.pop();
      if (next_val == goal) {
        UB = next_val.g;
        //std::cout << "Actions: ";
        //for (int i = 0; i < next_val.actions.size(); ++i) {
         // std::cout << std::to_string(next_val.actions[i]) << " ";
        //}
        //std::cout << std::endl;
        assert(next_val.h == 0);
        break;
      }

      ++expansions;
      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val.apply_action(i);
        auto it = closed.find(new_action);
        if (it == closed.end()) {
          open.push(new_action);
        }

#ifdef INCONSISTENT_HEURISTIC
        //only necessary for inconsistent heuristics
        else if ((*it).g > new_action.g) {
          closed.erase(it);
          open.push(new_action);
        }
#endif
      }


      auto in_closed = closed.find(next_val);
      if (in_closed != closed.end())
      {
#ifdef INCONSISTENT_HEURISTIC
        //only necessary for inconsistent heuristics
        if ((*in_closed).g > next_val.g) {
          closed.erase(in_closed);
        }
        else {
          continue;
        }
#else
        continue;
#endif
      }

      closed.insert(next_val);
    }

    return std::make_pair(UB, expansions);
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    Astar instance;
    return instance.run_search(start, goal);
  }
};

