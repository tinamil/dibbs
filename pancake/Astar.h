#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>


class Astar
{
  std::priority_queue<Pancake, std::vector<Pancake>, PancakeSort> open;
  std::unordered_set<Pancake, PancakeHash> closed;

  Astar() : open() {}

  std::pair<double, size_t> run_search(Pancake start) {
    size_t expansions = 0;
    open.push(start);

    double c_star = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      Pancake next_val = open.top();
      open.pop();
      ++expansions;
      if (next_val.is_solution()) {
        c_star = next_val.g;
        assert(next_val.h == 0);
        break;
      }

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val.apply_action(i);
        auto it = closed.find(new_action);
        if (it == closed.end()) {
          open.push(new_action);
        }//only necessary for inconsistent heuristics
        //else if ((*it).g > new_action.g) { 
        //  closed.erase(it);
        //  open.push(new_action);
        //}
      }


      auto in_closed = closed.find(next_val);
      if (in_closed != closed.end())
      {
        //only necessary for inconsistent heuristics
        //if ((*in_closed).g > next_val.g) { 
          //closed.erase(in_closed);
        //}
        //else {
        continue;
        //}
      }

      closed.insert(next_val);
    }

    return std::make_pair(c_star, expansions);
  }

public:

  static std::pair<double, size_t> search(Pancake start) {
    Astar instance;
    return instance.run_search(start);
  }
};

