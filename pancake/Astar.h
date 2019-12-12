#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <string>
#define NOMINMAX
#include <windows.h>
#include <Psapi.h>

class Astar
{
  std::priority_queue<Pancake, std::vector<Pancake>, PancakeFSort> open;
  std::unordered_set<Pancake, PancakeHash> closed;

  Astar() : open() {}

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    size_t expansions = 0;
    open.push(start);

    PROCESS_MEMORY_COUNTERS memCounter;
    double UB = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      Pancake next_val = open.top();
      open.pop();

      auto it = closed.find(next_val);
      if (it != closed.end())
      {
        continue;
      }
      closed.insert(next_val);

      if (next_val == goal) {
        UB = next_val.g;
#ifdef HISTORY
        std::cout << "\nSolution: ";
        for (int i = 0; i < next_val.actions.size(); ++i) {
          std::cout << std::to_string(next_val.actions[i]) << " ";
        }
        std::cout << std::endl;
#endif
        assert(next_val.h == 0);
        break;
      }

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      ++expansions;

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val.apply_action(i);
        it = closed.find(new_action);
        if (it == closed.end()) {
          open.push(new_action);
        }
      }
    }
    //std::cout << "Size: " << open.size() << '\n';
    return std::make_pair(UB, expansions);
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    Astar instance;
    return instance.run_search(start, goal);
  }
};

