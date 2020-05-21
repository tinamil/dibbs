#pragma once

#include "sliding_tile.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <string>
#include <windows.h>
#include <Psapi.h>

class Astar
{

public:
  std::priority_queue<SlidingTile, std::vector<SlidingTile>, FSort> open;
  std::unordered_set<SlidingTile, SlidingTileHash> closed;
  size_t memory;

  Astar() : open() {}

  std::tuple<double, size_t, size_t> run_search(SlidingTile start, SlidingTile goal) {
    size_t expansions = 0;
    memory = 0;
    open.push(start);

    PROCESS_MEMORY_COUNTERS memCounter;
    double UB = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      SlidingTile next_val = open.top();
      open.pop();

      auto it = closed.find(next_val);
      if (it != closed.end())
      {
        continue;
      }
      closed.insert(next_val);

      if (next_val == goal) {
        UB = next_val.g;
        //std::cout << "Solution: " << UB << '\n';
        //std::cout << "Actions: ";
        //for (int i = 0; i < next_val.actions.size(); ++i) {
        //  std::cout << std::to_string(next_val.actions[i]) << " ";
        //}
        //std::cout << std::endl;
        assert(next_val.h == 0);
        //break;
      }

      if (next_val.f > UB) {
        break;
      }

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      ++expansions;
      for (int i = 1, stop = next_val.num_actions_available(); i <= stop; ++i) {
        SlidingTile new_action = next_val.apply_action(i);
        it = closed.find(new_action);
        if (it == closed.end()) {
          open.push(new_action);
        }
      }
    }
    //std::cout << "Size: " << open.size() << '\n';
    return std::make_tuple(UB, expansions, memory);
  }

  static std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal) {
    Astar instance;
    return instance.run_search(start, goal);
  }
};

