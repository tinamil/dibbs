#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <string>
#include <windows.h>
#include <Psapi.h>
#include "StackArray.h"

class Astar
{

public:
  std::priority_queue<const Pancake*, std::vector<const Pancake*>, PancakeFSort> open;
  std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> closed;
  StackArray<Pancake> pancakes;
  size_t memory;

  Astar() : open() {}

  std::tuple<double, size_t, size_t> run_search(Pancake start, Pancake goal, std::vector<const Pancake*>* expansions_in_order = nullptr) {
    size_t expansions = 0;
    memory = 0;
    open.push(pancakes.push_back(start));

    PROCESS_MEMORY_COUNTERS memCounter;
    double UB = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      const Pancake* next_val = open.top();
      open.pop();

      auto it = closed.find(next_val);
      if (it != closed.end())
      {
        continue;
      }
      closed.insert(next_val);

      if (*next_val == goal && std::isinf(UB)) {
        UB = next_val->g;
#ifdef HISTORY
        /*std::cout << "\nSolution: ";
        for (int i = 0; i < next_val.actions.size(); ++i) {
          std::cout << std::to_string(next_val.actions[i]) << " ";
        }
        std::cout << std::endl;*/
#endif
        assert(next_val->h == 0);
        break;
      }

      /*if (next_val->f > UB) {
        break;
      }*/

      if (expansions_in_order != nullptr) {
        expansions_in_order->push_back(next_val);
      }

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      ++expansions;

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val->apply_action(i);
        it = closed.find(&new_action);
        if (it == closed.end()) {
          open.push(pancakes.push_back(new_action));
        }
      }
    }
    //std::cout << "Size: " << open.size() << '\n';
    return std::make_tuple(UB, expansions, memory);
  }

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal, std::vector<const Pancake*>* expansions_in_order = nullptr) {
    Astar instance;
    return instance.run_search(start, goal, expansions_in_order);
  }
};

