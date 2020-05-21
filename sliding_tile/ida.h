#pragma once

#include "sliding_tile.h"
#include <cstdint>
#include <stack>
#include <unordered_set>
#include <ctime>
#include <iostream>
#include <algorithm>


#include <windows.h>
#include <Psapi.h>

class IDAstar
{

private:
  size_t memory;
  typedef std::stack<SlidingTile, std::vector<SlidingTile>> stack;
  IDAstar() {}
  std::tuple<double, size_t, size_t> run_search(SlidingTile start, SlidingTile goal) {
    memory = 0;
    stack state_stack;

    if (start == goal)
    {
      std::cout << "Given a solved pancake problem.  Nothing to solve." << std::endl;
      return std::make_tuple(0, 0, 0);
    }

    uint8_t id_depth = start.f;

    state_stack.push(start);
    size_t UB;
    uint64_t expansions = 0;
    PROCESS_MEMORY_COUNTERS memCounter;
    bool done = false;
    while (done == false)
    {
      if (state_stack.empty())
      {
        id_depth += 1;
        state_stack.push(start);
        std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
      }

      auto next_val = state_stack.top();
      state_stack.pop();

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      ++expansions;

      for (int i = 1, stop = next_val.num_actions_available(); i <= stop; ++i) {
        SlidingTile new_action = next_val.apply_action(i);

        if (new_action.f <= id_depth) {

          if (new_action == goal) {
            UB = new_action.g;
            //std::cout << "Solution: " << UB << '\n';
            //std::cout << "Actions: ";
            //for (int i = 0; i < next_val.actions.size(); ++i) {
             // std::cout << std::to_string(next_val.actions[i]) << " ";
            //}
            //std::cout << std::endl;
            assert(new_action.h == 0);
            done = true;
            break;
          }
          state_stack.push(new_action);
        }
      }
    }
    return std::make_tuple(UB, expansions, memory);
  }


public:
  static inline std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal) {
    IDAstar instance;
    return instance.run_search(start, goal);
  }
  //std::pair<uint64_t, double> multithreaded_ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
};
