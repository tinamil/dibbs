#pragma once

#include "node.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <string>
#include <windows.h>
#include <Psapi.h>
#include "StackArray.h"

#define ASTAR

class Astar
{

public:
  std::priority_queue<const Node*, std::vector<const Node*>, NodeFSort> open;
  std::unordered_set<const Node*, NodeHash, NodeEqual> closed;
  StackArray<Node> pancakes;
  size_t memory;

  Astar() : open() {}

  std::tuple<double, size_t, size_t> run_search(Node start, Node goal)
  {
    size_t expansions = 0;
    memory = 0;
    open.push(pancakes.push_back(start));

    PROCESS_MEMORY_COUNTERS memCounter;
    double UB = std::numeric_limits<double>::infinity();
    while(open.size() > 0) {
      const Node* next_val = open.top();
      open.pop();
      auto it = closed.find(next_val);
      if(it != closed.end())
      {
        continue;
      }
      closed.insert(next_val);

      if(*next_val == goal && std::isinf(UB)) {
        UB = next_val->g;
        #ifdef HISTORY
        std::cout << "\nSolution: ";
        for(int i = 0; i < next_val.actions.size(); ++i) {
          std::cout << std::to_string(next_val.actions[i]) << " ";
        }
        std::cout << std::endl;
        #endif
        assert(next_val->h == 0);
        break;
      }

      /*if (next_val->f > UB) {
        break;
      }*/

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      ++expansions;

      for(int i = 0, j = Road::num_neighbors(next_val->vertex_index); i < j; ++i) {
        Node new_action = next_val->get_child(i);
        it = closed.find(&new_action);
        if(it == closed.end()) {
          open.push(pancakes.push_back(new_action));
        }
      }
    }
    //std::cout << "Size: " << open.size() << '\n';
    return std::make_tuple(UB, expansions, memory);
  }

  static std::tuple<double, size_t, size_t> search(Node start, Node goal)
  {
    Astar instance;
    return instance.run_search(start, goal);
  }
};

