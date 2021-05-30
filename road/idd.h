#pragma once

#include "node.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>

#include <iostream>
#include <windows.h>
#include <Psapi.h>

#define IDD
Node best_pancake_f, best_pancake_b;
class Idd
{
  typedef std::unordered_set<Node, NodeHash> hash_set;

  hash_set open_f, open_b;
  std::stack<Node> stack;
  size_t LB;
  size_t UB;
  size_t expansions;
  size_t memory;
  bool abort;

  Idd() : open_f(), open_b(), stack(), LB(0), UB(0), expansions(0), abort(false) {}

  void expand_layer(std::stack<Node>& stack, std::unordered_set<Node, NodeHash>& my_set, const std::unordered_set<Node, NodeHash>& other_set, const size_t iteration)
  {
    my_set.clear();
    PROCESS_MEMORY_COUNTERS memCounter;
    while(!stack.empty() && LB < UB) {
      Node current = stack.top();
      stack.pop();

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT) {
        abort = true;
        break;
      }

      expansions += 1;

      for(int i = 0, j = current.num_neighbors(); i < j; ++i) {
        Node new_node = current.get_child(i);

        //Check for intersection with other direction
        auto it = other_set.find(new_node);
        if(it != other_set.end()) {
          int tmp_UB = (*it).g + new_node.g;
          if(tmp_UB < UB) {
            UB = tmp_UB;
            if(current.dir == Direction::forward) {
              best_pancake_f = new_node;
              best_pancake_b = (*it);
            }
            else {
              best_pancake_b = new_node;
              best_pancake_f = (*it);
            }
          }
        }
        if(new_node.f_bar <= iteration) {
          stack.push(new_node);
        }
        else if(current.threshold) {
          //Inserts both current and new_node (current MUST be inserted, new_node is speculative)
          it = my_set.find(current);
          if(it == my_set.end()) {
            my_set.insert(current);
          }
          else if((it != my_set.end() && (*it).g > current.g))
          {
            my_set.erase(it);
            my_set.insert(current);
          }

          if(new_node.h2 - new_node.h > 1) {
            it = my_set.find(new_node);
            if(it == my_set.end()) {
              my_set.insert(new_node);
            }
            else if((it != my_set.end() && (*it).g > new_node.g))
            {
              my_set.erase(it);
              my_set.insert(new_node);
            }
          }
        }
      }
    }
  }

  void id_check_layer(std::stack<Node>& stack, const std::unordered_set<Node, NodeHash>& other_set, const size_t iteration)
  {
    while(!stack.empty() && LB < UB && !abort) {
      Node current = stack.top();
      stack.pop();
      expansions += 1;

      for(int i = 0, j = current.num_neighbors(); i < j; ++i) {
        Node new_node = current.get_child(i);

        auto it = other_set.find(new_node);
        if(it != other_set.end()) {
          int tmp_UB = (*it).g + new_node.g;
          if(tmp_UB < UB) {
            UB = tmp_UB;
            if(current.dir == Direction::forward) {
              best_pancake_f = new_node;
              best_pancake_b = (*it);
            }
            else {
              best_pancake_b = new_node;
              best_pancake_f = (*it);
            }
          }
        }
        if(new_node.f_bar <= iteration) {
          stack.push(new_node);
        }
      }
    }
  }

  void search_iteration(Node forward_origin, Node backward_origin, size_t& iteration, hash_set& forward_set, hash_set& backward_set, size_t& forward_expansions, size_t& backward_expansions)
  {
    size_t start_count = expansions;
    stack.push(forward_origin);
    expand_layer(stack, forward_set, backward_set, iteration);
    forward_expansions = expansions - start_count;

    if(UB <= LB || abort) return;

    if(forward_set.size() > 0) {
      stack.push(backward_origin);
      id_check_layer(stack, forward_set, iteration - 1);
    }

    iteration += 1;
    std::cout << iteration << "\n";
    LB = iteration;

    if(UB <= LB || abort) return;

    start_count = expansions;
    stack.push(backward_origin);
    expand_layer(stack, backward_set, forward_set, iteration - 1);
    backward_expansions = expansions - start_count;

    if(UB <= LB || abort) return;

    //Extra check, unnecessary but might find an early solution for next depth 
    if(backward_set.size() > 0) {
      stack.push(forward_origin);
      id_check_layer(stack, backward_set, iteration - 1);
    }

    if(UB <= LB || abort) return;
  }

  std::tuple<double, size_t, size_t> run_search(Node start, Node goal)
  {
    best_pancake_f = start;
    best_pancake_b = goal;
    if(start == goal) {
      return std::make_tuple(0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    UB = std::numeric_limits<size_t>::max();
    LB = 1;
    //assert(start.gap_lb(start.dir) == goal.gap_lb(goal.dir));
    size_t iteration = 0;
    size_t forward_expansions = 0;
    size_t backward_expansions = 0;

    open_b.insert(goal);

    stack.push(start);
    expand_layer(stack, open_f, open_b, iteration);

    stack.push(goal);
    expand_layer(stack, open_b, open_f, iteration);

    iteration = 1;
    LB = 1;

    while(LB < UB && !abort) {
      if(forward_expansions <= backward_expansions) {
        search_iteration(start, goal, iteration, open_f, open_b, forward_expansions, backward_expansions);
      }
      else {
        search_iteration(goal, start, iteration, open_b, open_f, backward_expansions, forward_expansions);
      }
    }

    #ifdef HISTORY
    if(LB >= UB) {
      std::cout << "\nSolution: ";
      for(int i = 0; i < best_pancake_f.actions.size(); ++i) {
        std::cout << std::to_string(best_pancake_f.actions[i]) << " ";
      }
      std::cout << " | ";
      for(int i = best_pancake_b.actions.size() - 1; i >= 0; --i) {
        std::cout << std::to_string(best_pancake_b.actions[i]) << " ";
      }
      std::cout << std::endl;
    }
    #endif
    if(LB >= UB)
      return std::make_tuple((double)UB, expansions, memory);
    else
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
  }

public:

  static std::tuple<double, size_t, size_t> search(Node start, Node goal)
  {
    Idd instance;
    return instance.run_search(start, goal);
  }
};

