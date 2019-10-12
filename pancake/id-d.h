#pragma once
#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>

Pancake best_pancake_f(Pancake::GetSortedStack(Direction::forward)), best_pancake_b(Pancake::GetSortedStack(Direction::backward));

class ID_D
{
  std::unordered_set<Pancake, PancakeHash> open_f, open_b;
  std::stack<Pancake> stack;

  ID_D() : open_f(), open_b() {}

  void expand_layer(std::stack<Pancake>& stack, std::unordered_set<Pancake, PancakeHash>& my_set, const std::unordered_set<Pancake, PancakeHash>& other_set, const size_t LB, size_t& UB, const size_t iteration, size_t& expansions) {
    my_set.clear();
    while (!stack.empty() && LB < UB) {
      Pancake current = stack.top();
      stack.pop();
      expansions += 1;

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_node = current.apply_action(i);

        auto it = other_set.find(new_node);
        if (it != other_set.end()) {
          int tmp_UB = (*it).g + new_node.g;
          if (tmp_UB < UB) {
            UB = tmp_UB;
            if (current.dir == Direction::forward) {
              best_pancake_f = new_node;
              best_pancake_b = (*it);
            }
            else {
              best_pancake_b = new_node;
              best_pancake_f = (*it);
            }
          }
        }

        if (new_node.f_bar <= iteration) {
          stack.push(new_node);
        }
        //TODO: Add strong threshold stores generated in addition to expanded
        else if (current.threshold) {
          it = my_set.find(current);
          if (it == my_set.end()) {
            my_set.insert(current);
          }
          else if ((it != my_set.end() && (*it).g > current.g))
          {
            my_set.erase(it);
            my_set.insert(current);
          }
        }
      }
    }
  }

  void id_check_layer(std::stack<Pancake>& stack, const std::unordered_set<Pancake, PancakeHash>& other_set, const size_t LB, size_t& UB, const size_t iteration, size_t& expansions) {
    while (!stack.empty() && LB < UB) {
      Pancake current = stack.top();
      stack.pop();
      expansions += 1;

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_node = current.apply_action(i);

        auto it = other_set.find(new_node);
        if (it != other_set.end()) {
          int tmp_UB = (*it).g + new_node.g;
          if (tmp_UB < UB) {
            UB = tmp_UB;
            if (current.dir == Direction::forward) {
              best_pancake_f = new_node;
              best_pancake_b = (*it);
            }
            else {
              best_pancake_b = new_node;
              best_pancake_f = (*it);
            }
          }
        }
        if (new_node.f_bar <= iteration) {
          stack.push(new_node);
        }
      }
    }
  }

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {

    if (start == goal) {
      return std::make_pair(0, 0);
    }
    size_t expansions = 0;
    size_t iteration = 0;
    size_t UB = std::numeric_limits<double>::max();
    size_t LB = 1;

    open_b.insert(goal);

    stack.push(start);
    expand_layer(stack, open_f, open_b, LB, UB, iteration, expansions);

    stack.push(goal);
    expand_layer(stack, open_b, open_f, LB, UB, iteration, expansions);

    iteration = 1;
    LB = 1;

    while (LB < UB) {
      //std::cout << "Expanding Forward: " << iteration << '\n';
      stack.push(start);
      expand_layer(stack, open_f, open_b, LB, UB, iteration, expansions);

      if (UB <= LB) break;

      if (open_f.size() > 0) {
        stack.push(goal);
        //std::cout << "ID Checking Backward: " << iteration << '\n';
        id_check_layer(stack, open_f, LB, UB, iteration - 1, expansions);
      }

      iteration += 1;
      LB = iteration;

      if (UB <= LB) break;

      stack.push(goal);
      //std::cout << "Expanding Backward: " << iteration << '\n';
      expand_layer(stack, open_b, open_f, LB, UB, iteration - 1, expansions);

      if (UB <= LB) break;

      //Extra check, unnecessary but might find an early solution for next depth 
      if (open_b.size() > 0) {
        stack.push(start);
        //std::cout << "ID Checking Forward: " << iteration << '\n';
        id_check_layer(stack, open_b, LB, UB, iteration - 1, expansions);
      }

      if (UB <= LB) break;
    }

    //std::cout << "Actions: ";
    //for (int i = 0; i < best_pancake_f.actions.size(); ++i) {
    //  std::cout << std::to_string(best_pancake_f.actions[i]) << " ";
    //}
    //for (int i = best_pancake_b.actions.size() - 1; i >= 0; --i) {
    //  std::cout << std::to_string(best_pancake_b.actions[i]) << " ";
    //}
    //std::cout << std::endl;

    return std::make_pair(UB, expansions);
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    ID_D instance;
    return instance.run_search(start, goal);
  }
};

