#pragma once

#include "node.h"
#include "Pancake.h"
#include <queue>
#include <unordered_set>


class Astar
{
  std::priority_queue<Node> open;
  std::unordered_set<Node> closed;

public:
  double search(Node start) {
    open.push(start);

    double c_star = std::numeric_limits<double>::infinity();
    while (open.size() > 0) {
      Node next_val = open.top();
      open.pop();

      if (next_val.is_goal()) {
        c_star = next_val.g;
        assert(next_val.h == 0);
        break;
      }

      for (int i = 0, j = next_val.get_num_actions(); i < j; ++i) {
        Node new_action = next_val.apply_action(i);
        if (closed.find(new_action) != closed.end()) {
          open.push(next_val.apply_action(i));
        }
      }

      auto in_closed = closed.find(next_val);
      if (in_closed != closed.end())
      {
        if ((*in_closed).g > next_val.g) {
          closed.erase(in_closed);
        }
        else {
          continue;
        }
      }

      closed.insert(next_val);
    }

    return c_star;
  }
};

