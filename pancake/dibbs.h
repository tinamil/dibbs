#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <tuple>
#include <string>
#include <algorithm>

class Dibbs
{
  typedef std::unordered_set<Pancake, PancakeHash> hash_set;
  typedef std::set<Pancake, PancakeFBarSortLowG> set;

  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;

  Dibbs() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0) {}

  void expand_node(set& open, hash_set& open_hash, const hash_set& other_open, std::unordered_set<Pancake, PancakeHash>& closed) {
    Pancake next_val = *open.begin();
    open.erase(next_val);
    open_hash.erase(next_val);


    ++expansions;

    closed.insert(next_val);

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);

      if (new_action.f > UB) {
        continue;
      }

      auto it_closed = closed.find(new_action);
      if (it_closed == closed.end()) {

        auto it_open = other_open.find(new_action);
        if (it_open != other_open.end()) {
          UB = std::min(UB, (size_t)(*it_open).g + new_action.g);
        }
        else {
          auto it_open = open_hash.find(new_action);
          if (it_open != open_hash.end()) {
            continue;
          }

          open.insert(new_action);
          open_hash.insert(new_action);
        }
      }
    }
  }

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    expansions = 0;

    open_f.insert(start);
    open_b.insert(goal);

    UB = std::numeric_limits<double>::infinity();
    while (open_f.size() > 0 && open_b.size() > 0 && UB > ceil(((*open_f.begin()).f_bar + (*open_b.begin()).f_bar) / 2.0) && (open_f.size() + open_b.size()) <= NODE_LIMIT) {

      if ((*open_f.begin()).f_bar < (*open_b.begin()).f_bar) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else if ((*open_f.begin()).f_bar > (*open_b.begin()).f_bar) {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b);
      }
      else if (open_f.size() <= open_b.size()) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b);
      }

    }
    if (UB > ceil(((*open_f.begin()).f_bar + (*open_b.begin()).f_bar) / 2.0)) {
      return std::make_pair(std::numeric_limits<double>::infinity(), expansions);
    }
    else {
      //std::cout << "Size: " << open.size() << '\n';
      return std::make_pair(UB, expansions);
    }
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    Dibbs instance;
    return instance.run_search(start, goal);
  }
};
