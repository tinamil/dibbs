#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include "StackArray.h"

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>

class Dibbs
{
  typedef std::set<const Pancake*, PancakeFBarSortLowG> set;
  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;

  StackArray<Pancake> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;


  Dibbs() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0) {}


  void expand_node(set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed) {
    const Pancake* next_val = *open.begin();

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(next_val);

    ++expansions;

    closed.insert(next_val);

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val->apply_action(i);

      if (new_action.f > UB) {
        continue;
      }

      auto it_closed = closed.find(&new_action);
      if (it_closed == closed.end()) {

        auto it_other = other_open.find(&new_action);
        if (it_other != other_open.end()) {
#ifdef HISTORY
          if (it_other->g + new_action.g < UB) {
            if (new_action.dir == Direction::forward) {
              best_f = new_action;
              best_b = *it_other;
            }
            else {
              best_f = *it_other;
              best_b = new_action;
            }
          }
#endif
          UB = std::min(UB, (size_t)(*it_other)->g + new_action.g);
        }
        auto it_open = open_hash.find(&new_action);
        if (it_open != open_hash.end())
        {
          if ((*it_open)->g <= new_action.g) {
            continue;
          }
          else {
            open.erase(&**it_open);
            open_hash.erase(it_open);
          }
        }

        auto ptr = storage.push_back(new_action);
        open.insert(ptr);
        open_hash.insert(ptr);
      }
    }
  }

#ifdef HISTORY
  Pancake best_f, best_b;
#endif

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    expansions = 0;

    auto ptr = storage.push_back(start);
    open_f.insert(ptr);
    open_f_hash.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.insert(ptr);
    open_b_hash.insert(ptr);

    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    while (open_f.size() > 0 && open_b.size() > 0 && UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      if ((*open_f.begin())->f_bar < (*open_b.begin())->f_bar) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else if ((*open_f.begin())->f_bar > (*open_b.begin())->f_bar) {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b);
      }
      else if (open_f.size() <= open_b.size()) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b);
      }

    }
#ifdef HISTORY
    std::cout << "Actions: ";
    for (int i = 0; i < best_f.actions.size(); ++i) {
      std::cout << std::to_string(best_f.actions[i]) << " ";
    }
    for (int i = 0; i < best_b.actions.size(); ++i) {
      std::cout << std::to_string(best_b.actions[i]) << " ";
    }
    std::cout << std::endl;
#endif
    if (UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {
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
