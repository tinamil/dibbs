#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>

const static int debug_nodes[][17] = {
  { 16, 10, 3, 12, 6, 7, 8, 4, 5, 15, 2, 9, 14, 13, 1, 11, 16 },
{ 16, 2, 15, 5, 4, 8, 7, 6, 12, 3, 10, 9, 14, 13, 1, 11, 16 },
{ 16, 13, 14, 9, 10, 3, 12, 6, 7, 8, 4, 5, 15, 2, 1, 11, 16 },
{ 16, 3, 10, 9, 14, 13, 12, 6, 7, 8, 4, 5, 15, 2, 1, 11, 16 },
{ 16, 8, 7, 6, 12, 13, 14, 9, 10, 3, 4, 5, 15, 2, 1, 11, 16 },
{ 16, 14, 13, 12, 6, 7, 8, 9, 10, 3, 4, 5, 15, 2, 1, 11, 16 },
{ 16, 5, 4, 3, 10, 9, 8, 7, 6, 12, 13, 14, 15, 2, 1, 11, 16 },
{ 16, 3, 4, 5, 10, 9, 8, 7, 6, 12, 13, 14, 15, 2, 1, 11, 16 },
{ 16, 15, 14, 13, 12, 6, 7, 8, 9, 10, 5, 4, 3, 2, 1, 11, 16 },
{ 16, 11, 1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16 },
{ 16, 6, 7, 8, 9, 10, 5, 4, 3, 2, 1, 11, 12, 13, 14, 15, 16 },
{ 16, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 12, 13, 14, 15, 16 },
{ 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
};

const static int debug[] = { 10, 9, 10, 8, 2, 6, 7, 1, 5, 4, 3 };

class Dibbs
{
  typedef std::set<Pancake, PancakeFBarSortLowG> set;
  typedef std::unordered_map<const Pancake*, set::const_iterator, PancakeHash, PancakeEqual> hash_set;

  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  std::unordered_set<Pancake, PancakeHash> closed_f, closed_b;
  size_t expansions;
  size_t UB;
 

  Dibbs() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0) {}


  void expand_node(set& open, hash_set& open_hash, const hash_set& other_open, std::unordered_set<Pancake, PancakeHash>& closed) {
    Pancake next_val = *open.begin();

    //for (int i = 0; i < 13; ++i) {
    //  bool match = true;
    //  for (int j = 1; j <= NUM_PANCAKES && match; ++j) {
    //    if (debug_nodes[i][j] != next_val.source[j]) match = false;
    //  }
    //  if (match) {
    //    std::cout << (next_val.dir == Direction::forward ? "F" : "B") << i << "\n";
    //    break;
    //  }
    //}

    auto it_hash = open_hash.find(&next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(next_val);

    ++expansions;

    closed.insert(next_val);

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);
      
      if (new_action.f > UB) {
        continue;
      }

      auto it_closed = closed.find(new_action);
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
          UB = std::min(UB, (size_t)it_other->first->g + new_action.g);
        }
        auto it_open = open_hash.find(&new_action);
        if (it_open != open_hash.end())
        {
          if (it_open->first->g <= new_action.g) {
            continue;
          }
          else {
            open.erase(it_open->second);
            open_hash.erase(it_open);
          }
        }

        auto insertion = open.insert(new_action);
        assert(insertion.second);
        open_hash.insert(std::make_pair(&*insertion.first, insertion.first));
      }
    }
  }

#ifdef HISTORY
  Pancake best_f, best_b;
#endif

  std::pair<double, size_t> run_search(Pancake start, Pancake goal) {
    expansions = 0;

    auto insertion = open_f.insert(start);
    open_f_hash.insert(std::make_pair(&*insertion.first, insertion.first));
    insertion = open_b.insert(goal);
    open_b_hash.insert(std::make_pair(&*insertion.first, insertion.first));

    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    while (open_f.size() > 0 && open_b.size() > 0 && UB > ceil((open_f.begin()->f_bar + open_b.begin()->f_bar) / 2.0)) {

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      if (open_f.begin()->f_bar < open_b.begin()->f_bar) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else if (open_f.begin()->f_bar > open_b.begin()->f_bar) {
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
    if (UB > ceil((open_f.begin()->f_bar + open_b.begin()->f_bar) / 2.0)) {
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
