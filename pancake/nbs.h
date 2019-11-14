#pragma once
#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>

typedef std::unordered_set<Pancake, PancakeHash> hash_set;
typedef std::set<Pancake, PancakeFSortLowG> waiting_set;
typedef std::set<Pancake, PancakeGSort> ready_set;

class Nbs {

  ready_set open_f_ready, open_b_ready;
  waiting_set open_f_waiting, open_b_waiting;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;

  Nbs() : open_f_ready(), open_b_ready(), open_f_waiting(), open_b_waiting(), open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  bool select_pair() {
    while (!open_f_waiting.empty() && open_f_waiting.begin()->f < lbmin) {
      open_f_ready.insert(*open_f_waiting.begin());
      open_f_waiting.erase(open_f_waiting.begin());
    }
    while (!open_b_waiting.empty() && open_b_waiting.begin()->f < lbmin) {
      open_b_ready.insert(*open_b_waiting.begin());
      open_b_waiting.erase(open_b_waiting.begin());
    }

    while (true) {
      if (open_f_ready.empty() && open_f_waiting.empty()) return false;
      if (open_b_ready.empty() && open_b_waiting.empty()) return false;

      if (!open_f_ready.empty() && !open_b_ready.empty() && open_f_ready.begin()->g + open_b_ready.begin()->g <= lbmin) return true;
      if (!open_f_waiting.empty() && open_f_waiting.begin()->f <= lbmin) {
        open_f_ready.insert(*open_f_waiting.begin());
        open_f_waiting.erase(open_f_waiting.begin());
      }
      else if (!open_b_waiting.empty() && open_b_waiting.begin()->f <= lbmin) {
        open_b_ready.insert(*open_b_waiting.begin());
        open_b_waiting.erase(open_b_waiting.begin());
      }
      else {
        size_t min_wf = std::numeric_limits<size_t>::max();
        if (!open_f_waiting.empty()) min_wf = open_f_waiting.begin()->f;
        size_t min_wb = std::numeric_limits<size_t>::max();
        if (!open_b_waiting.empty()) min_wb = open_b_waiting.begin()->f;
        size_t min_r = std::numeric_limits<size_t>::max();
        if (!open_f_ready.empty() && !open_b_ready.empty()) min_r = open_f_ready.begin()->g + open_b_ready.begin()->g;
        lbmin = std::min(std::min(min_wf, min_wb), min_r);
      }
    }
  }

  bool expand_node_forward() {
    Pancake next_val = *open_f_ready.begin();
    open_f_ready.erase(next_val);
    open_f_hash.erase(next_val);

    closed_f.insert(next_val);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);

      auto it_open = open_b_hash.find(new_action);
      if (it_open != open_b_hash.end()) {
        UB = std::min(UB, (size_t)it_open->g + new_action.g);
      }

      it_open = open_f_hash.find(new_action);
      if (it_open != open_f_hash.end() && it_open->g <= new_action.g) continue;
      auto it_closed = closed_f.find(new_action);
      if (it_closed != closed_f.end() && it_closed->g <= new_action.g) continue;

      open_f_waiting.insert(new_action);
      open_f_hash.insert(new_action);
    }
  }

  bool expand_node_backward() {
    Pancake next_val = *open_b_ready.begin();
    open_b_ready.erase(next_val);
    open_b_hash.erase(next_val);

    closed_b.insert(next_val);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);

      auto it_open = open_f_hash.find(new_action);
      if (it_open != open_f_hash.end()) {
        UB = std::min(UB, (size_t)it_open->g + new_action.g);
      }

      it_open = open_b_hash.find(new_action);
      if (it_open != open_b_hash.end() && it_open->g <= new_action.g) continue;
      auto it_closed = closed_b.find(new_action);
      if (it_closed != closed_b.end() && it_closed->g <= new_action.g) continue;

      open_b_waiting.insert(new_action);
      open_b_hash.insert(new_action);
    }
  }
  std::pair<double, size_t> run_search(Pancake start, Pancake goal)
  {
    if (start == goal) {
      return std::make_pair(0, 0);
    }

    expansions = 0;
    UB = std::numeric_limits<size_t>::max();

    open_f_waiting.insert(start);
    open_f_hash.insert(start);

    open_b_waiting.insert(goal);
    open_b_hash.insert(goal);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    while (select_pair())
    {
      if (lbmin >= UB) {
        finished = true;
        break;
      }

      expand_node_forward();
      expand_node_backward();
    }

    if (finished)  return std::make_pair(UB, expansions);
    else return std::make_pair(std::numeric_limits<double>::infinity(), expansions);
  }


public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    Nbs instance;
    return instance.run_search(start, goal);
  }
};