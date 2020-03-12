#pragma once

#include "sliding_tile.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>
#include <queue>
#include "StackArray.h"

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>


class DibbsNbs {

  typedef std::unordered_set<const SlidingTile*, SlidingTileHash, SlidingTileEqual> hash_set;
  typedef std::priority_queue<const SlidingTile*, std::vector<const SlidingTile*>, FSortHighDuplicate> waiting_set;
  typedef std::priority_queue<const SlidingTile*, std::vector<const SlidingTile*>, FBarSortHighGLowDuplicate> ready_set;

  StackArray<SlidingTile> storage;
  ready_set open_f_ready, open_b_ready;
  waiting_set open_f_waiting, open_b_waiting;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;

#ifdef HISTORY
  Pancake best_f;
  Pancake best_b;
#endif

  DibbsNbs() : open_f_ready(), open_b_ready(), open_f_waiting(), open_b_waiting(), open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  bool select_pair() {

    while (true) {
      while (!open_f_waiting.empty() && open_f_waiting.top()->f <= lbmin) {
        open_f_ready.push(open_f_waiting.top());
        open_f_waiting.pop();
      }
      while (!open_b_waiting.empty() && open_b_waiting.top()->f <= lbmin) {
        open_b_ready.push(open_b_waiting.top());
        open_b_waiting.pop();
      }
      if (!open_f_ready.empty() && closed_f.find(open_f_ready.top()) != closed_f.end()) open_f_ready.pop();
      else if (!open_b_ready.empty() && closed_b.find(open_b_ready.top()) != closed_b.end()) open_b_ready.pop();
      else if (open_f_ready.empty() && open_f_waiting.empty()) return false;
      else if (open_b_ready.empty() && open_b_waiting.empty()) return false;
      else if (!open_f_ready.empty() && !open_b_ready.empty() && open_f_ready.top()->f_bar + open_b_ready.top()->f_bar <= 2 * lbmin) return true;
      else {
        size_t min_wf = std::numeric_limits<size_t>::max();
        if (!open_f_waiting.empty()) min_wf = open_f_waiting.top()->f;
        size_t min_wb = std::numeric_limits<size_t>::max();
        if (!open_b_waiting.empty()) min_wb = open_b_waiting.top()->f;
        size_t min_r = std::numeric_limits<size_t>::max();
        if (!open_f_ready.empty() && !open_b_ready.empty()) min_r = (size_t)ceil(((open_f_ready.top())->f_bar + (open_b_ready.top())->f_bar) / 2.0f);
        lbmin = std::min(std::min(min_wf, min_wb), min_r);
      }
    }
  }

  bool expand_node(hash_set& hash, ready_set& ready, waiting_set& waiting, hash_set& closed, const hash_set& other_hash) {
    const SlidingTile* next_val = ready.top();
    ready.pop();
    auto removed = hash.erase(next_val);
    assert(removed == 1);

    auto insertion_result = closed.insert(next_val);
    assert(insertion_result.second);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    memory = std::max(memory, memCounter.PagefileUsage);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 1, stop = next_val->num_actions_available(); i <= stop; ++i) {
      SlidingTile new_action = next_val->apply_action(i);

      auto it_open = other_hash.find(&new_action);
      if (it_open != other_hash.end()) {
        size_t tmp_UB = (size_t)(*it_open)->g + new_action.g;
        if (tmp_UB < UB) {
          UB = tmp_UB;
#ifdef HISTORY
          best_f = new_action;
          best_b = **it_open;
#endif
        }
      }
      auto it_closed = closed.find(&new_action);
      if (it_closed != closed.end() && (*it_closed)->g <= new_action.g) continue;
      else if (it_closed != closed.end()) {
        closed.erase(it_closed);
        assert(false);
      }

      it_open = hash.find(&new_action);
      if (it_open != hash.end() && (*it_open)->g <= new_action.g) continue;
      else if (it_open != hash.end() && (*it_open)->g > new_action.g) {
        hash.erase(it_open);
      }

      auto ptr = storage.push_back(new_action);
      waiting.push(ptr);
      auto hash_insertion_result = hash.insert(ptr);
      assert(hash_insertion_result.second);
    }
    return true;
  }

  bool expand_node_forward() {
    return expand_node(open_f_hash, open_f_ready, open_f_waiting, closed_f, open_b_hash);
  }

  bool expand_node_backward() {
    return expand_node(open_b_hash, open_b_ready, open_b_waiting, closed_b, open_f_hash);
  }

  std::tuple<double, size_t, size_t> run_search(SlidingTile start, SlidingTile goal)
  {
    if (start == goal) {
      return std::make_tuple(0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    UB = std::numeric_limits<size_t>::max();

    auto ptr = storage.push_back(start);
    open_f_waiting.push(ptr);
    open_f_hash.insert(ptr);

    ptr = storage.push_back(goal);
    open_b_waiting.push(ptr);
    open_b_hash.insert(ptr);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    while (select_pair())
    {
      if (lbmin >= UB) {
        finished = true;
        break;
      }

      if (expand_node_forward() == false) break;
      if (expand_node_backward() == false) break;
    }

    if (finished) {
#ifdef HISTORY
      std::cout << "\nSolution: ";
      for (int i = 0; i < best_f.actions.size(); ++i) {
        std::cout << std::to_string(best_f.actions[i]) << " ";
      }
      std::cout << "|" << " ";
      for (int i = best_b.actions.size() - 1; i >= 0; --i) {
        std::cout << std::to_string(best_b.actions[i]) << " ";
      }
      std::cout << "\n";
#endif 
      return std::make_tuple(UB, expansions, memory);
    }
    else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
  }


public:

  static std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal) {
    DibbsNbs instance;
    return instance.run_search(start, goal);
  }
};