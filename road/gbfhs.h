#pragma once

#include "node.h"
#include "StackArray.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>

#include <windows.h>
#include <Psapi.h>

#define GBFHS

class Gbfhs
{

  typedef std::unordered_set<const Node*, NodeHash, NodeEqual> hash_set;
  typedef std::unordered_set<const Node*, NodeHash> closed_set;
  typedef std::set<const Node*, NodeFSortLowG> set;

  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  closed_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t f_lim; //Also LB
  size_t memory;
  StackArray<Node> storage;

  Gbfhs() : open_f(), open_b(), open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), f_lim(0) {}


  std::pair<size_t, size_t> gbfhs_split(const size_t f_lim)
  {
    auto glim_b = f_lim / 2;
    auto glim_a = f_lim - glim_b;

    return std::make_pair(glim_a, glim_b);
  }


  bool expand_level(const size_t glim_f, const size_t glim_b, const size_t f_lim)
  {

    PROCESS_MEMORY_COUNTERS memCounter;
    std::priority_queue<const Node*, std::vector<const Node*>, NodeGSortHigh> expandable;
    auto it = open_f.begin();
    while(it != open_f.end() && (*it)->f <= f_lim) {
      while(it != open_f.end() && (*it)->f <= f_lim && (*it)->g >= glim_f) ++it;
      if(it == open_f.end() || (*it)->f > f_lim) break;
      expandable.push(*it);
      ++it;
    }

    while(expandable.size() > 0) {
      const Node* next_val = expandable.top();
      expandable.pop();
      open_f_hash.erase(next_val);
      open_f.erase(next_val);

      assert(open_f_hash.size() == open_f_hash.size());

      closed_f.insert(next_val);

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT) {
        return false;
      }

      ++expansions;

      for(int i = 0, j = next_val->num_neighbors(); i < j; ++i) {
        Node new_action = next_val->get_child(i);

        auto it_open = open_f_hash.find(&new_action);
        if(it_open != open_f_hash.end()) {
          if((*it_open)->g <= new_action.g) continue;
          else {
            open_f.erase(*it_open);
            open_f_hash.erase(it_open);
          }
        }
        auto it_closed = closed_f.find(&new_action);
        if(it_closed != closed_f.end()) {
          if((*it_closed)->g <= new_action.g) continue;
          else closed_f.erase(it_closed);
        }

        auto ptr = storage.push_back(new_action);
        open_f.insert(ptr);
        open_f_hash.insert(ptr);

        if(new_action.f <= f_lim && new_action.g <= glim_f) expandable.push(ptr);

        it_open = open_b_hash.find(&new_action);
        if(it_open != open_b_hash.end()) {
          UB = std::min(UB, (size_t)(*it_open)->g + new_action.g);
          if(UB <= f_lim) return true;
        }
      }
    }

    it = open_b.begin();
    while(it != open_b.end() && (*it)->f <= f_lim) {
      while(it != open_b.end() && (*it)->f <= f_lim && (*it)->g >= glim_b) ++it;
      if(it == open_b.end() || (*it)->f > f_lim) break;
      expandable.push(*it);
      ++it;
    }

    while(expandable.size() > 0) {
      const Node* next_val = expandable.top();
      expandable.pop();
      open_b_hash.erase(next_val);
      open_b.erase(next_val);

      closed_b.insert(next_val);

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT) {
        return false;
      }

      ++expansions;

      for(int i = 0, j = next_val->num_neighbors(); i < j; ++i) {
        Node new_action = next_val->get_child(i);

        auto it_open = open_b_hash.find(&new_action);
        if(it_open != open_b_hash.end()) {
          if((*it_open)->g <= new_action.g) continue;
          else {
            open_b.erase(*it_open);
            open_b_hash.erase(it_open);
          }
        }
        auto it_closed = closed_b.find(&new_action);
        if(it_closed != closed_b.end()) {
          if((*it_closed)->g <= new_action.g) continue;
          else closed_b.erase(it_closed);
        }

        auto ptr = storage.push_back(new_action);
        open_b.insert(ptr);
        open_b_hash.insert(ptr);

        if(new_action.f <= f_lim && new_action.g <= glim_b) expandable.push(ptr);

        it_open = open_f_hash.find(&new_action);
        if(it_open != open_f_hash.end()) {
          UB = std::min(UB, (size_t)(*it_open)->g + new_action.g);
          if(UB <= f_lim) return true;
        }
      }
    }
    return true;
  }

  std::tuple<double, size_t, size_t> run_search(Node start, Node goal)
  {
    if(start == goal) {
      return std::make_tuple(0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    UB = std::numeric_limits<size_t>::max();

    size_t iteration = 0;

    auto start_ptr = storage.push_back(start);
    open_f.insert(start_ptr);
    open_f_hash.insert(start_ptr);

    auto goal_ptr = storage.push_back(goal);
    open_b.insert(goal_ptr);
    open_b_hash.insert(goal_ptr);

    f_lim = std::max(1ui32, std::max(start.h, goal.h));
    while(open_f.size() > 0 && open_b.size() > 0)
    {
      if(UB == f_lim) {
        break;
      }

      auto [glim_f, glim_b] = gbfhs_split(f_lim);
      if(!expand_level(glim_f, glim_b, f_lim)) {
        return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
      }

      if(UB == f_lim) {
        break;
      }
      f_lim += 1;
    }

    return std::make_tuple((double)UB, expansions, memory);
  }


public:

  static std::tuple<double, size_t, size_t> search(Node start, Node goal)
  {
    Gbfhs instance;
    return instance.run_search(start, goal);
  }
};