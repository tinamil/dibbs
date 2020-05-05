#pragma once
#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include <StackArray.h>

#include <windows.h>
#include <Psapi.h>

class AssymetricSearch
{
  template <typename T>
  struct FGDiff {
    bool operator()(T lhs, T rhs) const {
      if (lhs->f == rhs->f) {
        return lhs->g < rhs->g;
      }
      else {
        return lhs->f > rhs->f;
      }
    }
  };
  template <typename T>
  struct GDiff {
    bool operator()(T lhs, T rhs) const {
      if (lhs->delta == rhs->delta) {
        return lhs->g < rhs->g;
      }
      else {
        return lhs->delta > rhs->delta;
      }
    }
  };
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FGDiff<const Pancake*>> backward_queue;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GDiff<const Pancake*>> forward_queue;
  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;

  StackArray<Pancake> storage;
  forward_queue open_f;
  backward_queue open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t memory;
  size_t expansions_cstar = 0;


  AssymetricSearch() {}

  template <typename T>
  void expand_node(T& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed) {
    const Pancake* next_val = open.top();
    open.pop();

    if (closed.count(next_val) > 0) {
      return;
    }

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);

    ++expansions;
    ++expansions_cstar;

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
          if ((*it_other)->g + new_action.g < UB) {
            if (new_action.dir == Direction::forward) {
              best_f = new_action;
              best_b = **it_other;
            }
            else {
              best_f = **it_other;
              best_b = new_action;
            }
          }
#endif  
          size_t combined = (size_t)(*it_other)->g + new_action.g;
          if (combined < UB) {
            expansions_cstar = 0;
            UB = combined;
          }
        }
        auto it_open = open_hash.find(&new_action);
        if (it_open != open_hash.end())
        {
          if ((*it_open)->g <= new_action.g) {
            continue;
          }
          else {
            //open.erase(&**it_open);
            open_hash.erase(it_open);
          }
        }

        auto ptr = storage.push_back(new_action);
        open.push(ptr);
        open_hash.insert(ptr);
      }
    }
  }

#ifdef HISTORY
  Pancake best_f, best_b;
#endif

  std::tuple<double, size_t, size_t> run_search(Pancake start, Pancake goal) {
    expansions = 0;
    memory = 0;
    auto ptr = storage.push_back(start);
    open_f.push (ptr);
    open_f_hash.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.push(ptr);
    open_b_hash.insert(ptr);

    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    while (open_f.size() > 0 && open_b.size() > 0 && UB > (open_f.top()->delta + open_b.top()->f)) {

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }


      if (open_f.top()->delta == 0) {
        expand_node<forward_queue>(open_f, open_f_hash, open_b_hash, closed_f);
      }
      else {
        expand_node<backward_queue>(open_b, open_b_hash, open_f_hash, closed_b);
      }

    }
#ifdef HISTORY
    std::cout << "Actions: ";
    for (int i = 0; i < best_f.actions.size(); ++i) {
      std::cout << std::to_string(best_f.actions[i]) << " ";
    }
    std::cout << "|" << " ";
    for (int i = best_b.actions.size() - 1; i >= 0; --i) {
      std::cout << std::to_string(best_b.actions[i]) << " ";
    }
    std::cout << std::endl;
#endif
    if (UB > (open_f.top()->delta + open_b.top()->f)) {
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
    }
    else {
      std::cout << "Expansions cstar: " << expansions_cstar << " / " << expansions << '\n';
      return std::make_tuple((double)UB, expansions, memory);
    }
  }

public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal) {
    AssymetricSearch instance;
    return instance.run_search(start, goal);
  }
};
