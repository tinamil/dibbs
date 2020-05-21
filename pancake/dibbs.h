#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include <StackArray.h>
#include <tsl/hopscotch_set.h>

#include <windows.h>
#include <Psapi.h>

class Dibbs
{
  struct LocalitySearch {
    template <size_t s, size_t t>
    struct Hash {
      inline size_t operator()(const Pancake* x) const {
        return boost_hash(x->source + s, t);
      }
    };
    template <size_t s, size_t t>
    struct Equal {
      inline bool operator()(const Pancake* x, const Pancake* y) const {
        if (memcmp(x->source + s, y->source + s, t) == 0) return true;
        for (int i = s; i < t + s; ++i) {
          //if(x->source[s] ==
        }
      }
    };
    tsl::hopscotch_set<const Pancake*, Hash<1, 4>, Equal<1, 4>> set1;
    tsl::hopscotch_set<const Pancake*, Hash<5, 8>, Equal<5, 8>> set2;
    tsl::hopscotch_set<const Pancake*, Hash<9, 12>, Equal<9, 12>> set3;
    tsl::hopscotch_set<const Pancake*, Hash<13, 16>, Equal<13, 16>> set4;
    tsl::hopscotch_set<const Pancake*, Hash<17, 20>, Equal<17, 20>> set5;

    void Add_Node(const Pancake* p) {
      set1.insert(p);
      set2.insert(p);
      set3.insert(p);
      set4.insert(p);
      set5.insert(p);
    }
  };
public:

  typedef std::set<const Pancake*, PancakeFBarSortLowG> set;
  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  //typedef std::unordered_set<const Pancake*, PancakeNeighborHash, PancakeNeighborEqual> hash_set;

  StackArray<Pancake> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  hash_set neighbor_f, neighor_b;
  size_t expansions;
  size_t UB;
  size_t memory;
  size_t expansions_cstar = 0;


  Dibbs() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0) {}


  void expand_node(set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed, std::vector<Pancake>* expansions_in_order = nullptr) {
    const Pancake* next_val = *open.begin();

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(next_val);

    ++expansions;
    ++expansions_cstar;

    closed.insert(next_val);

    if (expansions_in_order) expansions_in_order->push_back(*next_val);

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

  std::tuple<double, size_t, size_t> run_search(Pancake start, Pancake goal, std::vector<Pancake>* expansions_in_order = nullptr) {
    expansions = 0;
    memory = 0;
    auto ptr = storage.push_back(start);
    open_f.insert(ptr);
    open_f_hash.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.insert(ptr);
    open_b_hash.insert(ptr);

    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    int lbmin = 0;
    bool forward = false;
    while (open_f.size() > 0 && open_b.size() > 0 && UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {

      if (lbmin < ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {
        expansions_cstar = 0;
        lbmin = ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0);
      }
      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        break;
      }

      if ((*open_f.begin())->f_bar < (*open_b.begin())->f_bar) {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f);
        forward = open_f.size() < open_b.size();
      }
      else if ((*open_f.begin())->f_bar > (*open_b.begin())->f_bar) {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b);
        forward = open_f.size() < open_b.size();
      }
      else if (forward) {
        //else if (open_f.size() <= open_b.size()) {
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
    std::cout << "|" << " ";
    for (int i = best_b.actions.size() - 1; i >= 0; --i) {
      std::cout << std::to_string(best_b.actions[i]) << " ";
    }
    std::cout << std::endl;
#endif
    if (UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, expansions_cstar);
    }
    else {
      return std::make_tuple((double)UB, expansions, expansions_cstar);
    }
  }

public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal, std::vector<Pancake>* expansions_in_order = nullptr) {
    Dibbs instance;
    return instance.run_search(start, goal, expansions_in_order);
  }
};
