#pragma once
#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>
#include <queue>
#include <StackArray.h>

#include <windows.h>
#include <Psapi.h>
#include <optional>

constexpr long EPSILON = 1;

template<typename T>
T variadic_min(T val) {
  return val;
}

template<typename T, typename... Ts>
T variadic_min(T val, Ts... other) {
  T other_min = (T)variadic_min(other...);
  return std::min(val, other_min);
}

template<typename T>
T variadic_max(T val) {
  return val;
}

template<typename T, typename... Ts>
T variadic_max(T val, Ts... other) {
  T other_max = (T)variadic_max(other...);
  return std::max(val, other_max);
}

class Pairs {

  struct FBarSortHighDuplicate {
    bool operator()(const Pancake& lhs, const Pancake& rhs) const {
      return operator()(&lhs, &rhs);
    }
    bool operator()(const Pancake* lhs, const Pancake* rhs) const {
      if (lhs->f_bar == rhs->f_bar) {
        return lhs->g > rhs->g;
      }
      else {
        return lhs->f_bar > rhs->f_bar;
      }
    }
  };
  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FSortHighDuplicate> f_set;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GSortHighDuplicate> g_set;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, DeltaSortHighDuplicate> d_set;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FBarSortHighDuplicate> fbar_set;

  g_set gset;
  f_set fset;
  d_set dset;
  fbar_set fbset;

  size_t size_val = 0;

public:
  inline void push_back(const Pancake* val) {
    gset.push(val);
    fset.push(val);
    dset.push(val);
    fbset.push(val);
    size_val += 1;
  }

  inline size_t size() {
    return size_val;
  }

  inline bool empty() {
    return size_val == 0;
  }

  static decltype(auto) query_pair(Pairs& front, Pairs& back, uint32_t lbmin, hash_set& closed_f, hash_set& closed_b) {
    uint32_t min = UINT32_MAX;
    uint32_t type1_min = UINT32_MAX, type2_min = UINT32_MAX, type3_min = UINT32_MAX, type4_min = UINT32_MAX;

    const Pancake* min_front = nullptr, * min_back = nullptr;

    while (closed_f.find(front.gset.top()) != closed_f.end()) {
      front.gset.pop();
    }
    while (closed_f.find(front.fset.top()) != closed_f.end()) {
      front.fset.pop();
    }
    while (closed_f.find(front.dset.top()) != closed_f.end()) {
      front.dset.pop();
    }
    while (closed_f.find(front.fbset.top()) != closed_f.end()) {
      front.fbset.pop();
    }
    while (closed_b.find(back.gset.top()) != closed_b.end()) {
      back.gset.pop();
    }
    while (closed_b.find(back.fset.top()) != closed_b.end()) {
      back.fset.pop();
    }
    while (closed_b.find(back.dset.top()) != closed_b.end()) {
      back.dset.pop();
    }
    while (closed_b.find(back.fbset.top()) != closed_b.end()) {
      back.fbset.pop();
    }

    uint32_t min1 = front.gset.top()->g + back.gset.top()->g + EPSILON;
    uint32_t min2 = front.fset.top()->f + back.dset.top()->g - back.dset.top()->h2;
    uint32_t min3 = front.dset.top()->g - front.dset.top()->h2 + back.fset.top()->f;
    uint32_t min4 = (uint32_t)(ceil((front.fbset.top()->f_bar + back.fbset.top()->f_bar) / 2.0));

    min = variadic_max(min4, min3, min2, min1);

    if (min == min4) {
      min_front = front.fbset.top();
      front.fbset.pop();
      min_back = back.fbset.top();
      back.fbset.pop();
      min = min4;
    }
    else if (min == min3) {
      min_front = front.dset.top();
      front.dset.pop();
      min_back = back.fset.top();
      back.fset.pop();
      min = min3;
    }
    else if (min == min2) {
      min_front = front.fset.top();
      front.fset.pop();
      min_back = back.dset.top();
      back.dset.pop();
      min = min2;
    }
    else if (min == min1) {
      min_front = front.gset.top();
      front.gset.pop();
      min_back = back.gset.top();
      back.gset.pop();
      min = min1;
    }
    front.size_val -= 1;
    back.size_val -= 1;

    return std::make_tuple(min_front, min_back, min);
  }
};

class DibbsNbs {

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;

  StackArray<Pancake> storage;
  Pairs open_f_data, open_b_data;
  //triple open_f_data;
  //triple open_b_data;
  //g_set open_f_gset, open_b_gset;
  //f_set open_f_fset, open_b_fset;
  //d_set open_f_dset, open_b_dset;
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

  std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair() {
    while (true) {
      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [front, back, min] = Pairs::query_pair(open_f_data, open_b_data, lbmin, closed_f, closed_b);
        if (min > lbmin) {
          lbmin = min;
        }
        assert(front != nullptr && back != nullptr);
        return std::make_tuple(front, back);
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, Pairs& data) {
    auto removed = hash.erase(next_val);
    if (removed == 0) return true;
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

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val->apply_action(i);

      auto it_open = other_hash.find(&new_action);
      if (it_open != other_hash.end()) {
        size_t tmp_UB = (size_t)(*it_open)->g + new_action.g;
        if (tmp_UB < UB) {
          UB = tmp_UB;
#ifdef HISTORY
          if (new_action.dir == Direction::forward) {
            best_f = new_action;
            best_b = (**it_open);
          }
          else {
            best_b = new_action;
            best_f = (**it_open);
          }
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
      data.push_back(ptr);
      auto hash_insertion_result = hash.insert(ptr);
      assert(hash_insertion_result.second);
    }
    return true;
  }

  bool expand_node_forward(const Pancake* pancake) {
    return expand_node(pancake, open_f_hash, closed_f, open_b_hash, open_f_data);
  }

  bool expand_node_backward(const Pancake* pancake) {
    return expand_node(pancake, open_b_hash, closed_b, open_f_hash, open_b_data);
  }

  std::tuple<double, size_t, size_t> run_search(Pancake start, Pancake goal)
  {
    if (start == goal) {
      return std::make_tuple(0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    UB = std::numeric_limits<size_t>::max();

    auto ptr = storage.push_back(start);
    open_f_data.push_back(ptr);
    open_f_hash.insert(ptr);

    ptr = storage.push_back(goal);
    open_b_data.push_back(ptr);
    open_b_hash.insert(ptr);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    std::optional<std::tuple<const Pancake*, const Pancake*>> pair;
    while ((pair = select_pair()).has_value())
    {
      if (lbmin >= UB) { //>= for first stop
        finished = true;
        break;
      }

      if (expand_node_forward(std::get<0>(pair.value())) == false) break;
      if (expand_node_backward(std::get<1>(pair.value())) == false) break;
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

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal) {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};