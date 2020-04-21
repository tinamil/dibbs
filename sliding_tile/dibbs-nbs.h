#pragma once
#pragma once

#include "sliding_tile.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>
#include <queue>
#include "StackArray.h"

#define NOMINMAX
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


template <typename T, typename THash, typename TEqual>
class Pairs {

public:
  struct FBarDiff {
    bool operator()(T lhs, T rhs) const {
      if (lhs->f_bar == rhs->f_bar) {
        return lhs->hdiff > rhs->hdiff;
      }
      else {
        return lhs->f_bar > rhs->f_bar;
      }
    }
  };
  struct GDiff {
    bool operator()(T lhs, T rhs) const {
      if (lhs->g == rhs->g) {
        return lhs->hdiff > rhs->hdiff;
      }
      else {
        return lhs->g > rhs->g;
      }
    }
  };

  template<typename Sort>
  struct queue_wrapper {
    std::priority_queue<T, std::vector<T>, Sort> data;
    void push(T val) {
      data.push(val);
    }
    bool empty() const {
      return data.empty();
    }
    T top() const {
      return data.top();
    }
    void pop() {
      data.pop();
    }
    decltype(auto) begin() {
      throw std::runtime_error("Cannot iterate queue");
    }
    decltype(auto) end() {
      throw std::runtime_error("Cannot iterate queue");
    }
    void erase(T val) {
      assert(val == data.top());
      pop();
    }
  };
  typedef std::unordered_set<T, THash, TEqual> hash_set;
  typedef queue_wrapper<FBarDiff> fbar_set;
  fbar_set fbset;
  size_t total_size = 0;
  std::vector<std::vector<std::vector<T>>> data;
  Pairs() {
    data.resize(255);
    for (int i = 0; i < data.size(); ++i) {
      data[i].resize(255);
    }
  }

  T query(T other, uint8_t lbmin) const {
    int max_delta = lbmin - other->f;
    int max_f = lbmin - other->delta;
    int target_fbar = 2 * lbmin - other->f_bar;
    int min_fbar = fbset.top()->f_bar;
    for (int target_f = lbmin - max_delta; target_f <= max_f; ++target_f) {
      for (int target_delta = std::max(min_fbar - target_f, 0); target_delta <= max_delta; ++target_delta) {
        auto val = data[target_f][target_delta];
        if (val.size() > 0) {
          return val.back();
        }
      }
    }
    return nullptr;
  }

  size_t size() const {
    return total_size;
  }

  void push_back(T val) {
    fbset.push(val);
    data[val->f][val->delta].push_back(val);
    total_size += 1;
  }

  void pop_back(T val) {
    data[val->f][val->delta].pop_back();
    total_size -= 1;
  }

  bool empty() const {
    return size() == 0;
  }

  static decltype(auto) query_pair(Pairs& front, Pairs& back, uint32_t lbmin, hash_set& closed_f, hash_set& closed_b)
  {
    uint32_t min = lbmin;
    T min_front = nullptr, min_back = nullptr;
    bool done = false;
    while (!front.fbset.empty() && closed_f.contains(front.fbset.top())) {
      front.fbset.pop();
    }
    while (!back.fbset.empty() && closed_b.contains(back.fbset.top())) {
      back.fbset.pop();
    }

    if (back.fbset.top()->f_bar <= front.fbset.top()->f_bar) {
      min_back = back.fbset.top();
      min_front = front.query(min_back, lbmin);
      while (min_front != nullptr && closed_f.contains(min_front)) {
        front.pop_back(min_front);
        min_front = front.query(min_back, lbmin);
      }
      if (min_front != nullptr && min_front->f_bar + min_back->f_bar <= 2 * lbmin) {
        back.fbset.pop();
        front.pop_back(min_front);
      }
      else {
        min_front = nullptr;
        min_back = nullptr;
      }
    }
    else {
      min_front = front.fbset.top();
      min_back = back.query(min_front, lbmin);
      while (min_back != nullptr && closed_b.contains(min_back)) {
        back.pop_back(min_back);
        min_back = back.query(min_front, lbmin);
      }
      if (min_back != nullptr && min_front->f_bar + min_back->f_bar <= 2 * lbmin) {
        front.fbset.pop();
        back.pop_back(min_back);
      }
      else {
        min_front = nullptr;
        min_back = nullptr;
      }
    }
    return std::make_tuple(min_front, min_back, min);
  }
};

class DibbsNbs {

  typedef Pairs<const SlidingTile*, SlidingTileHash, SlidingTileEqual> TypedPairs;
  typedef std::unordered_set<const SlidingTile*, SlidingTileHash, SlidingTileEqual> hash_set;

  StackArray<SlidingTile> storage;
  TypedPairs open_f_data, open_b_data;
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
  SlidingTile best_f;
  SlidingTile best_b;
#endif

  std::optional<std::tuple<const SlidingTile*, const SlidingTile*>> select_pair() {
    while (true) {
      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [front, back, min] = TypedPairs::query_pair(open_f_data, open_b_data, lbmin, closed_f, closed_b);
        if (front == nullptr && back == nullptr) {
          lbmin += 1;
          continue;
        }
        /*if (min > lbmin) {
          lbmin = min;
        }*/
        return std::make_tuple(front, back);
      }
    }
  }

  bool expand_node(const SlidingTile* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, TypedPairs& data) {
    if (next_val == nullptr) return true;
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

    for (int i = 1, stop = next_val->num_actions_available(); i <= stop; ++i) {
      SlidingTile new_action = next_val->apply_action(i);

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

  bool expand_node_forward(const SlidingTile* SlidingTile) {
    return expand_node(SlidingTile, open_f_hash, closed_f, open_b_hash, open_f_data);
  }

  bool expand_node_backward(const SlidingTile* SlidingTile) {
    return expand_node(SlidingTile, open_b_hash, closed_b, open_f_hash, open_b_data);
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
    open_f_data.push_back(ptr);
    open_f_hash.insert(ptr);

    ptr = storage.push_back(goal);
    open_b_data.push_back(ptr);
    open_b_hash.insert(ptr);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    std::optional<std::tuple<const SlidingTile*, const SlidingTile*>> pair;
    while ((pair = select_pair()).has_value())
    {
      if (lbmin > UB) { //>= for first stop
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

  static std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal) {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};