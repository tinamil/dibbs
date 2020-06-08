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

typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;


template <typename T, typename THash, typename TEqual>
class triple {

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
  //fbar_set fbset;
  size_t total_size = 0;
  std::vector<std::vector<std::vector<T>>> data;
  triple() {
    data.resize(255);
    for (int i = 0; i < data.size(); ++i) {
      data[i].resize(255);
    }
  }

  size_t query_size(int other_f, int other_delta, uint8_t lbmin) const {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
    //int target_fbar = 2 * lbmin - other->f_bar;
    //int min_fbar = fbset.top()->f_bar;
    for (int target_f = lbmin - max_delta; target_f <= max_f; ++target_f) {
      for (int target_delta = 0; target_delta <= max_delta; ++target_delta) {
        matches += data[target_f][target_delta].size();
      }
    }
    return matches;
  }

  size_t size() const {
    return total_size;
  }

  void push_back(T val) {
    //fbset.push(val);
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

  static decltype(auto) query_pair(triple& front, triple& back, uint32_t lbmin, hash_set& closed_f, hash_set& closed_b)
  {
    uint32_t min = lbmin;
    T min_front = nullptr, min_back = nullptr;
    bool done = false;

    float max_fsize = 0, max_bsize = 0;
    int front_f = -1, front_delta = -1;
    int back_f = -1, back_delta = -1;
    for (int f = 0; f <= lbmin; ++f) {
      for (int delta = 0; delta <= lbmin - f; ++delta) {
        if (front.data[f][delta].size() > 0)
        {
          float bsize = static_cast<float>(back.query_size(f, delta, lbmin)) / front.data[f][delta].size();
          if (bsize > max_bsize) {
            max_bsize = bsize;
            front_f = f;
            front_delta = delta;
          }
        }
        if (back.data[f][delta].size() > 0)
        {
          auto fsize = static_cast<float>(front.query_size(f, delta, lbmin)) / back.data[f][delta].size();

          if (fsize > max_fsize) {
            max_fsize = fsize;
            back_f = f;
            back_delta = delta;
          }
        }
      }
    }
    if (max_fsize >= max_bsize) {
      return std::make_tuple(back_f, back_delta, false);
    }
    else {
      return std::make_tuple(front_f, front_delta, true);
    }
  }
};

class DibbsNbs {

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FSortHighDuplicate> f_set;
  //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GSortHighDuplicate> g_set;

  StackArray<Pancake> storage;
  triple<const Pancake*, PancakeHash, PancakeEqual> open_f_data;
  triple<const Pancake*, PancakeHash, PancakeEqual> open_b_data;
  //g_set open_f_gset, open_b_gset;
  //f_set open_f_fset, open_b_fset;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;
  int expansions_at_cstar = 0;
  int f = 0, d = 0;
  bool dir = true;

#ifdef HISTORY
  Pancake best_f;
  Pancake best_b;
#endif

  DibbsNbs() : open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair() {
    while (true) {

      if (f >= 0 && dir && open_f_data.data[f][d].size() > 0) {
        auto val = open_f_data.data[f][d].back();
        open_f_data.data[f][d].pop_back();
        return std::make_tuple(val, nullptr);
      }
      else if (f >= 0 && open_b_data.data[f][d].size() > 0) {
        auto val = open_b_data.data[f][d].back();
        open_b_data.data[f][d].pop_back();
        return std::make_tuple(nullptr, val);
      }

      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [f1, d1, dir1] = triple<const Pancake*, PancakeHash, PancakeEqual>::query_pair(open_f_data, open_b_data, lbmin, closed_f, closed_b);
        f = f1;
        d = d1;
        dir = dir1;
        if (f == -1 && d == -1) {
          lbmin += 1;
          expansions_at_cstar = 0;
          continue;
        }
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, triple<const Pancake*, PancakeHash, PancakeEqual>& data) {
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
    ++expansions_at_cstar;

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
    expansions_at_cstar = 0;
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
      return std::make_tuple((double)UB, expansions, memory);
    }
    else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions_at_cstar, memory);
  }


public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal) {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};