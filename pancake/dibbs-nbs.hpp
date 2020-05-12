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
#include <exception>

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
    size_t size() const {
      return data.size();
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
  template<typename Sort>
  struct set_wrapper {
    std::set<T, Sort> data;
    void push(T val) {
      data.insert(val);
    }
    bool empty() const {
      return data.empty();
    }
    size_t size() const {
      return data.size();
    }
    T top() const {
      return (*data.begin());
    }
    void pop() {
      data.erase(data.begin());
    }
    void erase(T val) {
      data.erase(val);
    }

    decltype(auto) begin() {
      return data.begin();
    }
    decltype(auto) end() {
      return data.end();
    }
  };
  typedef std::unordered_set<T, THash, TEqual> hash_set;
  typedef queue_wrapper<FBarDiff> fbar_set;
  typedef queue_wrapper<GDiff> g_set;
  typedef set_wrapper<PancakeFBarSortLowG> fbar_backup;
  typedef set_wrapper<PancakeGSortLow> g_backup;


  fbar_set fbset;
  fbar_set fbset2;
  g_set gset;
  //g_backup gset2;


public:
  inline void push_back(T val) {
    if (val->hdiff > EPSILON) {
      fbset.push(val);
      //gset2.push(val);
    }
    else {
      gset.push(val);
      fbset2.push(val);
    }
  }

  inline bool empty() {
    return (fbset.empty() /*|| gset2.empty()*/) && (gset.empty() || fbset2.empty());
  }

  template<typename U>
  static std::tuple<bool, T, T> pop_pair(U& front, U& back, const hash_set& closed_f, const hash_set& closed_b) {
    if (closed_f.find(front.top()) != closed_f.end()) {
      front.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    else if (closed_b.find(back.top()) != closed_b.end()) {
      back.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    auto min_front = front.top();
    auto min_back = back.top();
    front.pop();
    back.pop();
    return std::make_tuple(true, min_front, min_back);
  }

  template<typename U>
  static std::tuple<bool, T, T> pop_minfbar(U& front, U& back, const hash_set& closed_f, const hash_set& closed_b) {
    if (closed_f.find(front.top()) != closed_f.end()) {
      front.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    else if (closed_b.find(back.top()) != closed_b.end()) {
      back.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    auto min_front = front.top();
    auto min_back = back.top();
    if (min_front->f_bar < min_back->f_bar) {
      front.pop();
      return std::make_tuple(true, min_front, nullptr);
    }
    else {
      back.pop();
      return std::make_tuple(true, nullptr, min_back);
    }
  }

  template<typename U>
  static std::tuple<bool, T, T> pop_ming(U& front, U& back, const hash_set& closed_f, const hash_set& closed_b) {
    if (closed_f.find(front.top()) != closed_f.end()) {
      front.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    else if (closed_b.find(back.top()) != closed_b.end()) {
      back.pop();
      return std::make_tuple(false, nullptr, nullptr);
    }
    auto min_front = front.top();
    auto min_back = back.top();
    if (min_front->g < min_back->g) {
      front.pop();
      return std::make_tuple(true, min_front, nullptr);
    }
    else {
      back.pop();
      return std::make_tuple(true, nullptr, min_back);
    }
  }

  static decltype(auto) query_pair(
    Pairs<T, THash, TEqual>& front,
    Pairs<T, THash, TEqual>& back,
    uint32_t lbmin,
    const hash_set& closed_f,
    const hash_set& closed_b)
  {
    uint32_t min = lbmin;
    T min_front = nullptr, min_back = nullptr;
    bool done = false;
    while (!done) {
      if (!front.fbset.empty() && !back.fbset.empty() && (front.fbset.top()->f_bar + back.fbset.top()->f_bar) <= (2 * min)) {
        auto [success, f, b] = pop_minfbar(front.fbset, back.fbset, closed_f, closed_b);
        done = success;
        min_front = f;
        min_back = b;
        continue;
      }
      if (!front.gset.empty() && !back.gset.empty() && (front.gset.top()->g + back.gset.top()->g + EPSILON) <= min) {
        auto [success, f, b] = pop_minfbar(front.gset, back.gset, closed_f, closed_b);
        done = success;
        min_front = f;
        min_back = b;
        continue;
      }
      if (!front.fbset.empty() && !back.fbset2.empty() && (front.fbset.top()->f_bar + back.fbset2.top()->f_bar) <= (2 * min)) {
        auto [success, f, b] = pop_minfbar(front.fbset, back.fbset2, closed_f, closed_b);
        done = success;
        min_front = f;
        min_back = b;
        continue;
      }
      if (!front.fbset2.empty() && !back.fbset.empty() && (front.fbset2.top()->f_bar + back.fbset.top()->f_bar) <= (2 * min)) {
        auto [success, f, b] = pop_minfbar(front.fbset2, back.fbset, closed_f, closed_b);
        done = success;
        min_front = f;
        min_back = b;
        continue;
      }

      done = true;
    }
    return std::make_tuple(min_front, min_back, min);
  }
};

class DibbsNbs {

  typedef Pairs<const Pancake*, PancakeHash, PancakeEqual> TypedPairs;
  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;

  StackArray<Pancake> storage;
  TypedPairs open_f_data, open_b_data;
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
        auto [front, back, min] = TypedPairs::query_pair(open_f_data, open_b_data, lbmin, closed_f, closed_b);
        //if (min > lbmin) {
        //  lbmin = min;
        //}
        if (front == nullptr && back == nullptr) {
          lbmin += 1;
          continue;
        }
        return std::make_tuple(front, back);
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, TypedPairs& data) {
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