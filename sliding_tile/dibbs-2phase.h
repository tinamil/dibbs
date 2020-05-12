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

typedef std::unordered_set<const SlidingTile*, SlidingTileHash, SlidingTileEqual> hash_set;


template <typename T, typename THash, typename TEqual>
class triple {

public:
  struct FBarDiff {
    bool operator()(T lhs, T rhs) const {
      if (lhs->f_bar == rhs->f_bar) {
        return lhs->g < rhs->g;
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
  struct MinH {
    bool operator()(T lhs, T rhs) const {
      if (lhs->h == rhs->h) {
        return lhs->f_bar > rhs->f_bar;
      }
      else {
        return lhs->h > rhs->h;
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
    size_t size() {
      return data.size();
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

  T query(T other, uint8_t lbmin) const {
    int max_delta = lbmin - other->f;
    int max_f = lbmin - other->delta;
    for (int target_f = lbmin - max_delta; target_f <= max_f; ++target_f) {
      for (int target_delta = 0; target_delta <= max_delta; ++target_delta) {
        auto val = data[target_f][target_delta];
        if (val.size() > 0) {
          return val.back();
        }
      }
    }
    return nullptr;
  }

  size_t query_size(int other_f, int other_delta, uint8_t lbmin) const {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
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

  decltype(auto) pop_min_fbar(uint32_t lbmin) {
    for (int f = 0; f <= lbmin; ++f) {
      for (int delta = 0; delta <= lbmin - f; ++delta) {
        if (data[f][delta].size() > 0) {
          return std::make_tuple(f, delta);
        }
      }
    }
    return std::make_tuple(-1, -1);
  }

  static decltype(auto) query_phase2(triple& front, triple& back, uint32_t lbmin, hash_set& closed_f, hash_set& closed_b)
  {
    float min_fsize = 0, min_bsize = 0;
    for (int f = 0; f <= lbmin; ++f) {
      for (int delta = 0; delta <= lbmin - f; ++delta) {
        min_fsize += front.data[f][delta].size();
        min_bsize += back.data[f][delta].size();
      }
    }
    return std::make_tuple(min_fsize, min_bsize);
  }

  static decltype(auto) query_pair(triple& front, triple& back, uint32_t lbmin, hash_set& closed_f, hash_set& closed_b, bool phase2)
  {
    T min_front = nullptr, min_back = nullptr;

    //size_t min_fsize = SIZE_MAX, min_bsize = SIZE_MAX;
    size_t min_fsize = 0, min_bsize = 0;
    float max_fsize = 0, max_bsize = 0;
    int front_f = -1, front_delta = -1;
    int back_f = -1, back_delta = -1;
    for (int f = 0; f <= lbmin; ++f) {
      for (int delta = 0; delta <= lbmin - f; ++delta) {
        if (front.data[f][delta].size() > 0)
        {
          if (phase2) {
            if (front.data[f][delta].size() > 0 /*<= min_fsize*/ && back.query_size(f, delta, lbmin) > 0) {
              min_fsize += front.data[f][delta].size();
              front_f = f;
              front_delta = delta;
            }
          }
          else {
            float bsize = static_cast<float>(back.query_size(f, delta, lbmin)) / front.data[f][delta].size();
            if (bsize > max_bsize) {
              max_bsize = bsize;
              front_f = f;
              front_delta = delta;
            }
          }
        }
        if (back.data[f][delta].size() > 0)
        {
          if (phase2) {
            if (back.data[f][delta].size() > 0 /*<= min_bsize*/ && front.query_size(f, delta, lbmin) > 0) {
              min_bsize += back.data[f][delta].size();
              back_f = f;
              back_delta = delta;
            }
          }
          else {
            auto fsize = static_cast<float>(front.query_size(f, delta, lbmin)) / back.data[f][delta].size();
            if (fsize > max_fsize) {
              max_fsize = fsize;
              back_f = f;
              back_delta = delta;
            }
          }
        }
      }
    }
    if (phase2) {
      if (min_bsize == 0 || (min_fsize > 0 && min_fsize < min_bsize)) {
        return std::make_tuple(front_f, front_delta, true);
      }
      else {
        return std::make_tuple(back_f, back_delta, false);
      }
    }
    else {
      if (max_fsize >= max_bsize) {
        return std::make_tuple(back_f, back_delta, false);
      }
      else {
        return std::make_tuple(front_f, front_delta, true);
      }
    }
  }
};

class DibbsNbs {

  typedef std::unordered_set<const SlidingTile*, SlidingTileHash, SlidingTileEqual> hash_set;

  StackArray<SlidingTile> storage;
  triple<const SlidingTile*, SlidingTileHash, SlidingTileEqual> open_f_data;
  triple<const SlidingTile*, SlidingTileHash, SlidingTileEqual> open_b_data;

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
  SlidingTile best_f;
  SlidingTile best_b;
#endif

  DibbsNbs() : open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  std::optional<std::tuple<const SlidingTile*, const SlidingTile*>> select_pair() {
    /*  if (UB == SIZE_MAX) {
        if ((open_f_data.fbset.top()->f_bar == open_b_data.fbset.top()->f_bar && open_f_data.fbset.size() < open_b_data.fbset.size()) || open_f_data.fbset.top()->f_bar < open_b_data.fbset.top()->f_bar) {
          auto p = open_f_data.fbset.top();
          open_f_data.fbset.pop();
          return std::make_tuple(p, nullptr);
        }
        else {
          auto p = open_b_data.fbset.top();
          open_b_data.fbset.pop();
          return std::make_tuple(nullptr, p);
        }
      }*/
    while (true) {

      /*if (f >= 0 && dir && open_f_data.data[f][d].size() > 0) {
        auto val = open_f_data.data[f][d].back();
        open_f_data.data[f][d].pop_back();
        return std::make_tuple(val, nullptr);
      }
      else if (f >= 0 && open_b_data.data[f][d].size() > 0) {
        auto val = open_b_data.data[f][d].back();
        open_b_data.data[f][d].pop_back();
        return std::make_tuple(nullptr, val);
      }*/
      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [f1, d1, dir1] = triple<const SlidingTile*, SlidingTileHash, SlidingTileEqual>::query_pair(open_f_data, open_b_data, lbmin, closed_f, closed_b, false /*UB < SIZE_MAX*/);
        f = f1;
        d = d1;
        dir = dir1;
        if (f == -1 && d == -1) {
          lbmin += 1;
          expansions_at_cstar = 0;
          continue;
        }
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
      }
    }
  }

  bool expand_node(const SlidingTile* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, triple<const SlidingTile*, SlidingTileHash, SlidingTileEqual>& data) {
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

    std::optional<std::tuple<const SlidingTile*, const SlidingTile*>> pair;
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
    else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
  }


public:

  static std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal) {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};