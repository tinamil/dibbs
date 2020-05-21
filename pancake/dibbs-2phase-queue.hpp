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

bool compare_one_off(const Pancake* lhs, const Pancake* rhs) {
  for (int i = NUM_PANCAKES; i >= 1; --i) {
    if (lhs->source[i] != rhs->source[i]) {
      //Trigger the reverse lookup once we find the first non-match
      int l_val = i, r_val = 1;
      while (l_val >= 1) {
        if (lhs->source[l_val--] != rhs->source[r_val++]) {
          return false;
        }
      }
      return true;
    }
  }
  //This only happens if the pancakes were identical
  std::cerr << "Identical pancakes in one-off comparison";
  return true;
};

struct MinH {
  bool operator()(Pancake* lhs, Pancake* rhs) const {
    if (lhs->h == rhs->h) {
      return lhs->f_bar > rhs->f_bar;
    }
    else {
      return lhs->h > rhs->h;
    }
  }
};

template <typename T, typename THash, typename TEqual, typename TLess>
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
  std::vector<std::vector<std::set<T, TLess>>> data;
  triple() {
    data.resize(255);
    for (int i = 0; i < data.size(); ++i) {
      data[i].resize(255);
    }
  }

  decltype(auto) query(int other_f, int other_delta, uint8_t lbmin, int glim) const {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
    for (int target_f = 0; target_f <= max_f; ++target_f) {
      for (int target_delta = 0; target_delta <= max_delta; ++target_delta) {
        if (data[target_f][target_delta].size() > 0) {
          if ((*data[target_f][target_delta].rbegin())->g <= glim) {
            return std::make_tuple(target_f, target_delta);
          }
        }
      }
    }
    return std::make_tuple(-1, -1);
  }

  size_t query_size(int other_f, int other_delta, uint8_t lbmin, int glim) const {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
    for (int target_f = 0; target_f <= max_f; ++target_f) {
      for (int target_delta = 0; target_delta <= max_delta; ++target_delta) {
        //if (target_f + target_delta >= lbmin) continue;
        matches += data[target_f][target_delta].size();
        for (auto x : data[target_f][target_delta]) {
          if (x->g > glim) matches -= 1;
          else break;
        }
      }
    }
    return matches;
  }

  size_t size() const {
    return total_size;
  }

  void push_back(T val) {
    //fbset.push(val);
    data[val->f][val->delta].insert(val);
    total_size += 1;
  }

  void pop_back(T val) {
    data[val->f][val->delta].erase(val);
    total_size -= 1;
  }

  bool empty() const {
    return size() == 0;
  }

  static decltype(auto) query_pair(triple& front, triple& back, uint32_t lbmin, uint32_t UB, hash_set& closed_f, hash_set& closed_b, bool phase2)
  {
    T min_front = nullptr, min_back = nullptr;

    static bool started_phase2 = false;
    static bool direction_forward = false;
    //if (!phase2) started_phase2 = false;
    //else {
    //  if (!started_phase2) {
    //    for (int fbar = 0; fbar < lbmin; ++fbar) {
    //      for (int f = fbar; f >= 0; --f) {
    //        int delta = fbar - f;
    //        if (front.data[f][delta].size() > 0) {
    //          return std::make_tuple(f, delta, true);
    //        }
    //        else if (back.data[f][delta].size() > 0) {
    //          return std::make_tuple(f, delta, false);
    //        }
    //      }
    //    }
    //    //If we've reached here then both front and back are up to fbar = lbmin - 1
    //    direction_forward = front.size() < back.size();
    //  }
    //  started_phase2 = true;
    //  for (int f = lbmin; f >= 0; --f) {
    //    int delta = lbmin - f;
    //    if (direction_forward && front.data[f][delta].size() > 0) {
    //      return std::make_tuple(f, delta, true);
    //    }
    //    else if (!direction_forward && back.data[f][delta].size() > 0) {
    //      return std::make_tuple(f, delta, false);
    //    }
    //  }
    //  //If we've reached here then we've cleared the chosen direction
    //  return std::make_tuple(-1, -1, direction_forward);
    //}
    const Pancake* ret_val = nullptr;


    //static bool forward_dir = false;
    for (int fbar = 0; fbar < lbmin; ++fbar) {
      for (int f = 0; f <= fbar; ++f) {
        int delta = fbar - f;
        if (front.data[f][delta].size() > 0) {
          //forward_dir = front.size() < back.size();
          return std::make_tuple(f, delta, (int)lbmin, true, ret_val);
        }
        else if (back.data[f][delta].size() > 0) {
          //forward_dir = front.size() < back.size();
          return std::make_tuple(f, delta, (int)lbmin, false, ret_val);
        }
      }
    }

    //for (int fbar = 0; fbar <= lbmin; ++fbar) {
    int fbar = lbmin;
    for (int f = fbar; f >= 0; --f) {
      int delta = fbar - f;
      for (auto front_val : front.data[f][delta]) {
        int max_delta = lbmin - f;
        int max_f = lbmin - delta;
        for (int target_f = max_f; target_f >= 0; --target_f) {
          for (int target_delta = 0; target_delta <= max_delta; ++target_delta) {
            for (auto back_val : back.data[target_f][target_delta]) {
              if (front_val->g + back_val->g + 1 == lbmin && compare_one_off(front_val, back_val)) {
                return std::make_tuple(f, delta, (int)front_val->g, true, front_val);
              }
            }
          }
        }
      }
    }

    /*size_t min_fsize = SIZE_MAX, min_bsize = SIZE_MAX;
    int front_f = -1, front_delta = -1;
    int back_f = -1, back_delta = -1;
    for (int f = 0; f <= lbmin; ++f) {
      int delta = lbmin - f;
      if (front.data[f][delta].size() > 0 && front.data[f][delta].size() < min_fsize) {
        front_f = f;
        front_delta = delta;
        min_fsize = front.data[f][delta].size();
      }
      if (back.data[f][delta].size() > 0 && back.data[f][delta].size() < min_bsize) {
        back_f = f;
        back_delta = delta;
        min_bsize = back.data[f][delta].size();
      }
    }
    if (min_bsize == 0 && min_fsize == 0)
      return std::make_tuple(-1, -1, direction_forward);
    else if (min_bsize <= min_fsize)
      return std::make_tuple(back_f, back_delta, false);
    else
      return std::make_tuple(front_f, front_delta, true);*/

      //size_t min_fsize = SIZE_MAX, min_bsize = SIZE_MAX;

    size_t min_fsize = 0, min_bsize = 0;
    float max_fsize = 0, max_bsize = 0;
    int front_f = -1, front_delta = -1, front_g = -1;
    int back_f = -1, back_delta = -1, back_g = -1;
    for (int f = lbmin; f >= 0; --f) {
      int delta = lbmin - f;
      //for (int delta = 0; delta <= lbmin - f; ++delta) {
      if (front.data[f][delta].size() > 0)
      {
        int glim = lbmin - 1 - (*front.data[f][delta].rbegin())->g;
        float fratio = static_cast<float>(back.query_size(f, delta, lbmin, glim)) / front.data[f][delta].size();

        if (/*forward_dir && */fratio > max_fsize) {
          max_fsize = fratio;
          front_f = f;
          front_delta = delta;
          auto [bf, bd] = back.query(f, delta, lbmin, glim);
          front_g = lbmin - 1 - (*back.data[bf][bd].rbegin())->g;
        }
        //else if (!forward_dir && fratio > 0 && (1. / fratio) > max_bsize) {
        //  max_bsize = 1. / fratio;
        //  auto [bf, bd] = back.query(f, delta, lbmin, glim);
        //  back_f = bf;
        //  back_delta = bd;
        //  back_g = glim;
        //}
      }
      if (back.data[f][delta].size() > 0)
      {
        int glim = lbmin - 1 - (*back.data[f][delta].rbegin())->g;
        auto bratio = static_cast<float>(front.query_size(f, delta, lbmin, glim)) / back.data[f][delta].size();
        if (/*!forward_dir && */bratio > max_bsize) {
          max_bsize = bratio;
          back_f = f;
          back_delta = delta;
          auto [ff, fd] = front.query(f, delta, lbmin, glim);
          back_g = lbmin - 1 - (*front.data[ff][fd].rbegin())->g;
        }
        /*else if (forward_dir && bratio > 0 && (1. / bratio) > max_fsize) {
          max_fsize = 1. / bratio;
          auto [ff, fd] = front.query(f, delta, lbmin, glim);
          front_f = ff;
          front_delta = fd;
          front_g = glim;
        }*/
      }
      //}
    }
    if (max_bsize >= max_fsize/*!forward_dir*/) {
      return std::make_tuple(back_f, back_delta, back_g, false, ret_val);
    }
    else {
      return std::make_tuple(front_f, front_delta, front_g, true, ret_val);
    }
  }
};

class DibbsNbs {

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;

  typedef triple<const Pancake*, PancakeHash, PancakeEqual, PancakeGSortHigh> pancake_triple;
  StackArray<Pancake> storage;

  pancake_triple open_f_data;
  pancake_triple open_b_data;

  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;
  int expansions_at_cstar = 0;
  int expansions_after_UB = 0;

#ifdef HISTORY
  Pancake best_f;
  Pancake best_b;
#endif

  DibbsNbs() : open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair() {
    while (true) {

      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [f, d, g, dir, ret_val] = pancake_triple::query_pair(open_f_data, open_b_data, lbmin, UB, closed_f, closed_b, false/*UB == lbmin + 1*/);

        if (f == -1 && d == -1) {
          lbmin += 1;
          expansions_at_cstar = 0;
          return std::make_tuple(nullptr, nullptr);
        }
        if (f >= 0 && dir && open_f_data.data[f][d].size() > 0) {
          if (ret_val != nullptr) {
            open_f_data.pop_back(ret_val);
            return std::make_tuple(ret_val, nullptr);
          }
          for (auto val : open_f_data.data[f][d]) {
            if (val->g <= g) {
              open_f_data.pop_back(val);
              return std::make_tuple(val, nullptr);
            }
          }
        }
        else if (f >= 0 && open_b_data.data[f][d].size() > 0) {
          if (ret_val != nullptr) {
            open_b_data.pop_back(ret_val);
            return std::make_tuple(nullptr, ret_val);
          }
          for (auto val : open_b_data.data[f][d]) {
            if (val->g <= g) {
              open_b_data.pop_back(val);
              return std::make_tuple(nullptr, val);
            }
          }
        }
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, pancake_triple& data) {
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
    ++expansions_after_UB;

    for (int i = 2, j = NUM_PANCAKES; i <= j && UB > lbmin; ++i) {
      Pancake new_action = next_val->apply_action(i);

      auto it_open = other_hash.find(&new_action);
      if (it_open != other_hash.end()) {
        size_t tmp_UB = (size_t)(*it_open)->g + new_action.g;
        if (tmp_UB < UB) {
          if (UB == SIZE_MAX) {
            expansions_after_UB = 0;
          }
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


      for (auto x : other_hash) {
        if (ptr->g + x->g + 1 < UB && compare_one_off(ptr, x)) {
          expand_node(ptr, hash, closed, other_hash, data);
        }
      }
    }
    return true;
  }

  bool expand_node_forward(const Pancake* pancake) {
    return expand_node(pancake, open_f_hash, closed_f, open_b_hash, open_f_data);
  }

  bool expand_node_backward(const Pancake* pancake) {
    return expand_node(pancake, open_b_hash, closed_b, open_f_hash, open_b_data);
  }

  std::tuple<double, size_t, size_t, size_t, size_t> run_search(Pancake start, Pancake goal)
  {
    if (start == goal) {
      return std::make_tuple(0, 0, 0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    expansions_at_cstar = 0;
    expansions_after_UB = 0;
    UB = std::numeric_limits<size_t>::max();
    Pancake* ptr;

    /*StackArray<Pancake> phase1_storage;
    std::priority_queue<Pancake*, std::vector<Pancake*>, MinH> phase1;
    phase1.push(phase1_storage.push_back(start));
    hash_set phase1_closed;
    while (phase1.size() > 0 && UB == SIZE_MAX) {
      Pancake* next_val = phase1.top();
      phase1.pop();

      auto it = phase1_closed.find(next_val);
      if (it != phase1_closed.end())
      {
        continue;
      }
      phase1_closed.insert(next_val);

      ++expansions;

      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
        Pancake new_action = next_val->apply_action(i);
        it = phase1_closed.find(&new_action);
        if (it == phase1_closed.end()) {
          phase1.push(phase1_storage.push_back(new_action));
        }
        if (new_action == goal) {
          UB = new_action.g;
          break;
        }
      }
    }

    while (!phase1.empty()) {
      ptr = storage.push_back(*phase1.top());
      phase1.pop();
      open_f_data.push_back(ptr);
      open_f_hash.insert(ptr);
    }
    phase1_storage.clear();
    phase1_closed.clear();*/

    ptr = storage.push_back(start);
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
      return std::make_tuple((double)UB, expansions, memory, expansions_at_cstar, expansions_after_UB);
    }
    else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory, expansions_at_cstar, expansions_after_UB);
  }


public:

  static std::tuple<double, size_t, size_t, size_t, size_t> search(Pancake start, Pancake goal) {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};