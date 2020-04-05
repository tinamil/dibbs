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

struct double_stacker {

  struct single_stacker {
    std::unordered_map<uint8_t, std::vector<const Pancake*>> single_stack;
    uint8_t begin_g = 0;
    uint8_t end_g = 0;
    size_t total_size = 0;

    void push_back(const Pancake* type) {
      if (single_stack.count(type->g) == 0) {
        single_stack[type->g] = std::vector<const Pancake*>();
      }

      single_stack[type->g].push_back(type);
      total_size += 1;
      if (total_size == 1) {
        begin_g = type->g;
        end_g = type->g + 1;
      }
      if (type->g < begin_g) begin_g = type->g;
      if (type->g >= end_g) end_g = type->g + 1;
    }

    const Pancake* top(uint8_t ask_max_g) {
      if (total_size == 0 || end_g == begin_g) return nullptr;
      for (uint8_t i = std::min<uint8_t>(ask_max_g + 1, end_g); i > begin_g; --i) {
        if (single_stack[i - 1].size() > 0) {
          return single_stack[i - 1].back();
        }
      }
      return nullptr;
    }

    const Pancake* pop_back(uint8_t ask_max_g) {
      if (total_size == 0 || end_g == begin_g) return nullptr;
      for (uint8_t i = std::min<uint8_t>(ask_max_g + 1, end_g); i > begin_g; --i) {
        if (single_stack[i - 1].size() > 0) {
          const Pancake* ret = single_stack[i - 1].back();
          single_stack[i - 1].pop_back();
          total_size -= 1;
          fix_bounds(i - 1);
          return ret;
        }
      }
      return nullptr;
    }

    void fix_bounds(uint8_t i) {
      if (total_size == 0) {
        begin_g = 0;
        end_g = 0;
      }
      else if (single_stack[i].size() == 0 && i == begin_g) {
        for (uint8_t i = begin_g + 1; i < end_g; ++i) {
          if (single_stack[i].size() > 0) {
            begin_g = i;
            break;
          }
        }
      }
      else if (single_stack[i].size() == 0 && i == end_g - 1) {
        for (uint8_t i = end_g - 1; i >= begin_g; --i) {
          if (single_stack[i].size() > 0) {
            end_g = i + 1;
            break;
          }
        }
      }
    }
  };

  std::unordered_map<uint8_t, single_stacker> double_stack;
  uint8_t begin_fbar = 0;
  uint8_t end_fbar = 0;
  size_t total_size = 0;

  void push_back(const Pancake* type) {
    if (double_stack.count(type->f_bar) == 0) {
      double_stack[type->f_bar] = single_stacker{};
    }

    double_stack[type->f_bar].push_back(type);
    total_size += 1;
    if (total_size == 1) {
      begin_fbar = type->f_bar;
      end_fbar = type->f_bar + 1;
    }
    if (type->f_bar < begin_fbar) begin_fbar = type->f_bar;
    if (type->f_bar >= end_fbar) end_fbar = type->f_bar + 1;
  }

  const Pancake* top(uint8_t ask_max_g = 254) {
    if (total_size == 0 || begin_fbar == end_fbar) return nullptr;
    for (uint8_t i = begin_fbar; i < end_fbar; ++i) {
      if (double_stack[i].total_size > 0) {
        const Pancake* ptr = double_stack[i].top(ask_max_g);
        if (ptr != nullptr) return ptr;
      }
    }
    return nullptr;
  }

  const Pancake* pop_back(uint8_t ask_max_g = 254) {
    if (total_size == 0 || begin_fbar == end_fbar) return nullptr;
    for (uint8_t i = begin_fbar; i < end_fbar; ++i) {
      if (double_stack[i].total_size > 0) {
        const Pancake* ptr = double_stack[i].top(ask_max_g);
        if (ptr != nullptr) {
          double_stack[i].pop_back(ask_max_g);
          total_size -= 1;
          fix_bounds(i);
          return ptr;
        }
      }
    }
    return nullptr;
  }

  void fix_bounds(uint8_t i) {
    if (total_size == 0) {
      begin_fbar = 0;
      end_fbar = 0;
    }
    else if (double_stack[i].total_size == 0 && i == begin_fbar) {
      for (uint8_t i = begin_fbar + 1; i < end_fbar; ++i) {
        if (double_stack[i].total_size > 0) {
          begin_fbar = i;
          break;
        }
      }
    }
    else if (double_stack[i].total_size == 0 && i == end_fbar - 1) {
      for (uint8_t i = end_fbar - 1; i >= begin_fbar; --i) {
        if (double_stack[i].total_size > 0) {
          end_fbar = i + 1;
          break;
        }
      }
    }
  }

  bool empty() const {
    return total_size == 0;
  }
};

class DibbsNbs {

  //const std::vector<uint8_t> meeting_point_b = { 20, 12, 13, 9, 8, 7, 11, 19, 2, 1, 3, 10, 6, 5, 4, 14, 15, 16, 17, 18, 20 };
  //const std::vector<uint8_t> meeting_point_f = { 20, 2, 19, 11, 7, 8, 9, 13, 12, 1, 3, 10, 6, 5, 4, 14, 15, 16, 17, 18, 20 };

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FSortHighDuplicate> f_set;
  //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GSortHighDuplicate> g_set;

  StackArray<Pancake> storage;
  double_stacker open_f_fbar;
  double_stacker open_b_fbar;
  //g_set open_f_gset, open_b_gset;
  f_set open_f_fset, open_b_fset;
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

  DibbsNbs() : open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair() {

    while (true) {
      while (!open_f_fset.empty() && closed_f.find(open_f_fset.top()) != closed_f.end()) open_f_fset.pop();
      while (!open_f_fset.empty() && open_f_fset.top()->f <= lbmin) {
        open_f_fbar.push_back(open_f_fset.top());
        //open_f_gset.push(open_f_fset.top());
        open_f_fset.pop();

        while (!open_f_fset.empty() && closed_f.find(open_f_fset.top()) != closed_f.end()) open_f_fset.pop();
      }

      while (!open_b_fset.empty() && closed_b.find(open_b_fset.top()) != closed_b.end()) open_b_fset.pop();
      while (!open_b_fset.empty() && open_b_fset.top()->f <= lbmin) {
        open_b_fbar.push_back(open_b_fset.top());
        //open_b_gset.push(open_b_fset.top());
        open_b_fset.pop();

        while (!open_b_fset.empty() && closed_b.find(open_b_fset.top()) != closed_b.end()) open_b_fset.pop();
      }

      //while (!open_f_gset.empty() && closed_f.find(open_f_gset.top()) != closed_f.end()) open_f_gset.pop();
      //while (!open_b_gset.empty() && closed_b.find(open_b_gset.top()) != closed_b.end()) open_b_gset.pop();
      //while (!open_f_gset.empty() && true /*((open_f_gset.top()->g + open_b_gset.top()->g + EPSILON) <= lbmin*/) {
      //  open_f_fbar.push(open_f_gset.top());
      //  open_f_gset.pop();

      //  while (!open_f_gset.empty() && closed_f.find(open_f_gset.top()) != closed_f.end()) open_f_gset.pop();
      //}
      //while (!open_b_gset.empty() && true /*((open_f_gset.top()->g + open_b_gset.top()->g + EPSILON) <= lbmin*/) {
      //  open_b_fbar.push(open_b_gset.top());
      //  open_b_gset.pop();

      //  while (!open_b_gset.empty() && closed_b.find(open_b_gset.top()) != closed_b.end()) open_b_gset.pop();
      //}

      while (!open_f_fbar.empty() && closed_f.find(open_f_fbar.top()) != closed_f.end()) open_f_fbar.pop_back();
      while (!open_b_fbar.empty() && closed_b.find(open_b_fbar.top()) != closed_b.end()) open_b_fbar.pop_back();

      if (open_f_fset.empty() /*&& open_f_gset.empty()*/ && open_f_fbar.empty()) return std::nullopt;
      else if (open_b_fset.empty() /*&& open_b_gset.empty() */ && open_b_fbar.empty()) return std::nullopt;
      else if (!open_f_fbar.empty() && !open_b_fbar.empty()) {

        const Pancake* top_f = open_f_fbar.top();
        const Pancake* top_b = open_b_fbar.top();

        for (int i = ceil(lbmin / 2.0); i < lbmin; ++i) {
          while (open_f_fbar.top(i) != nullptr && closed_f.find(open_f_fbar.top(i)) != closed_f.end()) open_f_fbar.pop_back(i);
          top_f = open_f_fbar.top(i);
          while (open_b_fbar.top(lbmin - 1 - i) != nullptr && closed_b.find(open_b_fbar.top(lbmin - 1 - i)) != closed_b.end()) open_b_fbar.pop_back(lbmin - 1 - i);
          top_b = open_b_fbar.top(lbmin - 1 - i);
          if (top_f != nullptr && top_b != nullptr && top_f->f_bar + top_b->f_bar <= 2 * lbmin) {
            open_f_fbar.pop_back(i);
            open_b_fbar.pop_back(lbmin - 1 - i);
            assert(top_f->g + top_b->g + 1 <= lbmin);
            return std::make_tuple(top_f, top_b);
          }

          while (open_f_fbar.top(lbmin - 1 - i) != nullptr && closed_f.find(open_f_fbar.top(lbmin - 1 - i)) != closed_f.end()) open_f_fbar.pop_back(lbmin - 1 - i);
          top_f = open_f_fbar.top(lbmin - 1 - i);
          while (open_b_fbar.top(i) != nullptr && closed_b.find(open_b_fbar.top(i)) != closed_b.end()) open_b_fbar.pop_back(i);
          top_b = open_b_fbar.top(i);
          if (top_f != nullptr && top_b != nullptr && top_f->f_bar + top_b->f_bar <= 2 * lbmin) {
            open_f_fbar.pop_back(lbmin - 1 - i);
            open_b_fbar.pop_back(i);
            assert(top_f->g + top_b->g + 1 <= lbmin);
            return std::make_tuple(top_f, top_b);
          }
        }
      }

      size_t min_g = std::numeric_limits<size_t>::max();
      if (!open_f_fbar.empty() && !open_b_fbar.empty()) {
        for (min_g = lbmin + 1; min_g < 255; ++min_g) {
          for (int i = 0; i <= min_g; ++i) {
            const Pancake* top_f = open_f_fbar.top(i);
            const Pancake* top_b = open_b_fbar.top(min_g - i);
            if (top_f != nullptr && top_b != nullptr) {
              min_g = std::max(min_g, (size_t)ceil((top_f->f_bar + top_b->f_bar) / 2.0));
              goto gdone;
            }
          }
        }
      gdone:;
      }
      size_t min_ff = std::numeric_limits<size_t>::max();
      if (!open_f_fset.empty()) min_ff = open_f_fset.top()->f;
      size_t min_fb = std::numeric_limits<size_t>::max();
      if (!open_b_fset.empty()) min_fb = open_b_fset.top()->f;
      lbmin = variadic_min(min_ff, min_fb, min_g);

    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, f_set& waiting, hash_set& closed, const hash_set& other_hash) {
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

      //if (memcmp(new_action.source, meeting_point_f.data(), NUM_PANCAKES + 1) == 0) {
      //  std::cout << "Found a meeting point node for forward";
      //}
      //if (memcmp(new_action.source, meeting_point_b.data(), NUM_PANCAKES + 1) == 0) {
      //  std::cout << "Found a meeting point node for backward";
      //}

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
      waiting.push(ptr);
      auto hash_insertion_result = hash.insert(ptr);
      assert(hash_insertion_result.second);
  }
    return true;
}

  bool expand_node_forward(const Pancake* pancake) {
    return expand_node(pancake, open_f_hash, open_f_fset, closed_f, open_b_hash);
  }

  bool expand_node_backward(const Pancake* pancake) {
    return expand_node(pancake, open_b_hash, open_b_fset, closed_b, open_f_hash);
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
    open_f_fset.push(ptr);
    open_f_hash.insert(ptr);

    ptr = storage.push_back(goal);
    open_b_fset.push(ptr);
    open_b_hash.insert(ptr);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    std::optional<std::tuple<const Pancake*, const Pancake*>> pair;
    while ((pair = select_pair()).has_value())
    {
      if (lbmin >= UB) {
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