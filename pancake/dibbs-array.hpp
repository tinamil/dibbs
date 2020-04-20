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

template<typename T>
T variadic_max(T val) {
  return val;
}

template<typename T, typename... Ts>
T variadic_max(T val, Ts... other) {
  T other_max = (T)variadic_max(other...);
  return std::max(val, other_max);
}

template <typename T, typename Val, typename Storage>
struct single_stacker {
  std::unordered_map<uint32_t, Storage> single_stack;
  uint32_t begin_val = 0;
  uint32_t end_val = 0;
  size_t total_size = 0;

  void push_back(const T type) {
    uint32_t v = static_cast<uint32_t>(Val{}(type));
    if (single_stack.count(v) == 0) {
      single_stack[v] = Storage();
    }

    single_stack[v].push_back(type);
    total_size += 1;
    if (total_size == 1) {
      begin_val = v;
      end_val = v + 1;
    }
    if (v < begin_val) begin_val = v;
    if (v >= end_val) end_val = v + 1;
  }

  std::optional<uint32_t> get_index(uint32_t val_limit) const {
    if (total_size == 0 || end_val == begin_val) return std::nullopt;
    for (uint32_t i = std::min<uint32_t>(val_limit + 1, end_val); i > begin_val; --i) {
      if (std::get<1>(*single_stack.find(i - 1)).size() > 0) {
        return i - 1;
      }
    }
    return std::nullopt;
  }

  const T top(const uint32_t ask_max_val) const {
    std::optional<uint32_t> val = get_index(ask_max_val);
    if (val.has_value() == false) return nullptr;
    else {
      const uint32_t index = val.value();
      Storage map = std::get<1>(*single_stack.find(index));
      return map.top(ask_max_val);
    }
  }

  size_t size() const {
    return total_size;
  }

  bool pop_back() {
    for (uint32_t i = std::min<uint32_t>(ask_max_val + 1, end_val); i > begin_val; --i) {
      if (single_stack[i - 1].size() > 0) {
        single_stack[i - 1].pop_back();
        total_size -= 1;
        fix_bounds(i - 1);
        return true;
      }
    }
    return false;
  }

  void fix_bounds(const uint32_t changed_index) {
    if (total_size == 0) {
      begin_val = 0;
      end_val = 0;
    }
    else if (single_stack[changed_index].size() == 0 && changed_index == begin_val) {
      for (uint32_t i = begin_val + 1; i < end_val; ++i) {
        if (single_stack[i].size() > 0) {
          begin_val = i;
          break;
        }
      }
    }
    else if (single_stack[changed_index].size() == 0 && changed_index == end_val - 1) {
      for (uint32_t i = end_val - 1; i >= begin_val; --i) {
        if (single_stack[i].size() > 0) {
          end_val = i + 1;
          break;
        }
      }
    }
  }

  bool empty() const {
    return total_size == 0;
  }
};

//struct double_stacker {
//
//  struct g_return {
//    uint8_t operator()(Pancake* p) {
//      return p->g;
//    }
//  };
//
//  struct h1_return {
//    uint8_t operator()(Pancake* p) {
//      return p->g;
//    }
//  };
//
//  typedef single_stacker<const Pancake*, g_return, std::vector<const Pancake*>> g_stack;
//  std::unordered_map<uint8_t, g_stack> double_stack;
//  uint8_t begin_fbar = 0;
//  uint8_t end_fbar = 0;
//  size_t total_size = 0;
//
//  void push_back(const Pancake* type) {
//    if (double_stack.count(type->f_bar) == 0) {
//      double_stack[type->f_bar] = g_stack{};
//    }
//
//    double_stack[type->f_bar].push_back(type);
//    total_size += 1;
//    if (total_size == 1) {
//      begin_fbar = type->f_bar;
//      end_fbar = type->f_bar + 1;
//    }
//    if (type->f_bar < begin_fbar) begin_fbar = type->f_bar;
//    if (type->f_bar >= end_fbar) end_fbar = type->f_bar + 1;
//  }
//
//  const Pancake* top(uint8_t ask_max_g = 254) {
//    if (total_size == 0 || begin_fbar == end_fbar) return nullptr;
//    for (uint8_t i = begin_fbar; i < end_fbar; ++i) {
//      if (double_stack[i].total_size > 0) {
//        const Pancake* ptr = double_stack[i].top(ask_max_g);
//        if (ptr != nullptr) return ptr;
//      }
//    }
//    return nullptr;
//  }
//
//  bool pop_back(uint8_t ask_max_g = 254) {
//    if (total_size == 0 || begin_fbar == end_fbar) return;
//    for (uint8_t i = begin_fbar; i < end_fbar; ++i) {
//      if (double_stack[i].total_size > 0) {
//        if (double_stack[i].pop_back(ask_max_g))
//        {
//          total_size -= 1;
//          fix_bounds(i);
//          return true;
//        }
//      }
//    }
//    return false;
//  }
//
//  void fix_bounds(uint8_t i) {
//    if (total_size == 0) {
//      begin_fbar = 0;
//      end_fbar = 0;
//    }
//    else if (double_stack[i].total_size == 0 && i == begin_fbar) {
//      for (uint8_t i = begin_fbar + 1; i < end_fbar; ++i) {
//        if (double_stack[i].total_size > 0) {
//          begin_fbar = i;
//          break;
//        }
//      }
//    }
//    else if (double_stack[i].total_size == 0 && i == end_fbar - 1) {
//      for (uint8_t i = end_fbar - 1; i >= begin_fbar; --i) {
//        if (double_stack[i].total_size > 0) {
//          end_fbar = i + 1;
//          break;
//        }
//      }
//    }
//  }
//
//  bool empty() const {
//    return total_size == 0;
//  }
//};

struct g_return {
  uint8_t operator()(const Pancake* p) {
    return p->g;
  }
};
struct h1_return {
  uint8_t operator()(const Pancake* p) {
    return p->h;
  }
};
struct h2_return {
  uint8_t operator()(const Pancake* p) {
    return p->h2;
  }
};

//typedef single_stacker<const Pancake*, g_return, StackArray<const Pancake*>> g_stack;
//typedef single_stacker<const Pancake*, h1_return, g_stack> h1_stack;
//typedef single_stacker<const Pancake*, h2_return, h1_stack> h2_stack;

struct triple {
  size_t total_size = 0;
  std::vector<std::vector<std::vector<std::vector<const Pancake*>>>> data;

  triple() {
    data.resize(255);
    for (int i = 0; i < data.size(); ++i) {
      data[i].resize(255);
      for (int j = 0; j < data.size(); ++j) {
        data[i][j].resize(255);
      }
    }
  }

  const Pancake* query(uint8_t g_lim, uint8_t h_lim, uint8_t h2_lim) const {
    //for (uint8_t g = 0; g <= g_lim; ++g) {
    //  for (uint8_t h = 0; h <= h_lim; ++h) {
    //    for (uint8_t h2 = 0; h2 <= h2_lim; ++h2) {
    auto val = data[g_lim][h_lim][h2_lim];
    if (val.size() > 0) {
      return val.back();
    }
    //     }
    //  }
  //}
    return nullptr;
  }

  size_t size() const {
    return total_size;
  }

  void push_back(const Pancake* val) {
    data[val->g][val->h][val->h2].push_back(val);
    total_size += 1;
  }

  void pop_back(const Pancake* val) {
    data[val->g][val->h][val->h2].pop_back();
    total_size -= 1;
  }

  bool empty() const {
    return size() == 0;
  }

  static decltype(auto) query_pair(const triple& front, const triple& back, uint32_t lbmin) {
    uint32_t min = UINT32_MAX;
    uint32_t type1_min = UINT32_MAX, type2_min = UINT32_MAX, type3_min = UINT32_MAX;

    const Pancake* min_front = nullptr, * min_back = nullptr;
    for (uint8_t gf = 0; gf <= 254; ++gf) {
      for (uint8_t hff = 0; hff <= 254; ++hff) {
        if (gf + hff > lbmin) continue;
        for (uint8_t hbf = 0; hbf <= 254; ++hbf) {
          auto front_pancake = front.query(gf, hff, hbf);
          if (front_pancake == nullptr) continue;
          for (uint8_t gb = 0; gb <= 254; ++gb) {
            for (uint8_t hbb = 0; hbb <= 254; ++hbb) {
              if (gb + hbb > lbmin) continue;
              for (uint8_t hfb = 0; hfb <= 254; ++hfb) {
                auto back_pancake = back.query(gb, hbb, hfb);
                if (back_pancake == nullptr) continue;
                if (back_pancake->f > lbmin) continue;
                uint32_t type1_val = front_pancake->g + back_pancake->g + abs(front_pancake->h - back_pancake->h2);
                uint32_t type2_val = front_pancake->g + back_pancake->g + EPSILON;
                uint32_t type3_val = front_pancake->g + back_pancake->g + abs(back_pancake->h - front_pancake->h2);
                auto possible_min = variadic_max(type1_val, type2_val, type3_val);
                if (possible_min <= lbmin) {
                  assert(front_pancake->g + front_pancake->h + back_pancake->g - back_pancake->h <= lbmin);
                  return std::make_tuple(front_pancake, back_pancake, possible_min);
                }
                else if (possible_min < min) {
                  //min_front = front_pancake;
                  //min_back = back_pancake;
                  min = possible_min;
                }
              }
            }
          }
        }
      }
    }
    return std::make_tuple(min_front, min_back, min);
  }
};

class DibbsNbs {

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FSortHighDuplicate> f_set;
  //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GSortHighDuplicate> g_set;

  StackArray<Pancake> storage;
  triple open_f_data;
  triple open_b_data;
  //g_set open_f_gset, open_b_gset;
  //f_set open_f_fset, open_b_fset;
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
      if (open_f_data.empty()) return std::nullopt;
      else if (open_b_data.empty()) return std::nullopt;
      else {
        auto [front, back, min] = triple::query_pair(open_f_data, open_b_data, lbmin);
        if (min > lbmin) {
          //lbmin = min;
        }
        if (front == nullptr || back == nullptr) {
          lbmin += 1;
          continue;
        }
        assert(front != nullptr && back != nullptr);
        open_f_data.pop_back(front);
        open_b_data.pop_back(back);
        return std::make_tuple(front, back);
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, triple& data) {
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

//    class DibbsNbs {
//
//    //const std::vector<uint8_t> meeting_point_b = { 20, 12, 13, 9, 8, 7, 11, 19, 2, 1, 3, 10, 6, 5, 4, 14, 15, 16, 17, 18, 20 };
//    //const std::vector<uint8_t> meeting_point_f = { 20, 2, 19, 11, 7, 8, 9, 13, 12, 1, 3, 10, 6, 5, 4, 14, 15, 16, 17, 18, 20 };
//
//    typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
//    typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, FSortHighDuplicate> f_set;
//    //typedef std::priority_queue<const Pancake*, std::vector<const Pancake*>, GSortHighDuplicate> g_set;
//
//    StackArray<Pancake> storage;
//    double_stacker open_f_fbar;
//    double_stacker open_b_fbar;
//    //g_set open_f_gset, open_b_gset;
//    f_set open_f_fset, open_b_fset;
//    hash_set open_f_hash, open_b_hash;
//    hash_set closed_f, closed_b;
//    size_t expansions;
//    size_t UB;
//    size_t lbmin;
//    size_t memory;
//
//#ifdef HISTORY
//    Pancake best_f;
//    Pancake best_b;
//#endif
//
//    DibbsNbs() : open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}
//
//    std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair() {
//
//      while (true) {
//        while (!open_f_fset.empty() && closed_f.find(open_f_fset.top()) != closed_f.end()) open_f_fset.pop();
//        while (!open_f_fset.empty() && open_f_fset.top()->f <= lbmin) {
//          open_f_fbar.push_back(open_f_fset.top());
//          //open_f_gset.push(open_f_fset.top());
//          open_f_fset.pop();
//
//          while (!open_f_fset.empty() && closed_f.find(open_f_fset.top()) != closed_f.end()) open_f_fset.pop();
//        }
//
//        while (!open_b_fset.empty() && closed_b.find(open_b_fset.top()) != closed_b.end()) open_b_fset.pop();
//        while (!open_b_fset.empty() && open_b_fset.top()->f <= lbmin) {
//          open_b_fbar.push_back(open_b_fset.top());
//          //open_b_gset.push(open_b_fset.top());
//          open_b_fset.pop();
//
//          while (!open_b_fset.empty() && closed_b.find(open_b_fset.top()) != closed_b.end()) open_b_fset.pop();
//        }
//
//        //while (!open_f_gset.empty() && closed_f.find(open_f_gset.top()) != closed_f.end()) open_f_gset.pop();
//        //while (!open_b_gset.empty() && closed_b.find(open_b_gset.top()) != closed_b.end()) open_b_gset.pop();
//        //while (!open_f_gset.empty() && true /*((open_f_gset.top()->g + open_b_gset.top()->g + EPSILON) <= lbmin*/) {
//        //  open_f_fbar.push(open_f_gset.top());
//        //  open_f_gset.pop();
//
//        //  while (!open_f_gset.empty() && closed_f.find(open_f_gset.top()) != closed_f.end()) open_f_gset.pop();
//        //}
//        //while (!open_b_gset.empty() && true /*((open_f_gset.top()->g + open_b_gset.top()->g + EPSILON) <= lbmin*/) {
//        //  open_b_fbar.push(open_b_gset.top());
//        //  open_b_gset.pop();
//
//        //  while (!open_b_gset.empty() && closed_b.find(open_b_gset.top()) != closed_b.end()) open_b_gset.pop();
//        //}
//
//        while (!open_f_fbar.empty() && closed_f.find(open_f_fbar.top()) != closed_f.end()) open_f_fbar.pop_back();
//        while (!open_b_fbar.empty() && closed_b.find(open_b_fbar.top()) != closed_b.end()) open_b_fbar.pop_back();
//
//        if (open_f_fset.empty() /*&& open_f_gset.empty()*/ && open_f_fbar.empty()) return std::nullopt;
//        else if (open_b_fset.empty() /*&& open_b_gset.empty() */ && open_b_fbar.empty()) return std::nullopt;
//        else if (!open_f_fbar.empty() && !open_b_fbar.empty()) {
//
//          const Pancake* top_f = open_f_fbar.top();
//          const Pancake* top_b = open_b_fbar.top();
//
//          for (int i = ceil(lbmin / 2.0); i < lbmin; ++i) {
//            while (open_f_fbar.top(i) != nullptr && closed_f.find(open_f_fbar.top(i)) != closed_f.end()) open_f_fbar.pop_back(i);
//            top_f = open_f_fbar.top(i);
//            while (open_b_fbar.top(lbmin - 1 - i) != nullptr && closed_b.find(open_b_fbar.top(lbmin - 1 - i)) != closed_b.end()) open_b_fbar.pop_back(lbmin - 1 - i);
//            top_b = open_b_fbar.top(lbmin - 1 - i);
//            if (top_f != nullptr && top_b != nullptr && top_f->f_bar + top_b->f_bar <= 2 * lbmin) {
//              open_f_fbar.pop_back(i);
//              open_b_fbar.pop_back(lbmin - 1 - i);
//              assert(top_f->g + top_b->g + 1 <= lbmin);
//              return std::make_tuple(top_f, top_b);
//            }
//
//            while (open_f_fbar.top(lbmin - 1 - i) != nullptr && closed_f.find(open_f_fbar.top(lbmin - 1 - i)) != closed_f.end()) open_f_fbar.pop_back(lbmin - 1 - i);
//            top_f = open_f_fbar.top(lbmin - 1 - i);
//            while (open_b_fbar.top(i) != nullptr && closed_b.find(open_b_fbar.top(i)) != closed_b.end()) open_b_fbar.pop_back(i);
//            top_b = open_b_fbar.top(i);
//            if (top_f != nullptr && top_b != nullptr && top_f->f_bar + top_b->f_bar <= 2 * lbmin) {
//              open_f_fbar.pop_back(lbmin - 1 - i);
//              open_b_fbar.pop_back(i);
//              assert(top_f->g + top_b->g + 1 <= lbmin);
//              return std::make_tuple(top_f, top_b);
//            }
//          }
//        }
//
//        size_t min_g = std::numeric_limits<size_t>::max();
//        if (!open_f_fbar.empty() && !open_b_fbar.empty()) {
//          for (min_g = lbmin + 1; min_g < 255; ++min_g) {
//            for (int i = 0; i <= min_g; ++i) {
//              const Pancake* top_f = open_f_fbar.top(i);
//              const Pancake* top_b = open_b_fbar.top(min_g - i);
//              if (top_f != nullptr && top_b != nullptr) {
//                min_g = std::max(min_g, (size_t)ceil((top_f->f_bar + top_b->f_bar) / 2.0));
//                goto gdone;
//              }
//            }
//          }
//        gdone:;
//        }
//        size_t min_ff = std::numeric_limits<size_t>::max();
//        if (!open_f_fset.empty()) min_ff = open_f_fset.top()->f;
//        size_t min_fb = std::numeric_limits<size_t>::max();
//        if (!open_b_fset.empty()) min_fb = open_b_fset.top()->f;
//        lbmin = variadic_min(min_ff, min_fb, min_g);
//
//      }
//    }
//
//    bool expand_node(const Pancake* next_val, hash_set& hash, f_set& waiting, hash_set& closed, const hash_set& other_hash) {
//      auto removed = hash.erase(next_val);
//      if (removed == 0) return true;
//      assert(removed == 1);
//
//      auto insertion_result = closed.insert(next_val);
//      assert(insertion_result.second);
//
//      PROCESS_MEMORY_COUNTERS memCounter;
//      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
//      assert(result);
//      memory = std::max(memory, memCounter.PagefileUsage);
//      if (memCounter.PagefileUsage > MEM_LIMIT) {
//        return false;
//      }
//
//      ++expansions;
//
//      for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
//        Pancake new_action = next_val->apply_action(i);
//
//        //if (memcmp(new_action.source, meeting_point_f.data(), NUM_PANCAKES + 1) == 0) {
//        //  std::cout << "Found a meeting point node for forward";
//        //}
//        //if (memcmp(new_action.source, meeting_point_b.data(), NUM_PANCAKES + 1) == 0) {
//        //  std::cout << "Found a meeting point node for backward";
//        //}
//
//        auto it_open = other_hash.find(&new_action);
//        if (it_open != other_hash.end()) {
//          size_t tmp_UB = (size_t)(*it_open)->g + new_action.g;
//          if (tmp_UB < UB) {
//            UB = tmp_UB;
//#ifdef HISTORY
//            if (new_action.dir == Direction::forward) {
//              best_f = new_action;
//              best_b = (**it_open);
//            }
//            else {
//              best_b = new_action;
//              best_f = (**it_open);
//            }
//#endif
//          }
//        }
//        auto it_closed = closed.find(&new_action);
//        if (it_closed != closed.end() && (*it_closed)->g <= new_action.g) continue;
//        else if (it_closed != closed.end()) {
//          closed.erase(it_closed);
//          assert(false);
//        }
//
//        it_open = hash.find(&new_action);
//        if (it_open != hash.end() && (*it_open)->g <= new_action.g) continue;
//        else if (it_open != hash.end() && (*it_open)->g > new_action.g) {
//          hash.erase(it_open);
//        }
//
//        auto ptr = storage.push_back(new_action);
//        waiting.push(ptr);
//        auto hash_insertion_result = hash.insert(ptr);
//        assert(hash_insertion_result.second);
//      }
//      return true;
//    }
//
//    bool expand_node_forward(const Pancake* pancake) {
//      return expand_node(pancake, open_f_hash, open_f_fset, closed_f, open_b_hash);
//    }
//
//    bool expand_node_backward(const Pancake* pancake) {
//      return expand_node(pancake, open_b_hash, open_b_fset, closed_b, open_f_hash);
//    }
//
//    std::tuple<double, size_t, size_t> run_search(Pancake start, Pancake goal)
//    {
//      if (start == goal) {
//        return std::make_tuple(0, 0, 0);
//      }
//      memory = 0;
//      expansions = 0;
//      UB = std::numeric_limits<size_t>::max();
//
//      auto ptr = storage.push_back(start);
//      open_f_fset.push(ptr);
//      open_f_hash.insert(ptr);
//
//      ptr = storage.push_back(goal);
//      open_b_fset.push(ptr);
//      open_b_hash.insert(ptr);
//
//      lbmin = std::max(1ui8, std::max(start.h, goal.h));
//
//      bool finished = false;
//      std::optional<std::tuple<const Pancake*, const Pancake*>> pair;
//      while ((pair = select_pair()).has_value())
//      {
//        if (lbmin >= UB) {
//          finished = true;
//          break;
//        }
//
//        if (expand_node_forward(std::get<0>(pair.value())) == false) break;
//        if (expand_node_backward(std::get<1>(pair.value())) == false) break;
//      }
//
//      if (finished) {
//#ifdef HISTORY
//        std::cout << "\nSolution: ";
//        for (int i = 0; i < best_f.actions.size(); ++i) {
//          std::cout << std::to_string(best_f.actions[i]) << " ";
//        }
//        std::cout << "|" << " ";
//        for (int i = best_b.actions.size() - 1; i >= 0; --i) {
//          std::cout << std::to_string(best_b.actions[i]) << " ";
//        }
//        std::cout << "\n";
//#endif 
//        return std::make_tuple(UB, expansions, memory);
//      }
//      else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
//    }
//
//
//    public:
//
//      static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal) {
//        DibbsNbs instance;
//        auto result = instance.run_search(start, goal);
//        return result;
//      }
//    };