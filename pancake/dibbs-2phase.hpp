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
#include "hash_table.h"

#include <windows.h>
#include <Psapi.h>
#include <optional>

constexpr long EPSILON = 1;
constexpr bool LATE_CLEANUP = false;
#define GSORT true

#define DIBBS_NBS "1phase"

//Doesn't work right now, don't enable without fixing
constexpr bool FORWARD_COMMIT = false;

//static bool compare_one_off(const Pancake* lhs, const Pancake* rhs) {
//  for (int i = NUM_PANCAKES; i >= 1; --i) {
//    if (lhs->source[i] != rhs->source[i]) {
//      //Trigger the reverse lookup once we find the first non-match
//      int l_val = i, r_val = 1;
//      while (l_val >= 1) {
//        if (lhs->source[l_val--] != rhs->source[r_val++]) {
//          return false;
//        }
//      }
//      return true;
//    }
//  }
//  //This only happens if the pancakes were identical
//  assert(false);
//  std::cerr << "Identical pancakes in one-off comparison";
//  return true;
//};

//inline size_t update_hash_value(const uint8_t seq[], int i, size_t hash_value)
//{
//  unsigned char  p1, pi, pi1;
//
//  assert((1 <= i) && (i <= NUM_PANCAKES));
//  p1 = seq[1];               assert((1 <= p1) && (p1 <= NUM_PANCAKES));
//  pi = seq[i];               assert((1 <= pi) && (pi <= NUM_PANCAKES));
//  if (i < NUM_PANCAKES)
//    pi1 = seq[i + 1];
//  else
//    return(hash_value);     // The hash_value does not change when the entire sequence is flipped.
//
//  if (hash_value > hash_table::hash(pi, pi1))
//    hash_value = hash_value - hash_table::hash(pi, pi1);
//  else
//    hash_value = hash_value - hash_table::hash(pi, pi1) + INT_MAX;
//  hash_value = (hash_value + hash_table::hash(p1, pi1)) % INT_MAX;
//
//  return hash_value;
//}

template <typename T, typename THash, typename TEqual, typename TLess>
class triple
{
public:
  
  typedef std::unordered_set<T, THash, TEqual> hash_set;

  size_t total_size = 0;

  #if GSORT
  std::vector<std::vector<std::set<T, TLess>>> data;
  #else
  std::vector<std::vector<std::vector<T>>> data;
  #endif

  triple()
  {
    data.resize(255);
    for(int i = 0; i < data.size(); ++i)
    {
      data[i].resize(255);
    }
  }

  decltype(auto) query(int other_f, int other_delta, uint8_t lbmin, int glim) const
  {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
    for(int target_f = 0; target_f <= max_f; ++target_f)
    {
      for(int target_delta = 0; target_delta <= max_delta; ++target_delta)
      {
        if(data[target_f][target_delta].size() > 0)
        {
          #if GSORT
          if((*data[target_f][target_delta].rbegin())->g <= glim)
          {
            return std::make_tuple(target_f, target_delta);
          }
          #else 
          return std::make_tuple(target_f, target_delta);
          #endif
        }
      }
    }
    return std::make_tuple(-1, -1);
  }

  size_t query_size(int other_f, int other_delta, uint8_t lbmin, int glim) const
  {
    size_t matches = 0;
    int max_delta = lbmin - other_f;
    int max_f = lbmin - other_delta;
    for(int target_f = 0; target_f <= max_f; ++target_f)
    {
      for(int target_delta = 0; target_delta <= max_delta; ++target_delta)
      {
        matches += data[target_f][target_delta].size();
        #if GSORT
        for(auto x = data[target_f][target_delta].begin(), stop = data[target_f][target_delta].end(); x != stop; ++x)
        {
          if((*x)->g > glim) matches -= 1;
          else break;
        }
        #endif
      }
    }
    return matches;
  }

  size_t size() const
  {
    return total_size;
  }

  void push_back(T val)
  {
    #if GSORT
    data[val->f][val->delta].insert(val);
    #else
    data[val->f][val->delta].push_back(val);
    #endif
    total_size += 1;
  }

  void pop_back(T val)
  {
    #if GSORT
    data[val->f][val->delta].erase(val);
    #else 
    data[val->f][val->delta].pop_back();
    #endif
    total_size -= 1;
  }

  void erase(T val)
  {
    std::vector<T>& ref = data[val->f][val->delta];
    for(auto x = ref.begin(); x != ref.end(); ++x)
    {
      if(PancakeEqual{}(*x, val))
      {
        ref.erase(x);
        total_size -= 1;
        return;
      }
    }
  }

  bool empty() const
  {
    return size() == 0;
  }


  static decltype(auto) query_pair(triple& front, triple& back, int lbmin, size_t UB, hash_set& closed_f, hash_set& closed_b)
  {
    T min_front = nullptr, min_back = nullptr;

    static bool forward_dir = false;
    T ret_val = nullptr;
    if constexpr(LATE_CLEANUP)
    {
      if((size_t)lbmin + 1 >= UB)
      {
        for(int fbar = 0; fbar < lbmin; ++fbar)
        {
          for(int f = 0; f <= fbar; ++f)
          {
            int delta = fbar - f;
            if(back.data[f][delta].size() > 0)
            {
              return std::make_tuple(f, delta, lbmin, false, ret_val);
            }
            else if(front.data[f][delta].size() > 0)
            {
              return std::make_tuple(f, delta, lbmin, true, ret_val);
            }
          }
        }
      }
    }
    if constexpr(FORWARD_COMMIT)
    {
      if(UB == (size_t)lbmin + 1)
      {
        size_t f_count_a = 0, f_count_b = 0, b_count_a = 0, b_count_b = 0;
        for(int f = 0; f <= UB; ++f)
        {
          for(int delta = 0; delta <= UB - f; ++delta)
          {
            if(front.data[f][delta].size() > 0)
            {
              f_count_a += front.data[f][delta].size();
              f_count_b += back.query_size(f, delta, lbmin);
            }
            if(back.data[f][delta].size() > 0)
            {
              b_count_a += back.data[f][delta].size();
              b_count_b += front.query_size(f, delta, lbmin);
            }
          }
        }
        float f_ratio = (float)f_count_b / f_count_a;
        float b_ratio = (float)b_count_b / b_count_a;
        forward_dir = f_ratio < b_ratio;
      }
    }

    size_t min_fsize = 0, min_bsize = 0;
    float max_fsize = 0, max_bsize = 0;
    int front_f = -1, front_delta = -1, front_g = -1;
    int back_f = -1, back_delta = -1, back_g = -1;

    for(int f = 0; f <= lbmin; ++f)
    {
      for(int delta = 0; delta <= lbmin - f; ++delta)
      {
        if((((size_t)lbmin + 1 < UB) || (!LATE_CLEANUP || front.size() < back.size())) && front.data[f][delta].size() > 0)
        {
          int glim = lbmin - 1 - (*front.data[f][delta].rbegin())->g;
          float fratio = static_cast<float>(back.query_size(f, delta, lbmin, glim)) / front.data[f][delta].size();

          if((!FORWARD_COMMIT || forward_dir) && fratio > max_fsize)
          {
            max_fsize = fratio;
            front_f = f;
            front_delta = delta;
            auto [bf, bd] = back.query(f, delta, lbmin, glim);
            front_g = lbmin - 1 - (*back.data[bf][bd].rbegin())->g;
          }
          else if(FORWARD_COMMIT && !forward_dir && fratio > 0 && (1. / fratio) > max_bsize)
          {
            max_bsize = 1. / fratio;
            auto [bf, bd] = back.query(f, delta, lbmin, glim);
            back_f = bf;
            back_delta = bd;
            back_g = glim;
          }
        }
        if((((size_t)lbmin + 1 < UB) || (!LATE_CLEANUP || front.size() >= back.size())) && back.data[f][delta].size() > 0)
        {
          int glim = lbmin - 1 - (*back.data[f][delta].rbegin())->g;
          auto bratio = static_cast<float>(front.query_size(f, delta, lbmin, glim)) / back.data[f][delta].size();
          if((!FORWARD_COMMIT || !forward_dir) && bratio > max_bsize)
          {
            max_bsize = bratio;
            back_f = f;
            back_delta = delta;
            auto [ff, fd] = front.query(f, delta, lbmin, glim);
            back_g = lbmin - 1 - (*front.data[ff][fd].rbegin())->g;
          }
          else if(FORWARD_COMMIT && forward_dir && bratio > 0 && (1. / bratio) > max_fsize)
          {
            max_fsize = 1. / bratio;
            auto [ff, fd] = front.query(f, delta, lbmin, glim);
            front_f = ff;
            front_delta = fd;
            front_g = glim;
          }
        }
      }
    }
    if((FORWARD_COMMIT && !forward_dir) || (!FORWARD_COMMIT && max_bsize >= max_fsize))
    {
      if(back_f == -1) forward_dir = front.size() < back.size();
      return std::make_tuple(back_f, back_delta, back_g, false, ret_val);
    }
    else
    {
      if(front_f == -1) forward_dir = front.size() < back.size();
      return std::make_tuple(front_f, front_delta, front_g, true, ret_val);
    }
  }
};

class DibbsNbs
{

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

  std::optional<std::tuple<const Pancake*, const Pancake*>> select_pair()
  {
    while(true)
    {
      if(open_f_data.empty()) return std::nullopt;
      else if(open_b_data.empty()) return std::nullopt;
      else
      {
        auto [f, d, g, dir, ret_val] = pancake_triple::query_pair(open_f_data, open_b_data, lbmin, UB, closed_f, closed_b);

        if(f == -1 && d == -1)
        {
          lbmin += 1;
          expansions_at_cstar = 0;
          return std::make_tuple(nullptr, nullptr);
        }
        if(f >= 0 && dir && open_f_data.data[f][d].size() > 0)
        {
          if(ret_val) return std::make_tuple(ret_val, nullptr);
          #if GSORT
          if(ret_val != nullptr)
          {
            open_f_data.pop_back(ret_val);
            return std::make_tuple(ret_val, nullptr);
          }
          for(auto val : open_f_data.data[f][d])
          {
            if(val->g <= g)
            {
              open_f_data.pop_back(val);
              return std::make_tuple(val, nullptr);
            }
          }
          #else
          auto val = open_f_data.data[f][d].back();
          open_f_data.pop_back(val);
          return std::make_tuple(val, nullptr);
          #endif
        }
        else if(f >= 0 && open_b_data.data[f][d].size() > 0)
        {
          #if GSORT
          if(ret_val != nullptr)
          {
            open_b_data.pop_back(ret_val);
            return std::make_tuple(nullptr, ret_val);
          }
          for(auto val : open_b_data.data[f][d])
          {
            if(val->g <= g)
            {
              open_b_data.pop_back(val);
              return std::make_tuple(nullptr, val);
            }
          }
          #else
          auto val = open_b_data.data[f][d].back();
          open_b_data.pop_back(val);
          return std::make_tuple(nullptr, val);
          #endif
        }
      }
    }
  }

  bool expand_node(const Pancake* next_val, hash_set& hash, hash_set& closed, const hash_set& other_hash, pancake_triple& data)
  {
    if(next_val == nullptr) return true;
    auto removed = hash.erase(next_val);
    if(removed == 0) return true;
    assert(removed == 1);

    auto insertion_result = closed.insert(next_val);
    assert(insertion_result.second);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    memory = std::max(memory, memCounter.PagefileUsage);
    if(memCounter.PagefileUsage > MEM_LIMIT)
    {
      return false;
    }

    ++expansions;
    ++expansions_at_cstar;
    ++expansions_after_UB;

    size_t starting_hash = hash_table::hash(next_val->source);
    for(int i = 2, j = NUM_PANCAKES; i <= j && UB > lbmin; ++i)
    {
      Pancake new_action = next_val->apply_action(i);

      auto it_open = other_hash.find(&new_action);
      if(it_open != other_hash.end())
      {
        size_t tmp_UB = (size_t)(*it_open)->g + new_action.g;
        if(tmp_UB < UB)
        {
          expansions_after_UB = 0;
          UB = tmp_UB;
          if(UB == lbmin) return true;
          #ifdef HISTORY
          if(new_action.dir == Direction::forward)
          {
            best_f = new_action;
            best_b = (**it_open);
          }
          else
          {
            best_b = new_action;
            best_f = (**it_open);
          }
          #endif
        }
      }
      auto it_closed = closed.find(&new_action);
      if(it_closed != closed.end() && (*it_closed)->g <= new_action.g) continue;
      else if(it_closed != closed.end())
      {
        closed.erase(it_closed);
        assert(false);
      }

      it_open = hash.find(&new_action);
      if(it_open != hash.end() && (*it_open)->g <= new_action.g) continue;
      else if(it_open != hash.end() && (*it_open)->g > new_action.g)
      {
        hash.erase(it_open);
      }

      const Pancake* ptr = storage.push_back(new_action);
      data.push_back(ptr);
      auto hash_insertion_result = hash.insert(ptr);
      assert(hash_insertion_result.second);
    }
    return true;
  }

  bool expand_node_forward(const Pancake* pancake)
  {
    return expand_node(pancake, open_f_hash, closed_f, open_b_hash, open_f_data);
  }

  bool expand_node_backward(const Pancake* pancake)
  {
    return expand_node(pancake, open_b_hash, closed_b, open_f_hash, open_b_data);
  }

  std::tuple<double, size_t, size_t, size_t, size_t> run_search(Pancake start, Pancake goal)
  {
    if(start == goal)
    {
      return std::make_tuple(0, 0, 0, 0, 0);
    }
    memory = 0;
    expansions = 0;
    expansions_at_cstar = 0;
    expansions_after_UB = 0;
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
    while((pair = select_pair()).has_value())
    {
      if(lbmin >= UB)
      { //>= for first stop
        finished = true;
        break;
      }

      if(expand_node_forward(std::get<0>(pair.value())) == false) break;
      if(expand_node_backward(std::get<1>(pair.value())) == false) break;
    }

    if(finished)
    {
      #ifdef HISTORY
      std::cout << "\nSolution: ";
      for(int i = 0; i < best_f.actions.size(); ++i)
      {
        std::cout << std::to_string(best_f.actions[i]) << " ";
      }
      std::cout << "|" << " ";
      for(int i = best_b.actions.size() - 1; i >= 0; --i)
      {
        std::cout << std::to_string(best_b.actions[i]) << " ";
      }
      std::cout << "\n";
      #endif 
      return std::make_tuple((double)UB, expansions, memory, expansions_at_cstar, expansions_after_UB);
    }
    else return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory, expansions_at_cstar, expansions_after_UB);
  }


public:

  static std::tuple<double, size_t, size_t, size_t, size_t> search(Pancake start, Pancake goal)
  {
    DibbsNbs instance;
    auto result = instance.run_search(start, goal);
    return result;
  }
};