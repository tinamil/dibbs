#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include <StackArray.h>
#include <tsl/hopscotch_map.h>
#include <tsl/hopscotch_set.h>
#include <vector>
#include "hash_table.h"

#include <windows.h>
#include <Psapi.h>
#include "ftf-pancake.h"

#define FTF_PANCAKE "FTF_Pancake"

class FTF_Dibbs
{
public:

  struct FTFPancakeFSortHighG
  {
    bool operator()(const FTF_Pancake& lhs, const FTF_Pancake& rhs) const
    {
      return operator()(&lhs, &rhs);
    }
    bool operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const
    {
      int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
      if(cmp == 0)
      {
        return false;
      }
      else if(lhs->f == rhs->f)
      {
        if(lhs->g == rhs->g)
          return cmp < 0;
        else
          return lhs->g > rhs->g;
      }
      else
      {
        return lhs->f < rhs->f;
      }
    }
  };

  struct FTFPancakeHash
  {
    inline std::size_t operator() (const FTF_Pancake& x) const
    {
      return operator()(&x);
    }
    inline std::size_t operator() (const FTF_Pancake* x) const
    {
      return SuperFastHash(x->source + 1, NUM_PANCAKES);
    }
  };

  struct FTFPancakeEqual
  {
    inline bool operator() (const FTF_Pancake* x, const FTF_Pancake* y) const
    {
      return memcmp(x->source, y->source, NUM_PANCAKES + 1) == 0;
    }
    inline bool operator() (const FTF_Pancake x, const FTF_Pancake y) const
    {
      return x == y;
    }
  };

  typedef std::set<const FTF_Pancake*, FTFPancakeFSortHighG> set;
  typedef std::unordered_set<const FTF_Pancake*, FTFPancakeHash, FTFPancakeEqual> hash_set;
  //typedef std::unordered_set<const Pancake*, PancakeNeighborHash, PancakeNeighborEqual> hash_set;

  StackArray<FTF_Pancake> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  ftf_matchstructure forward_index, backward_index;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;



  FTF_Dibbs() : open_f(), open_b(), closed_f(), closed_b(), forward_index(), backward_index(),
    open_f_hash(), open_b_hash(), expansions(0), UB(0), lbmin(0), memory(0)
  {

  }

  void expand_node(set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed,
                   ftf_matchstructure& my_index,
                   ftf_matchstructure& other_index)
  {
    const FTF_Pancake* next_val = *open.begin();

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(next_val);
    my_index.erase(next_val);

    ++expansions;

    closed.insert(next_val);

    for(int i = 2, j = NUM_PANCAKES; i <= j; ++i)
    {
      FTF_Pancake new_action = next_val->apply_action(i, other_index);

      if(new_action.f > UB)
      {
        continue;
      }

      auto it_closed = closed.find(&new_action);
      if(it_closed == closed.end())
      {

        auto it_other = other_open.find(&new_action);
        if(it_other != other_open.end())
        {
          #ifdef HISTORY
          if((*it_other)->g + new_action.g < UB)
          {
            if(new_action.dir == Direction::forward)
            {
              best_f = new_action;
              best_b = **it_other;
            }
            else
            {
              best_f = **it_other;
              best_b = new_action;
            }
          }
          #endif  
          size_t combined = (size_t)(*it_other)->g + new_action.g;
          if(combined < UB)
          {
            UB = combined;
          }
        }
        auto it_open = open_hash.find(&new_action);
        if(it_open != open_hash.end())
        {
          if((*it_open)->g <= new_action.g)
          {
            continue;
          }
          else
          {
            open.erase(&**it_open);
            open_hash.erase(it_open);
          }
        }

        auto ptr = storage.push_back(new_action);
        open.insert(ptr);
        open_hash.insert(ptr);
        my_index.insert(ptr);
      }
    }
  }

  #ifdef HISTORY
  Pancake best_f, best_b;
  #endif

  std::tuple<double, size_t, size_t> run_search(FTF_Pancake start, FTF_Pancake goal, std::vector<FTF_Pancake>* expansions_in_order = nullptr)
  {
    expansions = 0;
    memory = 0;
    auto ptr = storage.push_back(start);
    forward_index.insert(ptr);
    ptr->h = ptr->f = forward_index.match(&goal);
    goal.h = goal.f = ptr->h;
    open_f.insert(ptr);
    open_f_hash.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.insert(ptr);
    open_b_hash.insert(ptr);
    backward_index.insert(ptr);
    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    bool forward = false;
    while(open_f.size() > 0 && open_b.size() > 0 && UB > lbmin)
    {
      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT)
      {
        break;
      }

      if((*open_f.begin())->f < (*open_b.begin())->f)
      {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f, forward_index, backward_index);
        forward = open_f.size() < open_b.size();


        //lbmin = SIZE_MAX;
        //for(auto& pancake : open_f)
        //{
        //  auto pair = FTF_Pancake::gap_pair(pancake->hash_values, backward_index);
        //  size_t tmp = pancake->g + std::get<0>(pair);
        //  if(tmp < lbmin)
        //    lbmin = tmp;
        //}
      }
      else if((*open_f.begin())->f > (*open_b.begin())->f)
      {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b, backward_index, forward_index);
        forward = open_f.size() < open_b.size();


        //lbmin = SIZE_MAX;
        //for(auto& pancake : open_f)
        //{
        //  auto pair = FTF_Pancake::gap_pair(pancake->hash_values, backward_index);
        //  size_t tmp = pancake->g + std::get<0>(pair);
        //  if(tmp < lbmin)
        //    lbmin = tmp;
        //}
      }
      //else if(forward)
      else if(open_f.size() <= open_b.size())
      {
        expand_node(open_f, open_f_hash, open_b_hash, closed_f, forward_index, backward_index);
      }
      else
      {
        expand_node(open_b, open_b_hash, open_f_hash, closed_b, backward_index, forward_index);
      }

      lbmin = std::max((*open_f.begin())->f, (*open_b.begin())->f);
    }
    #ifdef HISTORY
    std::cout << "Actions: ";
    for(int i = 0; i < best_f.actions.size(); ++i)
    {
      std::cout << std::to_string(best_f.actions[i]) << " ";
    }
    std::cout << "|" << " ";
    for(int i = best_b.actions.size() - 1; i >= 0; --i)
    {
      std::cout << std::to_string(best_b.actions[i]) << " ";
    }
    std::cout << std::endl;
    #endif
    if(UB > (*open_f.begin())->f && UB > (*open_b.begin())->f)
    {
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
    }
    else
    {
      return std::make_tuple((double)UB, expansions, memory);
    }
  }

public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal, std::vector<Pancake>* expansions_in_order = nullptr)
  {
    FTF_Dibbs instance;
    return instance.run_search(FTF_Pancake(start.source, start.dir), FTF_Pancake(goal.source, goal.dir));
  }
};
