#pragma once
#include "Pancake.h"
#include "ftf-pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include "StackArray.h"
#include <tsl/hopscotch_set.h>

#include <windows.h>
#include <Psapi.h>
#include "ftf_cudastructure.h"

#define FTF_PANCAKE_HYBRID "FTF_Pancake_Hybrid"

class dibbs_ftf_hybrid
{
public:

  //typedef std::set<const Pancake*, PancakeFBarSort> set;
  typedef std::set<const FTF_Pancake*, FTFPancakeF_barSortHighG> set;
  typedef std::unordered_set<const FTF_Pancake*, FTFPancakeHash, FTFPancakeEqual> hash_set;
  //typedef std::unordered_set<const Pancake*, PancakeNeighborHash, PancakeNeighborEqual> hash_set;

  StackArray<FTF_Pancake> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  ftf_cudastructure<FTF_Pancake> front_cuda, back_cuda;
  size_t front_ftf_f = 0;
  size_t back_ftf_f = 0;
  size_t expansions;
  size_t UB;
  size_t memory;
  size_t expansions_cstar = 0;
  bool phase_2 = false;

  dibbs_ftf_hybrid() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0), memory(0) {}

  void update_ftf()
  {
    for(int i = front_cuda.pancakes.size() - 1; i >= 0; --i) {
      if(front_cuda.pancakes[i]->f_bar >= UB) {
        open_f.erase(front_cuda.pancakes[i]);
      }
    }
    for(int i = back_cuda.pancakes.size() - 1; i >= 0; --i) {
      if(back_cuda.pancakes[i]->f_bar >= UB) {
        open_b.erase(back_cuda.pancakes[i]);
      }
    }
    if(back_cuda.pancakes.size() < front_cuda.pancakes.size()) {
      front_cuda.match_all(back_cuda);
      size_t min_ftf = back_cuda.pancakes[0]->f;
      for(int i = 1; i < back_cuda.pancakes.size(); ++i) {
        FTF_Pancake* p = back_cuda.pancakes[i];
        //assert(p->f >= p->f_bar);
        if(p->f < min_ftf) min_ftf = p->f;
        if(p->f >= UB) {
          open_b.erase(p);
        }
      }
      back_ftf_f = min_ftf;
    }
    else {
      back_cuda.match_all(front_cuda);
      size_t min_ftf = front_cuda.pancakes[0]->f;
      for(int i = 1; i < front_cuda.pancakes.size(); ++i) {
        FTF_Pancake* p = front_cuda.pancakes[i];
        //assert(p->f >= p->f_bar);
        if(p->f < min_ftf) min_ftf = p->f;
        if(p->f >= UB) {
          open_f.erase(p);
        }
      }
      front_ftf_f = min_ftf;
    }
  }

  void expand_node(set& open, hash_set& open_hash, ftf_cudastructure<FTF_Pancake>& cuda, const hash_set& other_open, hash_set& closed, std::vector<FTF_Pancake>* expansions_in_order = nullptr)
  {
    const FTF_Pancake* next_val = *open.begin();

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(open.begin());
    cuda.erase(next_val);

    ++expansions;
    ++expansions_cstar;

    closed.insert(next_val);

    if(expansions_in_order) expansions_in_order->push_back(*next_val);

    for(int i = 2, j = NUM_PANCAKES; i <= j; ++i)
    {
      FTF_Pancake new_action = next_val->apply_action(i);

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
            phase_2 = true;
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
            cuda.erase(&**it_open);
            open_hash.erase(it_open);
          }
        }

        auto ptr = storage.push_back(new_action);
        open.insert(ptr);
        open_hash.insert(ptr);
        cuda.insert(ptr);
      }
    }
  }

  #ifdef HISTORY
  Pancake best_f, best_b;
  #endif

  std::tuple<double, size_t, size_t> run_search(FTF_Pancake start, FTF_Pancake goal, std::vector<Pancake>* expansions_in_order = nullptr)
  {
    expansions = 0;
    memory = 0;
    auto ptr = storage.push_back(start);
    open_f.insert(ptr);
    open_f_hash.insert(ptr);
    front_cuda.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.insert(ptr);
    open_b_hash.insert(ptr);
    back_cuda.insert(ptr);
    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    int lbmin = 0;
    bool forward = false;
    while(open_f.size() > 0 && open_b.size() > 0 && UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0) && UB > front_ftf_f && UB > back_ftf_f)
    {
      if(lbmin < ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0))
      {
        expansions_cstar = 0;
        lbmin = ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0);
      }
      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT)
      {
        break;
      }

      if((*open_f.begin())->f_bar < (*open_b.begin())->f_bar)
      {
        expand_node(open_f, open_f_hash, front_cuda, open_b_hash, closed_f);
        forward = open_f.size() < open_b.size();
      }
      else if((*open_f.begin())->f_bar > (*open_b.begin())->f_bar)
      {
        expand_node(open_b, open_b_hash, back_cuda, open_f_hash, closed_b);
        forward = open_f.size() < open_b.size();
      }
      //else if(forward)
      else if(open_f.size() <= open_b.size())
      {
        expand_node(open_f, open_f_hash, front_cuda, open_b_hash, closed_f);
      }
      else
      {
        expand_node(open_b, open_b_hash, back_cuda, open_f_hash, closed_b);
      }


      if(phase_2 && UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0)) {
        update_ftf();
        phase_2 = false;
      }
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
    if(open_f.size() > 0 && open_b.size() > 0 && UB > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0) && UB > front_ftf_f && UB > back_ftf_f)
    {
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, expansions_cstar);
    }
    else
    {
      return std::make_tuple((double)UB, expansions, expansions_cstar);
    }
  }

public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal)
  {
    dibbs_ftf_hybrid instance;
    return instance.run_search(FTF_Pancake(start.source, start.dir), FTF_Pancake(goal.source, goal.dir));
  }
};
