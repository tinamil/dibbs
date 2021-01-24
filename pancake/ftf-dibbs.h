#pragma once
#include "Pancake.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include "StackArray.h"
#include <tsl/hopscotch_map.h>
#include <tsl/hopscotch_set.h>
#include <vector>
#include "hash_table.h"

#include <windows.h>
#include <Psapi.h>
#include "ftf-pancake.h"
#include "ftf_cudastructure.h"

#define FTF_PANCAKE "FTF_Pancake"

class FTF_Dibbs
{
  static constexpr size_t CUDA_STREAMS_COUNT = 1;
public:

  typedef std::set<const FTF_Pancake*, FTFPancakeFSortHighG> set;
  typedef std::unordered_set<const FTF_Pancake*, FTFPancakeHash, FTFPancakeEqual> hash_set;
  //typedef std::unordered_set<const Pancake*, PancakeNeighborHash, PancakeNeighborEqual> hash_set;

  StackArray<FTF_Pancake> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  ftf_cudastructure<FTF_Pancake, FTFPancakeHash, FTFPancakeEqual> forward_index, backward_index;
  #ifdef FTF_HASH
  ftf_matchstructure f_match, b_match;
  #endif
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;
  static inline std::vector<mycuda> cuda_vector = std::vector<mycuda>(CUDA_STREAMS_COUNT);
  static inline std::vector<std::vector<FTF_Pancake*>> new_pancakes_vector = std::vector<std::vector<FTF_Pancake*>>(CUDA_STREAMS_COUNT);

  FTF_Dibbs() : expansions(0), UB(0), lbmin(0), memory(0)
  {

  }

  template<typename T>
  void expand_all_nodes(set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed, T& my_index, T& other_index)
  {
    auto f_val = (*open.begin())->f;
    auto g_val = (*open.begin())->g;
    #ifdef FTF_HASH
    ftf_matchstructure* my_match, * other_match;
    if((*open.begin())->dir == Direction::forward) {
      my_match = &f_match;
      other_match = &b_match;
    }
    else {
      my_match = &b_match;
      other_match = &f_match;
    }
    #endif
    int count = 0;
    for(int cuda_count = 0; cuda_count < CUDA_STREAMS_COUNT; ++cuda_count) {
      new_pancakes_vector[cuda_count].clear();
      while(UB > lbmin && !open.empty() && (*open.begin())->f == f_val/* && (*open.begin())->g == g_val*/ && ++count <= (BATCH_SIZE / NUM_PANCAKES / CUDA_STREAMS_COUNT))
      {
        const FTF_Pancake* next_val = *open.begin();
        
        auto it_hash = open_hash.find(next_val);
        assert(it_hash != open_hash.end());
        open_hash.erase(it_hash);
        open.erase(next_val);
        my_index.erase(next_val);
        assert(open_hash.size() == open.size());
        assert(open.size() == my_index.size());
        #ifdef FTF_HASH
        my_match->erase(next_val);
        #endif
        ++expansions;

        closed.insert(next_val);

        for(int i = 2, j = NUM_PANCAKES; i <= j; ++i)
        {
          FTF_Pancake new_action = next_val->apply_action(i);

          if((size_t)new_action.g + new_action.h >= UB) {
            continue;
          }
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
          auto it_closed = closed.find(&new_action);
          if(it_closed == closed.end())
          {
            auto ptr = storage.push_back(new_action);
            new_pancakes_vector[cuda_count].push_back(ptr);
          }
          else if((*it_closed)->g > new_action.g) {
            auto ptr = storage.push_back(new_action);
            new_pancakes_vector[cuda_count].push_back(ptr);
            closed.erase(it_closed);
          }
        }
      }
      if(new_pancakes_vector[cuda_count].size() > 0) {
        other_index.match(cuda_vector[cuda_count], new_pancakes_vector[cuda_count]);
        #ifdef FTF_HASH
        other_match->match(new_pancakes_vector[cuda_count]);
        #endif
      }
    }

    for(int cuda_count = 0; cuda_count < CUDA_STREAMS_COUNT && UB > lbmin; ++cuda_count) {
      uint8_t* answers = cuda_vector[cuda_count].get_answers();
      std::vector<FTF_Pancake*>& pancakes = new_pancakes_vector[cuda_count];
      for(int i = 0; i < pancakes.size(); ++i)
      {
        #ifdef FTF_HASH
        if(pancakes[i]->ftf_h != answers[i]) std::cout << "FTF Mismatch Error: " << std::to_string(pancakes[i]->ftf_h) << " did not equal the GPU value of " << answers[i] << "\n";
        assert(pancakes[i]->ftf_h == answers[i]);
        #endif
        pancakes[i]->ftf_h = answers[i];
        if(answers[i] == 255) std::cout << "ERROR";
        assert(pancakes[i]->ftf_h >= pancakes[i]->h);
        pancakes[i]->f = pancakes[i]->g + pancakes[i]->ftf_h;
      }
      for(int i = 0; i < pancakes.size(); ++i)
      {
        const FTF_Pancake* ptr = pancakes[i];
        assert(open_hash.size() == open.size());
        auto it_open = open_hash.find(ptr);
        if(it_open != open_hash.end())
        {
          if((*it_open)->g <= ptr->g)
          {
            pancakes[i] = pancakes.back();
            pancakes.resize(pancakes.size() - 1);
            i -= 1;
            continue;
          }
          else
          {
            size_t num_erased = open.erase(&**it_open);
            assert(num_erased == 1);
            my_index.erase(*it_open);
            #ifdef FTF_HASH
            my_match->erase(&**it_open);
            #endif
            open_hash.erase(it_open);
            assert(open_hash.size() == open.size());
          }
        }
        assert(my_index.contains(ptr) == false);
        auto [it, success] = open.insert(ptr);
        assert(success);
        auto [it2, success2] = open_hash.insert(ptr);
        assert(success2);
        assert(open_hash.size() == open.size());
        my_index.insert(ptr);
        assert(open.size() == my_index.size());
        #ifdef FTF_HASH
        my_match->insert(ptr);
        #endif
      }
      //Consistency errors with the pancakes vector, need to delete an earlier pancake
      //my_index.insert(pancakes);
    }
  }

  #ifdef HISTORY
  Pancake best_f, best_b;
  #endif
  std::tuple<double, size_t, size_t> run_search(FTF_Pancake start, FTF_Pancake goal, std::vector<FTF_Pancake>* expansions_in_order = nullptr)
  {
    expansions = 0;
    memory = 0;
    //assert(start.h == goal.h);
    auto ptr_start = storage.push_back(start);
    forward_index.insert(ptr_start);
    ptr_start->ftf_h = ptr_start->f = forward_index.match_one(cuda_vector[0], &goal);
    assert(ptr_start->ftf_h == MAX(ptr_start->h, goal.h));
    #ifdef FTF_HASH
    f_match.insert(ptr_start);
    #endif
    open_f.insert(ptr_start);
    open_f_hash.insert(ptr_start);
    auto ptr_goal = storage.push_back(goal);
    backward_index.insert(ptr_goal);
    ptr_goal->ftf_h = ptr_goal->f = ptr_start->ftf_h;
    open_b.insert(ptr_goal);
    open_b_hash.insert(ptr_goal);
    #ifdef FTF_HASH
    assert(f_match.match(ptr_goal) == ptr_start->ftf_h);
    b_match.insert(ptr_goal);
    #endif    

    UB = std::numeric_limits<size_t>::max();
    if(memcmp(start.source, goal.source, NUM_PANCAKES + 1) == 0) UB = 0;
    PROCESS_MEMORY_COUNTERS memCounter;
    bool forward = false;
    int i = 0;
    while(open_f.size() > 0 && open_b.size() > 0 && UB > lbmin)
    {
      if(++i % 500 == 0) std::cout << i << " ";
      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT)
      {
        break;
      }
      try {
        if((*open_f.begin())->f < (*open_b.begin())->f)
        {
          expand_all_nodes(open_f, open_f_hash, open_b_hash, closed_f, forward_index, backward_index);
          forward = open_f.size() < open_b.size();
        }
        else if((*open_f.begin())->f > (*open_b.begin())->f)
        {
          expand_all_nodes(open_b, open_b_hash, open_f_hash, closed_b, backward_index, forward_index);
          forward = open_f.size() < open_b.size();
        }
        else if(forward)
        //else if(open_f.size() <= open_b.size())
        {
          expand_all_nodes(open_f, open_f_hash, open_b_hash, closed_f, forward_index, backward_index);
          forward = open_f.size() < 2 * open_b.size();
        }
        else
        {
          expand_all_nodes(open_b, open_b_hash, open_f_hash, closed_b, backward_index, forward_index);
          forward = !(open_b.size() < 2 * open_f.size());
        }

        if(open_f.size() > 0 && open_b.size() > 0) {
          lbmin = std::max((*open_f.begin())->f, (*open_b.begin())->f);
        }
      }
      catch(std::exception e) {
        std::cout << "Failure at " << i << "\n";
        break;
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
