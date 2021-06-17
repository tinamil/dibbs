#pragma once
#include "node.h"
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
//#include "hash_table.h"

#include <windows.h>
#include <Psapi.h>
#include "ftf_node.h"
#include "ftf_cudastructure.h"

#define FFGBS "FFGBS"

class Ffgbs
{
  static constexpr size_t BATCH_SIZE = 254;
  static constexpr size_t CUDA_STREAMS_COUNT = 1;
  //int iteration_count = 0;
public:

  typedef std::set<const FTF_Node*, FTFNodeFSortHighG> set;
  typedef std::unordered_set<const FTF_Node*, FTFNodeHash, FTFNodeEqual> hash_set;

  StackArray<FTF_Node> storage;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  ftf_cudastructure<FTF_Node, FTFNodeHash, FTFNodeEqual> forward_index, backward_index;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  size_t memory;
  static inline std::vector<mycuda> cuda_vector = std::vector<mycuda>(CUDA_STREAMS_COUNT);
  static inline std::vector<std::vector<FTF_Node*>> new_pancakes_vector = std::vector<std::vector<FTF_Node*>>(CUDA_STREAMS_COUNT);
  Direction dir = Direction::forward;

  Ffgbs() : expansions(0), UB(0), lbmin(0), memory(0)
  {

  }

  template<typename T>
  void expand_all_nodes(set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed, T& my_index, T& other_index)
  {
    //iteration_count++;
    //if(iteration_count % 5000 == 0) std::cout << iteration_count << " " << open_hash.size() << " " << other_open.size() << " " << lbmin << " " << UB << "\n";
    for(int cuda_count = 0; cuda_count < CUDA_STREAMS_COUNT; ++cuda_count) {
      const double f_val = (*open.begin())->f * 1.0008;
      new_pancakes_vector[cuda_count].clear();
      while(UB > lbmin && !open.empty() && (*open.begin())->f <= f_val && (*open.begin())->f < UB && new_pancakes_vector[cuda_count].size() < BATCH_SIZE)
      {
        const FTF_Node* next_val = *open.begin();

        auto it_hash = open_hash.find(next_val);
        assert(it_hash != open_hash.end());
        open_hash.erase(it_hash);
        open.erase(next_val);
        my_index.erase(next_val);
        assert(open_hash.size() == open.size());
        assert(open.size() == my_index.size());
        ++expansions;

        closed.insert(next_val);

        for(int i = 0, j = next_val->num_neighbors(); i < j; ++i)
        {
          FTF_Node new_action = next_val->get_child(i);

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
      //std::cout << "Expanding iteration " << iteration_count << " with " << new_pancakes_vector[cuda_count].size() << " x " << other_index.size() << "\n";
      if(new_pancakes_vector[cuda_count].size() == 0) continue;
      /*else if(new_pancakes_vector[cuda_count].size() < BATCH_SIZE / 2) {
        for(int i = 0; i < new_pancakes_vector[cuda_count].size(); ++i) {
          auto& p = new_pancakes_vector[cuda_count][i];
          p->ftf_h = p->h;
        }
      }*/
      else {
        other_index.match(cuda_vector[cuda_count], new_pancakes_vector[cuda_count]);
      }
    }

    for(int cuda_count = 0; cuda_count < CUDA_STREAMS_COUNT && UB > lbmin; ++cuda_count) {
      uint32_t* answers = cuda_vector[cuda_count].get_answers();
      std::vector<FTF_Node*>& pancakes = new_pancakes_vector[cuda_count];
      for(int i = 0; i < pancakes.size(); ++i)
      {
        pancakes[i]->ftf_h = answers[i];
        if(answers[i] == UINT32_MAX) std::cout << "ERROR";
        //assert(pancakes[i]->ftf_h >= pancakes[i]->h);
        if(pancakes[i]->h > pancakes[i]->ftf_h) pancakes[i]->ftf_h = pancakes[i]->h;
        pancakes[i]->f = pancakes[i]->g + pancakes[i]->ftf_h;
      }
      for(int i = 0; i < pancakes.size(); ++i)
      {
        const FTF_Node* ptr = pancakes[i];
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
      }
    }
  }

  inline void choose_dir()
  {
    if((*open_f.begin())->f * 1.01 < (*open_b.begin())->f) {
      //if(dir == Direction::backward) std::cout << "Switching direction 1 backward to forward\n";
      dir = Direction::forward;
    }
    else if((*open_b.begin())->f * 1.01 < (*open_f.begin())->f)
    {
      //if(dir == Direction::forward) std::cout << "Switching direction 1 forward to backward\n";
      dir = Direction::backward;
    }
    else if(UB < SIZE_MAX) {
      //Do nothing, just expand dir until done
    }
    else if(dir == Direction::forward && open_f.size() > 2 * open_b.size())
    {
      //if(dir == Direction::forward) std::cout << "Switching direction 2 forward to backward\n";
      dir = Direction::backward;
    }
    else if(dir == Direction::backward && open_b.size() > 2 * open_f.size()) {
      //if(dir == Direction::backward) std::cout << "Switching direction 2 backward to forward\n";
      dir = Direction::forward;
    }
  }

  #ifdef HISTORY
  Pancake best_f, best_b;
  #endif
  std::tuple<double, size_t, size_t> run_search(FTF_Node start, FTF_Node goal)
  {
    expansions = 0;
    memory = 0;
    //assert(start.h == goal.h);
    auto ptr_start = storage.push_back(start);
    forward_index.insert(ptr_start);
    ptr_start->ftf_h = ptr_start->f = forward_index.match_one(cuda_vector[0], &goal);
    //assert(ptr_start->ftf_h == MAX(ptr_start->h, goal.h));
    open_f.insert(ptr_start);
    open_f_hash.insert(ptr_start);
    auto ptr_goal = storage.push_back(goal);
    backward_index.insert(ptr_goal);
    ptr_goal->ftf_h = ptr_goal->f = ptr_start->ftf_h;
    open_b.insert(ptr_goal);
    open_b_hash.insert(ptr_goal);

    UB = std::numeric_limits<size_t>::max();
    if(start.vertex_index == goal.vertex_index) UB = 0;
    PROCESS_MEMORY_COUNTERS memCounter;
    while(open_f.size() > 0 && open_b.size() > 0 && UB > lbmin)
    {
      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT)
      {
        break;
      }
      choose_dir();
      if(dir == Direction::forward)
      {
        expand_all_nodes(open_f, open_f_hash, open_b_hash, closed_f, forward_index, backward_index);
      }
      else
      {
        expand_all_nodes(open_b, open_b_hash, open_f_hash, closed_b, backward_index, forward_index);
      }

      if(open_f.size() > 0 && open_b.size() > 0) {
        lbmin = std::max((*open_f.begin())->f, (*open_b.begin())->f);
      }
    }

    //std::cout << "iterations: " << iteration_count << '\n';

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

  static std::tuple<double, size_t, size_t> search(Node start, Node goal)
  {
    Ffgbs instance;
    return instance.run_search(FTF_Node(start), FTF_Node(goal));
  }
};
