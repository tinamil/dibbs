#pragma once
#include <iostream>
#include "sliding_tile.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <string>
#include <algorithm>
#include "StackArray.h"
#include <tsl/hopscotch_map.h>
#include "hash_table.h"

#include <windows.h>
#include <Psapi.h>

class Dibbs
{
  constexpr static inline bool LOOKAHEAD = true;
  typedef std::set<const SlidingTile*, FBarSortHighG> set;
  typedef std::unordered_set<const SlidingTile*, SlidingTileHash, SlidingTileEqual> hash_set;
  typedef tsl::hopscotch_map<uint32_t, const SlidingTile*> neighbor_map;

  StackArray<SlidingTile> storage;
  neighbor_map neighbors_f, neighbors_b;
  set open_f, open_b;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t memory;

  Dibbs() : open_f(), open_b(), closed_f(), closed_b(), open_f_hash(), open_b_hash(), expansions(0), UB(0) {}


  void expand_node(const SlidingTile* next_val, set& open, hash_set& open_hash, const hash_set& other_open, hash_set& closed, neighbor_map& neighbors, neighbor_map& other_neighbors)
  {
    //if(expansions % 10000 == 0)
    //  std::cout << "Tile " << std::to_string(next_val->g) << " " << std::to_string(next_val->h) << " " << std::to_string(next_val->h2) << " " << std::to_string(next_val->f) << " " << std::to_string(next_val->delta) << " " << std::to_string(next_val->f_bar) << "\n";

    auto it_hash = open_hash.find(next_val);
    assert(it_hash != open_hash.end());
    open_hash.erase(it_hash);
    open.erase(next_val);

    closed.insert(next_val);
    if(next_val->f >= UB - 1)
    {
      return;
    }

    ++expansions;


    size_t starting_hash = hash_table::hash_configuration(next_val->source);
    for(int i = 1, stop = next_val->num_actions_available(); i <= stop; ++i)
    {
      SlidingTile new_action = next_val->apply_action(i);
      if(new_action.f >= UB - 1)
      {
        continue;
      }

      auto it_closed = closed.find(&new_action);
      if(it_closed == closed.end())
      {

        auto it_other = other_open.find(&new_action);
        bool already_matched = false;
        if(it_other != other_open.end())
        {
          already_matched = true;
          #ifdef HISTORY
          if(it_other->g + new_action.g < UB)
          {
            if(new_action.dir == Direction::forward)
            {
              best_f = new_action;
              best_b = *it_other;
            }
            else
            {
              best_f = *it_other;
              best_b = new_action;
            }
          }
          #endif
          UB = std::min(UB, (size_t)(*it_other)->g + new_action.g);
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

        if constexpr(LOOKAHEAD)
        {
          uint32_t new_hash = hash_table::update_hash_value(next_val->source, next_val->empty_location, SlidingTile::moves[next_val->empty_location][i], starting_hash);
          if(!already_matched)
          {
            if(other_neighbors.count(new_hash) > 0)
            {
              auto other_neighbor = other_neighbors[new_hash];
              if(other_neighbor->g + new_action.g + 1 < UB && hash_table::compare_one_off(other_neighbor, &new_action))
              {
                expand_node(&new_action, open, open_hash, other_open, closed, neighbors, other_neighbors);
              }
            }
          }
          for(int j = 1, stop = ptr->num_actions_available(); j <= stop; ++j)
          {
            uint32_t new_neighbor_hash = hash_table::update_hash_value(ptr->source, ptr->empty_location, SlidingTile::moves[ptr->empty_location][j], new_hash);
            neighbors[new_neighbor_hash] = ptr;
          }
        }
      }
    }
  }

  #ifdef HISTORY
  SlidingTile best_f, best_b;
  #endif

  std::tuple<double, size_t, size_t> run_search(SlidingTile start, SlidingTile goal)
  {
    expansions = 0;
    memory = 0;

    auto ptr = storage.push_back(start);
    open_f.insert(ptr);
    open_f_hash.insert(ptr);
    ptr = storage.push_back(goal);
    open_b.insert(ptr);
    open_b_hash.insert(ptr);

    UB = std::numeric_limits<size_t>::max();
    PROCESS_MEMORY_COUNTERS memCounter;
    bool forward = true;
    while(open_f.size() > 0 && open_b.size() > 0 && UB - 1 > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0))
    {

      BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if(memCounter.PagefileUsage > MEM_LIMIT)
      {
        break;
      }

      if((*open_f.begin())->f_bar < (*open_b.begin())->f_bar)
      {
        expand_node(*open_f.begin(), open_f, open_f_hash, open_b_hash, closed_f, neighbors_f, neighbors_b);
        forward = open_f.size() < open_b.size();
      }
      else if((*open_f.begin())->f_bar > (*open_b.begin())->f_bar)
      {
        expand_node(*open_b.begin(), open_b, open_b_hash, open_f_hash, closed_b, neighbors_b, neighbors_f);
        forward = open_f.size() < open_b.size();
      }
      else if(forward)
      //else if(open_f.size() <= open_b.size())
      {
        expand_node(*open_f.begin(), open_f, open_f_hash, open_b_hash, closed_f, neighbors_f, neighbors_b);
      }
      else
      {
        expand_node(*open_b.begin(), open_b, open_b_hash, open_f_hash, closed_b, neighbors_b, neighbors_f);
      }

    }
    #ifdef HISTORY
    std::cout << "Actions: ";
    for(int i = 0; i < best_f.actions.size(); ++i)
    {
      std::cout << std::to_string(best_f.actions[i]) << " ";
    }
    for(int i = 0; i < best_b.actions.size(); ++i)
    {
      std::cout << std::to_string(best_b.actions[i]) << " ";
    }
    std::cout << std::endl;
    #endif
    if(UB - 1 > ceil(((*open_f.begin())->f_bar + (*open_b.begin())->f_bar) / 2.0))
    {
      return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
    }
    else
    {
      std::cout << "Expansions: " << expansions << '\n';
      return std::make_tuple(UB, expansions, memory);
    }
  }

public:

  static std::tuple<double, size_t, size_t> search(SlidingTile start, SlidingTile goal)
  {
    Dibbs instance;
    return instance.run_search(start, goal);
  }
};
