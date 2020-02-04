#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>


class Dvcbs {

  typedef std::unordered_set<const Pancake*, PancakeHash, PancakeEqual> hash_set;
  typedef std::set<const Pancake*, PancakeFSortLowG> waiting_set;
  typedef std::set<const Pancake*, PancakeGSortLow> ready_set;

  StackArray<Pancake> storage;
  ready_set open_f_ready, open_b_ready;
  waiting_set open_f_waiting, open_b_waiting;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;
  PROCESS_MEMORY_COUNTERS memCounter;
  size_t memory;

  Dvcbs() : open_f_ready(), open_b_ready(), open_f_waiting(), open_b_waiting(), open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  bool expand_node(const Pancake* next_val, ready_set& ready, waiting_set& waiting, hash_set& hash, hash_set& closed, const hash_set& other_hash) {
    ready.erase(next_val);
    hash.erase(next_val);

    assert(hash.size() == (ready.size() + waiting.size()));

    closed.insert(next_val);

    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val->apply_action(i);

      auto it_open = other_hash.find(&new_action);
      if (it_open != other_hash.end()) {
        UB = std::min(UB, (size_t)(*it_open)->g + new_action.g);
      }

      it_open = hash.find(&new_action);
      if (it_open != hash.end() && (*it_open)->g <= new_action.g) continue;
      else if (it_open != hash.end()) {
        ready.erase(&**it_open);
        waiting.erase(&**it_open);
        hash.erase(it_open);
      }
      auto it_closed = closed.find(&new_action);
      if (it_closed != closed.end() && (*it_closed)->g <= new_action.g) continue;


      auto ptr = storage.push_back(new_action);
      waiting.insert(ptr);
      hash.insert(ptr);
    }
    return true;
  }

  bool expand_node_forward(const Pancake* next_val) {
    return expand_node(next_val, open_f_ready, open_f_waiting, open_f_hash, closed_f, open_b_hash);
  }

  bool expand_node_backward(const Pancake* next_val) {
    return expand_node(next_val, open_b_ready, open_b_waiting, open_b_hash, closed_b, open_f_hash);
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
    open_f_waiting.insert(ptr);
    open_f_hash.insert(ptr);

    ptr = storage.push_back(goal);
    open_b_waiting.insert(ptr);
    open_b_hash.insert(ptr);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));


    PROCESS_MEMORY_COUNTERS memCounter;
    while (true)
    {
      BOOL proc_result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      assert(proc_result);
      memory = std::max(memory, memCounter.PagefileUsage);
      if (memCounter.PagefileUsage > MEM_LIMIT) {
        return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
      }

      std::vector<const Pancake*> nForward, nBackward;
      bool result = getVertexCover(nForward, nBackward);
      // if failed, see if we have optimal path (but return)
      if (result == false)
      {
        if (UB == std::numeric_limits<double>::max())
        {
          return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
        }
        else {
          return std::make_tuple(UB, expansions, memory);
        }
      }


      if (lbmin >= UB) {
        return std::make_tuple(UB, expansions, memory);
      }

      else if (nForward.size() > 0 && nBackward.size() > 0)
      {
        std::unordered_map<const Pancake*, bool, PancakeHash, PancakeEqual> mapData;
        for (int i = 0; i < nForward.size(); i++) {
          mapData[nForward[i]] = true;
        }
        for (int j = 0; j < nBackward.size(); j++) {
          if (mapData.find(nBackward[j]) != mapData.end()) {
            return std::make_tuple(UB, expansions, memory);
          }
        }

      }

      auto currentLowerBound = lbmin;

      bool skip_loop = false;
      if (nForward.size() == 0) {
        for (int j = 0; j < ((int)nBackward.size()); j++) {
          double oldKey = (*open_b_ready.begin())->g;
          if (closed_b.find(nBackward[j]) == closed_b.end()) {
            if (expand_node_backward(nBackward[j]) == false)  return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
          }
          if (lbmin >= UB) {
            return std::make_tuple(UB, expansions, memory);
          }
          if (currentLowerBound != lbmin || open_b_ready.empty() || oldKey != (*open_b_ready.begin())->g) {
            skip_loop = true;
            break;
          }
        }
        if (skip_loop) continue;
      }

      else if (nBackward.size() == 0) {
        for (int i = 0; i < ((int)nForward.size()); i++) {
          double oldKey = (*open_f_ready.begin())->g;
          if (closed_f.find(nForward[i]) == closed_f.end()) {
            if (expand_node_forward(nForward[i]) == false)  return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
          }
          if (lbmin >= UB) {
            return std::make_tuple(UB, expansions, memory);
          }
          if (currentLowerBound != lbmin || open_f_ready.empty() || oldKey != (*open_f_ready.begin())->g) {
            skip_loop = true;
            break;
          }
        }
        if (skip_loop) continue;
      }
      else {
        int i = nForward.size() - 1;
        int j = nBackward.size() - 1;
        while (i >= 0 || j >= 0) {
          if (lbmin >= UB)
          {
            return std::make_tuple(UB, expansions, memory);
          }
          bool expandForward;
          if (i < 0) {
            expandForward = false;
          }
          else if (j < 0) {
            expandForward = true;
          }
          else {
            if (nForward[i]->g >= nBackward[j]->g) {
              expandForward = true;
            }
            else {
              expandForward = false;
            }
          }
          if (expandForward) {
            if (closed_f.find(nForward[i]) == closed_f.end()) {
            }
            i--;
          }
          else {
            if (closed_b.find(nBackward[j]) == closed_b.end()) {
              if (expand_node_backward(nBackward[j]) == false)  return std::make_tuple(std::numeric_limits<double>::infinity(), expansions, memory);
            }
            j--;
          }
          if (currentLowerBound != lbmin) {
            skip_loop = true;
            break;
          }
        }
        if (skip_loop) continue;
      }
    }
  }

  std::pair<int, int> computeSingleClusterMinNodesTieBreaking(std::vector<std::pair<int, int> >& minimalVertexCovers, std::vector<std::pair<double, uint64_t> >& forwardCluster, std::vector<std::pair<double, uint64_t> >& backwardCluster) {
    int maxF = INT_MAX;
    int maxB = INT_MAX;
    for (std::vector<std::pair<int, int> >::iterator it = minimalVertexCovers.begin(); it != minimalVertexCovers.end(); ++it) {
      if (maxF < INT_MAX && maxB < INT_MAX) {
        break;
      }
      if (it->first >= 0) {
        maxF = forwardCluster[0].second;
      }
      if (it->second >= 0) {
        maxB = backwardCluster[0].second;
      }
    }
    if (maxF < maxB) {
      return std::make_pair(0, -1);
    }
    else {
      return std::make_pair(-1, 0);
    }
  }

  bool getVertexCover(std::vector<const Pancake*>& nextForward, std::vector<const Pancake*>& nextBackward)
  {
    while (true)
    {
      if (open_f_waiting.size() == 0 && open_f_ready.size() == 0)
        return false;
      if (open_b_waiting.size() == 0 && open_b_ready.size() == 0)
        return false;

      while (!open_f_waiting.empty() && (*open_f_waiting.begin())->f <= lbmin) {
        open_f_ready.insert(*open_f_waiting.begin());
        open_f_waiting.erase(open_f_waiting.begin());
      }
      while (!open_b_waiting.empty() && (*open_b_waiting.begin())->f <= lbmin) {
        open_b_ready.insert(*open_b_waiting.begin());
        open_b_waiting.erase(open_b_waiting.begin());
      }

      if (!open_f_ready.empty() && !open_b_ready.empty() && (*open_f_ready.begin())->g + (*open_b_ready.begin())->g <= lbmin)
      {

        std::vector<std::pair<double, uint64_t> > forwardCluster;
        std::vector<std::pair<double, uint64_t> > backwardCluster;
        for (auto it = open_f_ready.begin(); it != open_f_ready.end() && (*it)->g <= lbmin - (*open_b_ready.begin())->g - 1 + 0.00001; it++) {
          if (forwardCluster.size() == 0 || forwardCluster.back().first != (*it)->g)
            forwardCluster.push_back(std::make_pair((*it)->g, 1));
          else {
            forwardCluster.back().second++;
          }
        }
        for (auto it = open_b_ready.begin(); it != open_b_ready.end() && (*it)->g <= lbmin - (*open_f_ready.begin())->g - 1 + 0.00001; it++) {
          if (backwardCluster.size() == 0 || backwardCluster.back().first != (*it)->g)
            backwardCluster.push_back(std::make_pair((*it)->g, 1));
          else {
            backwardCluster.back().second++;
          }
        }
        int minJ = INT_MAX;
        int minI = INT_MAX;
        int minValue = INT_MAX;
        uint64_t NumForwardInVC = 0;
        uint64_t NumBackwardInVC = 0;
        std::vector<std::pair<int, int> > minimalVertexCovers;
        for (int i = -1; i < ((int)forwardCluster.size()); i++) {
          if (i >= 0) {
            NumForwardInVC += forwardCluster[i].second;
          }
          else {
            NumForwardInVC = 0;
          }
          bool skip = false;
          for (int j = -1; j < ((int)backwardCluster.size()) && !skip; j++) {
            if (j >= 0) {
              NumBackwardInVC += backwardCluster[j].second;
            }
            else {
              NumBackwardInVC = 0;
            }
            if (i == ((int)forwardCluster.size()) - 1) {
              if (NumForwardInVC < minValue) {
                minimalVertexCovers.clear();
              }
              if (NumForwardInVC <= minValue) {
                minimalVertexCovers.push_back(std::make_pair(i, j));
                minValue = NumForwardInVC;
              }
              skip = true;
            }
            else if (j == ((int)backwardCluster.size()) - 1) {
              if (NumBackwardInVC < minValue) {
                minimalVertexCovers.clear();
              }
              if (NumBackwardInVC <= minValue) {
                minimalVertexCovers.push_back(std::make_pair(i, j));
                minValue = NumBackwardInVC;
              }
              skip = true;
            }
            else if (backwardCluster[j + 1].first + forwardCluster[i + 1].first + 1 > lbmin) {
              if (NumBackwardInVC + NumForwardInVC < minValue) {
                minimalVertexCovers.clear();
              }
              if (NumBackwardInVC + NumForwardInVC <= minValue) {
                minimalVertexCovers.push_back(std::make_pair(i, j));
                minValue = NumBackwardInVC + NumForwardInVC;
              }
              skip = true;
            }
          }
        }


        std::pair<int, int> chosenVC = computeSingleClusterMinNodesTieBreaking(minimalVertexCovers, forwardCluster, backwardCluster);

        for (int i = 0; i <= chosenVC.first; i++) {
          auto start = open_f_ready.begin();
          auto start_g = (*start)->g;
          while (start != open_f_ready.end() && (*start)->g == start_g) {
            nextForward.push_back(&**start);
            ++start;
          }
        }
        for (int i = 0; i <= chosenVC.second; i++) {
          auto start = open_b_ready.begin();
          auto start_g = (*start)->g;
          while (start != open_b_ready.end() && (*start)->g == start_g) {
            nextBackward.push_back(&**start);
            ++start;
          }
        }
        return true;
      }
      else
      {
        bool changed = false;
        if (open_b_waiting.size() != 0)
        {
          const auto i4 = (*open_b_waiting.begin())->f;
          if (i4 <= lbmin)
          {
            changed = true;
            while (!open_b_waiting.empty() && (*open_b_waiting.begin())->f == i4) {
              open_b_ready.insert(*open_b_waiting.begin());
              open_b_waiting.erase(open_b_waiting.begin());
            }
          }
        }
        if (open_f_waiting.size() != 0)
        {
          const auto i3 = (*open_f_waiting.begin())->f;
          if (i3 <= lbmin)
          {
            changed = true;
            while (!open_f_waiting.empty() && (*open_f_waiting.begin())->f == i3) {
              open_f_ready.insert(*open_f_waiting.begin());
              open_f_waiting.erase(open_f_waiting.begin());
            }
          }
        }
        if (!changed) {
          lbmin = std::numeric_limits<size_t>::max();
          if (open_f_waiting.size() != 0)
          {
            const auto i5 = (*open_f_waiting.begin())->f;
            lbmin = std::min(lbmin, (size_t)i5);
          }
          if (open_b_waiting.size() != 0)
          {
            const auto i6 = (*open_b_waiting.begin())->f;
            lbmin = std::min(lbmin, (size_t)i6);
          }
          if ((open_f_ready.size() != 0) && (open_b_ready.size() != 0))
            lbmin = std::min(lbmin, (size_t)((*open_f_ready.begin())->g + (*open_b_ready.begin())->g + 1));
        }
      }
    }
    return false;
  }


public:

  static std::tuple<double, size_t, size_t> search(Pancake start, Pancake goal) {
    Dvcbs instance;
    return instance.run_search(start, goal);
  }
};