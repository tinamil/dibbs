#pragma once
#pragma once
#pragma once

#include "Pancake.h"
#include <queue>
#include <unordered_set>
#include <tuple>
#include <stack>
#include <cmath>
#include <set>

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>

typedef std::unordered_set<Pancake, PancakeHash> hash_set;
typedef std::set<Pancake, PancakeFSortLowG> waiting_set;
typedef std::set<Pancake, PancakeGSort> ready_set;

class Dvcbs {

  ready_set open_f_ready, open_b_ready;
  waiting_set open_f_waiting, open_b_waiting;
  hash_set open_f_hash, open_b_hash;
  hash_set closed_f, closed_b;
  size_t expansions;
  size_t UB;
  size_t lbmin;

  Dvcbs() : open_f_ready(), open_b_ready(), open_f_waiting(), open_b_waiting(), open_f_hash(), open_b_hash(), closed_f(), closed_b(), expansions(0), UB(0), lbmin(0) {}

  bool select_pair() {
    while (!open_f_waiting.empty() && open_f_waiting.begin()->f < lbmin) {
      open_f_ready.insert(*open_f_waiting.begin());
      open_f_waiting.erase(open_f_waiting.begin());
    }
    while (!open_b_waiting.empty() && open_b_waiting.begin()->f < lbmin) {
      open_b_ready.insert(*open_b_waiting.begin());
      open_b_waiting.erase(open_b_waiting.begin());
    }

    while (true) {
      if (open_f_ready.empty() && open_f_waiting.empty()) return false;
      if (open_b_ready.empty() && open_b_waiting.empty()) return false;

      if (!open_f_ready.empty() && !open_b_ready.empty() && open_f_ready.begin()->g + open_b_ready.begin()->g <= lbmin) return true;
      if (!open_f_waiting.empty() && open_f_waiting.begin()->f <= lbmin) {
        open_f_ready.insert(*open_f_waiting.begin());
        open_f_waiting.erase(open_f_waiting.begin());
      }
      else if (!open_b_waiting.empty() && open_b_waiting.begin()->f <= lbmin) {
        open_b_ready.insert(*open_b_waiting.begin());
        open_b_waiting.erase(open_b_waiting.begin());
      }
      else {
        size_t min_wf = std::numeric_limits<size_t>::max();
        if (!open_f_waiting.empty()) min_wf = open_f_waiting.begin()->f;
        size_t min_wb = std::numeric_limits<size_t>::max();
        if (!open_b_waiting.empty()) min_wb = open_b_waiting.begin()->f;
        size_t min_r = std::numeric_limits<size_t>::max();
        if (!open_f_ready.empty() && !open_b_ready.empty()) min_r = open_f_ready.begin()->g + open_b_ready.begin()->g;
        lbmin = std::min(std::min(min_wf, min_wb), min_r);
      }
    }
  }

  bool expand_node_forward() {
    Pancake next_val = *open_f_ready.begin();
    open_f_ready.erase(next_val);
    open_f_hash.erase(next_val);

    closed_f.insert(next_val);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);

      auto it_open = open_b_hash.find(new_action);
      if (it_open != open_b_hash.end()) {
        UB = std::min(UB, (size_t)it_open->g + new_action.g);
      }

      it_open = open_f_hash.find(new_action);
      if (it_open != open_f_hash.end() && it_open->g <= new_action.g) continue;
      auto it_closed = closed_f.find(new_action);
      if (it_closed != closed_f.end() && it_closed->g <= new_action.g) continue;

      open_f_waiting.insert(new_action);
      open_f_hash.insert(new_action);
    }
  }

  bool expand_node_backward() {
    Pancake next_val = *open_b_ready.begin();
    open_b_ready.erase(next_val);
    open_b_hash.erase(next_val);

    closed_b.insert(next_val);

    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    assert(result);
    if (memCounter.PagefileUsage > MEM_LIMIT) {
      return false;
    }

    ++expansions;

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);

      auto it_open = open_f_hash.find(new_action);
      if (it_open != open_f_hash.end()) {
        UB = std::min(UB, (size_t)it_open->g + new_action.g);
      }

      it_open = open_b_hash.find(new_action);
      if (it_open != open_b_hash.end() && it_open->g <= new_action.g) continue;
      auto it_closed = closed_b.find(new_action);
      if (it_closed != closed_b.end() && it_closed->g <= new_action.g) continue;

      open_b_waiting.insert(new_action);
      open_b_hash.insert(new_action);
    }
  }
  std::pair<double, size_t> run_search(Pancake start, Pancake goal)
  {
    if (start == goal) {
      return std::make_pair(0, 0);
    }

    expansions = 0;
    UB = std::numeric_limits<double>::max();

    open_f_waiting.insert(start);
    open_f_hash.insert(start);

    open_b_waiting.insert(goal);
    open_b_hash.insert(goal);

    lbmin = std::max(1ui8, std::max(start.h, goal.h));

    bool finished = false;
    while (select_pair())
    {
      if (lbmin >= UB) {
        finished = true;
        break;
      }

      expand_node_forward();
      expand_node_backward();
    }

    if (finished)  return std::make_pair(UB, expansions);
    else return std::make_pair(std::numeric_limits<double>::infinity(), expansions);
  }

  std::pair<int, int> computeSingleClusterMinNodesTieBreaking(std::vector<std::pair<int, int> >& minimalVertexCovers, std::vector<std::pair<double, uint64_t> >& forwardCluster, std::vector<std::pair<double, uint64_t> >& backwardCluster) {
    int maxF = INT_MAX;
    int maxB = INT_MAX;
    std::pair<int, int> maxPair;
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
    return maxPair;
  }

  bool getVertexCover(std::vector<uint64_t>& nextForward, std::vector<uint64_t>& nextBackward)
  {
    while (true)
    {
      if (open_f_waiting.size() == 0 && open_f_ready.size() == 0)
        return false;
      if (open_b_waiting.size() == 0 && open_b_ready.size() == 0)
        return false;

      while (!open_f_waiting.empty() && open_f_waiting.begin()->f < lbmin) {
        open_f_ready.insert(*open_f_waiting.begin());
        open_f_waiting.erase(open_f_waiting.begin());
      }
      while (!open_b_waiting.empty() && open_b_waiting.begin()->f < lbmin) {
        open_b_ready.insert(*open_b_waiting.begin());
        open_b_waiting.erase(open_b_waiting.begin());
      }

      if (!open_f_ready.empty() && !open_b_ready.empty() && open_f_ready.begin()->g + open_b_ready.begin()->g <= lbmin)
      {

        std::vector<std::pair<double, uint64_t> > forwardCluster;
        std::vector<std::pair<double, uint64_t> > backwardCluster;
        for (auto it = open_f_ready.begin(); it != open_f_ready.end() && it->g <= lbmin - open_b_ready.begin()->g - 1 + 0.00001; it++) {
          if(forwardCluster.size() == 0 || forwardCluster.back().first != it->g)
            forwardCluster.push_back(std::make_pair(it->g, 1));
          else {
            forwardCluster.back().second++;
          }
        }
        for (auto it = open_b_ready.begin(); it != open_b_ready.end() && it->g <= lbmin - open_f_ready.begin()->g - 1 + 0.00001; it++) {
          if (backwardCluster.size() == 0 || backwardCluster.back().first != it->g)
            backwardCluster.push_back(std::make_pair(it->g, 1));
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
          auto v = forwardQueue.getNodesMapElements(kOpenReady, forwardCluster[i].first);
          nextForward.insert(nextForward.end(), v.begin(), v.end());
        }
        for (int j = 0; j <= chosenVC.second; j++) {
          auto v = backwardQueue.getNodesMapElements(kOpenReady, backwardCluster[j].first);
          nextBackward.insert(nextBackward.end(), v.begin(), v.end());
        }


        return true;
      }
      else
      {
        bool changed = false;
        if (/*backwardQueue.OpenReadySize() == 0 && */backwardQueue.OpenWaitingSize() != 0)
        {
          const auto i4 = backwardQueue.getFirstKey(kOpenWaiting);
          if (!fgreater(i4, CLowerBound))
          {
            changed = true;
            backwardQueue.PutToReady();
          }
        }
        if (/*forwardQueue.OpenReadySize() == 0 && */forwardQueue.OpenWaitingSize() != 0)
        {
          const auto i3 = forwardQueue.getFirstKey(kOpenWaiting);
          if (!fgreater(i3, CLowerBound))
          {
            changed = true;
            forwardQueue.PutToReady();
          }
        }
        if (!changed) {
          CLowerBound = DBL_MAX;
          if (forwardQueue.OpenWaitingSize() != 0)
          {
            const auto i5 = forwardQueue.getFirstKey(kOpenWaiting);
            CLowerBound = std::min(CLowerBound, i5);
          }
          if (backwardQueue.OpenWaitingSize() != 0)
          {
            const auto i6 = backwardQueue.getFirstKey(kOpenWaiting);
            CLowerBound = std::min(CLowerBound, i6);
          }
          if ((forwardQueue.OpenReadySize() != 0) && (backwardQueue.OpenReadySize() != 0))
            CLowerBound = std::min(CLowerBound, forwardQueue.getFirstKey(kOpenReady) + backwardQueue.getFirstKey(kOpenReady) + epsilon);
        }


      }


    }
    return false;
  }

  template <class state, class action, class environment, class dataStructure, class priorityQueue>
  bool ExpandAVertexCover(std::vector<state>& thePath)
  {
    std::vector<uint64_t> nForward, nBackward;
    bool result = queue.getVertexCover(nForward, nBackward, tieBreakingPolicy);
    // if failed, see if we have optimal path (but return)
    if (result == false)
    {
      if (currentCost == DBL_MAX)
      {
        thePath.resize(0);
        return true;
      }
      ExtractFromMiddle(thePath);
      return true;
    }


    if ((!isAllSolutions && !fless(queue.GetLowerBound(), currentCost)) || (isAllSolutions && !flesseq(queue.GetLowerBound(), currentCost))) {
      ExtractFromMiddle(thePath);
      return true;
    }

    else if (nForward.size() > 0 &&
      nBackward.size() > 0)
    {
      std::unordered_map<state*, bool> mapData;
      for (int i = 0; i < nForward.size(); i++) {
        mapData[&(queue.forwardQueue.Lookup(nForward[i]).data)] = true;
      }
      for (int j = 0; j < nBackward.size(); j++) {
        if (mapData.find(&(queue.backwardQueue.Lookup(nBackward[j]).data)) != mapData.end()) {
          ExtractFromMiddle(thePath);
          return true;
        }
      }

    }
    struct compareBackward {
      compareBackward(dataStructure currQueue) : queue(currQueue) {}
      bool operator () (uint64_t i, uint64_t j) { return (queue.backwardQueue.Lookup(i).h < queue.backwardQueue.Lookup(j).h); }
      dataStructure queue;
    };
    struct compareForward {
      compareForward(dataStructure currQueue) : queue(currQueue) {}
      bool operator () (uint64_t i, uint64_t j) { return (queue.forwardQueue.Lookup(i).h < queue.forwardQueue.Lookup(j).h); }
      dataStructure queue;
    };
    double currentLowerBound = queue.GetLowerBound();

    if (nForward.size() == 0) {
      for (int j = 0; j < ((int)nBackward.size()); j++) {
        double oldKey = queue.backwardQueue.getFirstKey(kOpenReady);
        if (queue.backwardQueue.Lookup(nBackward[j]).where != kClosed) {
          counts[currentLowerBound]++;
          Expand(nBackward[j], queue.backwardQueue, queue.forwardQueue, backwardHeuristic, start);
        }
        if ((!isAllSolutions && !fless(queue.GetLowerBound(), currentCost)) || (isAllSolutions && !flesseq(queue.GetLowerBound(), currentCost))) {
          ExtractFromMiddle(thePath);
          return true;
        }
        if (currentLowerBound != queue.GetLowerBound() || oldKey != queue.backwardQueue.getFirstKey(kOpenReady)) {
          return false;
        }
      }
    }

    else if (nBackward.size() == 0) {
      for (int i = 0; i < ((int)nForward.size()); i++) {
        double oldKey = queue.forwardQueue.getFirstKey(kOpenReady);
        if (queue.forwardQueue.Lookup(nForward[i]).where != kClosed) {
          counts[currentLowerBound]++;
          Expand(nForward[i], queue.forwardQueue, queue.backwardQueue, forwardHeuristic, goal);
        }
        if ((!isAllSolutions && !fless(queue.GetLowerBound(), currentCost)) || (isAllSolutions && !flesseq(queue.GetLowerBound(), currentCost))) {
          ExtractFromMiddle(thePath);
          return true;
        }
        if (currentLowerBound != queue.GetLowerBound() || oldKey != queue.forwardQueue.getFirstKey(kOpenReady)) {
          return false;
        }
      }
    }
    else {
      int i = nForward.size() - 1;
      int j = nBackward.size() - 1;
      while (i >= 0 || j >= 0) {
        if ((!isAllSolutions && !fless(queue.GetLowerBound(), currentCost)) || (isAllSolutions && !flesseq(queue.GetLowerBound(), currentCost)))
        {
          ExtractFromMiddle(thePath);
          return true;
        }
        bool expandForward;
        if (i < 0) {
          expandForward = false;
        }
        else if (j < 0) {
          expandForward = true;
        }
        else {
          if (queue.forwardQueue.Lookup(nForward[i]).g >= queue.backwardQueue.Lookup(nBackward[j]).g) {
            expandForward = true;
          }
          else {
            expandForward = false;
          }
        }
        if (expandForward) {
          if (queue.forwardQueue.Lookup(nForward[i]).where != kClosed) {
            counts[currentLowerBound]++;
            Expand(nForward[i], queue.forwardQueue, queue.backwardQueue, forwardHeuristic, goal);
          }
          i--;
        }
        else {
          if (queue.backwardQueue.Lookup(nBackward[j]).where != kClosed) {
            counts[currentLowerBound]++;
            Expand(nBackward[j], queue.backwardQueue, queue.forwardQueue, backwardHeuristic, start);
          }
          j--;
        }
        if (currentLowerBound != queue.GetLowerBound()) {
          return false;
        }
      }
    }
    return false;
  }

public:

  static std::pair<double, size_t> search(Pancake start, Pancake goal) {
    Dvcbs instance;
    return instance.run_search(start, goal);
  }
};