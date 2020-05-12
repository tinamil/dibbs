#pragma once

#include "Pancake.h"
#include "problems.h"
#include "Astar.h"
#include <deque>
#include <iostream>
#include <limits>

constexpr auto NUM_PROBLEMS = 100;


void generate_random_instance(double& seed, uint8_t problem[]);

std::vector<int> last, prev, head;
std::vector<int> dist, Q, matching;
std::vector<bool> used, vis;

inline void init(int _n1, int _n2) {
  last.clear();
  prev.clear();
  head.clear();
  dist.clear();
  Q.clear();
  last.resize(_n1);
  std::fill(last.begin(), last.end(), -1);
  dist.resize(_n1);
  Q.resize(_n1);
  matching.resize(_n2);
  used.resize(_n1);
  vis.resize(_n1);
}

inline void shrink(int new_size) {
  last.resize(new_size);
  dist.resize(new_size);
  Q.resize(new_size);
  used.resize(new_size);
  vis.resize(new_size);
}

inline void addEdge(int u, int v) {
  head.push_back(v);
  prev.push_back(last[u]);
  last[u] = head.size() - 1;
}

inline void bfs() {
  std::fill(dist.begin(), dist.end(), -1);
  int sizeQ = 0;
  for (int u = 0; u < dist.size(); ++u) {
    if (!used[u]) {
      Q[sizeQ++] = u;
      dist[u] = 0;
    }
  }
  for (int i = 0; i < sizeQ; i++) {
    int u1 = Q[i];
    for (int e = last[u1]; e >= 0; e = prev[e]) {
      int u2 = matching[head[e]];
      if (u2 >= 0 && dist[u2] < 0) {
        dist[u2] = dist[u1] + 1;
        Q[sizeQ++] = u2;
      }
    }
  }
}

inline bool dfs(int u1) {
  vis[u1] = true;
  for (int e = last[u1]; e >= 0; e = prev[e]) {
    int v = head[e];
    int u2 = matching[v];
    if (u2 < 0 || !vis[u2] && dist[u2] == dist[u1] + 1 && dfs(u2)) {
      matching[v] = u1;
      used[u1] = true;
      return true;
    }
  }
  return false;
}

inline int maxMatching() {
  std::fill(used.begin(), used.end(), false);
  std::fill(matching.begin(), matching.end(), -1);
  for (int res = 0;;) {
    bfs();
    std::fill(vis.begin(), vis.end(), false);
    int f = 0;
    for (int u = 0; u < vis.size(); ++u)
      if (!used[u] && dfs(u))
        ++f;
    if (!f)
      return res;
    res += f;
  }
}

//This version can suffer a stack overflow during DFS
//
//std::vector<int> pair_U, pair_V;
//std::vector<int> dist;
//std::vector<std::vector<int>> adjacency;
//void init(int l, int r) {
//  adjacency.clear();
//  adjacency.resize(l + 1);
//  pair_U.resize(l + 1);
//  std::fill(pair_U.begin(), pair_U.end(), 0);
//  pair_V.resize(r + 1);
//  std::fill(pair_V.begin(), pair_V.end(), 0);
//  dist.resize(l + 1);
//}
//
//void addEdge(int l, int r) {
//  adjacency[l + 1].push_back(r + 1);
//}
//
//bool BFS() {
//  std::queue<int> queue;
//  for (int u = 1; u < pair_U.size(); ++u) {
//    if (pair_U[u] == 0) {
//      dist[u] = 0;
//      queue.push(u);
//    }
//    else {
//      dist[u] = INT_MAX;
//    }
//  }
//  dist[0] = INT_MAX;
//  int u;
//  while (!queue.empty()) {
//    u = queue.front();
//    queue.pop();
//    if (dist[u] < dist[0]) {
//      for (auto v : adjacency[u]) {
//        if (dist[pair_V[v]] == INT_MAX) {
//          dist[pair_V[v]] = dist[u] + 1;
//          queue.push(pair_V[v]);
//        }
//      }
//    }
//  }
//  return dist[0] != INT_MAX;
//}
//
//bool DFS(int u) {
//  if (u != 0) {
//    for (auto v : adjacency[u]) {
//      if (dist[pair_V[v]] == dist[u] + 1 && DFS(pair_V[v])) {
//        pair_V[v] = u;
//        pair_U[u] = v;
//        return true;
//      }
//    }
//    dist[u] = INT_MAX;
//    return false;
//  }
//  else {
//    return true;
//  }
//}
//
//void cleanup() {
//  for (int u = pair_U.size() - 1; u >= 1; --u) {
//    if (adjacency[u].size() == 0) {
//      adjacency.erase(adjacency.begin() + u);
//    }
//  }
//  pair_U.resize(adjacency.size());
//  std::fill(pair_U.begin(), pair_U.end(), 0);
//}
//
//int maxMatching() {
//  cleanup();
//  int matching = 0;
//  while (BFS()) {
//    for (int u = 1; u < pair_U.size(); ++u) {
//      if (pair_U[u] == 0 && DFS(u)) {
//        matching += 1;
//      }
//    }
//  }
//  return matching;
//}

void GeneratePerfectCounts() {
  constexpr static bool findAll = true;
  double seed = 3.1567;
  uint8_t problem[NUM_PANCAKES + 1];
  std::cout << NUM_PANCAKES << " " << GAPX;

  if constexpr (findAll)
    std::cout << " All perfect: ";
  else
    std::cout << " First perfect: ";

  for (int i = 1; i <= NUM_PROBLEMS; ++i) {
    generate_random_instance(seed, problem);

    Pancake::Initialize_Dual(problem);
    Pancake start(problem, Direction::forward);
    Pancake goal = Pancake::GetSortedStack(Direction::backward);

    uint32_t cstar;
    {
      Astar backwardInstance;
      auto [cstar_b, expansions_b, memory_b] = backwardInstance.run_search(goal, start);
      cstar = cstar_b;

      Astar forwardInstance;
      auto [cstar_f, expansions_f, memory_f] = forwardInstance.run_search(start, goal);

      if (cstar_f != cstar) std::cout << "ERROR";

      std::vector<Pancake*> lPancakes, rPancakes;
      for (auto i = 0; i < forwardInstance.pancakes.size(); ++i) {
        auto f = &forwardInstance.pancakes[i];
        if (forwardInstance.closed.contains(f)) lPancakes.push_back(f);
      }
      for (auto i = 0; i < backwardInstance.pancakes.size(); ++i) {
        auto b = &backwardInstance.pancakes[i];
        if (backwardInstance.closed.contains(b)) rPancakes.push_back(b);
      }

      init(lPancakes.size(), rPancakes.size());

      int shrink_val = 0;
      int findex = 0;
      for (auto i = 0; i < lPancakes.size(); ++i) {
        auto f = lPancakes[i];

        int bindex = 0;
        bool match = false;
        for (auto j = 0; j < rPancakes.size(); ++j) {
          auto b = rPancakes[j];
          if constexpr (findAll) {
            if ((f->g + b->g + 1) <= cstar && (f->f_bar + b->f_bar) <= (2 * cstar) && (f->f + b->delta) <= cstar && (b->f + f->delta) <= cstar) {
              addEdge(findex, bindex);
              match = true;
            }
          }
          else {
            if (f->g + b->g + 1 < cstar && f->f_bar + b->f_bar < 2 * cstar && f->f + b->delta < cstar && b->f + f->delta < cstar) {
              addEdge(findex, bindex);
              match = true;
            }
          }
          bindex += 1;
        }
        if (match) {
          findex += 1;
        }
        else {
          shrink_val += 1;
        }
      }
      shrink(shrink_val);
      std::cout << maxMatching() << " ";
    }
  }
  std::cout << "\n";
}