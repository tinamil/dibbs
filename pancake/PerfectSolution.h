#pragma once

#include "Pancake.h"
#include "problems.h"
#include "Astar.h"
#include <deque>
#include <iostream>

constexpr auto NUM_PROBLEMS = 100;

void generate_random_instance(double& seed, uint8_t problem[]);

std::vector<int> last, prev, head;
std::vector<int> dist, Q, matching;
std::vector<bool> used, vis;

inline void init(int _n1, int _n2) {
  last.resize(_n1);
  std::fill(last.begin(), last.end(), -1);
  dist.resize(_n1);
  Q.resize(_n1);
  matching.resize(_n2);
  used.resize(_n1);
  vis.resize(_n1);
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

void GeneratePerfectCounts() {
  double seed = 3.1567;
  uint8_t problem[NUM_PANCAKES + 1];
  std::cout << NUM_PANCAKES << " " << GAPX << " All perfect: ";
  for (int i = 1; i <= NUM_PROBLEMS; ++i) {
    generate_random_instance(seed, problem);

    Pancake::Initialize_Dual(problem);
    Pancake bnode(problem, Direction::backward);
    Pancake goal = Pancake::GetSortedStack(Direction::backward);

    uint32_t cstar;
    std::unordered_set<Pancake, PancakeHash> closed_b, closed_f;
    {
      Astar backwardInstance;
      auto [cstar_b, expansions_b, memory_b] = backwardInstance.run_search(goal, bnode);
      cstar = cstar_b;
      closed_b = backwardInstance.closed;
    }

    Pancake fnode(problem, Direction::forward);
    goal = Pancake::GetSortedStack(Direction::forward);
    {
      Astar forwardInstance;
      auto [cstar_f, expansions_f, memory_f] = forwardInstance.run_search(fnode, goal);
      closed_f = forwardInstance.closed;

      if (cstar_f != cstar) std::cout << "ERROR";
    }

    int bsize = closed_b.size();
    init(closed_f.size(), closed_b.size());
    int findex = 0;
    for (const auto& f : closed_f) {
      int bindex = 0;
      for (const auto& b : closed_b) {
        if (f.g + b.g + 1 < cstar && f.f_bar + b.f_bar < 2 * cstar && f.f + b.delta < cstar && b.f + f.delta < cstar) {
          addEdge(findex, bindex);
        }
        bindex += 1;
      }
      findex += 1;
    }
    std::cout << maxMatching() << " ";
  }
}