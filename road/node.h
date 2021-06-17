#pragma once
#include <cstdint>
#include <cstddef>
#include "road.h"


enum class Direction { forward, backward };

struct Node
{
  static inline uint32_t start_node_index;
  static inline uint32_t goal_node_index;
  uint32_t vertex_index;
  uint32_t g;
  uint32_t h;
  uint32_t f;
  uint32_t h2;
  uint32_t f_bar;
  uint32_t delta;
  Direction dir;
  bool threshold = false;

  inline bool operator==(const Node& right) const
  {
    return vertex_index == right.vertex_index;
  }

  int num_neighbors() const
  {
    return Road::num_neighbors(vertex_index);
  }

  Node get_child(const int i) const
  {
    const Edge& edge = Road::get_edge(vertex_index, i);

    uint32_t heuristic_dist = 0, reverse_heuristic = 0;
    if(dir == Direction::forward) {
      heuristic_dist = Road::heuristic(goal_node_index, edge.other);
      reverse_heuristic = Road::heuristic(start_node_index, edge.other);
    }
    else {
      heuristic_dist = Road::heuristic(start_node_index, edge.other);
      reverse_heuristic = Road::heuristic(goal_node_index, edge.other);
    }

    Node n{
      .vertex_index = edge.other,
      .g = this->g + edge.cost,
      .h = heuristic_dist,
      .f = this->g + edge.cost + heuristic_dist,
      .h2 = reverse_heuristic,
      .f_bar = 2 * (this->g + edge.cost) + heuristic_dist - reverse_heuristic,
      .delta = this->g + edge.cost - reverse_heuristic,
      .dir = this->dir,
      .threshold = threshold || heuristic_dist <= reverse_heuristic
    };
    assert(reverse_heuristic <= n.g);
    return n;
  }
};

struct NodeFBarSortHighG
{
  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Node* lhs, const Node* rhs) const
  {
    int cmp = static_cast<int>(lhs->vertex_index) - rhs->vertex_index;
    if(cmp == 0)
    {
      return false;
    }
    else if(lhs->f_bar == rhs->f_bar)
    {
      if(lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g > rhs->g;
    }
    else
    {
      return lhs->f_bar < rhs->f_bar;
    }
  }
};

struct NodeFBarSortLowG
{
  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Node* lhs, const Node* rhs) const
  {
    int cmp = static_cast<int>(lhs->vertex_index) - rhs->vertex_index;
    if(cmp == 0)
    {
      return false;
    }
    else if(lhs->f_bar == rhs->f_bar)
    {
      if(lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else
    {
      return lhs->f_bar < rhs->f_bar;
    }
  }
};

//Returns smallest f value with largest g value
struct NodeFSort
{
  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Node* lhs, const Node* rhs) const
  {
    if(lhs->f == rhs->f)
    {
      return lhs->g > rhs->g;
    }
    return lhs->f > rhs->f;
  }
};


struct GSortHighDuplicate
{
  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Node* lhs, const Node* rhs) const
  {
    if(lhs->g == rhs->g)
    {
      return lhs->h < rhs->h;
    }
    return lhs->g > rhs->g;
  }
};

struct GSortLowDuplicate
{
  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const Node* lhs, const Node* rhs) const
  {
    return !GSortHighDuplicate{}(lhs, rhs);
  }
};

struct NodeGSortHigh
{

  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Node* lhs, const Node* rhs) const
  {
    int cmp = static_cast<int>(lhs->vertex_index) - rhs->vertex_index;
    if(cmp == 0)
    {
      return false;
    }
    if(lhs->g == rhs->g)
      return cmp > 0;
    else
      return lhs->g > rhs->g;
  }
};

//Returns smallest f value with smallest g value
struct NodeFSortLowG
{

  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Node* lhs, const Node* rhs) const
  {
    int cmp = static_cast<int>(lhs->vertex_index) - rhs->vertex_index;
    if(cmp == 0)
    {
      return false;
    }
    else if(lhs->f == rhs->f)
    {
      if(lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else
    {
      return lhs->f < rhs->f;
    }
  }
};

//Returns smallest g value
struct NodeGSortLow
{

  bool operator()(const Node& lhs, const Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const Node* lhs, const Node* rhs) const
  {
    int cmp = static_cast<int>(lhs->vertex_index) - rhs->vertex_index;
    if(cmp == 0)
    {
      return false;
    }
    if(lhs->g == rhs->g)
      return cmp < 0;
    else
      return lhs->g < rhs->g;
  }
};

struct NodeHash
{
  inline std::size_t operator() (const Node& x) const
  {
    return operator()(&x);
  }
  inline std::size_t operator() (const Node* x) const
  {
    return x->vertex_index;// SuperFastHash(lng->source + 1, NUM_PANCAKES);
  }
};

struct NodeEqual
{
  inline bool operator() (const Node* x, const Node* y) const
  {
    return x->vertex_index == y->vertex_index;
  }
  inline bool operator() (const Node& x, const Node& y) const
  {
    return x.vertex_index == y.vertex_index;
  }
};

