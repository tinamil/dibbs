#pragma once
#include <cstdint>
#include <unordered_map>
#include <set>
#include <tsl\hopscotch_map.h>
#include <string>
#include "node.h"


class FTF_Node
{
public:
  uint32_t vertex_index;
  uint32_t g;
  uint32_t h;
  uint32_t f;
  uint32_t h2;
  uint32_t f_bar;
  uint32_t delta;
  uint32_t ftf_h;
  Direction dir;
  bool threshold = false;

  FTF_Node() : vertex_index(0), g(0), h(0), f(0), h2(0), f_bar(0), delta(0), ftf_h(0), dir(Direction::forward), threshold(false) {}
  FTF_Node(Node n) : vertex_index(n.vertex_index), g(n.g), h(n.h), f(n.f), h2(n.h2), f_bar(n.f_bar), delta(n.delta),
    ftf_h(0), dir(n.dir), threshold(n.threshold)
  {}

  inline bool operator==(const FTF_Node& right) const
  {
    return this->vertex_index == right.vertex_index;
  }
  int num_neighbors() const
  {
    return Road::num_neighbors(vertex_index);
  }

  FTF_Node get_child(const int i) const
  {
    const Edge& edge = Road::get_edge(vertex_index, i);

    uint32_t heuristic_dist = 0, reverse_heuristic = 0;
    if(dir == Direction::forward) {
      heuristic_dist = Road::heuristic(Node::goal_node_index, edge.other);
      reverse_heuristic = Road::heuristic(Node::start_node_index, edge.other);
    }
    else {
      heuristic_dist = Road::heuristic(Node::start_node_index, edge.other);
      reverse_heuristic = Road::heuristic(Node::goal_node_index, edge.other);
    }

    FTF_Node n(
      Node{
      .vertex_index = edge.other,
      .g = this->g + edge.cost,
      .h = heuristic_dist,
      .f = this->g + edge.cost + heuristic_dist,
      .h2 = reverse_heuristic,
      .f_bar = 2 * (this->g + edge.cost) + heuristic_dist - reverse_heuristic,
      .delta = this->g + edge.cost - reverse_heuristic,
      .dir = this->dir,
      .threshold = threshold || heuristic_dist <= reverse_heuristic
      }
    );
    assert(reverse_heuristic <= n.g);
    return n;
  }
};

struct FTFNodeFSortHighG
{
  bool operator()(const FTF_Node& lhs, const FTF_Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const FTF_Node* lhs, const FTF_Node* rhs) const
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
        return lhs->g > rhs->g;
    }
    else
    {
      return lhs->f < rhs->f;
    }
  }
};

struct FTFNodeF_barSortHighG
{
  bool operator()(const FTF_Node& lhs, const FTF_Node& rhs) const
  {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const FTF_Node* lhs, const FTF_Node* rhs) const
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

struct FTFNodeHash
{
  inline std::size_t operator() (const FTF_Node& x) const
  {
    return operator()(&x);
  }
  inline std::size_t operator() (const FTF_Node* x) const
  {
    return x->vertex_index;
  }
};

struct FTFNodeEqual
{
  inline bool operator() (const FTF_Node* x, const FTF_Node* y) const
  {
    return x->vertex_index == y->vertex_index;
  }
  inline bool operator() (const FTF_Node& x, const FTF_Node& y) const
  {
    return x.vertex_index == y.vertex_index;
  }
};
