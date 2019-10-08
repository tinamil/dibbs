#pragma once

#include <memory>
#include "Pancake.h"
#include "hash.hpp"

class Node
{
  //Pancake data;
  std::shared_ptr<Node> parent;
  int action;

public:
  Pancake data;
  uint8_t g;
  uint8_t f;
  uint8_t h;

public:
  Node(uint8_t* src, uint8_t n) : data(src, n){
    
  }

  int get_num_actions() {
    return data.n;
  }

  Node apply_action(int i) {

  }

  bool operator<(Node const& right) {
    return f > right.f; //Inverted to make the smallest value the priority
  }

  bool is_goal() {
    //return data.is_solution();
  }
};

namespace std
{
  template<>
  struct hash<Node>
  {
    size_t operator()(const Node& x) const
    {
      SuperFastHash(x.data.source, x.data.n);
    }

  };
}