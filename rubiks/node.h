#pragma once
#include <string>
#include <cstdint>
#include <iostream>
#include <functional>
#include <iomanip>
#include "hash.hpp"
#include "rubiks.h"

#define HISTORY

struct Node
{
  #ifdef HISTORY
  std::shared_ptr<Node> parent;
  std::shared_ptr<Node> reverse_parent;
  #endif
  uint8_t state[40];
  uint8_t face;

  uint8_t depth;
  uint8_t combined;

  uint8_t f_bar;
  uint8_t heuristic;
  uint8_t reverse_heuristic;
  bool passed_threshold;

  Node();
  Node(const uint8_t* prev_state, const uint8_t* start_state, const Rubiks::PDB type);
  Node(const std::shared_ptr<Node> node_parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation, const bool reverse, const Rubiks::PDB type);
  Node(const Node& old_obj);

  uint8_t get_face() const;
  void set_reverse(const std::shared_ptr<Node> reverse);
  std::string print_state() const;
  std::string print_solution() const;
};

struct NodeCompare
{
  bool operator() (const Node* a, const Node* b) const;
  bool operator() (const Node& a, const Node& b) const;
  bool operator() (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) const;
};

struct NodeHash
{
  std::size_t operator() (const Node* s) const;
  std::size_t operator() (const Node& s) const;
  std::size_t operator() (const std::shared_ptr<Node> s) const;
};

struct NodeEqual
{
  bool operator() (const Node* a, const Node* b) const;
  bool operator() (const Node& a, const Node& b) const;
  bool operator() (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) const;
};
