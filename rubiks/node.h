#pragma once
#include <string>
#include <cstdint>
#include <iostream>
#include "hash.hpp"
#include "rubiks.h"

struct Node
{
  uint8_t state[40];
  uint8_t reverse_faces[20];
  uint8_t faces[20];
  uint8_t face;

  uint8_t depth;

  uint8_t combined;
  uint8_t reverse_depth;

  uint8_t f_bar;
  uint8_t heuristic;
  uint8_t reverse_heuristic;

  Node();
  Node(const uint8_t* prev_state, const uint8_t* start_state, const Rubiks::PDB type);
  Node(const Node& parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation, const bool reverse, const Rubiks::PDB type);
  Node(const Node& old_obj);

  uint8_t get_face() const;
  void set_reverse(const Node* reverse);
  std::string print_state() const;
  std::string print_solution() const;
};

struct NodeCompare
{
  bool operator() (const Node* a, const Node* b) const;
  bool operator() (const Node& a, const Node& b) const;
};

struct NodeHash
{
  std::size_t operator() (const Node* s) const;
  std::size_t operator() (const Node& s) const;
};

struct NodeEqual
{
  bool operator() (const Node* a, const Node* b) const;
  bool operator() (const Node& a, const Node& b) const;
};
