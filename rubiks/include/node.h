#pragma once
#include <cstring>
#include <cstdint>
#include <iostream>
#include "hash.hpp"

struct Node
{
  const uint8_t* state;
  uint8_t depth;
  uint8_t* faces;
  uint8_t* rotations;

  uint8_t* reverse_faces;
  uint8_t* reverse_rotations;
  uint8_t  reverse_depth;

  uint8_t heuristic;
  uint8_t reverse_heuristic;

  uint8_t combined;
  uint8_t f_bar;

  Node (Node* _parent, const uint8_t _state[], uint8_t _heuristic);
  Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _face, uint8_t _rotation);
  Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t reverse_heuristic,
        uint8_t _face, uint8_t _rotation);
  Node (const Node* old_obj);
  ~Node();

  uint8_t get_face() const;
  uint8_t get_rotation() const;
  void set_reverse (const Node* reverse);
};

struct NodeCompare
{
  bool operator() (const Node* a, const Node* b) const;
};

struct NodeHash
{
  std::size_t operator() (const Node* s) const;
};

struct NodeEqual
{
  bool operator() (const Node* a, const Node* b) const;
};
