#pragma once
#include <stdint.h>
#include <iostream>
#include "hash.hpp"

struct Node
{
  const uint8_t* state;
  uint8_t depth;
  uint8_t* faces;
  uint8_t* rotations;
  uint8_t heuristic;
  uint8_t combined;

  Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _face, uint8_t _rotation);
  ~Node();
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
    bool operator== (const Node* a, const Node* b) const;
}
