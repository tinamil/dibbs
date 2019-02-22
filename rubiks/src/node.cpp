#include "Node.h"

Node::Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _face, uint8_t _rotation)
{
  state = _state;
  depth = _depth;
  heuristic = _heuristic;
  combined = depth + heuristic;

  faces = new uint8_t[depth];
  rotations = new uint8_t[depth];

  if (_parent != NULL)
  {
    memcpy (faces, _parent->faces, depth - 1);
    memcpy (rotations, _parent->rotations, depth - 1);
    faces[depth - 1] = _face;
    rotations[depth - 1] = _rotation;
  }
}

Node::~Node()
{
  delete[] state;
  delete[] faces;
  delete[] rotations;
}

bool NodeCompare::operator() (const Node* a, const Node* b) const
{
  return a->combined < b->combined;
}

size_t NodeHash::operator() (const Node* s) const
{
  return SuperFastHash (s->state, 40);
}

bool NodeEqual::operator== (const Node* a, const Node* b) const
{
  return memcmp (a.state, b.state, 40) == 0;
}
