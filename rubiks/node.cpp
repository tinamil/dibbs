#include "Node.h"


Node::Node (Node* _parent, const uint8_t _state[], uint8_t _heuristic)
  : Node (_parent, _state, 0, _heuristic, 0, 0, 0) {}
Node::Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _face, uint8_t _rotation)
  : Node (_parent, _state, _depth, _heuristic, 0, _face, _rotation) {}

Node::Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _reverse_heuristic,
            uint8_t _face, uint8_t _rotation) : state (_state), depth (_depth), heuristic (_heuristic),
  combined (depth + heuristic), reverse_depth (0), reverse_heuristic (_reverse_heuristic), f_bar (combined + depth - reverse_heuristic)
{
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

Node::Node (const Node* old_node) : state (old_node->state), depth (old_node->depth), heuristic (old_node->heuristic),
  combined (old_node->combined), reverse_depth (old_node->reverse_depth), reverse_heuristic (old_node->reverse_heuristic), f_bar (old_node->f_bar)
{
  faces = new uint8_t[depth];
  rotations = new uint8_t[depth];

  memcpy (faces, old_node->faces, depth);
  memcpy (rotations, old_node->rotations, depth);

  if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    reverse_rotations = new uint8_t[reverse_depth];

    memcpy (reverse_faces, old_node->reverse_faces, reverse_depth);
    memcpy (reverse_rotations, old_node->reverse_rotations, reverse_depth);
  }
}

Node::~Node()
{
  delete[] state;
  delete[] faces;
  delete[] rotations;

  if (reverse_depth > 0)
  {
    delete[] reverse_faces;
    delete[] reverse_rotations;
  }
}

uint8_t Node::get_face() const
{
  return this->faces[this->depth - 1];
}

uint8_t Node::get_rotation() const
{
  return this->rotations[this->depth - 1];
}

void Node::set_reverse (const Node* reverse)
{
  reverse_depth = reverse->depth;
  if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    reverse_rotations = new uint8_t[reverse_depth];

    memcpy (reverse_faces, reverse->faces, reverse_depth);
    memcpy (reverse_rotations, reverse->rotations, reverse_depth);
  }
}

std::string Node::print_state()
{
  std::string result;
  for (int i = 0; i < 40; ++i)
  {
    result.append (std::to_string (state[i]) );
    result.append (" ");
  }
  return result;
}

bool NodeCompare::operator() (const Node* a, const Node* b) const
{
  return a->f_bar > b->f_bar;
}

size_t NodeHash::operator() (const Node* s) const
{
  return SuperFastHash (s->state, 40);
}

bool NodeEqual::operator() (const Node* a, const Node* b) const
{
  return memcmp (a->state, b->state, 40) == 0;
}
