#include "Node.h"

Node::Node(Node* _parent, const uint8_t _state[], uint8_t _heuristic)
  : Node(_parent, _state, 0, _heuristic, 0, 0, 0) {}
Node::Node(Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _face, uint8_t _rotation)
  : Node(_parent, _state, _depth, _heuristic, 0, _face, _rotation) {}

Node::Node(Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _heuristic, uint8_t _reverse_heuristic,
  uint8_t _face, uint8_t _rotation) : state(_state), depth(_depth), heuristic(_heuristic),
  combined(depth + heuristic), reverse_depth(0), reverse_heuristic(_reverse_heuristic), f_bar(depth + _heuristic + depth - _reverse_heuristic)
{
  faces = new uint8_t[depth];

  if (_parent != NULL)
  {
    memcpy(faces, _parent->faces, depth - 1);
    faces[depth - 1] = _face * 6 + _rotation;
  }
}

Node::Node(const Node* old_node) : state(old_node->state), depth(old_node->depth), heuristic(old_node->heuristic),
combined(old_node->combined), reverse_depth(old_node->reverse_depth), reverse_heuristic(old_node->reverse_heuristic), f_bar(old_node->f_bar)
{
  faces = new uint8_t[depth];
  memcpy(faces, old_node->faces, depth);

  if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    memcpy(reverse_faces, old_node->reverse_faces, reverse_depth);
  }
}

Node::~Node()
{
  delete[] state;
  delete[] faces;

  if (reverse_depth > 0)
  {
    delete[] reverse_faces;
  }
}

uint8_t Node::get_face() const
{
  return this->faces[this->depth - 1] / 6;
}

void Node::set_reverse(const Node* reverse)
{
  reverse_depth = reverse->depth;
  if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    memcpy(reverse_faces, reverse->faces, reverse_depth);
  }
}

std::string Node::print_state() const
{
  std::string result;
  for (int i = 0; i < 40; ++i)
  {
    result.append(std::to_string(state[i]));
    result.append(" ");
  }
  return result;
}

std::string Node::print_solution() const
{
  if ((depth > 0 && faces == nullptr) || (reverse_depth > 0 && reverse_faces == nullptr)) {
    throw new std::invalid_argument("Cannot print a solution if one of the faces/rotations arrays are null");
  }
  std::string solution;
  for (int i = 0; i < depth; ++i) {
    solution.append(Rubiks::_face_mapping[faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[faces[i] % 3]);
    solution.append(" ");
  }
  for (int i = reverse_depth - 1; i >= 0; --i) {
    solution.append(Rubiks::_face_mapping[reverse_faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[reverse_faces[i] % 3]);
    solution.append(" ");
  }
  return solution;
}

bool NodeCompare::operator() (const Node* a, const Node* b) const
{
  return a->f_bar > b->f_bar;
}

size_t NodeHash::operator() (const Node* s) const
{
  return boost_hash(s->state, 40); 
}

bool NodeEqual::operator() (const Node* a, const Node* b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}
