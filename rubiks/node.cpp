#include "Node.h"

Node::Node() : state(), face(0), depth(0), combined(0), f_bar(0), reverse_depth(0), heuristic(0),
reverse_heuristic(0), faces(), reverse_faces() {}

Node::Node(const uint8_t* prev_state, const uint8_t* start_state, const Rubiks::PDB type) : face(0), depth(0), reverse_depth(0),
faces(), reverse_faces() {
  memcpy(state, prev_state, 40);
  heuristic = Rubiks::pattern_lookup(state, start_state, type);
  reverse_heuristic = 0;
  f_bar = heuristic;
  combined = heuristic;
}

Node::Node(const Node& parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation,
  const bool reverse, const Rubiks::PDB type) : face(_face), depth(_depth), reverse_depth(0), reverse_faces()
{
  memcpy(state, parent.state, 40);
  Rubiks::rotate(state, _face, _rotation);

  heuristic = Rubiks::pattern_lookup(state, type);
  if (start_state != nullptr) {
    reverse_heuristic = Rubiks::pattern_lookup(state, start_state, type);
    if (reverse)
    {
      uint8_t tmp = heuristic;
      heuristic = reverse_heuristic;
      reverse_heuristic = tmp;
    }
    f_bar = depth + heuristic + depth - reverse_heuristic;
  }
  else {
    f_bar = 0;
    reverse_heuristic = 0;
  }
  combined = depth + heuristic;

    faces[depth - 1] = _face * 6 + _rotation;
  if (depth > 1) {
    memcpy(faces, parent.faces, depth - 1);
  }
}

Node::Node(const Node & old_node) : state(), faces(), face(old_node.face), depth(old_node.depth), combined(old_node.combined),
reverse_depth(old_node.reverse_depth), f_bar(old_node.f_bar), heuristic(old_node.heuristic),
reverse_heuristic(old_node.reverse_heuristic), reverse_faces()
{
  memcpy(state, old_node.state, 40);

  if (old_node.faces != nullptr) {
    memcpy(faces, old_node.faces, depth);
  }

  if (reverse_depth > 0)
  {
    memcpy(reverse_faces, old_node.reverse_faces, reverse_depth);
  }
}

uint8_t Node::get_face() const
{
  return face;// this->faces[this->depth - 1] / 6;
}

void Node::set_reverse(const Node * reverse)
{
  reverse_depth = reverse->depth;
  if (reverse_depth > 0)
  {
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
    solution.append(Rubiks::_face_mapping[faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[faces[i] % 3]);
  for (int i = 0; i < depth; ++i) {
    solution.append(" ");
  }
  for (int i = reverse_depth - 1; i >= 0; --i) {
    solution.append(Rubiks::_face_mapping[reverse_faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[reverse_faces[i] % 3]);
    solution.append(" ");
  }
  return solution;
  //return "Solutions disabled.";
}

bool NodeCompare::operator() (const Node * a, const Node * b) const
{
  return a->f_bar > b->f_bar;
}

bool NodeCompare::operator() (const Node & a, const Node & b) const
{
  int cmp = memcmp(a.state, b.state, 40);
  if (cmp == 0) {
    return false;
  }
  else if (a.f_bar == b.f_bar) {
    return cmp < 0;
  }
  else {
    return a.f_bar < b.f_bar;
  }
}

size_t NodeHash::operator() (const Node * s) const
{
  return boost_hash(s->state, 40);
}

size_t NodeHash::operator() (const Node & s) const
{
  return boost_hash(s.state, 40);
}

bool NodeEqual::operator() (const Node * a, const Node * b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}

bool NodeEqual::operator() (const Node & a, const Node & b) const
{
  return memcmp(a.state, b.state, 40) == 0;
}