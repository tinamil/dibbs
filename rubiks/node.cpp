#include "Node.h"

Node::Node() : state(), face(0), depth(0), combined(0), f_bar(0), reverse_depth(0), heuristic(0), reverse_heuristic(0) {}

Node::Node(const uint8_t* prev_state, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation,
  const bool reverse, const Rubiks::PDB type, const int min_heuristic, const int min_reverse_heuristic) : face(_face), depth(_depth), reverse_depth(0)
{
  memcpy(state, prev_state, 40);
  Rubiks::rotate(state, _face, _rotation);

  heuristic = Rubiks::pattern_lookup(state, type, min_heuristic);
  if (start_state != nullptr) {
    reverse_heuristic = Rubiks::pattern_lookup(state, start_state, type, min_reverse_heuristic);
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

  //faces = new uint8_t[depth];

  /*if (_parent != NULL)
  {
    memcpy(faces, _parent->faces, depth - 1);
    faces[depth - 1] = _face * 6 + _rotation;
  }*/
}

Node::Node(const Node &old_node) : face(old_node.face), depth(old_node.depth), combined(old_node.combined), reverse_depth(old_node.reverse_depth),
f_bar(old_node.f_bar), heuristic(old_node.heuristic), reverse_heuristic(old_node.reverse_heuristic)
{
  memcpy(state, old_node.state, 40);

  /*faces = new uint8_t[depth];
  memcpy(faces, old_node->faces, depth);

  if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    memcpy(reverse_faces, old_node->reverse_faces, reverse_depth);
  }*/
}

Node::~Node()
{
  /*delete[] state;
  delete[] faces;

  if (reverse_depth > 0)
  {
    delete[] reverse_faces;
  }*/
}

uint8_t Node::get_face() const
{
  return face;// this->faces[this->depth - 1] / 6;
}

void Node::set_reverse(const Node* reverse)
{
  reverse_depth = reverse->depth;
  /*if (reverse_depth > 0)
  {
    reverse_faces = new uint8_t[reverse_depth];
    memcpy(reverse_faces, reverse->faces, reverse_depth);
  }*/
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
  /*if ((depth > 0 && faces == nullptr) || (reverse_depth > 0 && reverse_faces == nullptr)) {
    throw new std::invalid_argument("Cannot print a solution if one of the faces/rotations arrays are null");
  }
  std::string solution;
  /*for (int i = 0; i < depth; ++i) {
    solution.append(Rubiks::_face_mapping[faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[faces[i] % 3]);
    solution.append(" ");
  }
  for (int i = reverse_depth - 1; i >= 0; --i) {
    solution.append(Rubiks::_face_mapping[reverse_faces[i] / 6]);
    solution.append(Rubiks::_rotation_mapping[reverse_faces[i] % 3]);
    solution.append(" ");
  }
  return solution;*/
  return "Solutions disabled.";
}

bool NodeCompare::operator() (const Node* a, const Node* b) const
{
  return a->f_bar > b->f_bar;
}

bool NodeCompare::operator() (const Node& a, const Node& b) const
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

size_t NodeHash::operator() (const Node* s) const
{
  return boost_hash(s->state, 40);
}

bool NodeEqual::operator() (const Node* a, const Node* b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}
