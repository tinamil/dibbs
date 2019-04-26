#include "Node.h"

Node::Node() : parent(nullptr), reverse_parent(nullptr), state(), face(0), depth(0), combined(0), f_bar(0), heuristic(0),
reverse_heuristic(0) {}

Node::Node(const uint8_t* prev_state, const uint8_t* start_state, const Rubiks::PDB type) : parent(nullptr), reverse_parent(nullptr), face(0), depth(0)
{
  memcpy(state, prev_state, 40);
  heuristic = Rubiks::pattern_lookup(state, start_state, type);
  reverse_heuristic = 0;
  f_bar = heuristic;
  combined = heuristic;
}

Node::Node(const std::shared_ptr<Node> node_parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation,
  const bool reverse, const Rubiks::PDB type) : reverse_parent(nullptr), face(_face * 6 + _rotation), depth(_depth)
{
  parent = node_parent;
  memcpy(state, parent->state, 40);
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
}

Node::Node(const Node& old_node) : parent(old_node.parent), reverse_parent(old_node.reverse_parent), state(),
face(old_node.face), depth(old_node.depth), combined(old_node.combined), f_bar(old_node.f_bar),
heuristic(old_node.heuristic), reverse_heuristic(old_node.reverse_heuristic)
{
  memcpy(state, old_node.state, 40);
}

uint8_t Node::get_face() const
{
  return face / 6;
}

void Node::set_reverse(const std::shared_ptr<Node> reverse)
{
  reverse_parent = reverse;
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
  const Node* this_parent = this;
  std::string solution;
  while (this_parent != nullptr) {
    solution.append(" ");
    if (this_parent->parent != nullptr) {
      solution.append(Rubiks::_rotation_mapping[this_parent->face % 3]);
      solution.append(Rubiks::_face_mapping[this_parent->face / 6]);
    }
    else {
      solution.append("tratS"); //Reversed "Start"
    }
    this_parent = this_parent->parent.get();
  }
  std::reverse(solution.begin(), solution.end());
  this_parent = reverse_parent.get();
  while (this_parent != nullptr && this_parent->parent != nullptr) {
    solution.append(Rubiks::_face_mapping[this_parent->face / 6]);
    solution.append(Rubiks::_rotation_mapping[this_parent->face % 3]);
    solution.append(" ");
    this_parent = this_parent->parent.get();
  }
  solution.append("Goal");
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

bool NodeCompare::operator() (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) const
{
  int cmp = memcmp(a->state, b->state, 40);
  if (cmp == 0) {
    return false;
  }
  else if (a->f_bar == b->f_bar) {
    return cmp < 0;
  }
  else {
    return a->f_bar < b->f_bar;
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

size_t NodeHash::operator() (const std::shared_ptr<Node> s) const
{
  return boost_hash(s->state, 40);
}

bool NodeEqual::operator() (const Node * a, const Node * b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}

bool NodeEqual::operator() (const Node & a, const Node & b) const
{
  return memcmp(a.state, b.state, 40) == 0;
}

bool NodeEqual::operator() (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}