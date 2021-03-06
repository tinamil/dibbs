#include "Node.h"

Node::Node() :
  #ifdef HISTORY 
  parent(nullptr), reverse_parent(nullptr),
  #endif 
  state(), face(0), depth(0), combined(0), f_bar(0), heuristic(0),
  reverse_heuristic(0), delta(0), passed_threshold(false) {}

Node::Node(const uint8_t* prev_state, const uint8_t* start_state, const Rubiks::PDB type) :
  #ifdef HISTORY 
  parent(nullptr), reverse_parent(nullptr),
  #endif 
  face(0), depth(0)
{
  memcpy(state, prev_state, 40);
  if (start_state != nullptr)
    heuristic = Rubiks::pattern_lookup(state, start_state, type);
  else {
    heuristic = 0;
  }
  reverse_heuristic = 0;
  f_bar = heuristic;
  combined = heuristic;
  delta = 0;
  passed_threshold = heuristic <= reverse_heuristic;
}

Node::Node(const std::shared_ptr<Node> node_parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation,
  const bool reverse, const Rubiks::PDB type) :
  #ifdef HISTORY 
  reverse_parent(nullptr),
  #endif 
  face(_face * 3 + _rotation), depth(_depth)
{
  #ifdef HISTORY
  parent = node_parent;
  #endif
  memcpy(state, node_parent->state, 40);
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
  delta = depth - reverse_heuristic;
  if (node_parent != nullptr && node_parent->passed_threshold) {
    passed_threshold = true;
  }
  else {
    passed_threshold = heuristic <= reverse_heuristic;
  }
  combined = depth + heuristic;
}

Node::Node(const Node* node_parent, const uint8_t* start_state, const uint8_t _depth, const uint8_t _face, const uint8_t _rotation,
  const bool reverse, const Rubiks::PDB type) :
#ifdef HISTORY 
  reverse_parent(nullptr),
#endif 
  face(_face * 3 + _rotation), depth(_depth)
{
#ifdef HISTORY
  parent = std::shared_ptr<const Node>(node_parent);
#endif
  memcpy(state, node_parent->state, 40);
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
  delta = depth - reverse_heuristic;
  if (node_parent != nullptr && node_parent->passed_threshold) {
    passed_threshold = true;
  }
  else {
    passed_threshold = heuristic <= reverse_heuristic;
  }
  combined = depth + heuristic;
}


Node::Node(const Node& old_node) :
  #ifdef HISTORY
  parent(old_node.parent), reverse_parent(old_node.reverse_parent),
  #endif
  state(), face(old_node.face), depth(old_node.depth), combined(old_node.combined), f_bar(old_node.f_bar),
  heuristic(old_node.heuristic), reverse_heuristic(old_node.reverse_heuristic), delta(old_node.delta), passed_threshold(old_node.passed_threshold)
{
  memcpy(state, old_node.state, 40);
}

uint8_t Node::get_face() const
{
  return face / 3;
}

void Node::set_reverse(const std::shared_ptr<Node> reverse)
{
  #ifdef HISTORY
  reverse_parent = reverse;
  #endif
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

#ifdef HISTORY
std::string generate_solution(const Node* node, const std::function<std::string(const Node*)> func, bool reverse = false) {

  const Node* this_parent = node;
  std::string solution;
  while (this_parent != nullptr) {
    std::string tmp = func(this_parent);
    if (reverse)
      std::reverse(tmp.begin(), tmp.end());
    solution.append(tmp);
    this_parent = this_parent->parent.get();
  }
  return solution;
}

std::string get_face_rotation(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr) {
    ss << std::setw(3) << std::setfill(' ');
    ss << Rubiks::_face_mapping[x->face / 3] + Rubiks::_rotation_mapping[x->face % 3];
  }
  ss << " **";
  return ss.str();
}

std::string get_depth(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr)
    ss << std::setw(6);
  else
    ss << std::setw(3);
  ss << std::setfill(' ') << std::to_string(x->depth);
  return ss.str();
}

std::string get_h1(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr)
    ss << std::setw(6);
  else
    ss << std::setw(3);
  ss << std::setfill(' ') << std::to_string(x->heuristic);
  return ss.str();
}
std::string get_h2(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr)
    ss << std::setw(6);
  else
    ss << std::setw(3);
  ss << std::setfill(' ') << std::to_string(x->reverse_heuristic);
  return ss.str();
}
std::string get_combined(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr)
    ss << std::setw(6);
  else
    ss << std::setw(3);
  ss << std::setfill(' ') << std::to_string(x->combined);
  return ss.str();
}
std::string get_fbar(const Node* x) {
  std::stringstream ss;
  if (x->parent != nullptr)
    ss << std::setw(6);
  else
    ss << std::setw(3);
  ss << std::setfill(' ') << std::to_string(x->f_bar);
  return ss.str();
}


std::string generate_bidirectional_solution(const Node* node, std::function<std::string(const Node*)> func, bool skip_middle = true) {
  std::string solution = generate_solution(node, func, true);
  std::reverse(solution.begin(), solution.end());
  solution.append(" |");
  if (skip_middle && node->reverse_parent != nullptr)
    solution.append(generate_solution(node->reverse_parent->parent.get(), func));
  else
    solution.append(generate_solution(node->reverse_parent.get(), func));
  solution.insert(0, "Start ");
  solution.append(" Goal");
  return solution;
}

std::string Node::print_solution() const
{

  std::string solution;
  solution += "\nmove=" + generate_bidirectional_solution(this, get_face_rotation, false);
  solution += "\ng=   " + generate_bidirectional_solution(this, get_depth);
  solution += "\nh =  " + generate_bidirectional_solution(this, get_h1);
  solution += "\nh'=  " + generate_bidirectional_solution(this, get_h2);
  solution += "\nf=   " + generate_bidirectional_solution(this, get_combined);
  solution += "\nfbar=" + generate_bidirectional_solution(this, get_fbar) + "\n";
  return solution;
}
#else
std::string Node::print_solution() const
{
  return "Solutions disabled.";
}
#endif

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

size_t NodeHash::operator() (const Node* s) const
{
  return SuperFastHash(s->state, 40);
}

size_t NodeHash::operator() (const Node& s) const
{
  return SuperFastHash(s.state, 40);
}

size_t NodeHash::operator() (const std::shared_ptr<Node> s) const
{
  return SuperFastHash(s->state, 40);
}

bool NodeEqual::operator() (const Node* a, const Node* b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}

bool NodeEqual::operator() (const Node& a, const Node& b) const
{
  return memcmp(a.state, b.state, 40) == 0;
}

bool NodeEqual::operator() (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) const
{
  return memcmp(a->state, b->state, 40) == 0;
}