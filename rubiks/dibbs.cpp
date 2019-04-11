#include "dibbs.h"


void expand(std::multiset<Node, NodeCompare>& front_multi,
  const std::multiset<Node, NodeCompare>& back_multi,
  uint8_t& upper_bound,
  Node*& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state)
{
  Node next_node = *front_multi.begin();
  front_multi.erase(next_node);
  for (int face = 0; face < 6; ++face)
  {
    if (next_node.depth > 0 && Rubiks::skip_rotations(next_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node new_node(next_node.state, start_state, next_node.depth + 1, face, rotation, reverse, type);

      uint8_t reverse_cost = 0;
      auto search = back_multi.find(new_node);
      if (search != back_multi.end())
      {
        reverse_cost = (*search).depth;
        if (new_node.depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node.depth + reverse_cost;
          if (best_node != nullptr)
          {
            delete best_node;
          }
          best_node = new Node(new_node);
          best_node->set_reverse(&*search);
          std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
        }
      }
      else {
        front_multi.insert(new_node);
      }
    }
  }
}

void search::dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "DIBBS" << std::endl;
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  std::multiset<Node, NodeCompare> front_multi, back_multi;

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  Node start;
  memcpy(start.state, start_state, 40);
  start.depth = 0;
  uint8_t h = Rubiks::pattern_lookup(start.state, pdb_type);
  start.reverse_depth = 0;
  start.combined = h;
  start.f_bar = h;
  front_multi.insert(start);

  Node goal;
  memcpy(goal.state, Rubiks::__goal, 40);
  goal.depth = 0;
  h = Rubiks::pattern_lookup(goal.state, start_state, pdb_type);
  goal.reverse_depth = 0;
  goal.combined = h;
  goal.f_bar = h;
  back_multi.insert(goal);

  bool explore_forward = true;
  Node* best_node = nullptr;
  int count = 0;

  uint8_t forward_fbar_min(0), backward_fbar_min(0);

  while (best_node == nullptr || upper_bound > (forward_fbar_min + uint64_t(backward_fbar_min)) / 2.0 - epsilon)
  {
    explore_forward = forward_fbar_min < backward_fbar_min;
    if (forward_fbar_min == backward_fbar_min && best_node == nullptr) {
      explore_forward = (*front_multi.begin()).combined < (*back_multi.begin()).combined;
    }

    if (explore_forward)
    {
      expand(front_multi, back_multi, upper_bound, best_node, false, pdb_type, start_state);
      forward_fbar_min = (*front_multi.begin()).f_bar;
    }
    else
    {
      expand(back_multi, front_multi, upper_bound, best_node, true, pdb_type, start_state);
      backward_fbar_min = (*back_multi.begin()).f_bar;
    }

    count += 1;

    if (count % 100000 == 0)
    {
      int h_balance = 0;
      for (auto id = front_multi.begin(); id != front_multi.end(); ++id) {
        Node n = *id;
        if (n.heuristic - 1 >= n.reverse_heuristic) {
          h_balance++;
        }
      }
      std::cout << unsigned(forward_fbar_min) << " " << unsigned(backward_fbar_min) << " ";
      std::cout << "Front: " << unsigned((*front_multi.begin()).depth) << " " << unsigned((*front_multi.begin()).heuristic) << " " << unsigned((*front_multi.begin()).reverse_heuristic);
      std::cout << " Back: " << unsigned((*back_multi.begin()).depth) << " " << unsigned((*back_multi.begin()).heuristic) << " " << unsigned((*back_multi.begin()).reverse_heuristic);
      std::cout << " " << count << " ";
      std::cout << "FQueue: " << front_multi.size() << " BQueue: " << back_multi.size();
      std::cout << " H-balance: " << h_balance << " " << double(h_balance) / front_multi.size() << '\n';
    }
  }

  std::cout << "Solved DIBBS: " << unsigned int(best_node->depth + best_node->reverse_depth) << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
}


void expand_backwards(std::stack<Node, std::vector<Node>> & front_stack,
  const std::unordered_set<Node, NodeHash, NodeEqual> & back_multi,
  uint8_t & upper_bound,
  Node * &best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t * start_state,
  const int id_depth)
{
  Node next_node = front_stack.top();
  front_stack.pop();
  for (int face = 0; face < 6; ++face)
  {
    if (next_node.depth > 0 && Rubiks::skip_rotations(next_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node new_node(next_node.state, start_state, next_node.depth + 1, face, rotation, reverse, type);

      if (new_node.f_bar <= id_depth) {
        front_stack.push(new_node);
      }

      uint8_t reverse_cost = 0;
      auto search = back_multi.find(new_node);
      if (search != back_multi.end())
      {
        reverse_cost = (*search).depth;
        if (new_node.depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node.depth + reverse_cost;
          if (best_node != nullptr)
          {
            delete best_node;
          }
          best_node = new Node(new_node);
          best_node->set_reverse(&*search);
          std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
        }
      }
    }
  }
}

bool expand_forward(std::stack<Node, std::vector<Node>> & front_stack,
  std::unordered_set<Node, NodeHash, NodeEqual> & front_multi,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t * start_state,
  const int id_depth,
  size_t & count,
  const uint8_t lambda)
{
  std::cout << "Expanding forward layer " << id_depth << '\n';
  while (front_stack.empty() == false) {
    Node next_node = front_stack.top();
    front_stack.pop();
    count += 1;
    if (count % 1000000 == 0) {
      std::cout << count << "\n";
    }
    for (int face = 0; face < 6; ++face)
    {
      if (next_node.depth > 0 && Rubiks::skip_rotations(next_node.get_face(), face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        Node new_node(next_node.state, start_state, next_node.depth + 1, face, rotation, reverse, type);

        if (new_node.f_bar <= id_depth) {
          front_stack.push(new_node);
        }

        if (new_node.heuristic <= new_node.reverse_heuristic + lambda) {
          auto existing = front_multi.find(new_node);
          if (existing == front_multi.end()) {
            front_multi.insert(new_node);
          }
          else if ((*existing).depth > new_node.depth) {
            front_multi.erase(existing);
            front_multi.insert(new_node);
          }
        }
      }
    }
  }
  std::cout << "Finished expanding forward layer " << id_depth << "; size= " << front_multi.size() << '\n';
  return true;
}

size_t search::id_dibbs(const uint8_t * start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "ID-DIBBS" << std::endl;
  const uint8_t epsilon = 1;
  const uint8_t lambda = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  std::unordered_set<Node, NodeHash, NodeEqual> front_multi;

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  std::stack<Node, std::vector<Node>> forward_stack, backward_stack;

  Node start(start_state, Rubiks::__goal, pdb_type);
  forward_stack.push(start);

  Node goal(Rubiks::__goal, start_state, pdb_type);
  backward_stack.push(goal);

  Node* best_node = nullptr;
  size_t count = 0;
  size_t id_limit = 15;

  unsigned int forward_fbar_min(0), backward_fbar_min(0);
  //epsilon is the smallest edge cost, must be >0 but can be infinitesmal
  while (best_node == nullptr || upper_bound >= (forward_fbar_min + backward_fbar_min) / 2.0f + epsilon)
  {
    if (backward_stack.empty()) {
      forward_stack.push(start);
      backward_stack.push(goal);
      if (forward_fbar_min < id_limit) {
        forward_fbar_min += 1;
        expand_forward(forward_stack, front_multi, false, pdb_type, start_state, forward_fbar_min, count, lambda);
      }
      backward_fbar_min += 1;
    }

    expand_backwards(backward_stack, front_multi, upper_bound, best_node, true, pdb_type, start_state, backward_fbar_min);

    count += 1;

    if (count % 1000000 == 0) {
      std::cout << count << "\n";
    }
  }

  std::cout << "Solved DIBBS: " << unsigned int(best_node->depth + best_node->reverse_depth) << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}
