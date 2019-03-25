#include "dibbs.h"


void expand(std::multiset<Node*, NodeCompare> &front_multi,
  const std::multiset<Node*, NodeCompare> &back_multi,
  uint8_t &upper_bound,
  Node* &best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state)
{
  Node* next_node = *front_multi.begin();
  front_multi.erase(next_node);
  for (int face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node* new_node = new Node(next_node->state, start_state, next_node->depth + 1, face, rotation, reverse, type, next_node->heuristic - 1, next_node->reverse_heuristic - 1);

      front_multi.insert(new_node);

      uint8_t reverse_cost = 0;
      auto search = back_multi.find(new_node);
      if (search != back_multi.end())
      {
        reverse_cost = (*search)->depth;
        if (new_node->depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node->depth + reverse_cost;
          if (best_node != nullptr)
          {
            delete best_node;
          }
          best_node = new Node(*new_node);
          best_node->set_reverse(*search);
          std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
        }
      }
    }
  }
  delete next_node;
}

void expand(std::priority_queue<Node*, std::vector<Node*>, NodeCompare> &frontier,
  std::unordered_set<Node*, NodeHash, NodeEqual> &frontier_set,
  const std::unordered_set<Node*, NodeHash, NodeEqual> &other_set,
  uint8_t &upper_bound,
  Node* &best_node,
  bool reverse,
  Rubiks::PDB type,
  const uint8_t* start_state)
{
  Node* next_node = frontier.top();
  frontier.pop();
  auto node_index = frontier_set.find(next_node);
  if (node_index != frontier_set.end())
  {
    frontier_set.erase(node_index);
  }
  for (int face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node* new_node = new Node(next_node->state, start_state, next_node->depth + 1, face, rotation, reverse, type, 0, 0);

      if (new_node->combined < next_node->combined) {
        std::cout << "DIBBS INCONSISTENCY ERROR: " << unsigned(new_node->combined) << " " << unsigned(next_node->combined) << " " << reverse << std::endl;
      }

      auto search = frontier_set.find(new_node);
      if (search != frontier_set.end()) {
        if ((*search)->depth <= new_node->depth) {
          delete new_node;
          continue;
        }
      }

      frontier_set.insert(new_node);
      frontier.push(new_node);

      uint8_t reverse_cost = 0;
      search = other_set.find(new_node);
      if (search != other_set.end())
      {
        reverse_cost = (*search)->depth;
        if (new_node->depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node->depth + reverse_cost;
          if (best_node != nullptr)
          {
            delete best_node;
          }
          best_node = new Node(*new_node);
          best_node->set_reverse(*search);
          std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
        }
      }
    }
  }
  delete next_node;
}

void search::dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "DIBBS" << std::endl;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  std::priority_queue<Node*, std::vector<Node*>, NodeCompare> front_queue, back_queue;
  std::unordered_set<Node*, NodeHash, NodeEqual> front_set, back_set;
  //std::multiset<Node*, NodeCompare> front_multi, back_multi;

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  Node* start = new Node();
  memcpy(start->state, start_state, 40);
  start->depth = 0;
  uint8_t h = Rubiks::pattern_lookup(start->state, pdb_type, 0);
  start->reverse_depth = 0;
  start->combined = h;
  start->f_bar = h;
  //front_multi.insert(start);
  front_queue.push(start);
  front_set.insert(start);

  Node* goal = new Node();
  memcpy(goal->state, Rubiks::__goal, 40);
  goal->depth = 0;
  h = Rubiks::pattern_lookup(goal->state, start_state, pdb_type, 0);
  goal->reverse_depth = 0;
  goal->combined = h;
  goal->f_bar = h;
  //back_multi.insert(goal);
  back_queue.push(goal);
  back_set.insert(goal);

  bool explore_forward = true;
  Node* best_node = nullptr;
  uint8_t forward_fbar_min = 0;
  uint8_t backward_fbar_min = 0;
  int count = 0;

  while (best_node == nullptr || upper_bound > (forward_fbar_min + backward_fbar_min) / 2)
  {

    if (forward_fbar_min == backward_fbar_min) {
      //explore_forward = front_multi.size() < back_multi.size();
      explore_forward = front_queue.size() < back_queue.size();
    }
    else {
      explore_forward = forward_fbar_min <= backward_fbar_min;
    }

    if (explore_forward)
    {
      //expand(front_multi, back_multi, upper_bound, best_node, false, pdb_type, start_state);
      //forward_fbar_min = (*front_multi.begin())->f_bar;
      expand(front_queue, front_set, back_set, upper_bound, best_node, false, pdb_type, start_state);
      forward_fbar_min = front_queue.top()->f_bar;
    }
    else
    {
      //expand(back_multi, front_multi, upper_bound, best_node, true, pdb_type, start_state);
      //backward_fbar_min = (*back_multi.begin())->f_bar;
      expand(back_queue, back_set, front_set, upper_bound, best_node, true, pdb_type, start_state);
      backward_fbar_min = back_queue.top()->f_bar;
    }

    count += 1;

    if (count % 100000 == 0)
    {
      std::cout << unsigned(forward_fbar_min) << " " << unsigned(backward_fbar_min) << " ";
      //std::cout << unsigned((*front_multi.begin())->depth) << " " << unsigned((*back_multi.begin())->depth) << " " << count << " ";
      //std::cout << "FQueue: " << front_multi.size() << " BQueue: " << back_multi.size() << "\n";
      std::cout << unsigned(front_queue.top()->depth) << " " << unsigned(back_queue.top()->depth) << " " << count << " ";
      std::cout << "FQueue: " << front_queue.size() << " " << front_set.size() << " BQueue: " << back_queue.size() << " " << back_set.size() << "\n";
    }
  }

  std::cout << "Solved DIBBS: " << unsigned int(best_node->depth + best_node->reverse_depth) << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;


  std::cout << "Cleaning up open/closed frontiers" << std::endl;
  /*while (front_multi.empty() == false)
  {
    Node* node = (*front_multi.begin());
    front_multi.erase(node);
    delete node;
  }

  while (back_multi.empty() == false)
  {
    Node* node = (*back_multi.begin());
    back_multi.erase(node);
    delete node;
  }*/
  while (front_queue.empty() == false) {
    delete front_queue.top();
    front_queue.pop();
  }
  while (back_queue.empty() == false) {
    delete back_queue.top();
    back_queue.pop();
  }
}

