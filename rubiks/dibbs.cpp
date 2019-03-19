#include "dibbs.h"


void expand(std::priority_queue<Node*, std::vector<Node*>, NodeCompare> &frontier,
  std::unordered_set<Node*, NodeHash, NodeEqual> &frontier_set,
  std::unordered_set<Node*, NodeHash, NodeEqual> &frontier_closed,
  const std::unordered_set<Node*, NodeHash, NodeEqual> &other_set,
  uint8_t &upper_bound, Node* &best_node, bool reverse, Rubiks::PDB type,
  const uint8_t start_state[])
{
  uint8_t* new_state;
  Node* next_node = frontier.top();
  frontier.pop();
  //frontier_closed.insert (next_node);
  auto node_index = frontier_set.find(next_node);
  if (node_index == frontier_set.end())
  {
    //std::cout << "BUG: " << frontier_set.size() << " " << frontier.size() << "NULL: " << next_node->print_state() << std::endl;
  }
  else
  {
    frontier_set.erase(node_index);
  }
  for (uint8_t face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
    {
      continue;
    }
    for (uint8_t rotation = 0; rotation < 3; ++rotation)
    {
      new_state = Rubiks::rotate(next_node->state, face, rotation);

      uint8_t new_state_heuristic = Rubiks::pattern_lookup(new_state, type);
      uint8_t reverse_heuristic = Rubiks::pattern_lookup(new_state, start_state, type);
      if (reverse)
      {
        std::swap(new_state_heuristic, reverse_heuristic);
      }

      Node* new_node = new Node(next_node, new_state, next_node->depth + 1,
        new_state_heuristic, reverse_heuristic, face, rotation);

      if (frontier_closed.count(new_node) > 0)
      {
        delete new_node;
        continue;
      }

      frontier_set.insert(new_node);
      frontier.push(new_node);
      uint8_t reverse_cost = 0;
      auto search = other_set.find(new_node);
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
          best_node = new Node(next_node);
          best_node->set_reverse(*search);
          std::cout << "New upper bound: " << upper_bound << std::endl;
        }
      }
    }
  }
}

void search::dibbs(const uint8_t start_state[], const Rubiks::PDB pdb_type)
{
  std::cout << "DIBBS" << std::endl;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  std::priority_queue<Node*, std::vector<Node*>, NodeCompare> front_queue, back_queue;
  std::unordered_set<Node*, NodeHash, NodeEqual> front_set, back_set;
  std::unordered_set<Node*, NodeHash, NodeEqual> front_closed, back_closed;

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  uint8_t* new_state = new uint8_t[40];
  memcpy(new_state, start_state, 40);
  front_queue.push(new Node(NULL, new_state, Rubiks::pattern_lookup(new_state, pdb_type)));
  front_set.insert(front_queue.top());

  new_state = new uint8_t[40];
  memcpy(new_state, Rubiks::__goal, 40);
  back_queue.push(new Node(NULL, new_state, Rubiks::pattern_lookup(new_state, start_state, pdb_type)));
  back_set.insert(back_queue.top());

  bool explore_forward = true;
  Node* best_node = nullptr;
  uint8_t forward_fbar_min = 0;
  uint8_t backward_fbar_min = 0;
  int count = 0;

  while (best_node == nullptr || upper_bound > (forward_fbar_min + backward_fbar_min) / 2)
  {

    explore_forward = forward_fbar_min <= backward_fbar_min;
    if (explore_forward)
    {
      expand(front_queue, front_set, front_closed, back_set, upper_bound, best_node, false, pdb_type, start_state);
      forward_fbar_min = front_queue.top()->f_bar;
    }
    else
    {
      expand(back_queue, back_set, back_closed, front_set, upper_bound, best_node, true, pdb_type, start_state);
      backward_fbar_min = back_queue.top()->f_bar;
    }

    count += 1;

    if (count % 100000 == 0)
    {
      std::cout << unsigned(forward_fbar_min) << " " << unsigned(backward_fbar_min) << " ";
      std::cout << unsigned(front_queue.top()->depth) << " " << unsigned(back_queue.top()->depth) << " " << count << "\n";
    }

  }

  std::cout << "Solved DIBBS: " << (best_node->depth + best_node->reverse_depth) << " Count = " << count << std::endl;


  std::cout << "Cleaning up open/closed frontiers" << std::endl;
  while (front_queue.empty() == false)
  {
    Node* node = front_queue.top();
    front_queue.pop();
    delete node;
  }

  while (back_queue.empty() == false)
  {
    Node* node = back_queue.top();
    back_queue.pop();
    delete node;
  }

  front_closed.erase(front_closed.begin(), front_closed.end());
  back_closed.erase(back_closed.begin(), back_closed.end());

}

