#include "dibbs.h"


void expand (std::priority_queue<Node*, std::vector<Node*>, NodeCompare> &frontier,
             std::unordered_multiset<Node*, NodeHash, NodeEqual> &frontier_set,
             const std::unordered_multiset<Node*, NodeHash, NodeEqual> &other_set,
             uint64_t &upper_bound, Node* &best_node, bool reverse)
{
  uint8_t* new_state;
  Node* next_node = frontier.top();
  frontier.pop();

  auto node_index = frontier_set.find (next_node);
  frontier_set.erase (node_index);

  for (uint8_t face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations (next_node->get_face(), face) )
    {
      continue;
    }
    for (uint8_t rotation = 0; rotation < 3; ++rotation)
    {
      new_state = new uint8_t[40];
      memcpy (new_state, next_node->state, 40);
      Rubiks::rotate (new_state, face, rotation);

      uint8_t new_state_heuristic = Rubiks::pattern_database_lookup (new_state);
      uint8_t reverse_heuristic = 0; // TODO
      if (reverse)
      {

      }

      Node* new_node = new Node (next_node, new_state, next_node->depth + 1,
                                 new_state_heuristic, reverse_heuristic, face, rotation);
      frontier_set.insert (new_node);
      frontier.push (new_node);
      uint8_t reverse_cost = 0;
      auto search = other_set.find (new_node);
      if (search != other_set.end() )
      {
        reverse_cost = (*search)->depth;
        if (new_node->depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node->depth + reverse_cost;
          if (best_node != nullptr)
          {
            delete best_node;
          }
          best_node = new Node (next_node);
          best_node->set_reverse (*search);
          std::cout << "New upper bound: " << upper_bound << std::endl;
        }
      }
    }
  }
  delete next_node;
}

void search::dibbs (const uint8_t state[])
{
  std::cout << "DIBBS" << std::endl;

  if (Rubiks::is_solved (state) )
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  std::priority_queue<Node*, std::vector<Node*>, NodeCompare> front_queue, back_queue;
  std::unordered_multiset<Node*, NodeHash, NodeEqual> front_set, back_set;

  uint64_t upper_bound = std::numeric_limits<uint64_t>::max();
  uint8_t r_heuristic = 0;

  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  front_queue.push (new Node (NULL, new_state, Rubiks::pattern_database_lookup (new_state) ) );
  front_set.insert (front_queue.top() );

  new_state = new uint8_t[40];
  memcpy (new_state, Rubiks::__goal, 40);
  back_queue.push (new Node (NULL, new_state, r_heuristic) );
  back_set.insert (back_queue.top() );

  bool explore_forward = true;
  Node* best_node;
  uint8_t forward_fbar_min = 0;
  uint8_t backward_fbar_min = 0;
  int count = 0;

  while (count < 1e6 && upper_bound > (forward_fbar_min + backward_fbar_min) / 2)
  {
    explore_forward = forward_fbar_min <= backward_fbar_min;
    if (explore_forward)
    {
      expand (front_queue, front_set, back_set, upper_bound, best_node, false);
      forward_fbar_min = front_queue.top()->f_bar;
    }
    else
    {
      expand (back_queue, back_set, front_set, upper_bound, best_node, true);
      backward_fbar_min = back_queue.top()->f_bar;
    }

    count += 1;

    if (count % 100000 == 0)
    {
      std::cout << unsigned(forward_fbar_min) << " " << unsigned(backward_fbar_min) << " ";
      std::cout << count << "\n";
    }

  }

  std::cout << "Solved DIBBS: " << (best_node->depth + best_node->reverse_depth) << " Count = " << count << std::endl;

  /*
  std::cout << "Cleaning up open frontiers" << std::endl;
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
  */
}

