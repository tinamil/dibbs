#include "dibbs.h"

typedef std::multiset<std::shared_ptr<Node>, NodeCompare> multi_set;

void expand(multi_set& front_multi,
  const multi_set& back_multi,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state)
{
  auto next_node = *front_multi.begin();
  front_multi.erase(next_node);
  for (int face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = std::make_shared<Node>(next_node, start_state, next_node->depth + 1, face, rotation, reverse, type);

      uint8_t reverse_cost = 0;
      auto search = back_multi.find(new_node);
      if (search != back_multi.end())
      {
        reverse_cost = (*search)->depth;
        if (new_node->depth + reverse_cost < upper_bound)
        {
          upper_bound = new_node->depth + reverse_cost;
          best_node = new_node;
          best_node->set_reverse(*search);
          std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
        }
      }
      else {
        front_multi.insert(new_node);
      }
    }
  }
}

std::pair<uint64_t, double> search::dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  auto c_start = clock();
  std::cout << "DIBBS" << std::endl;
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0,0);
  }

  multi_set front_multi, back_multi;

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  auto start = std::make_shared<Node>();
  memcpy(start->state, start_state, 40);
  start->depth = 0;
  uint8_t h = Rubiks::pattern_lookup(start->state, pdb_type);
  start->combined = h;
  start->f_bar = h;
  front_multi.insert(start);

  auto goal = std::make_shared<Node>();
  memcpy(goal->state, Rubiks::__goal, 40);
  goal->depth = 0;
  h = Rubiks::pattern_lookup(goal->state, start_state, pdb_type);
  goal->combined = h;
  goal->f_bar = h;
  back_multi.insert(goal);

  bool explore_forward = true;
  std::shared_ptr<Node> best_node = nullptr;
  int count = 0;

  uint8_t forward_fbar_min(0), backward_fbar_min(0);

  while (best_node == nullptr || upper_bound > ((forward_fbar_min + uint64_t(backward_fbar_min)) / 2.0 - epsilon))
  {
    explore_forward = forward_fbar_min < backward_fbar_min;
    if (forward_fbar_min == backward_fbar_min && best_node == nullptr) {
      explore_forward = (*front_multi.begin())->combined < (*back_multi.begin())->combined;
    }

    if (explore_forward)
    {
      expand(front_multi, back_multi, upper_bound, best_node, false, pdb_type, start_state);
      forward_fbar_min = (*front_multi.begin())->f_bar;
    }
    else
    {
      expand(back_multi, front_multi, upper_bound, best_node, true, pdb_type, start_state);
      backward_fbar_min = (*back_multi.begin())->f_bar;
    }

    count += 1;

    if (count % 100000 == 0)
    {
      int h_balance = 0;
      for (auto id = front_multi.begin(); id != front_multi.end(); ++id) {
        auto n = (*id);
        if (n->heuristic - 1 >= n->reverse_heuristic) {
          h_balance++;
        }
      }
      std::cout << unsigned(forward_fbar_min) << " " << unsigned(backward_fbar_min) << " ";
      std::cout << "Front: " << unsigned((*front_multi.begin())->depth) << " " << unsigned((*front_multi.begin())->heuristic) << " " << unsigned((*front_multi.begin())->reverse_heuristic);
      std::cout << " Back: " << unsigned((*back_multi.begin())->depth) << " " << unsigned((*back_multi.begin())->heuristic) << " " << unsigned((*back_multi.begin())->reverse_heuristic);
      std::cout << " " << count << " ";
      std::cout << "FQueue: " << front_multi.size() << " BQueue: " << back_multi.size();
      std::cout << " H-balance: " << h_balance << " " << double(h_balance) / front_multi.size() << '\n';
    }
  }

  std::cout << "Solved DIBBS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(count, time_elapsed);
}
