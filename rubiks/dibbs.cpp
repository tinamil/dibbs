#include "dibbs.h"

typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node> > > stack;
typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;
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

size_t search::dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "DIBBS" << std::endl;
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
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
  return count;
}

std::shared_ptr<Node> make_node(const hash_set& other_set,
  std::shared_ptr<Node> prev_node,
  const uint8_t* start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::shared_ptr<Node>& best_node,
  uint8_t& upper_bound)
{
  auto new_node = std::make_shared<Node>(prev_node, start_state, prev_node->depth + 1, face, rotation, reverse, type);

  uint8_t reverse_cost = 0;
  auto search = other_set.find(new_node);
  if (search != other_set.end())
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

  return new_node;
}

void expand_node(std::shared_ptr<Node> next_node,
  stack& my_stack,
  hash_set& my_set,
  const hash_set& other_set,
  const unsigned int id_depth,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  size_t& count) {
  count += 1;
  if (count % 1000000 == 0) {
    std::cout << count << "\n";
  }

  for (int face = 0; face < 6; ++face)
  {
    if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = make_node(other_set, next_node, start_state, face, rotation, reverse, type, best_node, upper_bound);
      if (new_node->f_bar <= id_depth) {
        my_stack.push(new_node);
      }
      //-1 is maximum difference between heuristic estimates (g(a-b)
      else if(new_node->heuristic + -1 <= new_node->reverse_heuristic){
        auto existing = my_set.find(next_node);
        if (existing == my_set.end()) {
          my_set.insert(next_node);
        }
        else if ((*existing)->depth > next_node->depth) {
          //Must check because we are searching in DFS order, not BFS
          my_set.erase(existing);
          my_set.insert(next_node);
        }
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  hash_set& my_set,
  hash_set& other_set,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  unsigned int& id_depth,
  const unsigned int other_depth,
  size_t& count,
  int& lambda,
  const int epsilon,
  const size_t node_limit)
{
  std::cout << "Expanding layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  my_set.clear();
  const auto other_size = other_set.size();
  float termination = (id_depth + other_depth) / 2.0f + epsilon;

  while (!my_stack.empty() && upper_bound >= termination) {
    std::shared_ptr<Node> next_node = my_stack.top();
    my_stack.pop();
    expand_node(next_node, my_stack, my_set, other_set, id_depth, upper_bound, best_node, reverse, type, start_state, count);
  }

  std::cout << "Finished expanding layer " << id_depth << "; size= " << my_set.size() << '\n';
  id_depth += 1;
  //std::cout << "Speculatively expanding layer " << id_depth << '\n';
  //termination = (id_depth + other_depth) / 2.0f + epsilon;
  //const std::function<void(const std::shared_ptr<Node>)> speculative_func = [&](std::shared_ptr<Node> new_node) {
  //  if (new_node->f_bar == id_depth + 1)// && new_node->heuristic + lambda <= new_node->reverse_heuristic)
  //  {
  //    auto existing = my_set.find(new_node);
  //    if (existing == my_set.end()) {
  //      my_set.insert(new_node);
  //    }
  //    else if ((*existing)->depth > new_node->depth) {
  //      //Must check because we are searching in DFS order, not BFS
  //      my_set.erase(existing);
  //      my_set.insert(new_node);
  //    }
  //  }
  //};
  //for (auto set_iterator = tmpNodes.begin(), end = tmpNodes.end(); set_iterator != end && upper_bound >= termination; set_iterator++) {
  //  auto next_node = *set_iterator;
  //  expand_node(next_node, other_set, upper_bound, best_node, reverse, type, start_state, count, speculative_func);
  //}
  return true;
}

size_t search::id_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "ID-DIBBS" << std::endl;
  const uint8_t epsilon = 1;
  int lambda = -1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  stack forward_stack, backward_stack;

  auto start = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  auto goal = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);

  hash_set front_set, back_set;
  front_set.insert(start);
  back_set.insert(goal);

  std::shared_ptr<Node> best_node(nullptr);
  size_t count = 0;
  const size_t node_limit = (size_t)1e15;//1e8;

  unsigned int forward_fbar_min(1), backward_fbar_min(1);
  bool reverse = false;

  //epsilon is the smallest edge cost, must be >0 but can be infinitesmal
  while (best_node == nullptr || upper_bound >= (forward_fbar_min + backward_fbar_min) / 2.0f + epsilon)
  {
    if (reverse) {
      backward_stack.push(goal);
      expand_layer(backward_stack, back_set, front_set, upper_bound, best_node, reverse, pdb_type, start_state, backward_fbar_min, forward_fbar_min, count, lambda, epsilon, node_limit);
    }
    else {
      forward_stack.push(start);
      expand_layer(forward_stack, front_set, back_set, upper_bound, best_node, reverse, pdb_type, start_state, forward_fbar_min, backward_fbar_min, count, lambda, epsilon, node_limit);
    }
    reverse = !reverse;
  }

  std::cout << "Solved DIBBS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}
