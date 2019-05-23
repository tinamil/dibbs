#include "id_dibbs.h"

typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node> > > stack;
typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;

//TODO: 
//Verify H1 <= H2 (vs -1)
//Implement memory sweep when running out
//  1) Dump the current frontier when checking the other frontier
//  2) Sweep fractions of the frontier
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
      else if (new_node->passed_threshold) {
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

bool is_done(const uint8_t upper_bound,
  const unsigned int id_depth,
  const unsigned int other_depth,
  const int epsilon) {
  return upper_bound >= (id_depth + other_depth) / 2.0f + epsilon;
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

  while (!my_stack.empty() && !is_done(upper_bound, id_depth, other_depth, epsilon)) {
    std::shared_ptr<Node> next_node = my_stack.top();
    my_stack.pop();
    expand_node(next_node, my_stack, my_set, other_set, id_depth, upper_bound, best_node, reverse, type, start_state, count);
  }

  std::cout << "Finished expanding layer " << id_depth << "; size= " << my_set.size() << '\n';
  id_depth += 1;
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

  unsigned int forward_fbar_min(start->f_bar), backward_fbar_min(goal->f_bar);
  bool reverse = false;

  //epsilon is the smallest edge cost, must be >0 but can be infinitesmal
  while (best_node == nullptr || !is_done(upper_bound, forward_fbar_min, backward_fbar_min, epsilon))
  {
    if (reverse) {
      backward_stack.push(goal);
      expand_layer(backward_stack, back_set, front_set, upper_bound, best_node, reverse, pdb_type, start_state, backward_fbar_min, forward_fbar_min, count, lambda, epsilon, node_limit);
    }
    else {
      forward_stack.push(start);
      expand_layer(forward_stack, front_set, back_set, upper_bound, best_node, reverse, pdb_type, start_state, forward_fbar_min, backward_fbar_min, count, lambda, epsilon, node_limit);
    }
    reverse = backward_fbar_min <= forward_fbar_min;
  }

  std::cout << "Solved DIBBS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}
