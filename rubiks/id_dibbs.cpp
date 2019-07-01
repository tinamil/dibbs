#include "id_dibbs.h"

typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node> > > stack;
typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;

std::shared_ptr<Node> make_node(const hash_set* other_set,
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
  if (other_set != nullptr) {
    uint8_t reverse_cost = 0;
    auto search = other_set->find(new_node);
    if (search != other_set->end())
    {
      reverse_cost = (*search)->depth;
      if (new_node->depth + reverse_cost < upper_bound)
      {
        upper_bound = new_node->depth + reverse_cost;
        if (reverse == false) {
          best_node = new_node;
          best_node->set_reverse(*search);
        }
        else {
          best_node = *search;
          best_node->set_reverse(new_node);
        }
        std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
      }
    }
  }
  return new_node;
}

void expand_node(std::shared_ptr<Node> prev_node,
  stack& my_stack,
  hash_set* my_set,
  const hash_set* other_set,
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
    if (prev_node->depth > 0 && Rubiks::skip_rotations(prev_node->get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = make_node(other_set, prev_node, start_state, face, rotation, reverse, type, best_node, upper_bound);
      if (new_node->f_bar <= id_depth) {
        my_stack.push(new_node);
      }
      //else if (my_set != nullptr && (new_node->depth < 16 || (prev_node->passed_threshold && new_node->reverse_heuristic == new_node->depth && upper_bound > id_depth + 1))) {
      //  auto existing = my_set->find(new_node);
      //  if (existing == my_set->end()) {
      //    my_set->insert(new_node);
      //  }
      //  else if ((*existing)->depth > new_node->depth) {
      //    //Must check because we are searching in DFS order, not BFS
      //    my_set->erase(existing);
      //    my_set->insert(new_node);
      //  }
      //}
      else if (my_set != nullptr && prev_node->passed_threshold) {
        auto existing = my_set->find(prev_node);
        if (existing == my_set->end()) {
          my_set->insert(prev_node);
        }
        else if ((*existing)->depth > prev_node->depth) {
          //Must check because we are searching in DFS order, not BFS
          my_set->erase(existing);
          my_set->insert(prev_node);
        }
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  hash_set* my_set,
  const hash_set* other_set,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  size_t& count,
  const size_t node_limit)
{
  std::cout << "Expanding layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  my_set->clear();

  while (!my_stack.empty() && upper_bound > c_star) {
    std::shared_ptr<Node> next_node = my_stack.top();
    my_stack.pop();
    expand_node(next_node, my_stack, my_set, other_set, id_depth, upper_bound, best_node, reverse, type, start_state, count);

    if ((my_set->size() + other_set->size()) > node_limit) {
      return false;
    }
  }
  std::cout << "Finished expanding layer " << id_depth << "; size= " << my_set->size() << '\n';
  return true;
}

bool id_check_layer(stack& my_stack,
  const hash_set* other_set,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  size_t& count)
{
  std::cout << "ID checking layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';

  while (!my_stack.empty() && upper_bound > c_star) {
    std::shared_ptr<Node> next_node = my_stack.top();
    my_stack.pop();
    expand_node(next_node, my_stack, nullptr, other_set, id_depth, upper_bound, best_node, reverse, type, start_state, count);
  }

  std::cout << "Finished ID checking layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  return upper_bound <= c_star;
}

bool store_layer(stack& my_stack,
  hash_set* my_set,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  size_t& count,
  const size_t node_limit)
{
  my_set->clear();
  if (my_stack.empty()) return true;
  std::cout << "Storing layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  uint8_t tmp;
  while (!my_stack.empty() && my_set->size() < node_limit) {
    std::shared_ptr<Node> next_node = my_stack.top();
    my_stack.pop();
    expand_node(next_node, my_stack, my_set, nullptr, id_depth, tmp, best_node, reverse, type, start_state, count);
  }

  std::cout << "Finished storing layer " << id_depth << "; size= " << my_set->size() << '\n';
  return my_stack.empty();
}

bool iterative_expand_then_test(
  stack& my_stack,
  stack& other_stack,
  std::shared_ptr<Node> other_stack_initializer,
  hash_set* my_set,
  const unsigned int id_depth,
  const unsigned int other_depth,
  const unsigned int c_star,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  size_t& count,
  const size_t node_limit)
{
  while (store_layer(my_stack, my_set, best_node, reverse, pdb_type, start_state, id_depth, count, node_limit) == false || my_set->size() > 0) {
    other_stack.push(other_stack_initializer);
    if (id_check_layer(other_stack, my_set, upper_bound, best_node, !reverse, pdb_type, start_state, other_depth, c_star, count)) {
      return true;
    }
  }
  my_set->clear();
  return false;
}

bool iterative_layer(stack my_stack,
  std::shared_ptr<Node> my_stack_initializer,
  stack other_stack,
  std::shared_ptr<Node> other_stack_initializer,
  hash_set* my_set,
  hash_set* other_set,
  unsigned int& iteration,
  unsigned int& c_star,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  size_t& count,
  size_t& my_last_count,
  size_t& other_last_count,
  const size_t node_limit)
{
  size_t start_count;
  if (iteration < 18) {
    start_count = count;
    my_stack.push(my_stack_initializer);
    expand_layer(my_stack, my_set, other_set, upper_bound, best_node, reverse, pdb_type, start_state, iteration, c_star, count, std::numeric_limits<size_t>::max());
    my_last_count = count - start_count;

    if (upper_bound <= c_star) return true;

    if (my_set->size() > 0) {
      other_stack.push(other_stack_initializer);
      if (id_check_layer(other_stack, my_set, upper_bound, best_node, !reverse, pdb_type, start_state, iteration - 1, c_star, count)) {
        return true;
      }
    }

    iteration += 1;
    c_star = iteration;

    if (upper_bound <= c_star) return true;

    start_count = count;
    other_stack.push(other_stack_initializer);
    expand_layer(other_stack, other_set, my_set, upper_bound, best_node, !reverse, pdb_type, start_state, iteration - 1, c_star, count, std::numeric_limits<size_t>::max());
    other_last_count = count - start_count;

    if (upper_bound <= c_star) return true;

    //Extra check, unnecessary but might find an early solution for next depth and ~14^2 times faster than checking iteration depth
    if (other_set->size() > 0) {
      my_stack.push(my_stack_initializer);
      if (id_check_layer(my_stack, other_set, upper_bound, best_node, reverse, pdb_type, start_state, iteration - 1, c_star, count)) {
        std::cout << "FOUND SOLUTION DURING 2nd EXTRA CHECK!!!!!\n";
        return true;
      }
    }    
  }
  else {
    my_set->clear();
    other_set->clear();

    my_stack.push(my_stack_initializer);
    if (iterative_expand_then_test(my_stack, other_stack, other_stack_initializer, my_set, iteration, iteration - 1, c_star, upper_bound, best_node, reverse, pdb_type, start_state, count, node_limit)) {
      return true;
    }

    other_stack.push(other_stack_initializer);
    if (iterative_expand_then_test(other_stack, my_stack, my_stack_initializer, my_set, iteration - 1, iteration, c_star, upper_bound, best_node, !reverse, pdb_type, start_state, count, node_limit)) {
      return true;
    }

    iteration += 1;
    c_star = iteration;
  }
  return false;
}


size_t search::id_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "ID-DIBBS" << std::endl;
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  stack forward_stack, backward_stack;

  auto start = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  auto goal = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);

  std::shared_ptr<Node> best_node(nullptr);
  size_t count = 0;
  const size_t node_limit = (size_t)2e8;

  hash_set forward_set, backward_set;
  hash_set* storage_set = &forward_set;
  hash_set* other_set = &backward_set;

  unsigned int iteration = 1;
  unsigned int c_star = 1;

  size_t last_forward_size, last_backward_size;
  size_t start_count;

  start_count = count;
  forward_stack.push(start);
  expand_layer(forward_stack, storage_set, other_set, upper_bound, best_node, false, pdb_type, start_state, 0, c_star, count, std::numeric_limits<size_t>::max());
  last_forward_size = count - start_count;

  start_count = count;
  backward_stack.push(goal);
  expand_layer(backward_stack, other_set, storage_set, upper_bound, best_node, true, pdb_type, start_state, 0, c_star, count, std::numeric_limits<size_t>::max());
  last_backward_size = count - start_count;

  while (best_node == nullptr || upper_bound > c_star)
  {
    if (last_forward_size <= last_backward_size) {
      std::cout << last_forward_size << " <= " << last_backward_size << " ; Searching forward first\n";
      iterative_layer(forward_stack, start, backward_stack, goal, storage_set, other_set, iteration, c_star, upper_bound, best_node, false, pdb_type, start_state, count, last_forward_size, last_backward_size, node_limit);
    }
    else {
      std::cout << last_forward_size << " > " << last_backward_size << " ; Searching backward first\n";
      iterative_layer(backward_stack, goal, forward_stack, start, other_set, storage_set, iteration, c_star, upper_bound, best_node, true, pdb_type, start_state, count, last_backward_size, last_forward_size, node_limit);
    }
  }

  std::cout << "Solved DIBBS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}
