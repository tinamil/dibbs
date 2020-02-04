#include "multithreaded_id_dibbs.h"

using namespace search;

#ifndef HISTORY

//std::atomic_uint64_t hmiddle, total, small_diff, large_diff;

Node make_node(const concurrent_set* other_set,
  const Node* prev_node,
  const uint8_t* start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::atomic_uint8_t& upper_bound)
{
  Node new_node(prev_node, start_state, prev_node->depth + 1, face, rotation, reverse, type);

  if (other_set != nullptr) {
    uint8_t reverse_cost = 0;
    auto search = other_set->find(new_node);
    if (search != other_set->end())
    {
      reverse_cost = (*search).depth;
      if (new_node.depth + reverse_cost < upper_bound)
      {
        upper_bound = new_node.depth + reverse_cost;
        std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
      }
    }
  }
  return new_node;
}

void expand_node(const Node prev_node,
  stack& my_stack,
  concurrent_set* my_set,
  std::mutex& my_set_mutex,
  const concurrent_set* other_set,
  const unsigned int id_depth,
  std::atomic_uint8_t& upper_bound,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count) {

  ++count;
  //if (count % 1000000 == 0) {
  //  std::cout << count << "\n";
  //}

  for (int face = 0; face < 6; ++face)
  {
    if (prev_node.depth > 0 && Rubiks::skip_rotations(prev_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = make_node(other_set, &prev_node, start_state, face, rotation, reverse, type, upper_bound);
      if (new_node.f_bar <= id_depth) {
        my_stack.push((new_node));
      }
      else if (my_set != nullptr && prev_node.passed_threshold) {
        //hmiddle++;
        //total++;
        if (id_depth < 17 && (new_node.reverse_heuristic - new_node.heuristic) > 1) {
          //large_diff++;
          auto [existing, success] = my_set->insert(new_node);
          if (!success && (*existing).depth > new_node.depth) {
            my_set_mutex.lock();
            //Must check because we are searching in DFS order, not BFS
            my_set->unsafe_erase(existing);
            my_set->insert(new_node);
            my_set_mutex.unlock();
          }
        }
        else {
          //small_diff++;
          auto [existing, success] = my_set->insert(prev_node);
          if (!success && (*existing).depth > prev_node.depth) {
            my_set_mutex.lock();
            //Must check because we are searching in DFS order, not BFS
            my_set->unsafe_erase(existing);
            my_set->insert(prev_node);
            my_set_mutex.unlock();
          }
        }
      }
      else {
        //total++;
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  concurrent_set* my_set,
  const concurrent_set* other_set,
  std::atomic_uint8_t& upper_bound,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  std::atomic_uint64_t& count,
  const size_t thread_count)
{
  std::cout << (my_set == nullptr ? "ID-Checking" : (other_set == nullptr ? "Storing" : "Expanding")) << " layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  if (my_set != nullptr) {
    my_set->clear();
  }

  if (my_stack.empty() || upper_bound <= c_star) return true;

  tstack tstack;
  while (!my_stack.empty()) {
    tstack.push(my_stack.top());
    my_stack.pop();
  }

  std::mutex my_set_mutex;

  while (tstack.size() < thread_count) {
    while (!tstack.empty()) {
      auto [success, node] = tstack.pop();
      expand_node(node, my_stack, my_set, my_set_mutex, other_set, id_depth, upper_bound, reverse, type, start_state, count);
    }
    move_nodes(my_stack, tstack);
    if (tstack.empty()) break;
  }

  std::thread* thread_array = new std::thread[thread_count];

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i] = std::thread([&my_stack, &tstack, my_set, &my_set_mutex, other_set, &upper_bound, reverse, type, start_state, id_depth, c_star, &count]() {
      stack this_stack;
      while (upper_bound > c_star) {
        if (this_stack.empty()) {
          auto [success, node] = tstack.pop();
          if (success == false) { return; }
          this_stack.push(node);
        }
        Node next_node = this_stack.top();
        this_stack.pop();
        expand_node(next_node, this_stack, my_set, my_set_mutex, other_set, id_depth, upper_bound, reverse, type, start_state, count);
      }
      });
  }

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;

  while (!tstack.empty()) {
    auto [success, node] = tstack.pop();
    my_stack.push(node);
  }

  if (my_stack.size() > 0) {
    std::cout << "ERROR, stack wasn't empty, shouldn't have stopped.";
    return false;
  }

  std::cout << "Finished " << (my_set == nullptr ? "ID-checking" : (other_set == nullptr ? "storing" : "expanding")) << " layer " << id_depth << '\n';
  return true;
}

bool id_check_layer(stack& my_stack,
  const concurrent_set* other_set,
  std::atomic_uint8_t& upper_bound,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  std::atomic_uint64_t& count,
  const size_t thread_count)
{
  expand_layer(my_stack, nullptr, other_set, upper_bound, reverse, type, start_state, id_depth, c_star, count, thread_count);
  return upper_bound <= c_star;
}

bool iterative_layer(stack my_stack,
  const Node& my_stack_initializer,
  stack other_stack,
  const Node& other_stack_initializer,
  concurrent_set* my_set,
  concurrent_set* other_set,
  unsigned int& iteration,
  unsigned int& LB,
  std::atomic_uint8_t& UB,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count,
  size_t& my_last_count,
  size_t& other_last_count,
  const size_t thread_count)
{
  size_t start_count;
  start_count = count;
  my_stack.push(my_stack_initializer);
  if (!expand_layer(my_stack, my_set, other_set, UB, reverse, pdb_type, start_state, iteration, LB, count, thread_count)) {
    return false;
  }
  my_last_count = count - start_count;

  if (UB <= LB) return true;

  if (my_set->size() > 0) {
    other_stack.push(other_stack_initializer);
    if (id_check_layer(other_stack, my_set, UB, !reverse, pdb_type, start_state, iteration - 1, LB, count, thread_count)) {
      return true;
    }
  }

  iteration += 1;
  LB = iteration;

  if (UB <= LB) return true;

  start_count = count;
  other_stack.push(other_stack_initializer);
  if (!expand_layer(other_stack, other_set, my_set, UB, !reverse, pdb_type, start_state, iteration - 1, LB, count, thread_count)) {
    return false;
  }
  other_last_count = count - start_count;

  if (UB <= LB) return true;

  //Extra check, unnecessary but might find an early solution for next depth 
  if (other_set->size() > 0) {
    my_stack.push(my_stack_initializer);
    if (id_check_layer(my_stack, other_set, UB, reverse, pdb_type, start_state, iteration - 1, LB, count, thread_count)) {
      std::cout << "FOUND SOLUTION DURING 2nd EXTRA CHECK!!!!!\n";
      return true;
    }
  }

  return true;
}

bool reached_depth_limit(unsigned int iteration, const Rubiks::PDB pdb_type) {
  switch (pdb_type)
  {
  case Rubiks::PDB::a1997:
    return iteration >= 17;
  case Rubiks::PDB::a888:
    return iteration >= 18;
  default:
    return iteration >= 19;
  }
}

std::pair<uint64_t, double> search::multithreaded_id_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  auto c_start = clock();
  const unsigned int thread_count = std::thread::hardware_concurrency();

  std::cout << "ID-DIBBS" << std::endl;
  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  std::atomic_uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  stack forward_stack, backward_stack;

  auto start = Node(start_state, Rubiks::__goal, pdb_type);
  auto goal = Node(Rubiks::__goal, start_state, pdb_type);

  std::atomic_uint64_t count = 0;

  concurrent_set forward_set, backward_set;
  concurrent_set* storage_set = &forward_set;
  concurrent_set* other_set = &backward_set;

  unsigned int iteration = 1;
  unsigned int c_star = 1;

  size_t last_forward_size, last_backward_size;
  size_t start_count;

  start_count = count;
  forward_stack.push(start);
  expand_layer(forward_stack, storage_set, other_set, upper_bound, false, pdb_type, start_state, 0, c_star, count, thread_count);
  last_forward_size = count - start_count;

  start_count = count;
  backward_stack.push(goal);
  expand_layer(backward_stack, other_set, storage_set, upper_bound, true, pdb_type, start_state, 0, c_star, count, thread_count);
  last_backward_size = count - start_count;

  while (upper_bound > c_star)
  {
    if (forward_set.size() >= backward_set.size()) {
      iterative_layer(forward_stack, start, backward_stack, goal, storage_set, other_set, iteration, c_star, upper_bound, false, pdb_type, start_state, count, last_forward_size, last_backward_size, thread_count);
    }
    else {
      iterative_layer(backward_stack, goal, forward_stack, start, other_set, storage_set, iteration, c_star, upper_bound, true, pdb_type, start_state, count, last_backward_size, last_forward_size, thread_count);
    }
    /*if (upper_bound > c_star && reached_depth_limit(iteration, pdb_type)) {
      std::cout << "Reached node limit: " << std::to_string(storage_set->size()) << " + " << std::to_string(other_set->size()) << "; switching to disk" << std::endl;
      std::thread thread1 = std::thread([&forward_set]() {
        forward_set.clear();
        });
      std::thread thread2 = std::thread([&backward_set]() {
        backward_set.clear();
        });
      auto result = search::solve_disk_dibbs(start_state, pdb_type, iteration, upper_bound);
      thread1.join();
      thread2.join();
      upper_bound = c_star = result.first;
      count += result.second;
    }*/
  }

  std::cout << "Solved DIBBS: " << std::to_string(upper_bound) << " Count = " << std::to_string(count) << std::endl;
  //std::cout << "Stats: " << "total=" << std::to_string(total) << "; hmiddle=" << std::to_string(hmiddle) << "; small_diff=" << std::to_string(small_diff) << "; large_diff=" << std::to_string(large_diff) << std::endl;
  std::cout << "Set sizes: forward= " << std::to_string(forward_set.size()) << "; backward= " << std::to_string(backward_set.size()) << std::endl;
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(uint64_t(count), time_elapsed);
}

#else

std::shared_ptr<Node> make_node(const concurrent_set* other_set,
  std::shared_mutex* other_set_mutex,
  const std::shared_ptr<Node>& prev_node,
  const uint8_t* start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::shared_ptr<Node>& best_node,
  std::atomic_uint8_t& upper_bound)
{
  auto new_node = std::make_shared<Node>(prev_node, start_state, prev_node->depth + 1, face, rotation, reverse, type);

  if (other_set != nullptr) {
    other_set_mutex->lock_shared();
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
    other_set_mutex->unlock_shared();
  }
  return new_node;
}

void expand_node(std::shared_ptr<Node> prev_node,
  stack& my_stack,
  concurrent_set* my_set,
  std::shared_mutex* my_set_mutex,
  const concurrent_set* other_set,
  std::shared_mutex* other_set_mutex,
  const unsigned int id_depth,
  std::atomic_uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count) {

  ++count;
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
      auto new_node = make_node(other_set, other_set_mutex, prev_node, start_state, face, rotation, reverse, type, best_node, upper_bound);
      if (new_node->f_bar <= id_depth) {
        my_stack.push(std::move(new_node));
      }
      else if (my_set != nullptr && prev_node->passed_threshold) {
        auto insertion_node = prev_node;
        if ((new_node->reverse_heuristic - new_node->heuristic) > 1) {
          insertion_node = new_node;
        }
        auto [existing, success] = my_set->insert(insertion_node);
        if (!success && (*existing)->depth > insertion_node->depth) {
          my_set_mutex->lock();
          //Must check because we are searching in DFS order, not BFS
          my_set->unsafe_erase(*existing);
          my_set->insert(insertion_node);
          my_set_mutex->unlock();
        }
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  concurrent_set* my_set,
  std::shared_mutex* my_set_mutex,
  const concurrent_set* other_set,
  std::shared_mutex* other_set_mutex,
  std::atomic_uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  std::atomic_uint64_t& count,
  const size_t node_limit,
  const size_t thread_count)
{
  std::cout << (my_set == nullptr ? "ID-Checking" : (other_set == nullptr ? "Storing" : "Expanding")) << " layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  if (my_set != nullptr) {
    my_set->clear();
  }

  if (my_stack.empty() || upper_bound <= c_star) return true;

  tstack tstack;
  while (!my_stack.empty()) {
    tstack.push(my_stack.top());
    my_stack.pop();
  }

  while (tstack.size() < thread_count) {
    while (!tstack.empty()) {
      auto [success, node] = tstack.pop();
      expand_node(std::move(node), my_stack, my_set, my_set_mutex, other_set, other_set_mutex, id_depth, upper_bound, best_node, reverse, type, start_state, count);
    }
    move_nodes(my_stack, tstack);
    if (tstack.empty()) break;
  }

  std::thread* thread_array = new std::thread[thread_count];

  std::shared_mutex stack_mutex;
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i] = std::thread([&my_stack, &tstack, my_set, my_set_mutex, other_set, other_set_mutex, &upper_bound, &best_node, reverse, type, start_state, id_depth, c_star, &count, node_limit, &stack_mutex]() {
      stack this_stack;
      while (upper_bound > c_star) {
        if (this_stack.empty()) {
          auto [success, node] = tstack.pop();
          if (success == false) { return; }
          this_stack.push(std::move(node));
        }
        std::shared_ptr<Node> next_node = std::move(this_stack.top());
        this_stack.pop();
        expand_node(next_node, this_stack, my_set, my_set_mutex, other_set, other_set_mutex, id_depth, upper_bound, best_node, reverse, type, start_state, count);

        if (my_set != nullptr && my_set->size() > node_limit) {
          stack_mutex.lock();
          move_nodes(this_stack, my_stack);
          stack_mutex.unlock();
          return;
        }
      }
      });
  }

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;

  while (!tstack.empty()) {
    auto [success, node] = tstack.pop();
    my_stack.push(std::move(node));
  }

  if (my_stack.size() > 0) {
    return false;
  }

  std::cout << "Finished " << (my_set == nullptr ? "ID-checking" : (other_set == nullptr ? "storing" : "expanding")) << " layer " << id_depth << '\n';
  return true;
}

bool id_check_layer(stack& my_stack,
  const concurrent_set* other_set,
  std::shared_mutex* other_set_mutex,
  std::atomic_uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  std::atomic_uint64_t& count,
  const size_t thread_count)
{
  expand_layer(my_stack, nullptr, nullptr, other_set, other_set_mutex, upper_bound, best_node, reverse, type, start_state, id_depth, c_star, count, std::numeric_limits<size_t>::max(), thread_count);
  return upper_bound <= c_star;
}

//Create a buffer thread to read values and write them
bool store_layer(stack& my_stack,
  concurrent_set* my_set,
  std::shared_mutex* my_set_mutex,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  std::atomic_uint64_t& count,
  const size_t node_limit,
  const size_t thread_count)
{
  std::atomic_uint8_t tmp;
  return expand_layer(my_stack, my_set, my_set_mutex, nullptr, nullptr, tmp, best_node, reverse, pdb_type, start_state, id_depth, 0, count, node_limit, thread_count);
}


bool iterative_store_then_check(
  stack& my_stack,
  stack& other_stack,
  const std::shared_ptr<Node>& other_stack_initializer,
  concurrent_set* my_set,
  std::shared_mutex* my_set_mutex,
  const unsigned int id_depth,
  const unsigned int other_depth,
  const unsigned int c_star,
  std::atomic_uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count,
  const size_t node_limit,
  const size_t thread_count)
{
  while (store_layer(my_stack, my_set, my_set_mutex, best_node, reverse, pdb_type, start_state, id_depth, count, node_limit, thread_count) == false || my_set->size() > 0) {
    other_stack.push(other_stack_initializer);
    if (id_check_layer(other_stack, my_set, my_set_mutex, upper_bound, best_node, !reverse, pdb_type, start_state, other_depth, c_star, count, thread_count)) {
      return true;
    }
  }
  my_set->clear();
  return false;
}

bool iterative_layer(stack my_stack,
  const std::shared_ptr<Node>& my_stack_initializer,
  stack other_stack,
  const std::shared_ptr<Node>& other_stack_initializer,
  concurrent_set* my_set,
  std::shared_mutex* my_set_mutex,
  concurrent_set* other_set,
  std::shared_mutex* other_set_mutex,
  unsigned int& iteration,
  unsigned int& c_star,
  std::atomic_uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB pdb_type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count,
  size_t& my_last_count,
  size_t& other_last_count,
  const size_t node_limit,
  const size_t thread_count)
{
  size_t start_count;
  if (iteration < 18) {
    start_count = count;
    my_stack.push(my_stack_initializer);
    expand_layer(my_stack, my_set, my_set_mutex, other_set, other_set_mutex, upper_bound, best_node, reverse, pdb_type, start_state, iteration, c_star, count, std::numeric_limits<size_t>::max(), thread_count);
    my_last_count = count - start_count;

    if (upper_bound <= c_star) return true;

    if (my_set->size() > 0) {
      other_stack.push(other_stack_initializer);
      if (id_check_layer(other_stack, my_set, my_set_mutex, upper_bound, best_node, !reverse, pdb_type, start_state, iteration - 1, c_star, count, thread_count)) {
        return true;
      }
    }

    iteration += 1;
    c_star = iteration;

    if (upper_bound <= c_star) return true;

    start_count = count;
    other_stack.push(other_stack_initializer);
    expand_layer(other_stack, other_set, other_set_mutex, my_set, my_set_mutex, upper_bound, best_node, !reverse, pdb_type, start_state, iteration - 1, c_star, count, std::numeric_limits<size_t>::max(), thread_count);
    other_last_count = count - start_count;

    if (upper_bound <= c_star) return true;

    //Extra check, unnecessary but might find an early solution for next depth 
    if (other_set->size() > 0) {
      my_stack.push(my_stack_initializer);
      if (id_check_layer(my_stack, other_set, other_set_mutex, upper_bound, best_node, reverse, pdb_type, start_state, iteration - 1, c_star, count, thread_count)) {
        std::cout << "FOUND SOLUTION DURING 2nd EXTRA CHECK!!!!!\n";
        return true;
      }
    }
  }
  else {
    if (iteration == 18) {
      //Depth 17 already stored in memory in both directions.  If forward was smaller, then iteratively expand forward and check against backward.
      my_set->clear();
      my_stack.push(my_stack_initializer);
      if (id_check_layer(my_stack, other_set, other_set_mutex, upper_bound, best_node, reverse, pdb_type, start_state, iteration, c_star, count, thread_count)) {
        return true;
      }
      other_set->clear();
    }
    else {
      other_stack.push(other_stack_initializer);
      if (iterative_store_then_check(other_stack, my_stack, my_stack_initializer, my_set, my_set_mutex, iteration - 1, iteration, c_star, upper_bound, best_node, !reverse, pdb_type, start_state, count, node_limit, thread_count)) {
        return true;
      }
    }

    my_stack.push(my_stack_initializer);
    if (iterative_store_then_check(my_stack, other_stack, other_stack_initializer, my_set, my_set_mutex, iteration, iteration - 1, c_star, upper_bound, best_node, reverse, pdb_type, start_state, count, node_limit, thread_count)) {
      return true;
    }
    iteration += 1;
    c_star = iteration;
  }
  return false;
}


std::pair<uint64_t, double> search::multithreaded_id_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  auto c_start = clock();
  const unsigned int thread_count = std::thread::hardware_concurrency();

  std::cout << "ID-DIBBS" << std::endl;
  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  std::atomic_uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  stack forward_stack, backward_stack;

  auto start = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  auto goal = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);

  std::shared_ptr<Node> best_node(nullptr);
  std::atomic_uint64_t count = 0;
  const size_t node_limit = (size_t)2e8;

  concurrent_set forward_set, backward_set;
  concurrent_set* storage_set = &forward_set;
  concurrent_set* other_set = &backward_set;

  unsigned int iteration = 1;
  unsigned int c_star = 1;

  size_t last_forward_size, last_backward_size;
  size_t start_count;


  std::shared_mutex* my_set_mutex = &std::shared_mutex();
  std::shared_mutex* other_set_mutex = &std::shared_mutex();

  start_count = count;
  forward_stack.push(start);
  expand_layer(forward_stack, storage_set, my_set_mutex, other_set, other_set_mutex, upper_bound, best_node, false, pdb_type, start_state, 0, c_star, count, std::numeric_limits<size_t>::max(), thread_count);
  last_forward_size = count - start_count;

  start_count = count;
  backward_stack.push(goal);
  expand_layer(backward_stack, other_set, other_set_mutex, storage_set, my_set_mutex, upper_bound, best_node, true, pdb_type, start_state, 0, c_star, count, std::numeric_limits<size_t>::max(), thread_count);
  last_backward_size = count - start_count;

  while (best_node == nullptr || upper_bound > c_star)
  {
    if (forward_set.size() >= backward_set.size()) {
      iterative_layer(forward_stack, start, backward_stack, goal, storage_set, my_set_mutex, other_set, other_set_mutex, iteration, c_star, upper_bound, best_node, false, pdb_type, start_state, count, last_forward_size, last_backward_size, node_limit, thread_count);
    }
    else {
      iterative_layer(backward_stack, goal, forward_stack, start, other_set, other_set_mutex, storage_set, my_set_mutex, iteration, c_star, upper_bound, best_node, true, pdb_type, start_state, count, last_backward_size, last_forward_size, node_limit, thread_count);
    }
  }

  std::cout << "Solved DIBBS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;

  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(uint64_t(count), time_elapsed);
}
#endif