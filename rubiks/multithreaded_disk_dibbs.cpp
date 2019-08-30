#include "multithreaded_id_dibbs.h"

using namespace search;

void expand_node(const Node prev_node,
  stack& my_stack,
  disk_set* my_set,
  const unsigned int id_depth,
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
    if (prev_node.depth > 0 && Rubiks::skip_rotations(prev_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node new_node(&prev_node, start_state, prev_node.depth + 1, face, rotation, reverse, type);
      if (new_node.f_bar <= id_depth) {
        my_set->insert(new_node);
        my_stack.push(new_node);
      }
      else {
        my_set->insert(new_node);
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  disk_set* my_set,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  std::atomic_uint64_t& count,
  const size_t thread_count,
  bool do_cleanup = true)
{
  std::cout << "Storing layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  if (do_cleanup) {
    my_set->open();
  }
  tstack tstack;
  while (!my_stack.empty()) {
    tstack.push(my_stack.top());
    my_stack.pop();
  }

  while (tstack.size() < thread_count) {
    while (!tstack.empty()) {
      auto [success, node] = tstack.pop();
      expand_node(node, my_stack, my_set, id_depth, reverse, type, start_state, count);
    }
    move_nodes(my_stack, tstack);
    if (tstack.empty()) break;
  }

  std::thread* thread_array = new std::thread[thread_count];

  std::mutex stack_mutex;
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i] = std::thread([&my_stack, &tstack, my_set, reverse, type, start_state, id_depth, &count, &stack_mutex]() {
      stack this_stack;
      while (true) {
        if (this_stack.empty()) {
          auto [success, node] = tstack.pop();
          if (success == false) { return; }
          this_stack.push(node);
        }
        Node next_node = this_stack.top();
        this_stack.pop();
        expand_node(next_node, this_stack, my_set, id_depth, reverse, type, start_state, count);
      }
      });
  }

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;

  if (do_cleanup) {
    my_set->close();
  }

  if (my_stack.size() > 0) {
    return false;
  }
  std::cout << "Finished storing layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  return true;
}

void check_results(std::vector<std::pair<Node, Node> > results, uint8_t& upper_bound) {
  for (size_t i = 0; i < results.size(); ++i) {
    auto my_cost = results[i].first.depth;
    auto reverse_cost = results[i].second.depth;
    if (my_cost + reverse_cost < upper_bound)
    {
      upper_bound = my_cost + reverse_cost;
      std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
    }
  }
}

std::pair<uint8_t, uint64_t> search::solve_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type, unsigned int iteration = 1, uint8_t upper_bound = 255)
{
  const size_t thread_count = std::thread::hardware_concurrency();

  if (iteration == 0) {
    std::cout << "Iteration cannot be equal to 0 to start, start at 1";
    throw std::exception("Iteration cannot be equal to 0 to start, start at 1");
  }

  stack forward_stack, backward_stack;

  disk_set forward_set("forward/forward"), backward_set("backward/backward");

  auto start = Node(start_state, Rubiks::__goal, pdb_type);
  auto goal = Node(Rubiks::__goal, start_state, pdb_type);

  std::atomic_uint64_t count = 0;

  forward_stack.push(start);
  expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration - 1, count, thread_count);
  backward_stack.push(goal);
  expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration - 1, count, thread_count);
  auto results = forward_set.compare_hash(backward_set);
  check_results(results, upper_bound);

  while (upper_bound > iteration)
  {
    if (forward_set.size() < backward_set.size()) {
      forward_stack.push(start);
      expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration, count, thread_count);
      auto results = backward_set.compare_hash(forward_set);
      check_results(results, upper_bound);

      iteration += 1;
      if (upper_bound <= iteration) {
        break;
      }

      backward_stack.push(goal);
      expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration - 1, count, thread_count);
      results = forward_set.compare_hash(backward_set);
      check_results(results, upper_bound);
    }
    else {
      backward_stack.push(goal);
      expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration, count, thread_count);
      auto results = forward_set.compare_hash(backward_set);
      check_results(results, upper_bound);

      iteration += 1;
      if (upper_bound <= iteration) {
        break;
      }

      forward_stack.push(start);
      expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration - 1, count, thread_count);
      results = backward_set.compare_hash(forward_set);
      check_results(results, upper_bound);
    }
  }

  uint64_t size = forward_set.disk_size() + backward_set.disk_size();
  std::cout << "Size used = " << std::to_string(size / 1024.0 / 1024 / 1024) << " GB\n";

  return std::make_pair(upper_bound, uint64_t(count));
}

std::pair<uint64_t, double> search::multithreaded_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  auto c_start = clock();

  std::cout << "ID-DIBBS" << std::endl;
  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  auto [upper_bound, count] = solve_disk_dibbs(start_state, pdb_type);

  std::cout << "Solved DIBBS: Length = " << std::to_string(upper_bound) << " Count = " << std::to_string(count) << std::endl;

  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(uint64_t(count), time_elapsed);
}
