#include "multithreaded_id_dibbs.h"
#include <bitset>
#include<filesystem>

using namespace search;

constexpr unsigned int bucket_bits = 8;
constexpr unsigned int group_bucket_count = 1 << bucket_bits;
constexpr unsigned int bucket_mask = group_bucket_count - 1;
constexpr unsigned int num_costs = 21;
constexpr unsigned int all_bucket_count = group_bucket_count * num_costs;

void expand_node(const Node prev_node,
  stack& my_stack,
  disk_set* my_set,
  const unsigned int id_depth,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count) {

  ++count;

  for (int face = 0; face < 6; ++face)
  {
    if (prev_node.depth > 0 && Rubiks::skip_rotations(prev_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      Node new_node(&prev_node, start_state, prev_node.depth + 1, face, rotation, reverse, type);

      uint64_t edge_hash = Rubiks::get_edge_index12(new_node.state);
      uint64_t corner_hash = Rubiks::get_corner_index(new_node.state);
      size_t hash = bucket_mask & corner_hash;
      size_t bucket = hash * num_costs + new_node.depth;
      uint64_t data = (corner_hash >> bucket_bits) * 1961990553600ui64 + edge_hash;
      my_set->insert(data, bucket);
      if (new_node.f_bar <= id_depth) {
        my_stack.push(new_node);
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
  const size_t thread_count)
{
  std::cout << "Expanding layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  my_set->open();
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

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i] = std::thread([&my_stack, &tstack, my_set, reverse, type, start_state, id_depth, &count]() {
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

  my_set->close();

  if (my_stack.size() > 0) {
    std::cout << "Error with expanding, stack not empty\n";
    return false;
  }
  //std::cout << "Finished storing layer " << id_depth << " in " << (reverse ? "backward" : "forward") << '\n';
  return true;
}

void compare_hash(const disk_set& left, const disk_set& right, const size_t thread_count, const uint8_t g_lower_bound, uint8_t& upper_bound)
{

  //std::cout << "Comparing hash for g >= " << std::to_string(g_lower_bound) << '\n';
  std::thread* thread_array = new std::thread[thread_count];
  std::mutex ub_lock;
  std::mutex stack_lock;
  std::queue<size_t> buckets_remaining;
  for (size_t i = 0; i < group_bucket_count; i++) {
    buckets_remaining.push(i * num_costs);
  }

  //Split work up across threads
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([thread_number, &left, &right, g_lower_bound, &ub_lock, &upper_bound, &stack_lock, &buckets_remaining]() {

      tsl::hopscotch_set<uint64_t> left_set;
      std::vector<uint64_t> right_values;

      while (true) {
        //Each thread takes a bucket and repeats until all buckets exhausted (each bucket has num_costs sub-buckets skipped over)
        stack_lock.lock();
        if (buckets_remaining.empty()) {
          stack_lock.unlock();
          return;
        }
        size_t zero_bucket_index = buckets_remaining.front();
        buckets_remaining.pop();
        stack_lock.unlock();

        //Check all of the g cost sub-buckets from the left side against the valid combinations on the right side
        for (int left_g_index = 0; left_g_index < num_costs; ++left_g_index) {
          left.load_bucket(zero_bucket_index + left_g_index, left_set);

          //Only check nodes for which left_g + right_g >= g_lower_bound
          for (int right_g_index = num_costs - 1; right_g_index >= std::max(0, (int)g_lower_bound - left_g_index); --right_g_index) {
            right.load_vector(zero_bucket_index + right_g_index, right_values);
            for (size_t vec_index = 0; vec_index < right_values.size(); ++vec_index) {
              auto it = left_set.find(right_values[vec_index]);
              if (it != left_set.end()) {
                uint8_t new_g_cost = left_g_index + right_g_index;
                ub_lock.lock();
                if (new_g_cost < upper_bound)
                {
                  upper_bound = new_g_cost;
                }
                ub_lock.unlock();
              }
            }

            //Early exit check for all threads to save time once a solution is found
            if (upper_bound == g_lower_bound) {
              return;
            }
          }
        }
      }
      });
  }
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;
}

std::pair<uint8_t, uint64_t> search::solve_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  const size_t thread_count = (size_t)(std::thread::hardware_concurrency());

  stack forward_stack, backward_stack;

  disk_set forward_set("forward/forward", all_bucket_count), backward_set("backward/backward", all_bucket_count);

  auto start = Node(start_state, Rubiks::__goal, pdb_type);
  auto goal = Node(Rubiks::__goal, start_state, pdb_type);

  int iteration = std::max(start.combined, goal.combined);
  uint8_t upper_bound = 255;

  std::atomic_uint64_t count = 0;

  forward_stack.push(start);
  expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration - 1, count, thread_count);
  backward_stack.push(goal);
  expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration - 1, count, thread_count);
  compare_hash(forward_set, backward_set, thread_count, 0, upper_bound);

  while (upper_bound > iteration)
  {
    //if (forward_set.size() > backward_set.size()) {
    //  forward_stack.push(start);
    //  expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration, count, thread_count);
    //  compare_hash(backward_set, forward_set, thread_count, iteration, upper_bound);

    //  iteration += 1;
    //  if (upper_bound <= iteration) {
    //    break;
    //  }

    //  backward_stack.push(goal);
    //  expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration - 1, count, thread_count);
    //  compare_hash(forward_set, backward_set, thread_count, iteration, upper_bound);
    //}
    //else {
    backward_stack.push(goal);
    expand_layer(backward_stack, &backward_set, true, pdb_type, start_state, iteration, count, thread_count);
    compare_hash(forward_set, backward_set, thread_count, iteration, upper_bound);

    iteration += 1;
    if (upper_bound <= iteration) {
      break;
    }

    forward_stack.push(start);
    expand_layer(forward_stack, &forward_set, false, pdb_type, start_state, iteration - 1, count, thread_count);
    compare_hash(backward_set, forward_set, thread_count, iteration, upper_bound);
    //}
  }

  uint64_t size = forward_set.disk_size() + backward_set.disk_size();
  std::cout << "Size used = " << std::to_string(size / 1024.0 / 1024 / 1024) << " GB\n";

  return std::make_pair(upper_bound, uint64_t(count));
}

std::pair<uint64_t, double> search::multithreaded_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  const char* prefix1 = "forward/", * prefix2 = "backward/";

  for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
    std::filesystem::remove(entry);
  }
  for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
    std::filesystem::remove(entry);
  }

  auto c_start = clock();

  std::cout << "DISK-DIBBS" << std::endl;
  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  auto [upper_bound, count] = solve_disk_dibbs(start_state, pdb_type);

  std::cout << "Solved DIBBS: Length = " << std::to_string(upper_bound) << " Count = " << std::to_string(count) << std::endl;

  size_t total_size = 0;
  for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
    total_size += std::filesystem::file_size(entry);
  }
  for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
    total_size += std::filesystem::file_size(entry);
  }
  std::cout << "Total size: " << (total_size / (double)(1ui64 << 30)) << "GB\n";

  for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
    std::filesystem::remove(entry);
  }
  for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
    std::filesystem::remove(entry);
  }

  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(uint64_t(count), time_elapsed);
}
