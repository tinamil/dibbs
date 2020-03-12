#include "multithreaded_id_dibbs.h"

#define NOMINMAX
#include <windows.h>
#include <Psapi.h>
using namespace search;
#include <queue>

constexpr unsigned int bucket_bits = 8;
constexpr unsigned int group_bucket_count = 1 << bucket_bits;
constexpr unsigned int bucket_mask = group_bucket_count - 1;
constexpr unsigned int num_costs = 21;
constexpr unsigned int all_bucket_count = group_bucket_count * num_costs;

struct BestCompare
{
  bool operator() (const Node& a, const Node& b) const {
    int cmp = memcmp(a.state, b.state, 40);
    if (cmp == 0) {
      return false;
    }
    else if (a.f_bar == b.f_bar) {
      if (a.depth == b.depth) {
        return cmp < 0;
      }
      else {
        return a.depth < b.depth;
      }
    }
    else {
      return a.f_bar < b.f_bar;
    }
  }
};

int best_fbar(0);
int best_depth(0);

Node make_node(const concurrent_int_set* other_set,
  const Node* prev_node,
  const uint8_t* start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::atomic_uint8_t& upper_bound,
  std::uint8_t lower_bound)
{
  Node new_node(prev_node, start_state, prev_node->depth + 1, face, rotation, reverse, type);

  if (other_set != nullptr) {
    const uint64_t edge_hash = Rubiks::get_edge_index12(new_node.state);
    const uint64_t corner_hash = Rubiks::get_corner_index(new_node.state);
    const size_t hash = bucket_mask & corner_hash;
    const uint64_t data = (corner_hash >> bucket_bits) * 1961990553600ui64 + edge_hash;
    const size_t bucket_group = hash * num_costs;

    for (int right_g_index = std::min((int)num_costs - 1, upper_bound - new_node.depth - 1); right_g_index >= std::max(0, (int)lower_bound - new_node.depth); --right_g_index) {
      const size_t bucket = bucket_group + right_g_index;
      auto search = other_set[bucket].find(data);
      if (search != other_set[bucket].end())
      {
        upper_bound = new_node.depth + right_g_index;
        std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
      }
    }
  }
  return new_node;
}

void expand_node(const Node prev_node,
  stack& my_stack,
  concurrent_int_set* my_set,
  const concurrent_int_set* other_set,
  const unsigned int id_depth,
  std::atomic_uint8_t& upper_bound,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  std::atomic_uint64_t& count,
  const unsigned int c_star,
  std::mutex* my_set_mutex) {

  ++count;

  if (my_set != nullptr && prev_node.f_bar >= id_depth && prev_node.passed_threshold) {
    const uint64_t edge_hash = Rubiks::get_edge_index12(prev_node.state);
    const uint64_t corner_hash = Rubiks::get_corner_index(prev_node.state);
    const size_t hash = bucket_mask & corner_hash;
    const uint64_t data = (corner_hash >> bucket_bits) * 1961990553600ui64 + edge_hash;
    const size_t bucket = hash * num_costs + prev_node.depth;
    my_set_mutex[bucket].lock();
    my_set[bucket].insert(data);
    my_set_mutex[bucket].unlock();
  }
  for (int face = 0; face < 6; ++face)
  {
    if (prev_node.depth > 0 && Rubiks::skip_rotations(prev_node.get_face(), face))
    {
      continue;
    }
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = make_node(other_set, &prev_node, start_state, face, rotation, reverse, type, upper_bound, c_star);
      if (new_node.f_bar <= id_depth) {
        my_stack.push((new_node));
      }
      else if (my_set != nullptr && new_node.f_bar >= id_depth && prev_node.passed_threshold) {
        if (new_node.reverse_heuristic - new_node.heuristic > 1) {
          const uint64_t edge_hash = Rubiks::get_edge_index12(new_node.state);
          const uint64_t corner_hash = Rubiks::get_corner_index(new_node.state);
          const size_t hash = bucket_mask & corner_hash;
          const uint64_t data = (corner_hash >> bucket_bits) * 1961990553600ui64 + edge_hash;
          const size_t bucket = hash * num_costs + new_node.depth;
          my_set_mutex[bucket].lock();
          my_set[bucket].insert(data);
          my_set_mutex[bucket].unlock();
        }
      }
    }
  }
}

bool expand_layer(stack& my_stack,
  concurrent_int_set* my_set,
  const concurrent_int_set* other_set,
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

  std::thread* thread_array = new std::thread[thread_count];

  if (my_stack.empty() || upper_bound <= c_star) return true;

  tstack tstack;
  while (!my_stack.empty()) {
    tstack.push(my_stack.top());
    my_stack.pop();
  }

  std::mutex* my_set_mutex = new std::mutex[all_bucket_count];

  while (tstack.size() < thread_count) {
    while (!tstack.empty()) {
      auto [success, node] = tstack.pop();
      expand_node(node, my_stack, my_set, other_set, id_depth, upper_bound, reverse, type, start_state, count, c_star, my_set_mutex);
    }
    move_nodes(my_stack, tstack);
    if (tstack.empty()) break;
  }

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i] = std::thread([&my_stack, &my_set_mutex, &tstack, my_set, other_set, &upper_bound, reverse, type, start_state, id_depth, c_star, &count]() {
      stack this_stack;
      while (upper_bound > c_star) {
        if (this_stack.empty()) {
          auto [success, node] = tstack.pop();
          if (success == false) { return; }
          this_stack.push(node);
        }
        Node next_node = this_stack.top();
        this_stack.pop();
        expand_node(next_node, this_stack, my_set, other_set, id_depth, upper_bound, reverse, type, start_state, count, c_star, my_set_mutex);
      }
      });
  }

  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] my_set_mutex;
  delete[] thread_array;
#ifndef NDEBUG
  while (!tstack.empty()) {
    auto [success, node] = tstack.pop();
    my_stack.push(node);
  }

  if (my_stack.size() > 0 && upper_bound > c_star) {
    std::cout << "ERROR, stack wasn't empty, shouldn't have stopped.";
    return false;
  }
#endif

  //std::cout << "Finished " << (my_set == nullptr ? "ID-checking" : (other_set == nullptr ? "storing" : "expanding")) << " layer " << id_depth << '\n';
  return true;
}

bool id_check_layer(stack& my_stack,
  const concurrent_int_set* other_set,
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

bool store_layer(stack& my_stack,
  concurrent_int_set* my_set,
  std::atomic_uint8_t& upper_bound,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  const unsigned int id_depth,
  const unsigned int c_star,
  std::atomic_uint64_t& count,
  const size_t thread_count)
{
  expand_layer(my_stack, my_set, nullptr, upper_bound, reverse, type, start_state, id_depth, c_star, count, thread_count);
  return upper_bound <= c_star;
}

bool iterative_layer(stack my_stack,
  const Node& my_stack_initializer,
  stack other_stack,
  const Node& other_stack_initializer,
  concurrent_int_set* my_set,
  concurrent_int_set* other_set,
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

  //if (my_set->size() > 0) {
  other_stack.push(other_stack_initializer);
  if (id_check_layer(other_stack, my_set, UB, !reverse, pdb_type, start_state, iteration - 1, LB, count, thread_count)) {
    return true;
  }
  //}

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
  //if (other_set->size() > 0) {
  my_stack.push(my_stack_initializer);
  if (id_check_layer(my_stack, other_set, UB, reverse, pdb_type, start_state, iteration - 1, LB, count, thread_count)) {
    std::cout << "FOUND SOLUTION DURING 2nd EXTRA CHECK!!!!!\n";
    return true;
  }
  //}

  return true;
}

std::tuple<uint64_t, double, size_t> search::multithreaded_compressed_id_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type)
{
  auto c_start = clock();
  const unsigned int thread_count = std::thread::hardware_concurrency();

  std::cout << "ID-DIBBS" << std::endl;
  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_tuple(0, 0, 0);
  }

  std::atomic_uint8_t upper_bound = std::numeric_limits<uint8_t>::max();

  stack forward_stack, backward_stack;

  auto start = Node(start_state, Rubiks::__goal, pdb_type);
  auto goal = Node(Rubiks::__goal, start_state, pdb_type);

  std::atomic_uint64_t count = 0;

  concurrent_int_set* forward_set = new concurrent_int_set[all_bucket_count];
  concurrent_int_set* backward_set = new concurrent_int_set[all_bucket_count];
  concurrent_int_set* storage_set = forward_set;
  concurrent_int_set* other_set = backward_set;

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
    /*size_t forward_size(0), backward_size(0);
    for (int i = 0; i < all_bucket_count; ++i) {
      forward_size += forward_set[i].size();
      backward_size += backward_set[i].size();
    }*/
    //if (forward_size <= backward_size) {
    //  iterative_layer(forward_stack, start, backward_stack, goal, storage_set, other_set, iteration, c_star, upper_bound, false, pdb_type, start_state, count, last_forward_size, last_backward_size, thread_count);
    //}
    //else {
    iterative_layer(backward_stack, goal, forward_stack, start, other_set, storage_set, iteration, c_star, upper_bound, true, pdb_type, start_state, count, last_backward_size, last_forward_size, thread_count);
    //}
  }

  PROCESS_MEMORY_COUNTERS memCounter;
  BOOL result = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
  assert(result);

  std::cout << "Solved DIBBS: " << std::to_string(upper_bound) << " Count = " << std::to_string(count) << std::endl;
  //std::cout << "Stats: " << "total=" << std::to_string(total) << "; hmiddle=" << std::to_string(hmiddle) << "; small_diff=" << std::to_string(small_diff) << "; large_diff=" << std::to_string(large_diff) << std::endl;
  //std::cout << "Set sizes: forward= " << std::to_string(forward_set.size()) << "; backward= " << std::to_string(backward_set.size()) << std::endl;
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  delete[] forward_set;
  delete[] backward_set;
  return std::make_tuple(uint64_t(count), time_elapsed, memCounter.PagefileUsage);
}
