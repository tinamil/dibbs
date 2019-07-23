#include <stdint.h>
#include <stack>
#include <queue>
#include <atomic>
#include "a_star.h"
#include "rubiks.h"
#include "node.h"
#include "thread_safe_stack.hpp"

typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>> node_stack;
typedef thread_safe_stack<std::shared_ptr<Node>> thread_safe_node_stack;

std::pair<uint64_t, double> search::ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, bool reverse)
{
  auto c_start = clock();
  std::cout << "IDA*" << std::endl;
  node_stack state_stack;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  std::shared_ptr<Node> original_node;

  if (reverse) {
    original_node = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);
  }
  else {
    original_node = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  }

  uint8_t id_depth = original_node->combined;

  state_stack.push(original_node);
  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;
  uint64_t count = 0;
  bool done = false;
  while (done == false)
  {
    if (state_stack.empty())
    {
      id_depth += 1;
      state_stack.push(original_node);
      std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
    }

    std::shared_ptr<Node> next_node = state_stack.top();
    state_stack.pop();

    count += 1;

    if (count % 1000000 == 0)
    {
      std::cout << count << std::endl;
    }
    for (int face = 0; face < 6; ++face)
    {

      if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
      {
        continue;
      }

      for (int rotation = 0; rotation < 3; ++rotation)
      {
        std::shared_ptr<Node> new_node = std::make_shared<Node>(next_node, start_state, next_node->depth + 1, face, rotation, reverse, pdb_type);

        if (new_node->combined < next_node->combined) {
          std::cout << "Consistency error: " << unsigned(new_node->combined) << " < " << unsigned(next_node->combined) << " " << std::endl;
        }

        if (new_node->combined > id_depth)
        {
          continue;
        }

        if ((reverse && Rubiks::is_solved(new_node->state, start_state)) || (!reverse && Rubiks::is_solved(new_node->state, Rubiks::__goal)))
        {
          std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
          std::cout << "Solution: " << new_node->print_solution() << std::endl;
          done = true;
        }
        state_stack.push(new_node);
      }
    }
  }
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(count, time_elapsed);
}

void expand_node(
  thread_safe_node_stack& shared_stack,
  uint8_t id_depth,
  std::atomic_uint64_t& count,
  const uint8_t* start_state,
  const Rubiks::PDB pdb_type,
  bool reverse,
  std::shared_ptr<Node>* optimal_node
) {
  node_stack state_stack;
  while (std::atomic_load(optimal_node) == nullptr) {
    if (state_stack.empty()) {
      auto [result, node] = shared_stack.pop();
      if (result == false) { return; }
      state_stack.push(node);
    }

    std::shared_ptr<Node> next_node = state_stack.top();
    state_stack.pop();

    ++count;
    if (count % 1000000 == 0)
    {
      std::cout << count << std::endl;
    }

    for (int face = 0; face < 6; ++face)
    {
      if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        std::shared_ptr<Node> new_node = std::make_shared<Node>(next_node, start_state, next_node->depth + 1, face, rotation, reverse, pdb_type);

        if (new_node->combined < next_node->combined) {
          std::cout << "Consistency error: " << unsigned(new_node->combined) << " < " << unsigned(next_node->combined) << " " << std::endl;
        }

        if (new_node->combined > id_depth)
        {
          continue;
        }

        if ((reverse && Rubiks::is_solved(new_node->state, start_state)) || (!reverse && Rubiks::is_solved(new_node->state, Rubiks::__goal)))
        {
          std::atomic_store(optimal_node, new_node);
          return;
        }
        state_stack.push(new_node);
      }
    }
  }

}

std::pair<uint64_t, double> search::multithreaded_ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, bool reverse)
{
  auto c_start = clock();
  std::cout << "IDA*" << std::endl;
  thread_safe_node_stack state_stack;
  const unsigned int thread_count = std::thread::hardware_concurrency();

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  std::shared_ptr<Node> original_node;
  std::shared_ptr<Node> optimal_node;

  if (reverse) {
    original_node = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);
  }
  else {
    original_node = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  }

  std::atomic_uint64_t count = 1;


  uint8_t id_depth = original_node->combined;

  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;
  std::thread* thread_array = new std::thread[thread_count];
  while (optimal_node == nullptr)
  {
    id_depth += 1;
    std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;

    //Expand original node, then expand all 18 of the rotations of that node and store them all in state_stack for threads to process
    for (int face = 0; face < 6; ++face)
    {
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        std::shared_ptr<Node> new_node = std::make_shared<Node>(original_node, start_state, 1, face, rotation, reverse, pdb_type);
        if ((reverse && Rubiks::is_solved(new_node->state, start_state)) || (!reverse && Rubiks::is_solved(new_node->state, Rubiks::__goal)))
        {
          std::cout << "Solved IDA*: 1" << std::endl;
          std::cout << "Solution: " << new_node->print_solution() << std::endl;
          auto c_end = clock();
          auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
          return std::make_pair(count, time_elapsed);
        }
        if (new_node->combined > id_depth)
        {
          continue;
        }
        count++;
        for (int face2 = 0; face2 < 6; ++face2)
        {
          for (int rotation2 = 0; rotation2 < 3; ++rotation2)
          {
            std::shared_ptr<Node> new_node2 = std::make_shared<Node>(new_node, start_state, 2, face2, rotation2, reverse, pdb_type);
            if ((reverse && Rubiks::is_solved(new_node2->state, start_state)) || (!reverse && Rubiks::is_solved(new_node2->state, Rubiks::__goal)))
            {
              std::cout << "Solved IDA*: 2" << std::endl;
              std::cout << "Solution: " << new_node2->print_solution() << std::endl;
              auto c_end = clock();
              auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
              return std::make_pair(count, time_elapsed);
            }
            if (new_node2->combined > id_depth)
            {
              continue;
            }
            state_stack.push(new_node2);
          }
        }
      }
    }

    for (size_t i = 0; i < thread_count; ++i) {
      thread_array[i] = std::thread(expand_node, std::ref(state_stack), id_depth, std::ref(count), start_state, pdb_type, reverse, &optimal_node);
    }

    for (size_t i = 0; i < thread_count; ++i) {
      thread_array[i].join();
    }
  }

  std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
  std::cout << "Solution: " << optimal_node->print_solution() << std::endl;
  delete[] thread_array;
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair(count, time_elapsed);
}
