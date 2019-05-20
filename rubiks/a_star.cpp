#include "a_star.h"

bool all_done(const std::atomic_bool* done_array) {
  for (int i = 0; i < omp_get_num_threads(); ++i) {
    if (done_array[i] == false) {
      return false;
    }
  }
  return true;
}

uint64_t search::a_star(const uint8_t* state, const Rubiks::PDB pdb_type)
{
  std::cout << "IDA*" << std::endl;
  thread_safe_stack<std::shared_ptr<Node>> shared_stack, base_stack;

  if (Rubiks::is_solved(state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  auto original_node = std::make_shared<Node>();
  memcpy(original_node->state, state, 40);
  uint8_t id_depth = Rubiks::pattern_lookup(original_node->state, pdb_type);
  original_node->heuristic = id_depth;
  original_node->combined = id_depth;
  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;
  std::atomic_bool done(false);
  uint64_t count = 0;

  for (int face = 0; face < 6; ++face)
  {
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      auto new_node = std::make_shared<Node>(original_node, state, 1, face, rotation, false, pdb_type);
      count += 1;
      if (Rubiks::is_solved(new_node->state))
      {
        std::cout << "Solved IDA*: 1 Count = " << unsigned long long(count) << std::endl;
        std::cout << "Solution: " << new_node->print_solution() << std::endl;
        return count;
      }
      base_stack.push(new_node);
    }
  }

  std::atomic_bool* done_array = new std::atomic_bool[omp_get_max_threads()];

  #pragma omp parallel default(none) reduction(+: count) shared(shared_stack, base_stack, done, id_depth, done_array, std::cout, state)
  {
    std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node> > > my_stack;
    while (done == false)
    {
      if (my_stack.empty()) {
        //Extract the next result to begin searching
        auto result = shared_stack.pop();
        //Check to make sure we received a valid result, only true if shared_stack was not empty and a valid result was returned
        if (result.first) {
          my_stack.push(result.second);
        }
        else {
          //Shared_stack was empty, so wait for all threads to finish before searching the next largest id_depth
          done_array[omp_get_thread_num()] = true;
          while (all_done(done_array) == false);

          //Check to make sure another thread didn't find the solution while we were waiting,
          //in which case one or more threads would have already exited the while loop 
          if (done == false) {
            //Only have one thread change the id_depth, reset the shared_stack, and reset the done_array.
            if (omp_get_thread_num() == 0) {
              id_depth += 1;
              shared_stack.copy(base_stack);
              std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
              for (int i = 0; i < omp_get_num_threads(); ++i) {
                done_array[i] = false;
              }
            }
            //All threads need to wait until the done_array is finished being reset.
            while (all_done(done_array) == true);
          }
          //my_stack is still empty, so restart the while loop to try to pull a value from shared_stack again
          continue;
        }
      }
      auto next_node = my_stack.top();
      my_stack.pop();

      count += 1;

      if (count % 1000000 == 0) {
        #pragma omp single nowait
        {
          std::cout << "In-progress: " << next_node->print_solution() << std::endl;
        }
      }

      for (int face = 0; face < 6; ++face)
      {
        if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
        {
          continue;
        }

        for (int rotation = 0; rotation < 3; ++rotation)
        {
          auto new_node = std::make_shared<Node>(next_node, state, next_node->depth + 1, face, rotation, false, pdb_type);

          if (new_node->combined < next_node->combined) {
            std::cout << "Consistency error: " << unsigned(new_node->combined) << " < " << unsigned(next_node->combined) << " " << std::endl;
          }

          if (new_node->combined > id_depth)
          {
            continue;
          }

          if (Rubiks::is_solved(new_node->state))
          {
            done = true;
            std::cout << "Solved IDA*: " << unsigned int(id_depth) << std::endl;
            std::cout << "Solution: " << new_node->print_solution() << std::endl;
          }

          my_stack.push(new_node);
        }
      }
    }
    //Let any waiting threads know that this thread has exited the while loop 
    done_array[omp_get_thread_num()] = true;
  }
  delete[] done_array;
  std::cout << "Count = " << unsigned long long(count) << std::endl;
  return count;
}
