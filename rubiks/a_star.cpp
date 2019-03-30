#include "a_star.h"


uint64_t search::a_star(const uint8_t* state, const Rubiks::PDB pdb_type)
{
  std::cout << "IDA*" << std::endl;
  thread_safe_stack<Node> shared_stack;

  if (Rubiks::is_solved(state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  Node original_node;
  memcpy(original_node.state, state, 40);
  uint8_t id_depth = std::max(1ui8, Rubiks::pattern_lookup(original_node.state, pdb_type, 0));
  original_node.heuristic = id_depth;
  original_node.combined = id_depth;
  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;

  uint64_t count = 0;

  bool done = false;

  #pragma omp parallel default(none) reduction(+: count) shared(shared_stack, done, id_depth, std::cout, original_node)
  {
    std::stack<Node, std::vector<Node>> my_stack;
    while (done == false)
    {
      if (my_stack.empty()) {
        auto result = shared_stack.pop();
        if (result.first) {
          my_stack.push(result.second);
        }
        else {
          //Shared_stack is empty, so wait for all threads to finish before searching the next largest id_depth
          #pragma omp barrier
          #pragma omp single
          {
            id_depth += 1;
            for (int face = 0; face < 6; ++face)
            {
              for (int rotation = 0; rotation < 3; ++rotation)
              {
                Node new_node(original_node.state, nullptr, 1, face, rotation, false, pdb_type, 0, 0);
                count += 1;
                if (Rubiks::is_solved(new_node.state))
                {
                  std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
                  std::cout << "Solution: " << new_node.print_solution() << std::endl;
                  done = true;
                  #pragma omp flush(done)
                }
                shared_stack.push(new_node);
              }
            }
            std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << " and count = " << unsigned(count) << std::endl;
          }
          //my_stack is still empty, so restart the while loop to try to pull a value from shared_stack again
          continue;
        }
      }
      Node next_node = my_stack.top();
      my_stack.pop();

      count += 1;

      for (int face = 0; face < 6; ++face)
      {

        if (next_node.depth > 0 && Rubiks::skip_rotations(next_node.get_face(), face))
        {
          continue;
        }

        for (int rotation = 0; rotation < 3; ++rotation)
        {
          Node new_node(next_node.state, nullptr, next_node.depth + 1, face, rotation, false, pdb_type, 0, 0);

          if (new_node.combined < next_node.combined) {
            std::cout << "Consistency error: " << unsigned(new_node.combined) << " < " << unsigned(next_node.combined) << " " << std::endl;
          }

          if (new_node.combined > id_depth)
          {
            continue;
          }

          if (Rubiks::is_solved(new_node.state))
          {
            std::cout << "Solved IDA*: " << unsigned int(id_depth) << std::endl;
            std::cout << "Solution: " << new_node.print_solution() << std::endl;
            done = true;
            #pragma omp flush (done)
          }

          my_stack.push(new_node);
        }
      }
    }
  }
  std::cout << "Count = " << unsigned long long(count) << std::endl;
  return count;
}
