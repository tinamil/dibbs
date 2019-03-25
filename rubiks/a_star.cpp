#include "a_star.h"


void search::a_star(const uint8_t* state, const Rubiks::PDB pdb_type)
{
  std::cout << "IDA*" << std::endl;
  std::stack<Node, std::vector<Node>> state_stack;

  if (Rubiks::is_solved(state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  Node original_node;
  memcpy(original_node.state, state, 40);
  uint8_t id_depth = Rubiks::pattern_lookup(original_node.state, pdb_type, 0);
  original_node.heuristic = id_depth;
  original_node.combined = id_depth;
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

    Node next_node = state_stack.top();
    state_stack.pop();

    count += 1;

    if (count % 1000000 == 0)
    {
      std::cout << count << std::endl;
    }
    #pragma omp parallel for
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
          std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
          std::cout << "Solution: " << new_node.print_solution() << std::endl;
          done = true;
        }
        #pragma omp critical (stack)
        {
          state_stack.push(new_node);
        }
      }
    }
  }
}
