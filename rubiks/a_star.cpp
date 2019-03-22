#include "a_star.h"


void search::a_star(const uint8_t state[], const Rubiks::PDB pdb_type)
{
  std::cout << "IDA*" << std::endl;
  std::stack<Node*, std::vector<Node*>> state_stack;

  if (Rubiks::is_solved(state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  uint8_t* new_state = new uint8_t[40];
  memcpy(new_state, state, 40);
  uint8_t id_depth = Rubiks::pattern_lookup(new_state, pdb_type);
  state_stack.push(new Node(NULL, new_state, id_depth));
  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;
  uint64_t count = 0;
  Node* next_node;
  while (true)
  {
    if (state_stack.empty())
    {
      id_depth += 1;
      new_state = new uint8_t[40];
      memcpy(new_state, state, 40);
      state_stack.push(new Node(NULL, new_state, id_depth));
      std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
    }

    next_node = state_stack.top();
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
        new_state = new uint8_t[40];
        memcpy(new_state, next_node->state, 40);
        Rubiks::rotate(new_state, face, rotation);

        uint8_t new_state_heuristic = Rubiks::pattern_lookup(new_state, pdb_type);
        uint8_t new_state_cost = next_node->depth + 1 + new_state_heuristic;

        if (new_state_cost > id_depth)
        {
          delete[] new_state;
          continue;
        }

        if (Rubiks::is_solved(new_state))
        {
          std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
          Node n(next_node, new_state, next_node->depth + 1, new_state_heuristic, face, rotation);
          std::cout << "Solution: " << n.print_solution() << std::endl;
          return;
        }
        state_stack.push(new Node(next_node, new_state, next_node->depth + 1, new_state_heuristic, face, rotation));
      }
    }
    delete next_node;
  }
}
