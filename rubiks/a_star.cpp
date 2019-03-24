#include "a_star.h"


void search::a_star(const uint8_t* state, const Rubiks::PDB pdb_type)
{
  std::cout << "IDA*" << std::endl;
  std::stack<Node*, std::vector<Node*>> state_stack;

  if (Rubiks::is_solved(state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  Node* new_node = new Node();
  memcpy(new_node->state, state, 40);
  uint8_t id_depth = Rubiks::pattern_lookup(new_node->state, pdb_type, 0);
  new_node->heuristic = id_depth;
  new_node->combined = id_depth;
  state_stack.push(new_node);
  std::cout << "Minimum number of moves to solve: " << unsigned int(id_depth) << std::endl;
  uint64_t count = 0;
  Node* next_node;
  while (true)
  {
    if (state_stack.empty())
    {
      id_depth += 1;
      new_node = new Node();
      memcpy(new_node->state, state, 40);
      new_node->heuristic = Rubiks::pattern_lookup(new_node->state, pdb_type, 0);
      new_node->combined = new_node->heuristic;
      state_stack.push(new_node);
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
        new_node = new Node(next_node->state, nullptr, next_node->depth + 1, face, rotation, false, pdb_type, next_node->heuristic - 1, 0);

        if (new_node->combined > id_depth)
        {
          delete new_node;
          continue;
        }

        if (Rubiks::is_solved(new_node->state))
        {
          std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
          std::cout << "Solution: " << new_node->print_solution() << std::endl;

          delete new_node;
          while (state_stack.empty() == false) {
            next_node = state_stack.top();
            state_stack.pop();
            delete next_node;
          }
          return;
        }
        state_stack.push(new_node);
      }
    }
    delete next_node;
  }
}
