#include "a_star.h"


void search::a_star (const uint8_t state[])
{
  std::cout << "A*" << std::endl;
  std::stack<Node*> state_stack;

  if (Rubiks::is_solved (state) )
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  int id_depth = Rubiks::pattern_database_lookup (new_state);
  state_stack.push (new Node (NULL, new_state, 0, id_depth, 0, 0) );
  std::cout << "Minimum number of moves to solve: " << id_depth << std::endl;
  int count = 0;
  Node* next_node;
  while (count < 1e8)
  {
    if (state_stack.empty() )
    {
      id_depth += 1;
      new_state = new uint8_t[40];
      memcpy (new_state, state, 40);
      state_stack.push (new Node (NULL, new_state, 0, id_depth, 0, 0) );
      std::cout << "Incrementing id-depth to " << id_depth << std::endl;
    }

    next_node = state_stack.top();
    state_stack.pop();

    for (uint8_t face = 0; face < 6; ++face)
    {

      if (next_node->depth > 0 && Rubiks::skip_rotations (next_node->faces[next_node->depth - 1], face) )
      {
        continue;
      }

      for (uint8_t rotation = 0; rotation < 3; ++rotation)
      {
        new_state = new uint8_t[40];
        memcpy (new_state, next_node->state, 40);
        Rubiks::rotate (new_state, face, rotation);


        uint8_t new_state_heuristic = Rubiks::pattern_database_lookup (new_state);
        uint8_t new_state_cost = next_node->depth + 1 + new_state_heuristic;

        count += 1;

        if (count % 1000000 == 0)
        {
          std::cout << count << std::endl;
        }

        if (new_state_cost > id_depth)
        {
          delete[] new_state;
          continue;
        }

        if (Rubiks::is_solved (new_state) )
        {
          //flip(new_faces);
          //flip(new_rots);
          std::cout << "Solved IDA*: " << id_depth << " Count = " << count << std::endl;
          return;
          //return new_faces, new_rots, count
        }
        state_stack.push (new Node (next_node, new_state, next_node->depth + 1, new_state_heuristic, face, rotation) );
      }
    }
    delete next_node;

  }
}

void expand()
{

}

void search::dibbs (const uint8_t state[])
{
  std::cout << "DIBBS" << std::endl;
  std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_front, open_back;
  std::unordered_set<Node*, NodeHash, NodeEqual> front_nodes, back_nodes;
  if (Rubiks::is_solved (state) )
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return;
  }

  uint64_t upper_bound = std::numeric_limits<uint64_t>::max();
  uint8_t r_heuristic = 0;

  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  open_front.push (new Node (NULL, new_state, 0, Rubiks::pattern_database_lookup (new_state), 0, 0) );

  new_state = new uint8_t[40];
  memcpy (new_state, Rubiks::__goal, 40);
  open_back.push (new Node (NULL, new_state, 0, r_heuristic, 0, 0) );

  bool explore_forward = true;
  Node* best_node, *next_node;
  uint8_t forward_fbar_min = 0;
  uint8_t backward_fbar_min = 0;
  uint8_t f_combined = 0;
  uint8_t b_combined = 0;
  uint8_t best_f_fbar = 0;
  uint8_t best_b_fbar = 0;
  uint8_t f_cost = 0;
  uint8_t b_cost = 0;
  int count = 0;

  while (count < 1e8 && upper_bound > (forward_fbar_min + backward_fbar_min) / 2)
  {

  }
}
