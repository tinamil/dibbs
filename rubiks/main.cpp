#include <iostream>
#include <stack>
#include "rubiks.h"

using namespace std;

void a_star (const uint8_t state[]);
int main()
{
  cout << "Hello world!" << endl;
  const uint8_t start_state[] =
  {
    2,  2,  8,  1, 17, 1,  9,  0, 15,  0,  7,  1, 18,  1, 14,  0,  3,  0, 13,  1,  1,  0, 10,  0,
    12,  0,  6,  1,  5,  1,  4,  1, 11,  0,  0,  2, 16,  1, 19,  1
  };
  a_star (start_state);
  return 0;
}

struct Node
{
  const Node* parent;
  const uint8_t* state;
  uint8_t depth;
  uint8_t face;
  uint8_t rotation;

  Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _face, uint8_t _rotation)
  {
    parent = _parent;
    state = _state;
    depth = _depth;
    face = _face;
    rotation = _rotation;
  }
};

uint8_t heuristic_func (const uint8_t state[])
{
  return 0;
}

using namespace Rubiks;
void a_star (const uint8_t state[])
{
  std::stack<Node*> state_stack;

  if (is_solved (state) )
  {
    cout << "A* given a solved cube.  Nothing to solve." << endl;
    return;
  }

  Node* n = new Node (NULL, state, 0, 0, 0);
  state_stack.push (n);
  int id_depth = heuristic_func (state);
  cout << "Minimum number of moves to solve: " << id_depth << endl;
  int count = 0;
  while (true)
  {

    if (state_stack.empty() )
    {
      id_depth += 1;
      state_stack.push (new Node (NULL, state, 0, 0, 0) );
      cout << "Incrementing id-depth to " << id_depth << endl;
    }

    Node* next_node = state_stack.top();
    state_stack.pop();

    for (uint8_t face = 0; face < 6; ++face)
    {

      if (next_node->depth > 0 && skip_rotations (next_node->face, face) )
      {
        continue;
      }

      for (uint8_t rotation = 0; rotation < 3; ++rotation)
      {
        uint8_t* new_state = new uint8_t[40];
        memcpy (new_state, next_node->state, 40);
        rotate (new_state, face, rotation);


        uint8_t new_state_heuristic = heuristic_func (new_state);
        uint8_t new_state_cost = next_node->depth + 1 + new_state_heuristic;

        count += 1;

        if (new_state_cost > id_depth)
        {
          continue;
        }

        if (is_solved (new_state) )
        {
          //flip(new_faces);
          //flip(new_rots);
          cout << "Solved IDA*: " << id_depth << " Count = " << count << endl;
          //return new_faces, new_rots, count
        }
        state_stack.push (new Node (next_node, new_state, next_node->depth + 1, face, rotation) );
      }
    }
    delete[] next_node;
  }
}
