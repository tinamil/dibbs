#include <iostream>
#include <stack>
#include <vector>
#include "rubiks.h"

using namespace std;

void a_star (const uint8_t state[]);
int main()
{
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
  const uint8_t* state;
  uint8_t depth;
  uint8_t* faces;
  uint8_t* rotations;

  Node (Node* _parent, const uint8_t _state[], uint8_t _depth, uint8_t _face, uint8_t _rotation)
  {
    state = _state;
    depth = _depth;

    faces = new uint8_t[depth];
    rotations = new uint8_t[depth];

    if(_parent != NULL)
    {
      memcpy(faces, _parent->faces, depth-1);
      memcpy(rotations, _parent->rotations, depth-1);
      faces[depth-1] = _face;
      rotations[depth-1] = _rotation;
    }
  }

  ~Node()
  {
    delete[] state;
    delete[] faces;
    delete[] rotations;
  }
};

using namespace Rubiks;
void a_star (const uint8_t state[])
{

  std::stack<Node*> state_stack;

  if (is_solved (state) )
  {
    cout << "A* given a solved cube.  Nothing to solve." << endl;
    return;
  }

  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  state_stack.push (new Node (NULL, new_state, 0, 0, 0));
  int id_depth = pattern_database_lookup(new_state);
  cout << "Minimum number of moves to solve: " << id_depth << endl;
  int count = 0;
  Node* next_node;
  while (count < 5e7)
  {
    if (state_stack.empty() )
    {
      id_depth += 1;
      new_state = new uint8_t[40];
      memcpy (new_state, state, 40);
      state_stack.push (new Node (NULL, new_state, 0, 0, 0) );
      cout << "Incrementing id-depth to " << id_depth << endl;
    }

    next_node = state_stack.top();
    state_stack.pop();

    for (uint8_t face = 0; face < 6; ++face)
    {

      if (next_node->depth > 0 && skip_rotations (next_node->faces[next_node->depth-1], face) )
      {
        continue;
      }

      for (uint8_t rotation = 0; rotation < 3; ++rotation)
      {
        new_state = new uint8_t[40];
        memcpy (new_state, next_node->state, 40);
        rotate (new_state, face, rotation);


        uint8_t new_state_heuristic = pattern_database_lookup(new_state);
        uint8_t new_state_cost = next_node->depth + 1 + new_state_heuristic;

        count += 1;

        if(count % 1000000 == 0)
        {
          cout << count << endl;
        }

        if (new_state_cost > id_depth)
        {
          delete[] new_state;
          continue;
        }

        if (is_solved (new_state) )
        {
          //flip(new_faces);
          //flip(new_rots);
          cout << "Solved IDA*: " << id_depth << " Count = " << count << endl;
          return;
          //return new_faces, new_rots, count
        }
        state_stack.push (new Node (next_node, new_state, next_node->depth + 1, face, rotation) );
      }
    }
    delete next_node;

  }
}
