#include "rubiks.h"

void Rubiks::rotate (uint8_t state[], const uint8_t face, const uint8_t rotation)
{
  uint8_t rotation_index = 6 * rotation + face;
  if (rotation == 2)
  {
    for (uint8_t i = 0; i < 40; i += 2)
    {
      state[i] = __turn_position_lookup[state[i]][rotation_index];
    }
  }
  else
  {
    for (uint8_t i = 0; i < 40; i += 2)
    {
      if (__turn_lookup[state[i]][face])
      {
        if (__corner_booleans[i])
        {
          state[i + 1] = __corner_rotation[face][state[i + 1]];
        }
        else if ( face == 2 or face == 5 ) // Face left and right
        {
          state[i + 1] = 1 - state[i + 1];
        }
        state[i] = __turn_position_lookup[state[i]][rotation_index];
      }
    }
  }
}


uint32_t Rubiks::get_corner_index (const uint8_t state[])
{
  const static int base3[] = {729, 243, 81, 27, 9, 3, 1};
  const static int size = 8;
  uint8_t corners[size];
  for (uint8_t i = 0; i < size; ++i)
  {
    corners[i] = __cube_translations[state[__corner_pos_indices[i]]];
  }

  uint32_t corner_index = (uint32_t) mr::get_rank (size, corners);
  corner_index *= 2187;

  for (uint8_t i = 0; i < size - 1; ++i)
  {
    corner_index += state[__corner_rot_indices[i]] * base3[i];
  }
  return corner_index;
}

uint64_t FactorialUpperK (int n, int k)
{
  static const uint64_t result[13][13] =
  {
    {1}, // n = 0
    {1, 1}, // n = 1
    {2, 2, 1}, // n = 2
    {6, 6, 3, 1}, // n = 3
    {24, 24, 12, 4, 1}, // n = 4
    {120, 120, 60, 20, 5, 1}, // n = 5
    {720, 720, 360, 120, 30, 6, 1}, // n = 6
    {5040, 5040, 2520, 840, 210, 42, 7, 1}, // n = 7
    {40320, 40320, 20160, 6720, 1680, 336, 56, 8, 1}, // n = 8
    {362880, 362880, 181440, 60480, 15120, 3024, 504, 72, 9, 1}, // n = 9
    {3628800, 3628800, 1814400, 604800, 151200, 30240, 5040, 720, 90, 10, 1}, // n = 10
    {39916800, 39916800, 19958400, 6652800, 1663200, 332640, 55440, 7920, 990, 110, 11, 1}, // n = 11
    {479001600, 479001600, 239500800, 79833600, 19958400, 3991680, 665280, 95040, 11880, 1320, 132, 12, 1} // n = 12
  };
  return result[n][k];
}

uint64_t Rubiks::get_edge_index (const uint8_t state[], int size, const uint8_t edges[],
                                 const uint8_t edge_rot_indices[])
{
  uint8_t edge_pos[12];
  for (uint8_t i = 0; i < 12; ++i)
  {
    edge_pos[i] = __cube_translations[state[edge_pos_indices_12[i]]];
  }
  int8_t puzzle[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  uint8_t dual[12];
  uint8_t newdual[12];
  for (int x = 0; x < 12; x++)
    dual[edge_pos[x]] = x;
  for (int x = 0; x < size; x++)
  {
    newdual[x] = dual[edges[x]];
    puzzle[dual[edges[x]]] = x;
  }
  uint64_t edge_index = mr::k_rank (puzzle, newdual, size, 12);

  edge_index *= 1 << size;

  for (uint8_t i = 0; i < size; ++i)
  {
    edge_index += state[edge_rot_indices[i]] * 1 << (size - i - 1);
  }
  return edge_index;
}

bool Rubiks::is_solved (const uint8_t cube[])
{
  for (uint8_t i = 0; i < sizeof __goal; ++i)
  {
    if (__goal[i] != cube[i])
    {
      return false;
    }
  }
  return true;
}

uint8_t Rubiks::pattern_database_lookup (const uint8_t state[])
{
  static bool initialized = false;
  static std::vector<char> edge_8a, edge_8b;
  static std::vector<uint8_t> corner_db;

  if (initialized == false)
  {
    initialized = true;
    std::vector<uint64_t> shape { 1 };
    npy::LoadArrayFromNumpy<uint8_t> ("corner_db.npy", shape, corner_db);

    shape.clear();
    shape.push_back (1);
    npy::LoadArrayFromNumpy<char> ("edge_db_8a.npy", shape, edge_8a);

    shape.clear();
    shape.push_back (1);
    npy::LoadArrayFromNumpy<char> ("edge_db_8b.npy", shape, edge_8b);
  }
  uint8_t best = corner_db[get_corner_index (state)];
  uint8_t val = edge_8a[get_edge_index (state, 8, edges_8a, edge_rot_indices_8a)];
  if (val > best)
  {
    best = val;
  }
  val = edge_8b[get_edge_index (state, 8, edges_8b, edge_rot_indices_8b)];
  if (val > best)
  {
    best = val;
  }
  return best;
}

uint64_t npr (int n, int r)
{
  return Rubiks::__factorial_lookup[n] / Rubiks::__factorial_lookup[n - r];
}

void Rubiks::generate_edges_pattern_database (std::string filename,
    const uint8_t state[],
    const uint8_t max_depth,
    const uint8_t size,
    const uint8_t edges[],
    const uint8_t edge_rot_indices[])
{
  std::cout << "Generating edges db\n";
  std::stack<RubiksIndex*> stack;
  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  stack.push (new RubiksIndex (new_state, 0, 0) );

  uint64_t all_edges = npr (12, size) * pow (2, size);
  std::cout << "Edges: " << all_edges << "\n";
  std::vector<uint8_t> pattern_lookup (all_edges, max_depth);
  uint64_t new_state_index = get_edge_index (state, size, edges, edge_rot_indices);
  pattern_lookup[new_state_index] = 0;
  uint8_t* found_index_stack = new uint8_t[all_edges];
  std::fill_n (found_index_stack, all_edges, max_depth);
  unsigned int id_depth = 1;
  unsigned int count = 1;

  uint8_t new_state_depth = 0;
  while (count < all_edges && id_depth < max_depth)
  {

    if ( stack.empty() )
    {
      id_depth += 1;
      std::fill_n (found_index_stack, all_edges, max_depth);
      new_state = new uint8_t[40];
      memcpy (new_state, state, 40);
      stack.push (new RubiksIndex (new_state, 0, 0) );
      std::cout << "Incrementing id-depth to " << id_depth << "\n";
    }

    RubiksIndex* ri = stack.top();
    stack.pop();
    for (uint8_t face = 0; face < 6; ++face)
    {
      if (ri->depth > 0 && Rubiks::skip_rotations (ri->last_face, face) )
      {
        continue;
      }
      for (uint8_t rotation = 0; rotation < 3; ++rotation)
      {
        new_state = new uint8_t[40];
        memcpy (new_state, ri->state, 40);
        rotate (new_state, face, rotation);
        new_state_index = get_edge_index (new_state, size, edges, edge_rot_indices);
        new_state_depth = ri->depth + 1;
        if (new_state_depth == id_depth && pattern_lookup[new_state_index] == max_depth)
        {
          pattern_lookup[new_state_index] = new_state_depth;
          count += 1;
          if (count % 100000 == 0 || count > (all_edges * .99) )
          {
            std::cout << count << "\n";
          }
          delete[] new_state;
        }
        else if (new_state_depth < id_depth && new_state_depth < found_index_stack[new_state_index])
        {
          found_index_stack[new_state_index] = new_state_depth;
          stack.push (new RubiksIndex (new_state, new_state_depth, face) );
        }
        else
        {
          delete[] new_state;
        }
      }
    }
    delete ri;
  }
  delete[] found_index_stack;
  const uint64_t shape[] = {all_edges};
  npy::SaveArrayAsNumpy (filename, false, 1, shape, pattern_lookup);
}

void Rubiks::generate_corners_pattern_database (std::string filename, const uint8_t state[], const uint8_t max_depth)
{
  std::cout << "Generating corners db\n";
  std::stack<RubiksIndex*> stack;
  uint8_t* new_state = new uint8_t[40];
  memcpy (new_state, state, 40);
  stack.push (new RubiksIndex (new_state, 0, 0) );

  //# 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
  uint32_t all_corners = 88179840;
  std::vector<uint8_t> pattern_lookup (all_corners, max_depth);
  pattern_lookup[get_corner_index (state)] = 0;
  uint8_t* found_index_stack = new uint8_t[all_corners];
  std::fill_n (found_index_stack, all_corners, max_depth);
  unsigned int id_depth = 1;
  unsigned int count = 1;

  uint32_t new_state_index = 0;
  uint8_t new_state_depth = 0;
  while (count < all_corners && id_depth < max_depth)
  {

    if ( stack.empty() )
    {
      id_depth += 1;
      std::fill_n (found_index_stack, all_corners, max_depth);
      new_state = new uint8_t[40];
      memcpy (new_state, state, 40);
      stack.push (new RubiksIndex (new_state, 0, 0) );
      std::cout << "Incrementing id-depth to " << id_depth << "\n";
    }

    RubiksIndex* ri = stack.top();
    stack.pop();
    for (uint8_t face = 0; face < 6; ++face)
    {
      if (ri->depth > 0 && Rubiks::skip_rotations (ri->last_face, face) )
      {
        continue;
      }
      for (uint8_t rotation = 0; rotation < 3; ++rotation)
      {
        new_state = new uint8_t[40];
        memcpy (new_state, ri->state, 40);
        rotate (new_state, face, rotation);
        new_state_index = get_corner_index (new_state);
        new_state_depth = ri->depth + 1;
        if (new_state_depth == id_depth && pattern_lookup[new_state_index] == max_depth)
        {
          pattern_lookup[new_state_index] = new_state_depth;
          count += 1;
          if (count % 1000000 == 0 || count > 88100000)
          {
            std::cout << count << "\n";
          }
          delete[] new_state;
        }
        else if (new_state_depth < id_depth && new_state_depth < found_index_stack[new_state_index])
        {
          found_index_stack[new_state_index] = new_state_depth;
          stack.push (new RubiksIndex (new_state, new_state_depth, face) );
        }
        else
        {
          delete[] new_state;
        }
      }
    }
    delete ri;
  }
  delete[] found_index_stack;

  const uint64_t shape [] = {all_corners};
  npy::SaveArrayAsNumpy (filename, false, 1, shape, pattern_lookup);
}

void Rubiks::generate_all_dbs()
{
// generate_corners_pattern_database ("corner_db.npy", __goal, corner_max_depth);
//  generate_edges_pattern_database ("edge_db_6a.npy", __goal, edge_6_max_depth, 6, edges_6a, edge_rot_indices_6a);
//  generate_edges_pattern_database ("edge_db_6b.npy", __goal, edge_6_max_depth, 6, edges_6b, edge_rot_indices_6b);
  generate_edges_pattern_database ("edge_db_8a.npy", __goal, edge_8_max_depth, 8, edges_8a, edge_rot_indices_8a);
  generate_edges_pattern_database ("edge_db_8b.npy", __goal, edge_8_max_depth, 8, edges_8b, edge_rot_indices_8b);
}
