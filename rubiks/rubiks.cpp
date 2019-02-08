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
  const static int size = sizeof __corner_pos_indices;
  uint8_t corners[size];
  for (uint8_t i = 0; i < size; ++i)
  {
    corners[i] = __cube_translations[state[__corner_pos_indices[i]]];
  }

  uint32_t corner_index = (uint32_t) mr::get_rank(size, corners);
  corner_index *= 2187;

  for (uint8_t i = 0; i < size - 1; ++i)
  {
    corner_index += state[__corner_rot_indices[i]] * base3[i];
  }
  return corner_index;
}

uint64_t Rubiks::get_edge_index(const uint8_t state[], int size, const uint8_t edge_pos_indices[], const uint8_t edge_rot_indices[])
{
  uint8_t edge_pos[size];
  for (uint8_t i = 0; i < size; ++i)
  {
    edge_pos[i] = __cube_translations[state[edge_pos_indices[i]]];
  }

  uint64_t edge_index = mr::get_rank(size, edge_pos);
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

uint8_t Rubiks::pattern_database_lookup(const uint8_t state[])
{
  static bool initialized = false;
  static std::vector<char> corner_db, edge_8a, edge_8b;

  if(initialized == false)
  {
    initialized = true;
    std::vector<uint64_t> shape { 1 };
    npy::LoadArrayFromNumpy<char>("C:\\Users\\John\\git\\dibbs\\Python\\Rubiks\\corner_db.npy", shape, corner_db);

    shape.clear();
    shape.push_back(1);
    npy::LoadArrayFromNumpy<char>("C:\\Users\\John\\git\\dibbs\\Python\\Rubiks\\edge_db_8a.npy", shape, edge_8a);

    shape.clear();
    shape.push_back(1);
    npy::LoadArrayFromNumpy<char>("C:\\Users\\John\\git\\dibbs\\Python\\Rubiks\\edge_db_8b.npy", shape, edge_8b);
  }
  uint8_t best = corner_db[get_corner_index(state)];
  char val = edge_8a[get_edge_index(state, sizeof edge_pos_indices_8a, edge_pos_indices_8a, edge_rot_indices_8a)];
  if(val > best)
  {
    best = val;
  }
  val = edge_8b[get_edge_index(state, sizeof edge_pos_indices_8b, edge_pos_indices_8b, edge_rot_indices_8b)];
  if(val > best)
  {
    best = val;
  }
  return best;
}
