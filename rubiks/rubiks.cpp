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


uint32_t Rubiks::get_corner_index (uint8_t state[])
{
  /*
  Gets the unique index of the corners of this cube.
  Finds the permutation of 8 corner cubies using a factorial number system by counting the number of inversions
  per corner. https://en.wikipedia.org/wiki/Factorial_number_system

  Each given permutation of corners has 2187 possible rotations of corners (3^7),
  so multiply by 2187 and then calculate the rotation by assuming each rotation is a digit in base 3.
  */

  //Select all of the even (position) corner indices

  uint8_t corners[sizeof __corner_pos_indices];
  for (uint8_t i = 0; i < sizeof __corner_pos_indices; ++i)
  {
    corners[i] = state[__corner_pos_indices[i]];
  }

  //Count the number of inversions in the corner table per element
  uint8_t inversions[7] = {};
  for (uint8_t i = 0; i < 7; ++i)
  {
    for (uint8_t j = i + 1; j < 8; ++j)
    {
      if (corners[i] > corners[j])
      {
        inversions[i] += 1;
      }
    }
  }

  uint32_t corner_index = inversions[0] * 5040 ;
  corner_index += inversions[1] * 720 ;
  corner_index += inversions[2] * 120  ;
  corner_index += inversions[3] * 24  ;
  corner_index += inversions[4] * 6 ;
  corner_index += inversions[5] * 2  ;
  corner_index += inversions[6]  ;

  //Index into the specific corner rotation that we're in
  corner_index *= 2187;

  //View the odd (rotation) corner indices then convert them from a base 3 to base 10 number
  for (uint8_t i = 0; i < sizeof __corner_rot_indices; ++i)
  {
    corners[i] = state[__corner_rot_indices[i]];
  }
  corner_index += corners[0] * 729;
  corner_index += corners[1] * 243;
  corner_index += corners[2] * 81;
  corner_index += corners[3] * 27;
  corner_index += corners[4] * 9;
  corner_index += corners[5] * 3;
  corner_index += corners[6] ;
  return corner_index;
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
