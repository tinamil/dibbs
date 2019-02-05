#pragma once
#include <stdint.h>
#include <cstring>

namespace Rubiks
{

const uint8_t __corner_cubies[] = {0, 2, 5, 7, 12, 14, 17, 19};

const uint8_t __edge_cubies[] = {1, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18};

const uint8_t __turn_position_lookup[][18] = {{0, 0, 5, 2, 12, 0, 0, 0, 12, 5, 2, 0, 0, 0, 17, 7, 14, 0},
  {1, 1, 1, 4, 8, 1, 1, 1, 1, 3, 9, 1, 1, 1, 1, 6, 13, 1},
  {2, 2, 2, 7, 0, 14, 2, 2, 2, 0, 14, 7, 2, 2, 2, 5, 12, 19},
  {3, 3, 10, 1, 3, 3, 3, 3, 8, 6, 3, 3, 3, 3, 15, 4, 3, 3},
  {4, 4, 4, 6, 4, 9, 4, 4, 4, 1, 4, 11, 4, 4, 4, 3, 4, 16},
  {5, 7, 17, 0, 5, 5, 5, 17, 0, 7, 5, 5, 5, 19, 12, 2, 5, 5},
  {6, 11, 6, 3, 6, 6, 6, 10, 6, 4, 6, 6, 6, 18, 6, 1, 6, 6},
  {7, 19, 7, 5, 7, 2, 7, 5, 7, 2, 7, 19, 7, 17, 7, 0, 7, 14},
  {8, 8, 3, 8, 13, 8, 8, 8, 15, 8, 1, 8, 8, 8, 10, 8, 9, 8},
  {9, 9, 9, 9, 1, 16, 9, 9, 9, 9, 13, 4, 9, 9, 9, 9, 8, 11},
  {10, 6, 15, 10, 10, 10, 10, 18, 3, 10, 10, 10, 10, 11, 8, 10, 10, 10},
  {11, 18, 11, 11, 11, 4, 11, 6, 11, 11, 11, 16, 11, 10, 11, 11, 11, 9},
  {17, 12, 0, 12, 14, 12, 14, 12, 17, 12, 0, 12, 19, 12, 5, 12, 2, 12},
  {15, 13, 13, 13, 9, 13, 16, 13, 13, 13, 8, 13, 18, 13, 13, 13, 1, 13},
  {12, 14, 14, 14, 2, 19, 19, 14, 14, 14, 12, 2, 17, 14, 14, 14, 0, 7},
  {18, 15, 8, 15, 15, 15, 13, 15, 10, 15, 15, 15, 16, 15, 3, 15, 15, 15},
  {13, 16, 16, 16, 16, 11, 18, 16, 16, 16, 16, 9, 15, 16, 16, 16, 16, 4},
  {19, 5, 12, 17, 17, 17, 12, 19, 5, 17, 17, 17, 14, 7, 0, 17, 17, 17},
  {16, 10, 18, 18, 18, 18, 15, 11, 18, 18, 18, 18, 13, 6, 18, 18, 18, 18},
  {14, 17, 19, 19, 19, 7, 17, 7, 19, 19, 19, 14, 12, 5, 19, 19, 19, 2}
};

const bool __turn_lookup[][6] = {{false, false, true, true, true, false},
  {false, false, false, true, true, false},
  {false, false, false, true, true, true},
  {false, false, true, true, false, false},
  {false, false, false, true, false, true},
  {false, true, true, true, false, false},
  {false, true, false, true, false, false},
  {false, true, false, true, false, true},
  {false, false, true, false, true, false},
  {false, false, false, false, true, true},
  {false, true, true, false, false, false},
  {false, true, false, false, false, true},
  {true, false, true, false, true, false},
  {true, false, false, false, true, false},
  {true, false, false, false, true, true},
  {true, false, true, false, false, false},
  {true, false, false, false, false, true},
  {true, true, true, false, false, false},
  {true, true, false, false, false, false},
  {true, true, false, false, false, true}
};

const uint8_t __corner_rotation[][3] = {{0, 2, 1},
  {1, 0, 2},
  {2, 1, 0},
  {0, 2, 1},
  {1, 0, 2},
  {2, 1, 0}
};

const bool __corner_booleans[] = {true, true, false, false, true, true, false, false,
                                  false, false, true, true, false, false, true, true,
                                  false, false, false, false, false, false, false, false,
                                  true, true, false, false, true, true, false, false,
                                  false, false, true, true, false, false, true, true
                                 };

const uint8_t __corner_pos_indices[] = {0, 4, 10, 14, 24, 28, 34, 38};
const uint8_t __corner_rot_indices[] = {1, 5, 11, 15, 25, 29, 35, 39};

const uint8_t corner_max_depth = 11;
const uint8_t edge_6_max_depth = 10;
const uint8_t edge_8_max_depth = 12;

const uint8_t edge_pos_indices_6a[] = {2, 6, 8, 12, 16, 18};
const uint8_t edge_rot_indices_6a[] = {3, 7, 9, 13, 17, 19};
const uint8_t edge_pos_indices_6b[] = {20, 22, 26, 30, 32, 36};
const uint8_t edge_rot_indices_6b[] = {21, 23, 27, 31, 33, 37};
const uint8_t edge_pos_indices_8a[] = {2, 6, 8, 12, 16, 18, 20, 22};
const uint8_t edge_rot_indices_8a[] = {3, 7, 9, 13, 17, 19, 21, 23};
const uint8_t edge_pos_indices_8b[] = {16, 18, 20, 22, 26, 30, 32, 36};
const uint8_t edge_rot_indices_8b[] = {17, 19, 21, 23, 27, 31, 33, 37};

const uint8_t __edge_translations[] = {0, 0, 0, 1, 2, 0, 3, 0, 4, 5, 6, 7, 0, 8, 0, 9, 10, 0, 11, 0};

const uint64_t __factorial_lookup[] =
{
  1, 1, 2, 6, 24, 120, 720, 5040, 40320,
  362880, 3628800, 39916800, 479001600,
  6227020800, 87178291200, 1307674368000,
  20922789888000, 355687428096000, 6402373705728000,
  121645100408832000, 2432902008176640000
};


const uint8_t __goal[] = {0, 0, 1, 0, 2, 0, 3, 0,
                          4, 0, 5, 0, 6, 0, 7, 0,
                          8, 0, 9, 0, 10, 0, 11, 0,
                          12, 0, 13, 0, 14, 0, 15, 0,
                          16, 0, 17, 0, 18, 0, 19, 0
                         };

inline uint64_t fast_factorial (uint8_t n)
{
  return __factorial_lookup[n];
}

inline bool skip_rotations (uint8_t last_face, uint8_t face)
{
  return last_face == face || (last_face == 3 && face == 0) ||
        (last_face == 5 && face == 2) || (last_face == 4 && face == 1);
}

extern void rotate (uint8_t state[], uint8_t face, uint8_t rotation);
extern uint32_t get_corner_index (uint8_t state[]);
extern bool is_solved(const uint8_t state[]);
}
