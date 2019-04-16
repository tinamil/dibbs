#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stack>
#include <unordered_map>
#include "npy.hpp"
#include "mr_rank.h"
#include "utility.h"
#include "hash.hpp"

namespace Rubiks
{

  struct StateHash {
    size_t operator() (const uint8_t* s) const
    {
      return boost_hash(s, 40);
    }

    bool operator() (const uint8_t* a, const uint8_t* b) const
    {
      return memcmp(a, b, 40) == 0;
    }
  };

  enum PDB
  {
    a1997, a888, zero, a12
  };

  enum Face
  {
    front = 0,
    up = 1,
    left = 2,
    back = 3,
    down = 4,
    right = 5
  };

  enum Rotation
  {
    clockwise = 0,
    counterclockwise = 1,
    half = 2
  };
  const std::string _face_mapping[] = { "F", "U", "L", "B", "D", "R" };
  const std::string _rotation_mapping[] = { "", "'", "2" };

  const uint8_t __corner_cubies[] = { 0, 2, 5, 7, 12, 14, 17, 19 };

  const uint8_t __edge_cubies[] = { 1, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18 };

  const uint8_t __turn_position_lookup[][18] =
  {
    { 0, 0, 5, 2, 12, 0, 0, 0, 12, 5, 2, 0, 0, 0, 17, 7, 14, 0 },
  { 1, 1, 1, 4, 8, 1, 1, 1, 1, 3, 9, 1, 1, 1, 1, 6, 13, 1 },
  { 2, 2, 2, 7, 0, 14, 2, 2, 2, 0, 14, 7, 2, 2, 2, 5, 12, 19 },
  { 3, 3, 10, 1, 3, 3, 3, 3, 8, 6, 3, 3, 3, 3, 15, 4, 3, 3 },
  { 4, 4, 4, 6, 4, 9, 4, 4, 4, 1, 4, 11, 4, 4, 4, 3, 4, 16 },
  { 5, 7, 17, 0, 5, 5, 5, 17, 0, 7, 5, 5, 5, 19, 12, 2, 5, 5 },
  { 6, 11, 6, 3, 6, 6, 6, 10, 6, 4, 6, 6, 6, 18, 6, 1, 6, 6 },
  { 7, 19, 7, 5, 7, 2, 7, 5, 7, 2, 7, 19, 7, 17, 7, 0, 7, 14 },
  { 8, 8, 3, 8, 13, 8, 8, 8, 15, 8, 1, 8, 8, 8, 10, 8, 9, 8 },
  { 9, 9, 9, 9, 1, 16, 9, 9, 9, 9, 13, 4, 9, 9, 9, 9, 8, 11 },
  { 10, 6, 15, 10, 10, 10, 10, 18, 3, 10, 10, 10, 10, 11, 8, 10, 10, 10 },
  { 11, 18, 11, 11, 11, 4, 11, 6, 11, 11, 11, 16, 11, 10, 11, 11, 11, 9 },
  { 17, 12, 0, 12, 14, 12, 14, 12, 17, 12, 0, 12, 19, 12, 5, 12, 2, 12 },
  { 15, 13, 13, 13, 9, 13, 16, 13, 13, 13, 8, 13, 18, 13, 13, 13, 1, 13 },
  { 12, 14, 14, 14, 2, 19, 19, 14, 14, 14, 12, 2, 17, 14, 14, 14, 0, 7 },
  { 18, 15, 8, 15, 15, 15, 13, 15, 10, 15, 15, 15, 16, 15, 3, 15, 15, 15 },
  { 13, 16, 16, 16, 16, 11, 18, 16, 16, 16, 16, 9, 15, 16, 16, 16, 16, 4 },
  { 19, 5, 12, 17, 17, 17, 12, 19, 5, 17, 17, 17, 14, 7, 0, 17, 17, 17 },
  { 16, 10, 18, 18, 18, 18, 15, 11, 18, 18, 18, 18, 13, 6, 18, 18, 18, 18 },
  { 14, 17, 19, 19, 19, 7, 17, 7, 19, 19, 19, 14, 12, 5, 19, 19, 19, 2 }
  };

  const bool __turn_lookup[][6] =
  {
    { false, false, true, true, true, false },
  { false, false, false, true, true, false },
  { false, false, false, true, true, true },
  { false, false, true, true, false, false },
  { false, false, false, true, false, true },
  { false, true, true, true, false, false },
  { false, true, false, true, false, false },
  { false, true, false, true, false, true },
  { false, false, true, false, true, false },
  { false, false, false, false, true, true },
  { false, true, true, false, false, false },
  { false, true, false, false, false, true },
  { true, false, true, false, true, false },
  { true, false, false, false, true, false },
  { true, false, false, false, true, true },
  { true, false, true, false, false, false },
  { true, false, false, false, false, true },
  { true, true, true, false, false, false },
  { true, true, false, false, false, false },
  { true, true, false, false, false, true }
  };

  const uint8_t __corner_rotation[][3] =
  {
    { 0, 2, 1 },
  { 1, 0, 2 },
  { 2, 1, 0 },
  { 0, 2, 1 },
  { 1, 0, 2 },
  { 2, 1, 0 }
  };

  const bool __corner_booleans[] = { true, true, false, false, true, true, false, false,
    false, false, true, true, false, false, true, true,
    false, false, false, false, false, false, false, false,
    true, true, false, false, true, true, false, false,
    false, false, true, true, false, false, true, true
  };

  const uint8_t __corner_pos_indices[] = { 0, 4, 10, 14, 24, 28, 34, 38 };
  const uint8_t __corner_rot_indices[] = { 1, 5, 11, 15, 25, 29, 35, 39 };

  const uint8_t corner_max_depth = 11;
  const uint8_t edge_6_max_depth = 10;
  const uint8_t edge_8_max_depth = 12;
  const uint8_t edge_12_pos_max_depth = 10;
  const uint8_t edge_20_rot_max_depth = 9;

  const uint8_t edge_pos_indices_6a[] = { 2, 6, 8, 12, 16, 18 };
  const uint8_t edge_rot_indices_6a[] = { 3, 7, 9, 13, 17, 19 };
  const uint8_t edge_pos_indices_6b[] = { 20, 22, 26, 30, 32, 36 };
  const uint8_t edge_rot_indices_6b[] = { 21, 23, 27, 31, 33, 37 };
  const uint8_t edge_pos_indices_8a[] = { 2, 6, 8, 12, 16, 18, 20, 22 };
  const uint8_t edge_rot_indices_8a[] = { 3, 7, 9, 13, 17, 19, 21, 23 };
  const uint8_t edge_pos_indices_8b[] = { 16, 18, 20, 22, 26, 30, 32, 36 };
  const uint8_t edge_rot_indices_8b[] = { 17, 19, 21, 23, 27, 31, 33, 37 };
  const uint8_t edge_pos_indices_12[] = { 2, 6, 8, 12, 16, 18, 20, 22, 26, 30, 32, 36 };
  const uint8_t edge_rot_indices_12[] = { 3, 7, 9, 13, 17, 19, 21, 23, 27, 31, 33, 37 };

  const uint8_t __cube_translations[] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 4, 8, 5, 9, 10, 6, 11, 7 };
  const uint8_t edges_6a[] = { 0, 1, 2, 3, 4, 5 };
  const uint8_t edges_6b[] = { 6, 7, 8, 9, 10, 11 };
  const uint8_t edges_8a[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const uint8_t edges_8b[] = { 4, 5, 6, 7, 8, 9, 10, 11 };

  const static int base2[] = { 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
  const static int base3[] = { 729, 243, 81, 27, 9, 3, 1 };

  const uint8_t __goal[] = { 0, 0, 1, 0, 2, 0, 3, 0,
    4, 0, 5, 0, 6, 0, 7, 0,
    8, 0, 9, 0, 10, 0, 11, 0,
    12, 0, 13, 0, 14, 0, 15, 0,
    16, 0, 17, 0, 18, 0, 19, 0
  };


  inline bool skip_rotations(uint8_t last_face, uint8_t face)
  {
    return last_face == face || (last_face == 3 && face == 0) ||
      (last_face == 5 && face == 2) || (last_face == 4 && face == 1);
  }

  extern void rotate(uint8_t * new_state, const uint8_t face, const uint8_t rotation);
  extern uint32_t get_corner_index(const uint8_t * state);

  extern uint64_t get_edge_index(const uint8_t * state, bool a, PDB type);
  extern uint64_t get_edge_index(const uint8_t * state, int size, const uint8_t * edges, const uint8_t * edge_rot_indices);
  extern uint64_t get_new_edge_pos_index(const uint8_t * state);
  extern uint64_t get_new_edge_rot_index(const uint8_t * state);
  extern bool is_solved(const uint8_t * state);
  extern uint8_t pattern_lookup(const uint8_t * state, const uint8_t * start_state, PDB type);
  inline uint8_t pattern_lookup(const uint8_t * state, PDB type)
  {
    return pattern_lookup(state, __goal, type);
  }
  extern void generate_corners_pattern_database(std::string filename, const uint8_t * state, const uint8_t max_depth);
  extern void generate_edges_pattern_database(std::string filename, const uint8_t * state, const uint8_t max_depth, const uint8_t size, const uint8_t * edge_pos_indices, const uint8_t * edge_rot_indices);
  extern void generate_edges_pos_pattern_database(std::string filename, const uint8_t * state, const uint8_t max_depth);
  extern void generate_rotations_pattern_database(std::string filename, const uint8_t * state, const uint8_t max_depth);
  extern void generate_goal_dbs();
  struct RubiksIndex
  {
    uint8_t state[40];
    const uint8_t depth;
    const uint8_t last_face;
    RubiksIndex(const uint8_t depth, const uint8_t last_face) : state(), depth(depth), last_face(last_face) {}
    RubiksIndex(const RubiksIndex& original) : state(), depth(original.depth), last_face(original.last_face) {
      memcpy(state, original.state, 40);
    }
  };

  struct PDBVectors
  {
    std::vector<uint8_t> edge_a, edge_b;
    std::vector<uint8_t> corner_db;
  };

  struct RubiksEdgeStateHash
  {
    inline std::size_t operator() (const uint8_t* s) const {
      if (s == nullptr) {
        throw std::invalid_argument("received null pointer value in RubiksStateHash operator ()");
      }
      return boost_hash(s, 24);
    }
  };

  struct RubiksEdgeStateEqual
  {
    inline bool operator() (const uint8_t* a, const uint8_t* b) const {
      {
        if (a == nullptr || b == nullptr) {
          throw std::invalid_argument("received null pointer value in RubiksStateEqual operator ()");
        }
        return memcmp(a, b, 24) == 0;
      }
    }
  };
}
