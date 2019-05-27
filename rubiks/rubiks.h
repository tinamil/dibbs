#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stack>
#include <queue>
#include <functional>
#include <unordered_map>
#include <thread>
#include <utility>
#include "npy.hpp"
#include "mr_rank.h"
#include "utility.h"
#include "hash.hpp"
#include "cameron314/blockingconcurrentqueue.h"

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

  static constexpr uint64_t __factorial_lookup[] =
  {
    1ll, 1ll, 2ll, 6ll, 24ll, 120ll, 720ll, 5040ll, 40320ll,
    362880ll, 3628800ll, 39916800ll, 479001600ll,
    6227020800ll, 87178291200ll, 1307674368000ll,
    20922789888000ll, 355687428096000ll, 6402373705728000ll,
    121645100408832000ll, 2432902008176640000ll
  };

  inline constexpr uint64_t npr(int n, int r) { return __factorial_lookup[n] / __factorial_lookup[n - r]; }

  const bool __corner_booleans[] = { true, true, false, false, true, true, false, false,
    false, false, true, true, false, false, true, true,
    false, false, false, false, false, false, false, false,
    true, true, false, false, true, true, false, false,
    false, false, true, true, false, false, true, true
  };

  const uint8_t __corner_pos_indices[] = { 0, 4, 10, 14, 24, 28, 34, 38 };
  const uint8_t __corner_rot_indices[] = { 1, 5, 11, 15, 25, 29, 35, 39 };

  constexpr uint8_t corner_max_depth = 11;
  constexpr uint8_t edge_6_max_depth = 10;
  constexpr uint8_t edge_8_max_depth = 12;
  constexpr uint8_t edge_12_pos_max_depth = 10;
  constexpr uint8_t edge_20_rot_max_depth = 9;

  constexpr size_t corner_max_count = 88179840;
  constexpr size_t edge_6_max_count = npr(12, 6) * 64;
  constexpr size_t edge_8_max_count = npr(12, 8) * 256;
  constexpr size_t edge_12_pos_max_count = npr(12, 12);
  constexpr size_t edge_20_rot_max_count = 4478976;

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

  extern void rotate(uint8_t* new_state, const uint8_t face, const uint8_t rotation);
  extern uint32_t get_corner_index(const uint8_t* state);

  extern uint64_t get_edge_index(const uint8_t* state, const bool a, const PDB type);
  extern uint64_t get_edge_index(const uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices);

  inline uint64_t get_edge_index6a(const uint8_t* state) { return get_edge_index(state, 6, edges_6a, edge_rot_indices_6a); }
  inline uint64_t get_edge_index6b(const uint8_t* state) { return get_edge_index(state, 6, edges_6b, edge_rot_indices_6b); }
  inline uint64_t get_edge_index8a(const uint8_t* state) { return get_edge_index(state, 8, edges_8a, edge_rot_indices_8a); }
  inline uint64_t get_edge_index8b(const uint8_t* state) { return get_edge_index(state, 8, edges_8b, edge_rot_indices_8b); }

  extern uint64_t get_new_edge_pos_index(const uint8_t* state);
  extern uint64_t get_new_edge_rot_index(const uint8_t* state);
  extern bool is_solved(const uint8_t* state);
  extern uint8_t pattern_lookup(const uint8_t* state, const uint8_t* start_state, PDB type);
  inline uint8_t pattern_lookup(const uint8_t* state, PDB type)
  {
    return pattern_lookup(state, __goal, type);
  }
  extern void generate_pattern_database(std::string filename, const uint8_t* state, const uint8_t max_depth, const size_t max_count, const std::function<size_t(const uint8_t* state)> func);
  extern void generate_pattern_database_multithreaded(std::string filename, const uint8_t* state, const uint8_t max_depth, const size_t max_count, const std::function<size_t(const uint8_t* state)> func);

  struct RubiksIndex
  {
    uint8_t state[40];
    uint8_t depth;
    uint8_t last_face;

    RubiksIndex() : state(), depth(0), last_face(0) {  }
    RubiksIndex(const uint8_t* original_state, const uint8_t depth, const uint8_t last_face) : depth(depth), last_face(last_face) {
      memcpy(state, original_state, 40);
    }
    RubiksIndex(const uint8_t* original_state, const uint8_t depth, const uint8_t face, const uint8_t rotation) : state(), depth(depth), last_face(face) {
      memcpy(state, original_state, 40);
      rotate(state, face, rotation);
    }
    RubiksIndex(const RubiksIndex& original) : state(), depth(original.depth), last_face(original.last_face) {
      memcpy(state, original.state, 40);
    }
    RubiksIndex(RubiksIndex&& original) noexcept : state(), depth(original.depth), last_face(original.last_face) {
      memcpy(state, original.state, 40);
    }
    RubiksIndex& operator=(const RubiksIndex& rhs) {
      memcpy(state, rhs.state, 40);
      depth = rhs.depth;
      last_face = rhs.last_face;
      return *this;
    }
  };

  struct PDB_Value {
    size_t index;
    uint8_t value;

    PDB_Value() : index(0), value(0) {}
    PDB_Value(size_t idx, uint8_t val) noexcept : index(idx), value(val) {}
  };


  extern void pdb_expand_nodes(
    moodycamel::ConcurrentQueue<PDB_Value>& results_queue,
    moodycamel::ConcurrentQueue<RubiksIndex>& input_queue,
    std::vector<uint8_t>& pattern_lookup,
    const std::function<size_t(const uint8_t* state)> lookup_func,
    const uint8_t id_depth,
    std::atomic_bool& finished
  );

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
