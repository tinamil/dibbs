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
#include <mutex>
#include "npy.hpp"
#include "mr_rank.h"
#include "utility.h"
#include "hash.hpp"
#include "cameron314/blockingconcurrentqueue.h"

/*
 6 center cubies are fixed position (rotation doesn't change) and define the solution color for that face
 8 corner cubies are defined by an integer defining their rotation: 0 to 2186 (3^7-1, 8th cube is defined by the other 7) and an integer defining their positions: 0 to 40319 (8! - 1)
 12 edge cubies have rotation integer 0 to 2047 (2^11-1) and position integer 0 to 12!-1

 A cube state is an array of 20 sets of (position, rotation), values ranging from 0-19 and 0-2 respectively, values of 20 and 3 denote a blank cubie.
 Rotations are 0-1 for edge pieces.  Center face pieces are fixed and not included.
 Cubies are numbered from 0 to 19 throughout the code, as shown here ( credit to https://github.com/brownan/Rubiks-Cube-Solver/blob/master/cube.h )
     5----6----7
     |         |\
     3    Y    4 \
     |         |  \
     0----1----2   \
      \             \
       \   10---R---11
        \  |         |\
         \ B    X    G \
          \|         |  \
           8----O----9   \
            \             \
             \   17--18---19
              \  |         |
               \ 15   W   16
                \|         |
                 12--13---14
*/
namespace Rubiks
{

  struct StateHash {
    uint64_t operator() (const uint8_t* s) const
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
    a1997, a888, zero, a12, a81220, clear_state
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

  static constexpr uint64_t __factorial_lookup[] =
  {
    1ll, 1ll, 2ll, 6ll, 24ll, 120ll, 720ll, 5040ll, 40320ll,
    362880ll, 3628800ll, 39916800ll, 479001600ll,
    6227020800ll, 87178291200ll, 1307674368000ll,
    20922789888000ll, 355687428096000ll, 6402373705728000ll,
    121645100408832000ll, 2432902008176640000ll
  };

  inline constexpr uint64_t npr(int n, int r) { return __factorial_lookup[n] / __factorial_lookup[n - r]; }


  const uint8_t __corner_pos_indices[] = { 0, 2, 5, 7, 12, 14, 17, 19 };
  const uint8_t __corner_rot_indices[] = { 20, 22, 25, 27, 32, 34, 37, 39 };

  //2^4 - 1, largest possible unsigned 4 bit value, and larger than any PDB ever gets for Rubiks
  constexpr uint8_t pdb_initialization_value = 15;

  constexpr uint64_t corner_max_count = 88179840;
  constexpr uint64_t edge_6_max_count = npr(12, 6) * 64;
  constexpr uint64_t edge_8_max_count = npr(12, 8) * 256;
  constexpr uint64_t edge_12_pos_max_count = npr(12, 12);
  constexpr uint64_t edge_20_rot_max_count = 4478976;

  const uint8_t edge_pos_indices_6a[] = { 1, 3, 4, 6, 8, 9 };
  const uint8_t edge_rot_indices_6a[] = { 21, 23, 24, 26, 28, 29 };
  const uint8_t edge_pos_indices_6b[] = { 10, 11, 13, 15, 16, 18 };
  const uint8_t edge_rot_indices_6b[] = { 30, 31, 33, 35, 36, 38 };
  const uint8_t edge_pos_indices_8a[] = { 1, 3, 4, 6, 8, 9, 10, 11 };
  const uint8_t edge_rot_indices_8a[] = { 21, 23, 24, 26, 28, 29, 30, 31 };
  const uint8_t edge_pos_indices_8b[] = { 8, 9, 10, 11, 13, 15, 16, 18 };
  const uint8_t edge_rot_indices_8b[] = { 28, 29, 30, 31, 33, 35, 36, 38 };
  const uint8_t edge_pos_indices_12[] = { 1, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18 };
  const uint8_t edge_rot_indices_12[] = { 21, 23, 24, 26, 28, 29, 30, 31, 33, 35, 36, 38 };

  const uint8_t __cube_translations[] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 4, 8, 5, 9, 10, 6, 11, 7, 12 };
  const uint8_t edges_6a[] = { 0, 1, 2, 3, 4, 5 };
  const uint8_t edges_6b[] = { 6, 7, 8, 9, 10, 11 };
  const uint8_t edges_8a[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const uint8_t edges_8b[] = { 4, 5, 6, 7, 8, 9, 10, 11 };

  constexpr uint32_t base3[] = { 729, 243, 81, 27, 9, 3, 1 };

  const uint8_t __goal[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  extern void rotate(uint8_t* __restrict new_state, const uint8_t face, const uint8_t rotation);

  struct RubiksIndex
  {
    uint8_t state[40];
    uint8_t depth;
    uint8_t last_face;
    uint64_t index;

    RubiksIndex() : state(), depth(0), last_face(0), index(0) {  }
    RubiksIndex(const uint8_t* original_state, const uint8_t depth, const uint8_t last_face) : depth(depth), last_face(last_face), index(UINT64_MAX) {
      memcpy(state, original_state, 40);
    }
    RubiksIndex(const uint8_t* original_state, const uint8_t depth, const uint8_t face, const uint8_t rotation) : state(), depth(depth), last_face(face), index(UINT64_MAX) {
      memcpy(state, original_state, 40);
      rotate(state, face, rotation);
    }
    RubiksIndex(const RubiksIndex& original) : state(), depth(original.depth), last_face(original.last_face), index(original.index) {
      memcpy(state, original.state, 40);
    }
    RubiksIndex(RubiksIndex&& original) noexcept : state(), depth(original.depth), last_face(original.last_face), index(original.index) {
      memcpy(state, original.state, 40);
    }
    RubiksIndex& operator=(const RubiksIndex& rhs) {
      memcpy(state, rhs.state, 40);
      depth = rhs.depth;
      last_face = rhs.last_face;
      index = rhs.index;
      return *this;
    }
  };

  struct PDB_Value {
    uint64_t index;
    uint8_t value;

    PDB_Value() : index(0), value(0) {}
    PDB_Value(uint64_t idx, uint8_t val) noexcept : index(idx), value(val) {}
  };

  struct RubiksEdgeStateHash
  {
    inline uint64_t operator() (const uint8_t* s) const {
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

  inline bool skip_rotations(uint8_t last_face, uint8_t face)
  {
    return last_face == face || (last_face == 3 && face == 0) ||
      (last_face == 5 && face == 2) || (last_face == 4 && face == 1);
  }

  extern uint64_t get_index(const uint8_t* state, const int order, const Rubiks::PDB type);

  extern uint32_t get_corner_index(const uint8_t* state);
  extern void restore_corner(const uint64_t index, uint8_t* state);

  extern uint64_t get_edge_index(const uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices);

  inline uint64_t get_edge_index6a(const uint8_t* state) { return get_edge_index(state, 6, edges_6a, edge_rot_indices_6a); }
  inline uint64_t get_edge_index6b(const uint8_t* state) { return get_edge_index(state, 6, edges_6b, edge_rot_indices_6b); }
  inline uint64_t get_edge_index8a(const uint8_t* state) { return get_edge_index(state, 8, edges_8a, edge_rot_indices_8a); }
  inline uint64_t get_edge_index8b(const uint8_t* state) { return get_edge_index(state, 8, edges_8b, edge_rot_indices_8b); }


  extern uint64_t get_new_edge_pos_index(const uint8_t* state);
  extern void restore_new_edge_pos_index(const uint64_t index, uint8_t* state);
  extern uint64_t get_new_edge_rot_index(const uint8_t* state);
  extern void restore_new_edge_rot_index(const uint64_t index, uint8_t* state);

  extern void restore_state_from_index(const uint64_t hash_index, uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices);
  inline void restore_index6a(const uint64_t index, uint8_t* state) { restore_state_from_index(index, state, 6, edges_6a, edge_rot_indices_6a); }
  inline void restore_index6b(const uint64_t index, uint8_t* state) { restore_state_from_index(index, state, 6, edges_6b, edge_rot_indices_6b); }
  inline void restore_index8a(const uint64_t index, uint8_t* state) { restore_state_from_index(index, state, 8, edges_8a, edge_rot_indices_8a); }
  inline void restore_index8b(const uint64_t index, uint8_t* state) { restore_state_from_index(index, state, 8, edges_8b, edge_rot_indices_8b); }

  extern bool is_solved(const uint8_t* state);
  extern bool is_solved(const uint8_t* cube, const uint8_t* target);
  extern uint8_t pattern_lookup(const uint8_t* state, const uint8_t* start_state, PDB type);
  inline uint8_t pattern_lookup(const uint8_t* state, PDB type)
  {
    return pattern_lookup(state, __goal, type);
  }

  extern void generate_pattern_database(std::string filename, const uint8_t* state, const uint8_t max_depth, const uint64_t max_count, const std::function<uint64_t(const uint8_t* state)> func);
  extern void generate_pattern_database_multithreaded(
    std::string filename,
    const uint8_t* state,
    const uint64_t max_count,
    const std::function<uint64_t(const uint8_t* state)> func,
    const std::function<void(const uint64_t hash, uint8_t* state)> reverse_func
  );
  extern void process_buffer(std::vector<uint8_t>& pattern_lookup, std::atomic_uint64_t& count, const std::vector<uint64_t>& local_results_buffer, const uint8_t id_depth, const uint64_t max_count);

  extern void pdb_expand_nodes(
    moodycamel::ConcurrentQueue<std::pair<uint64_t, uint64_t>>& input_queue,
    std::atomic_uint64_t& count,
    const uint64_t max_count,
    std::mutex& pattern_lookup_mutex,
    std::vector<uint8_t>& pattern_lookup,
    const std::function<uint64_t(const uint8_t* state)> lookup_func,
    const std::function<void(const uint64_t hash, uint8_t* state)> reverse_func,
    const uint8_t id_depth,
    const bool reverse_direction
  );


  extern void set_4byte_value(std::vector<uint8_t>& data, const uint64_t position, const uint8_t value);
  extern uint8_t get_4byte_value(const std::vector<uint8_t>& data, const uint64_t position);
}
