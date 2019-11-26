#pragma once

#include <cstdint>
#include <cassert>
#include <cstring>
#include <cmath>
#include "hash.hpp"

constexpr int NUM_TILES = 16;
constexpr size_t MEM_LIMIT = 100ui64 * 1024 * 1024 * 1024; //100GB

enum class Direction { forward, backward };
inline Direction OppositeDirection(Direction d) {
  if (d == Direction::forward) return Direction::backward;
  else return Direction::forward;
}

class SlidingTile {

private:
  static uint8_t distances[NUM_TILES][NUM_TILES];            // distances[i][j] = Manhattan distance between location i and location j.
  static uint8_t moves[NUM_TILES][5];                // moves[i] = list of possible ways to move the empty tile from location i.
  static uint8_t starting[NUM_TILES];
  static uint8_t*& DUAL_SOURCE() { static uint8_t* I = nullptr; return I; };
  uint8_t empty_location;

public:
  Direction dir;
#ifdef HISTORY
  std::vector<uint8_t> actions;
#endif
  uint8_t source[NUM_TILES];
  uint8_t g;
  uint8_t h;
  uint8_t h2;
  uint8_t f;
  uint8_t f_bar;
  bool threshold;

  uint8_t compute_manhattan() {
    if (dir == Direction::forward) {
      return compute_manhattan_forward();
    }
    else {
      return compute_manhattan_backward();
    }
  }

  uint8_t compute_manhattan_opposite() {
    if (dir == Direction::forward) {
      return compute_manhattan_backward();
    }
    else {
      return compute_manhattan_forward();
    }
  }

  uint8_t compute_manhattan_forward()
  {
    uint8_t LB = 0;
    for (int i = 0; i < NUM_TILES; ++i) {
      if (source[i] != 0) {
        LB += distances[i][source[i]];
      }
    }
    return LB;
  }

  //_________________________________________________________________________________________________

  uint8_t compute_manhattan_backward()
  {
    uint8_t location_of_tile1[NUM_TILES];
    uint8_t location_of_tile2[NUM_TILES];
    for (int i = 0; i < NUM_TILES; i++) {
      location_of_tile1[source[i]] = i;
      location_of_tile2[starting[i]] = i;
    }

    uint8_t LB = 0;
    for (int i = 0; i < NUM_TILES; i++) {
      LB += distances[location_of_tile1[i]][location_of_tile2[i]];
    }

    return LB;
  }

  /*************************************************************************************************/

  static void initialize(uint8_t initial_state[])
  {
    int            cnt, d, i_row, i_col, j, j_row, j_col;

    memcpy(starting, initial_state, NUM_TILES);

    if (DUAL_SOURCE() == nullptr) DUAL_SOURCE() = new uint8_t[NUM_TILES];
    for (int i = 0; i < NUM_TILES; ++i) DUAL_SOURCE()[initial_state[i]] = i;

    uint8_t n_rows = (uint8_t)sqrt(NUM_TILES);
    uint8_t n_cols = n_rows;
    assert(n_cols * n_rows == NUM_TILES);

    for (int i = 0; i < NUM_TILES; ++i) {
      i_col = 1 + (i % n_rows);
      cnt = 0;
      // Up.
      j = i - n_cols;
      if ((0 <= j) && (j < NUM_TILES)) moves[i][++cnt] = j;
      // Left
      if (i_col > 1) {
        j = i - 1;
      }
      else {
        j = -1;
      }
      if ((0 <= j) && (j < NUM_TILES)) moves[i][++cnt] = j;
      // Right.
      if (i_col < n_cols) {
        j = i + 1;
      }
      else {
        j = -1;
      }
      if ((0 <= j) && (j < NUM_TILES)) moves[i][++cnt] = j;
      // Down.
      j = i + n_cols;
      if ((0 <= j) && (j < NUM_TILES)) moves[i][++cnt] = j;
      moves[i][0] = cnt;
    }

    // Initializes distances.
    // distances[i][j] = Manhattan distance between location i and location j.

    for (int i = 0; i < NUM_TILES; ++i) {
      i_row = 1 + (i / n_cols);
      i_col = 1 + (i % n_rows);
      for (j = i; j < NUM_TILES; j++) {
        j_row = 1 + (j / n_cols);
        j_col = 1 + (j % n_rows);
        distances[i][j] = abs(i_row - j_row) + abs(i_col - j_col);
        distances[j][i] = distances[i][j];
      }
    }
  }

  SlidingTile(const uint8_t* data, Direction dir) : dir(dir), g(0), h(0), h2(0), f(0), f_bar(0)
  {
    memcpy(source, data, NUM_TILES);
    h = compute_manhattan();
    f = h;
    f_bar = f;
    threshold = h == 0;

    // Find the location of the empty tile.
    for (int i = 0; i < NUM_TILES; i++) {
      if (source[i] == 0) {
        empty_location = i;
        break;
      }
    }
  }

  SlidingTile(const SlidingTile& copy) : dir(copy.dir), g(copy.g), h(copy.h), h2(copy.h2), f(copy.f), f_bar(copy.f_bar), threshold(copy.threshold), empty_location(copy.empty_location)
#ifdef HISTORY
    , actions(copy.actions)
#endif
  {
    memcpy(source, copy.source, NUM_TILES);
  }

  inline static SlidingTile GetSolvedPuzzle(Direction dir) {
    uint8_t puzzle[NUM_TILES];
    for (int i = 0; i < NUM_TILES; ++i) {
      puzzle[i] = i;
    }
    return SlidingTile(puzzle, dir);
  }

  inline bool operator==(const SlidingTile& right) const {
    return memcmp(source, right.source, NUM_TILES) == 0;
  }

  void apply_move(int move) {
    uint8_t new_location = moves[empty_location][move];
    uint8_t tile = source[new_location];
    source[empty_location] = tile;
    source[new_location] = 0;

    h = h + distances[empty_location][tile] - distances[new_location][tile];
    //assert(h == compute_manhattan());

    h2 = h2 + distances[empty_location][DUAL_SOURCE()[tile]] - distances[new_location][DUAL_SOURCE()[tile]];
    //assert(h2 == compute_manhattan_opposite());
  }

  size_t num_actions_available() {
    return moves[empty_location][0];
  }

  SlidingTile apply_action(int i) const {
    SlidingTile new_node(*this);
#ifdef HISTORY
    new_node.actions.push_back(i);
#endif
    new_node.apply_move(i);
    new_node.g = g + 1;
    new_node.f = new_node.g + new_node.h;
    new_node.f_bar = 2 * new_node.g + new_node.h - new_node.h2;
    new_node.threshold = threshold || new_node.h <= new_node.h2;
    assert(new_node.f >= f); //Consistency check
    return new_node;
  }
};

//Returns smallest f value with largest g value
struct FSort {
  bool operator()(const SlidingTile& lhs, const SlidingTile& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const SlidingTile* lhs, const SlidingTile* rhs) const {
    if (lhs->f == rhs->f) {
      return lhs->g < rhs->g;
    }
    return lhs->f > rhs->f;
  }
};

//Returns smallest f value with smallest g value
struct FSortLowG {

  bool operator()(const SlidingTile& lhs, const SlidingTile& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const SlidingTile* lhs, const SlidingTile* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_TILES);
    if (cmp == 0) {
      return false;
    }
    else if (lhs->f == rhs->f) {
      if (lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else {
      return lhs->f < rhs->f;
    }
  }
};

//Returns smallest g value
struct GSortLow {

  bool operator()(const SlidingTile& lhs, const SlidingTile& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const SlidingTile* lhs, const SlidingTile* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_TILES);
    if (cmp == 0) {
      return false;
    }
    if (lhs->g == rhs->g)
      return cmp < 0;
    else
      return lhs->g < rhs->g;
  }
};

struct GSortHigh {

  bool operator()(const SlidingTile& lhs, const SlidingTile& rhs) const {
    return operator()(&lhs, &rhs);
  }

  bool operator()(const SlidingTile* lhs, const SlidingTile* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_TILES);
    if (cmp == 0) {
      return false;
    }
    if (lhs->g == rhs->g)
      return cmp < 0;
    else
      return lhs->g > rhs->g;
  }
};

//Returns smallest fbar with smallest g value
struct FBarSortLowG {
  bool operator()(const SlidingTile& lhs, const SlidingTile& rhs) const {
    return operator()(&lhs, &rhs);
  }
  bool operator()(const SlidingTile* lhs, const SlidingTile* rhs) const {
    int cmp = memcmp(lhs->source, rhs->source, NUM_TILES);
    if (cmp == 0) {
      return false;
    }
    else if (lhs->f_bar == rhs->f_bar) {
      if (lhs->g == rhs->g)
        return cmp < 0;
      else
        return lhs->g < rhs->g;
    }
    else {
      return lhs->f_bar < rhs->f_bar;
    }
  }
};

struct SlidingTileHash
{
  inline std::size_t operator() (const SlidingTile& x) const
  {
    return operator()(&x);
  }
  inline std::size_t operator() (const SlidingTile* x) const
  {
    return SuperFastHash(x->source, NUM_TILES);
  }
};

struct SlidingTileEqual
{
  inline bool operator() (const SlidingTile* x, const SlidingTile* y) const
  {
    return *x == *y;
  }
  inline bool operator() (const SlidingTile x, const SlidingTile y) const
  {
    return x == y;
  }
};
