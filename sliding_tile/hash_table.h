#pragma once
#include "sliding_tile.h"
#include <array>
#include <cassert>
#include <random>

class hash_table {
private:
  // hash_values[i][j] =random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
  static inline std::array <std::array<uint32_t, NUM_TILES + 1>, NUM_TILES + 1> hash_values;
public:
  static void initialize_hash_values() {
    int seed = 1;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> gen(0, UINT32_MAX);
    for (int i = 0; i < NUM_TILES; ++i) {
      for (int j = i; j <= NUM_TILES; ++j) {
        hash_values[i][j] = gen(rng);
        hash_values[j][i] = hash_values[i][j];
      }
    }
  }

  static inline uint32_t hash(const uint8_t data[], int i, int j)
  {
    assert(i <= NUM_TILES && j <= NUM_TILES && i >= 0 && j >= 0);
    if(data[i] < data[j])
      return hash_values[data[i]][data[j]];
    else
      return hash_values[data[j]][data[i]];
  }

  static inline uint32_t hash(int i, int j) {
    assert(i <= NUM_TILES && j <= NUM_TILES && i >= 0 && j >= 0);
    return hash_values[i][j];
  }

  static inline uint32_t hash(const uint8_t data[]) {
    uint32_t hash_val = 0;
    for (int i = 0; i < NUM_TILES; ++i) {
      hash_val += hash_values[data[i]][data[i + 1]];
    }
    return hash_val;
  }


//All squares should be the same, except for the empty tile and one adjacent tile
  static bool compare_one_off(const SlidingTile* lhs, const SlidingTile* rhs)
  {
    for(int i = 0; i < NUM_TILES; ++i)
    {
      if(lhs->source[i] != rhs->source[i])
      {
        if(lhs->empty_location != i && rhs->empty_location != i)
        {
          return false;
        }
      }
    }
    return true;
  };

  static inline uint32_t hash_configuration(const uint8_t tile_in_location[])
  /*
     1. This routine computes the hash value of a configuration.
     2. It uses a simple modular hash function.
     3. tile_in_location[i] = the tile that is in location i.
        The elements of tile_in_location are stored beginning in tile_in_location[0].
     4. hash_value[t][i] = random hash value associated with tile t assigned to location i: U[0,HASH_SIZE).
     5. The hash value is returned.
     6. Written 11/20/12.
  */
  {
    uint64_t  i, index, t;

    index = 0;
    for(i = 0; i < NUM_TILES; i++)
    {
      t = tile_in_location[i];
      assert((0 <= t) && (t < NUM_TILES));
      index = (index + hash(t, i)) % UINT32_MAX;
    }
    return(index);
  }

  static inline uint32_t update_hash_value(const uint8_t tile_in_location[NUM_TILES], uint8_t empty_location, uint8_t new_location, uint64_t index)
  /*
     1. This routine updates the hash value of a configuration when the empty tile is moved from empty_location to new_location.
     2. tile_in_location[i] = the tile that is in location i.
        The elements of tile_in_location are stored beginning in tile_in_location[0].
     3. empty_location = location of the empty tile.
     4. new_location = new location of the empty tile.
     5. index = the hash value corresponding to the configuration represented by tile_in_location prior to the move.
     4. hash_value[t][i] = random hash value associated with tile t assigned to location i: U[0,HASH_SIZE).
     5. The hash value is returned.
     6. Written 11/20/12.
  */
  {
    int      tile;

    assert((0 <= empty_location) && (empty_location < NUM_TILES));
    assert((0 <= new_location) && (new_location < NUM_TILES));
    assert(tile_in_location[empty_location] == 0);
    tile = tile_in_location[new_location];       assert((0 <= tile) && (tile < NUM_TILES));

    if(index > hash(0, empty_location))
      index = index - hash(0, empty_location);
    else
      index = index - hash(0, empty_location) + UINT32_MAX;

    if(index > hash(tile, new_location))
      index = index - hash(tile, new_location);
    else
      index = index - hash(tile, new_location) + UINT32_MAX;
    
    index = (index + hash(0, new_location)) % UINT32_MAX;
    index = (index + hash(tile, empty_location)) % UINT32_MAX;

    return(index);
  }

};

