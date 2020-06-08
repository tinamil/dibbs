#pragma once
#include <array>
#include "Pancake.h"
#include <random>

class hash_table {
private:
  // hash_values[i][j] =random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
  static inline std::array <std::array<size_t, NUM_PANCAKES + 1>, NUM_PANCAKES + 1> hash_values;
public:
  static void initialize_hash_values() {
    int seed = 1;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> gen(0, SIZE_MAX);
    for (int i = 1; i < NUM_PANCAKES; ++i) {
      for (int j = i + 1; j <= NUM_PANCAKES; ++j) {
        hash_values[i][j] = gen(rng);
        hash_values[j][i] = hash_values[i][j];
      }
    }
  }

  static inline size_t hash(const uint8_t data[], int i, int j)
  {
    assert(i <= NUM_PANCAKES && j <= NUM_PANCAKES && i >= 0 && j >= 0);
    if(data[i] < data[j])
      return hash_values[data[i]][data[j]];
    else
      return hash_values[data[j]][data[i]];
  }

  static inline size_t hash(int i, int j) {
    assert(i <= NUM_PANCAKES && j <= NUM_PANCAKES && i >= 0 && j >= 0);
    return hash_values[i][j];
  }

  static inline size_t hash(const uint8_t data[]) {
    size_t hash_val = 0;
    for (int i = 1; i < NUM_PANCAKES; ++i) {
      hash_val += hash_values[data[i]][data[i + 1]];
    }
    return hash_val;
  }
};

