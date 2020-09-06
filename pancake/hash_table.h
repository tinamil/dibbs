#pragma once
#include <array>
#include "Pancake.h"
#include <random>

typedef uint32_t hash_t;
constexpr size_t HASH_MAX = UINT32_MAX;
constexpr bool RANDOM = false;

class hash_table
{
private:
  // hash_values[i][j] =random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
  static inline hash_t hash_values[NUM_PANCAKES + 2][NUM_PANCAKES + 2];
public:
  static void initialize_hash_values()
  {
    int seed = 1;
    if constexpr(RANDOM)
    {
      std::mt19937 rng(seed);
      std::uniform_int_distribution<uint64_t> gen(0, HASH_MAX);
      for(int i = 0; i <= NUM_PANCAKES; ++i)
      {
        for(int j = i + 1; j <= NUM_PANCAKES + 1; ++j)
        {
          hash_values[i][j] = gen(rng);
          hash_values[j][i] = hash_values[i][j];
        }
      }
    }
    else
    {
      int count = 0;
      for(int i = 1; i <= NUM_PANCAKES; ++i)
      {
        for(int j = i + 1; j <= NUM_PANCAKES + 1; ++j)
        {
          hash_values[i][j] = count++;
          hash_values[j][i] = hash_values[i][j];
        }
      }
      assert(count == ((NUM_PANCAKES) * (NUM_PANCAKES + 1) / 2));
    }
  }

  static inline hash_t hash(int i, int j)
  {
    assert(i <= NUM_PANCAKES && j <= NUM_PANCAKES + 1 && i >= 0 && j >= 0);
    return hash_values[i][j];
  }

  static inline uint64_t compress(hash_t hash_values[NUM_PANCAKES + 1])
  {
    uint64_t compressed(0);

    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      uint8_t offset = 0;
      while(compressed & (1ui64 << (hash_values[i] % 64) + offset))
      {
        offset += 1;
      }
      compressed |= 1ui64 << ((hash_values[i] % 64) + offset);
    }

    return compressed;
  }

  static inline hash_t hash(const uint8_t data[])
  {
    hash_t hash_val = 0;
    for(int i = 1; i <= NUM_PANCAKES; ++i)
    {
      hash_val += hash_values[data[i]][data[i + 1]];
    }
    return hash_val;
  }
};

