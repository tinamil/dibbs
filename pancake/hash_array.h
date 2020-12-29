#pragma once

#include "Pancake.h"
#include <cstdint>

struct hash_array
{
  uint32_t hash[NUM_INTS_PER_PANCAKE];

  void clear_hash()
  {
    for(int i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      hash[i] = 0;
    }
  }

  void set_hash(uint32_t hash_val)
  {
    uint32_t which_int = hash_val / 32u;
    uint32_t hash_bit = 1u << (hash_val % 32u);
    assert((hash_bit & hash[which_int]) == 0);
    hash[which_int] ^= hash_bit;
  }

  void unset_hash(uint32_t hash_val)
  {
    uint32_t which_int = hash_val / 32u;
    uint32_t hash_bit = 1u << (hash_val % 32u);
    assert((hash_bit & hash[which_int]) > 0);
    hash[which_int] ^= hash_bit;
  }

  uint32_t count()
  {
    uint32_t tmp = 0;
    for(int i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      tmp += __popcnt(hash[i]);
    }
    return tmp;
  }
};