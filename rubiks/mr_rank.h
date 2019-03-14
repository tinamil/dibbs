/*
https://rosettacode.org/wiki/Permutations/Rank_of_a_permutation
*/

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

namespace mr
{
  void _mr_unrank1(uint64_t rank, int n, uint8_t *vec);
  uint64_t _mr_rank1(int n, uint8_t *vec, uint8_t *inv);
  void get_permutation(uint64_t rank, int n, uint8_t *vec);
  uint64_t get_rank(int n, uint8_t *vec);
  uint64_t k_rank(int8_t *locs, uint8_t *dual, unsigned int distinctSize, unsigned int puzzleSize);
  template <typename T>
  inline void swap(T &a, T &b)
  {
    T tmp = a;
    a = b;
    b = tmp;
  }
}
