#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define _MR_SWAP(a,b) do{t=(a);(a)=(b);(b)=t;}while(0)
namespace mr
{
void _mr_unrank1(uint64_t rank, int n, uint8_t *vec);
uint64_t _mr_rank1(int n, uint8_t *vec, uint8_t *inv);
void get_permutation(uint64_t rank, int n, uint8_t *vec);
uint64_t get_rank(int n, uint8_t *vec);
}
