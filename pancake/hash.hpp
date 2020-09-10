#pragma once
#include <array>
#include <random>

/*
http://www.azillionmonkeys.com/qed/hash.html
Licensed under LGPL-2.1
*/
#include <stdint.h>
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif


//https://www.boost.org/doc/libs/1_69_0/boost/container_hash/hash.hpp


inline void hash_combine_impl(uint64_t& h, uint64_t k)
{
  const uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
  const int r = 47;

  k *= m;
  k ^= k >> r;
  k *= m;

  h ^= k;
  h *= m;

  // Completely arbitrary number, to prevent 0's
  // from hashing to 0.
  h += 0xe6546b64;
}

uint64_t inline boost_hash(const uint8_t* data, const int64_t len) {
  uint64_t hash = 0;
  if (len >= 0) {
    for (size_t i = 0; i < len; ++i) {
      hash_combine_impl(hash, data[i]);
    }
  }
  else {
    for (int64_t i = len; i >= 0; --i) {
      hash_combine_impl(hash, data[i]);
    }
  }
  return hash;
}

uint32_t inline SuperFastHash(const unsigned char* data, int len) {
  uint32_t hash = len, tmp;
  int rem;

  if (len <= 0 || data == NULL) return 0;

  rem = len & 3;
  len >>= 2;

  /* Main loop */
  for (; len > 0; len--) {
    hash += get16bits(data);
    tmp = (get16bits(data + 2) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    data += 2 * sizeof(uint16_t);
    hash += hash >> 11;
  }

  /* Handle end cases */
  switch (rem) {
  case 3: hash += get16bits(data);
    hash ^= hash << 16;
    hash ^= ((signed char)data[sizeof(uint16_t)]) << 18;
    hash += hash >> 11;
    break;
  case 2: hash += get16bits(data);
    hash ^= hash << 11;
    hash += hash >> 17;
    break;
  case 1: hash += (signed char)*data;
    hash ^= hash << 10;
    hash += hash >> 1;
  }

  /* Force "avalanching" of final 127 bits */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;
}