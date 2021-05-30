#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <intrin.h>

class int_bitset
{
private:
  const uint32_t num_bits;
  const uint32_t num_ints;
  std::vector<uint32_t> data;

public:
  //Instantiates a bitset with all zeros
  int_bitset(int number_bits) : num_bits(number_bits), num_ints((num_bits + 31) / 32)
  {
    //num_ints is initialized with an integer division that rounds up
    data.resize(num_ints);
    clear();
  }

  //Copies a bitset
  int_bitset(const int_bitset& copy_from) : num_bits(copy_from.num_bits), num_ints(copy_from.num_ints)
  {
    data.resize(num_ints);
    for(int i = 0; i < num_ints; ++i) {
      data[i] = copy_from.data[i];
    }
  }

  // Returns true if both bitsets have identical size and bits
  bool equals(const int_bitset& other) const
  {
    if(num_bits != other.num_bits) return false;
    for(int i = 0; i < num_ints; ++i) {
      if(data[i] != other.data[i]) return false;
    }
    return true;
  }

  // Sets all bits to zero
  void clear()
  {
    for(int i = 0; i < data.size(); ++i) {
      data[i] = 0;
    }
  }

  // Flips the specified bit (0 => 1 and 1 => 0)
  void flip_bit(const uint32_t which_bit)
  {
    assert(which_bit < num_bits);
    uint32_t which_int = which_bit / 32u;
    uint32_t which_bit32 = 1u << (which_bit % 32u);
    data[which_int] ^= which_bit32;
  }

  // Sets the specified bit to 1
  void set_bit(const uint32_t which_bit)
  {
    assert(which_bit < num_bits);
    uint32_t which_int = which_bit / 32u;
    uint32_t which_bit32 = 1u << (which_bit % 32u);
    data[which_int] |= which_bit32;
  }

  // Sets the specified bit to 0
  void unset_bit(const uint32_t which_bit)
  {
    assert(which_bit < num_bits);
    uint32_t which_int = which_bit / 32u;
    uint32_t which_bit32 = 1u << (which_bit % 32u);
    data[which_int] &= ~which_bit32;
  }

  // Returns the state of the specified bit
  bool check_bit(const uint32_t which_bit) const
  {
    assert(which_bit < num_bits);
    uint32_t which_int = which_bit / 32u;
    uint32_t which_bit32 = 1u << (which_bit % 32u);
    return((which_bit32 & data[which_int]) > 0);
  }

  // Returns the number of bits with value of 1
  uint32_t count_ones() const
  {
    uint32_t tmp = 0;
    for(int i = 0; i < num_ints; ++i) {
      tmp += __popcnt(data[i]);
    }
    return tmp;
  }

  // Returns the number of bits thare are set to 1 in both bitsets.  
  // If the bitsets are different sizes, then the smaller bitset 
  // should call this function and pass in the larger bitset.
  uint32_t count_intersection(const int_bitset& other_bitset) const
  {
    assert(num_bits <= other_bitset.num_bits);
    uint32_t tmp = 0;
    for(int i = 0; i < num_ints; ++i) {
      tmp += __popcnt(data[i] & other_bitset.data[i]);
    }
    return tmp;
  }
};