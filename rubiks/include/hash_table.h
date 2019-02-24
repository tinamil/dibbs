#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include "node.h"

#define  HASH_SIZE 400000043  // Must be a prime number.  Currently using linear
//#define  HASH_SIZE 400000000  // Must be a prime number.  Currently using linear
// probing, but if quadratic probing is used, then
// it must be a prime of the form 4j+3.  4000039=4*1000009+3  c=4*10000000+3
// 400000043=4*100000010+3   1000000007 = 4*250000001+3

int randomi (int n, double *dseed);

// The following classes and functions implement a min-max heap, which can be used for implementing a
// distributed best bound first search.
// See Fundamentals of Data Structure by Horowitz and Sahni and Algorithms in
// C++ by Sedgwick for explanations of hashing.
/*
   1. HASH_SIZE = the length of the hash table.  It must be a prime number.
      Currently using linear probing, but if quadratic probing is used, then
      it must be a prime of the form 4j+3.  1019=4*254+3   4000039=4*1000009+3
   2. hash_record does not store a key.  Instead it stores state_index, which is the index (in states) of the corresponding state.
   3. The state_index of every entry in the hash table is initialized to -1.
      If -1 is a valid state_index, then some other value must be used.
   4. n = # of entries stored in the hash table.
   5. This implements a very simple hash table where the hash record contains only a state_index.
   6. Created 10/3/12 by modifying c:\sewell\research\statmatch\boss_cpp\hash_table.cpp, which was based on the hash tables in:
      a. c:\sewell\research\cdc\phase2\search06\search06.cpp (which was written as classes).
      b. c:\sewell\research\schedule\rtardy\bbr\memoroy.cpp.
*/

/*************************************************************************************************/

class hash_record
{
public:
  int      state_index;
  hash_record()
  {
    state_index = -1; // This assumes all legitimate state indices are >= 0
  }
  void  clear()
  {
    state_index = -1;
  }
};

/*************************************************************************************************/

class Hash_table                 // Note: I am capitalizing this class because the compiler will not let me declare a variable with the exact same spelling.
{
public:
  Hash_table()
  {
    n = 0;
    table = NULL;
    hash_value = NULL;
  }
  ~Hash_table()
  {
    delete [] table;
    for (int i = 0; i < N_LOCATIONS; i++)
      delete [] hash_value[i];
    delete [] hash_value;
  }
  void  initialize()
  {
    table = new hash_record[HASH_SIZE];
    if (table == NULL)
    {
      fprintf (stderr, "Out of space for hash table\n");
      exit (1);
    }
    n = 0;
    hash_value = new int*[N_LOCATIONS];
    if (hash_value == NULL)
    {
      fprintf (stderr, "Out of space for hash_value\n");
      exit (1);
    }
    for (int i = 0; i < N_LOCATIONS; i++)
    {
      hash_value[i] = new int[N_LOCATIONS];
      if (hash_value[i] == NULL)
      {
        fprintf (stderr, "Out of space for hash_value\n");
        exit (1);
      }
    }
    initialize_hash_value();
    //print_hash_values();
  };
  hash_record  operator[] (int i) const
  {
    assert ( (0 <= i) && (i < HASH_SIZE) );
    return table[i];
  }
  int   size()
  {
    return n;
  }
  int   table_size()
  {
    return HASH_SIZE;
  }
  void  clear()
  {
    for (int i = 0; i < HASH_SIZE; i++)
    {
      table[i].clear();
    }
    n = 0;
  }
  int   hash_configuration (unsigned char  *tile_in_location);
  int   update_hash_value (unsigned char *tile_in_location, unsigned char empty_location, unsigned char new_location, int index);
  void  initialize_hash_value()
  {
    double   seed;
    seed = 3.1567;
    for (int t = 0; t < N_LOCATIONS; t++)
    {
      for (int i = 0; i < N_LOCATIONS; i++)
      {
        hash_value[t][i] = randomi (HASH_SIZE, &seed);
      }
    }
  }
  int   insert (int state_index, unsigned char *tile_in_location, int hash_index, states_array *states);
  int   find (unsigned char *tile_in_location, int hash_value, int *hash_index, states_array *states);
  int   insert_at_index (int state_index, int hash_index);
  int   replace_at_index (int state_index, int hash_index);
  //void  print();
  void  print_hash_values();
  int   find_bistate (unsigned char *tile_in_location, int hash_value, int *hash_index, bistates_array *bistates);
  int   insert_bistate (int state_index, unsigned char *tile_in_location, int hash_index, bistates_array *bistates);
private:
  int            n;
  int            **hash_value;     // hash_value[t][i] = random hash value associated with tile t assigned to location i: U[0,HASH_SIZE).
  //hash_record    table[HASH_SIZE];
  hash_record    *table;           // Use dynamic allocation because the compiler does not permit a declared array with more than 2^31 bytes.
};
typedef Hash_table* Hash_tablepnt;
