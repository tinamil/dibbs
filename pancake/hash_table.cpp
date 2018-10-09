
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include "hash_table.h"

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
   6. Created 7/21/17 by modifying c:\sewell\research\15puzzle\15puzzle_code2\hash_table.cpp, which was based on the hash tables in:
      a. c:\sewell\research\cdc\phase2\search06\search06.cpp (which was written as classes).
      b. c:\sewell\research\schedule\rtardy\bbr\memoroy.cpp.
      c. c:\sewell\research\statmatch\boss_cpp\hash_table.cpp
*/

/*************************************************************************************************/

//_________________________________________________________________________________________________

int Hash_table::hash_seq(unsigned char *seq)
/*
   1. This routine computes the hash value of a sequence.
   2. It uses a simple modular hash function.
   3. seq[i] = the number of the pancake that is in position i.
      The elements of seq are stored beginning in seq[1].
   4. hash_values[i][j] = random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
   5. The hash value is returned.
   6. Written 11/20/12.
   7. Created 7/21/17 by modifying find_configuration in c:\sewell\research\15puzzle\15puzzle_code2\hash_table.cpp.
*/
{
   int            i, index;
   unsigned char  pi, pi1;

   index = 0;
   for(i = 1; i < n_pancakes; i++) {
      pi = seq[i];
      pi1 = seq[i + 1];
      assert((1 <= pi) && (pi <= n_pancakes));
      index = (index + hash_values[pi][pi1]) % HASH_SIZE;
   }
   return(index);
}

//_________________________________________________________________________________________________

int Hash_table::update_hash_value(unsigned char *seq, int i, int hash_value)
/*
   1. This routine updates the hash value of a sequence when a flip is made at position i.
   2. seq[i] = the number of the pancake that is in position i.
      The elements of seq are stored beginning in seq[1].
   3. i = position where the flip is to be made.  I.e., the new sequence is obtained by reverse_vector(1, i, n_seq, seq).
   4. index = the hash value corresponding to seq before the flip.
   5. hash_values[i][j] = random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
   6. The hash value is returned.
   7. Written 11/20/12.
   8. Created 7/21/17 by modifying update_hash_value in c:\sewell\research\15puzzle\15puzzle_code2\hash_table.cpp.
*/
{
   unsigned char  p1, pi, pi1;

   assert((1 <= i) && (i <= n_pancakes));
   p1 = seq[1];               assert((1 <= p1) && (p1 <= n_pancakes));
   pi = seq[i];               assert((1 <= pi) && (pi <= n_pancakes));
   if(i < n_pancakes)
      pi1 = seq[i + 1];
   else
      return(hash_value);     // The hash_value does not change when the entire sequence is flipped.

   if(hash_value > hash_values[pi][pi1])
      hash_value = hash_value - hash_values[pi][pi1];
   else
      hash_value = hash_value - hash_values[pi][pi1] + HASH_SIZE;
   hash_value = (hash_value + hash_values[p1][pi1]) % HASH_SIZE;

   return(hash_value);
}

//_________________________________________________________________________________________________

int Hash_table::insert_at_index(int state_index, int hash_index)
/*
   1. This routine inserts an entry into the hash table at a particular index.
      It is designed to be used in a situation where the correct index in the hash table has already
      been found (by the find function).  This avoids a second search through the hash table to
      find the correct location.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. -1 is returned if the hash table is full,
      -2 is returned if the hash table already has an entry in the given index,
      o.w. 1 is returned.
   4. Written 10/5/12.
*/
{
   if (n >= HASH_SIZE) {
      fprintf(stderr, "Hash table is full\n");
      exit(1);
   }

   assert((0 <= hash_index) && (hash_index < HASH_SIZE));
   if (table[hash_index].state_index != -1) {
      fprintf(stderr, "The hash table already contains an entry in this location\n");
      return(-2);
   }

   table[hash_index].state_index = state_index;
   n++;

   return(1);
}

//_________________________________________________________________________________________________

int Hash_table::replace_at_index(int state_index, int hash_index)
/*
   1. This routine replaces an entry into the hash table at a particular index.
      It is designed to be used in a situation where the correct index in the hash table has already
      been found (by the find function).  This avoids a second search through the hash table to
      find the correct location.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. The program is aborted if the hash table is full,
      -2 is returned if there is no entry in the hash table in the given index,
      o.w. 1 is returned.
   4. Written 12/14/12.
*/
{
   if (n >= HASH_SIZE) {
      fprintf(stderr, "Hash table is full\n");
      exit(1);
   }

   assert((0 <= hash_index) && (hash_index < HASH_SIZE));
   if (table[hash_index].state_index == -1) {
      fprintf(stderr, "The hash table does not contain an entry in this location\n");
      return(-2);
   }

   table[hash_index].state_index = state_index;

   return(1);
}

//_________________________________________________________________________________________________

void Hash_table::print_hash_values()
{
   printf("\n");
   for(int i = 1; i <= n_pancakes; i++) {
      for(int j = 1; j <= n_pancakes; j++) {
         printf("%9d ", hash_values[i][j]);
      }
      printf("\n");
   }
   printf("\n");
}

//_________________________________________________________________________________________________

int Hash_table::find_bistate(unsigned char *seq, int hash_value, int *hash_index, bistates_array *bistates)
/*
   1. This routine searches the hash table for a subproblem whose tile_in_location is the same as the input arguement.
   2. It uses linear probing.
   3. If the subproblem is found, the index where it is stored in the table is returned, 
      o.w., -1 is returned.
   4. hash_value = the index in the hash table to begin searching for the subprolem.
   5. Created 10/4/12 by modifying find_key in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
   6. Modified 11/21/12 to return a status and the hash index.
      a. Status
         1 = the subproblem was found,
        -1 = the subproblem was not found.
      b. The index (in the hash table) is returned in hash_index.
         hash_index = the index where the subproblem is, if it was found.
                    = the index where the subproblem should be added to the hash table, if it was not found.
   7. Created 2/17/15 by modifying find.
   8. Created 7/21/17 by modifying find_bistate in c:\sewell\research\15puzzle\15puzzle_code2\hash_table.cpp.
*/
{
   int      index, state_index;

   if (n >= HASH_SIZE) {
      fprintf(stderr, "Hash table is full\n");
      exit(1);
   }

   assert((0 <= hash_value) && (hash_value < HASH_SIZE));
   index = hash_value;

   while ((state_index = table[index].state_index) != -1) {
      if (memcmp((*bistates)[state_index].seq, seq, n_pancakes + 1) == 0) {
         *hash_index = index;
         return(1);
      } else {
         index = (index + 1) % HASH_SIZE;
      }
   }
   *hash_index = index;
   return(-1);
}

//_________________________________________________________________________________________________

int Hash_table::insert_bistate(int state_index, unsigned char *seq, int hash_index, bistates_array *bistates)
/*
   1. This routine uses the linear probing method to insert an entry into the hash table.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. -1 is returned if the hash table is full,
      -2 is returned if the key is already in the hash table,
      o.w. 1 is returned.
   4. Created 10/5/12 by modifying insert in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
   5. Created 2/17/15 by modifying insert.
   6. Created 7/21/17 by modifying insert_bistate in c:\sewell\research\15puzzle\15puzzle_code2\hash_table.cpp.
*/
{
   int      index;

   if (n >= HASH_SIZE) {
      fprintf(stderr, "Hash table is full\n");
      exit(1);
   }

   assert((0 <= hash_index) && (hash_index < HASH_SIZE));
   index = hash_index;

   while ((state_index = table[index].state_index) != -1) {
      if (memcmp((*bistates)[state_index].seq, seq, n_pancakes + 1) == 0) {
         fprintf(stderr, "Already in the hash table\n");
         return(-2);
      } else {
         index = (index + 1) % HASH_SIZE;
      }
   }

   table[index].state_index = state_index;
   n++;

   return(1);
}
