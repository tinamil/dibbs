#ifndef _bistates_
#define _bistates_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
//#include "heap_record.h"
#include <cstring>

#define  N_LOCATIONS 16             // Number of locations

/*
   The following class and functions implement an array that can be used to store states in a bidirectional search algorithm.
   1. The array supports the following operations in constant time:
      a. Insert a state into the array.
      b. Replace a state in the array, given the index of the state to be replaced.
   2. The array is a simple array of states.
      a. last = the index in states where the last state is stored.
   3. A states_array consists of the following.
      a. last = the index in states where the last state is stored.
      b. max_size = maximum number of states that can be stored in the array.
      c. n_stored = total number of states that have been stored, including replacements.
      c. states: The states are stored in states[0], ..., states[last].
   4. Created 2/17/15 by modifying c:\sewell\research\15puzzle\15puzzle_code\states.h
*/

/*************************************************************************************************/

class bistate {
public:
   void  fill(const unsigned char g1i, const unsigned char h1i, const unsigned char open1i, const unsigned char empty_locationi, const unsigned char prev_location1i, const int parent1i,
              const unsigned char g2i, const unsigned char h2i, const unsigned char open2i, const unsigned char prev_location2i, const int parent2i,
              const int hash_valuei, const unsigned char *tile_in_locationi, const int n_tiles )
   {
      g1 = g1i;
      h1 = h1i;
      open1 = open1i;
      empty_location = empty_locationi;
      prev_location1 = prev_location1i;
      parent1 = parent1i;
      g2 = g2i;
      h2 = h2i;
      open2 = open2i;
      prev_location2 = prev_location2i;
      parent2 = parent2i;
      hash_value = hash_valuei;
      memcpy(tile_in_location, tile_in_locationi, n_tiles + 1);
   };

   unsigned char  g1;                  // = number of moves that have been made so far in the forward direction
   unsigned char  h1;                  // = lower bound on the number of moves needed to reach the goal postion
   unsigned char  open1;               // = 2 if this subproblem has not yet been generated in the forward direction
                                       // = 1 if this subproblem is open in the forward direction
                                       // = 0 if this subproblem closed in the forward direction
   unsigned char  empty_location;      // = location of the empty tile
   unsigned char  prev_location1;      // = location of the empty tile in the parent of this subproblem in the forward direction
   int            parent1;             // = index of the parent subproblem in the forward direction
   unsigned char  g2;                  // = number of moves that have been made so far in the reverse direction
   unsigned char  h2;                  // = lower bound on the number of moves needed to reach the source postion
   unsigned char  open2;               // = 2 if this subproblem has not yet been generated in the reverse direction
                                       // = 1 if this subproblem is open in the reverse direction
                                       // = 0 if this subproblem closed in the reverse direction
   unsigned char  prev_location2;      // = location of the empty tile in the parent of this subproblem in the reverse direction
   int            parent2;             // = index of the parent subproblem in the reverse direction
   int            hash_value;          // = the index in the hash table to begin searching for the state.
   unsigned char  tile_in_location[N_LOCATIONS];   // tile_in_location[i] = the tile that is in location i
};

/*************************************************************************************************/

class bistates_array {
public:
   bistates_array()  {  last = -1; max_size = 0; n_stored = 0; bistates = NULL;}
   bistates_array(const bistates_array&); // copy constructor
   ~bistates_array() { delete [] bistates;}
   void  initialize(const int maximum_size)  {  assert(maximum_size > 0);
                                                bistates = new bistate[maximum_size + 1];
                                                if(bistates == NULL) {
                                                   fprintf(stderr, "Out of space for bistates\n");
                                                   exit(1);
                                                }
                                                assert(bistates != NULL);
                                                max_size = maximum_size;
                                                last = -1;
                                                n_stored = 0;
                                             };
   bistate& operator[] (int i) const {assert((0 <= i) && (i <= last)); return bistates[i];}
   int   n_of_states()  const {return last + 1;}
   void  null()         {last = -1; max_size = 0; n_stored = 0; bistates = NULL;}
   void  clear()        {last = -1; n_stored = 0;}
   bool  empty()        const {return(last == -1);}
   bool  is_full()      const {return(last >= max_size - 1);}
   bool  is_not_full()  const {return(last < max_size - 1);}
   bool  is_null()      const {return(bistates == NULL);}
   int   maximum_size() const {return max_size;}
   __int64  stored()    const {return n_stored;}
   void  swap(const int i, const int j)   {  bistate temp;
                                             assert((0 <= i) && (i <= last));
                                             assert((0 <= j) && (j <= last));
                                             temp = bistates[i];
                                             bistates[i] = bistates[j];
                                             bistates[j] = temp;
                                          };
   void  print();
   int   add_bistate(const bistate *bistate) {
                                          last++;
                                          if (last >= max_size) {
                                             //fprintf (stderr, "Out of space for bistate\n");
                                             //exit(1);
                                             return(-1);
                                          }
                                          n_stored++;
                                          bistates[last] = *bistate;

                                          return(last);
                                       }
   void  replace_bistate(const int index, const bistate *bistate) {
                                                                     assert((0 <= index) && (index <= last));
                                                                     n_stored++;
                                                                     bistates[index] = *bistate;
                                                                  }
   void  print_bistate(const int index)   {
                                             assert((0 <= index) && (index <= last));
                                             printf("%10d %3d %3d %3d %10d: ", index, bistates[index].g1, bistates[index].h1, bistates[index].empty_location, bistates[index].prev_location1, bistates[index].parent1);
                                             for(int j = 0; j < N_LOCATIONS; j++) printf("%2d ", bistates[index].tile_in_location[j]);
                                             printf("\n");
                                          }
private:
   int            last;             // = last index in states that is being used.
                                    // = number of states currently in states - 1.
   int            max_size;         // = max number of items in the heap.
   __int64        n_stored;         // = total number of states that have been stored, including replacements.
   bistate        *bistates;        // The states are stored in bistates[0], ..., bistates[last].
};

#endif

