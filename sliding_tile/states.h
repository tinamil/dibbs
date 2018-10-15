#ifndef _states_
#define _states_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
//#include "heap_record.h"
#include <cstdlib>

#define  N_LOCATIONS 16             // Number of locations

/*
   The following class and functions implement an array that can be used to store states in a cyclic best first search algorithm.
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
   4. Written 12/2/11.  Based on the states in c:\sewell\research\facility\facility_cbfns\memory.cpp.
*/

/*************************************************************************************************/

class state {
public:
   unsigned char  z;                   // = objective function value = number of moves that have been made so far
   unsigned char  LB;                  // = lower bound on the number of moves needed to reach the goal postion
   unsigned char  empty_location;      // = location of the empty tile
   unsigned char  prev_location;       // = location of the empty tile in the parent of this subproblem
   int            parent;              // = index of the parent subproblem
   unsigned char  tile_in_location[N_LOCATIONS];   // tile_in_location[i] = the tile that is in location i
};

/*************************************************************************************************/

class states_array {
public:
   states_array()    {  last = -1; max_size = 0; n_stored = 0; states = NULL;}
   states_array(const states_array&); // copy constructor
   ~states_array()   { delete [] states;}
   void  initialize(const int maximum_size)  {  assert(maximum_size > 0);
                                                states = new state[maximum_size + 1];
                                                if(states == NULL) {
                                                   fprintf(stderr, "Out of space for states\n");
                                                   exit(1);
                                                }
                                                assert(states != NULL);
                                                max_size = maximum_size;
                                                last = -1;
                                                n_stored = 0;
                                             };
   state& operator[] (int i) const {assert((0 <= i) && (i <= last)); return states[i];}
   int   n_of_states()  const {return last + 1;}
   void  null()         {last = -1; max_size = 0; n_stored = 0; states = NULL;}
   void  clear()        {last = -1; n_stored = 0;}
   bool  empty()        const {return(last == -1);}
   bool  is_full()      const {return(last >= max_size - 1);}
   bool  is_not_full()  const {return(last < max_size - 1);}
   bool  is_null()      const {return(states == NULL);}
   int   maximum_size() const {return max_size;}
   __int64  stored()    const {return n_stored;}
   void  swap(const int i, const int j)   {  state temp;
                                             assert((0 <= i) && (i <= last));
                                             assert((0 <= j) && (j <= last));
                                             temp = states[i];
                                             states[i] = states[j];
                                             states[j] = temp;
                                          };
   void  print();
   int   add_state(const state *state) {
                                          last++;
                                          if (last >= max_size) {
                                             //fprintf (stderr, "Out of space for state\n");
                                             //exit(1);
                                             return(-1);
                                          }
                                          n_stored++;
                                          states[last] = *state;

                                          return(last);
                                       }
   void  replace_state(const int index, const state *state) {
                                                               assert((0 <= index) && (index <= last));
                                                               n_stored++;
                                                               states[index] = *state;
                                                            }
   void  check_for_dominated_states()  {
                                          int   cnt, k;
                                          for(int i = 0; i < last; i++) {
                                             cnt = 0;
                                             for(int j = i + 1; j <= last; j++) {
                                                for(k = 0; (k < N_LOCATIONS) && (states[i].tile_in_location[k] == states[j].tile_in_location[k]); k++);
                                                if(k == N_LOCATIONS) {
                                                   cnt++;
                                                   this->print_state(i);
                                                   this->print_state(j);
                                                }
                                             }
                                             //if(cnt > 0) {
                                             //   printf("%10d %10d: ", i, cnt);
                                             //   for(int j = 0; j < N_LOCATIONS; j++) printf("%2d ", states[i].tile_in_location[j]);
                                             //   printf("\n");
                                             //}
                                          }
                                       }
   void  print_state(const int index)  {
                                          assert((0 <= index) && (index <= last));
                                          printf("%10d %3d %3d %3d %3d %10d: ", index, states[index].z, states[index].LB, states[index].empty_location, states[index].prev_location, states[index].prev_location, states[index].parent);
                                          for(int j = 0; j < N_LOCATIONS; j++) printf("%2d ", states[index].tile_in_location[j]);
                                          printf("\n");
                                       }
private:
   int            last;             // = last index in states that is being used.
                                    // = number of states currently in states - 1.
   int            max_size;         // = max number of items in the heap.
   __int64        n_stored;         // = total number of states that have been stored, including replacements.
   state          *states;          // The states are stored in states[0], ..., states[last].
};

#endif

