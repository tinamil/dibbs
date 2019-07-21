#ifndef _bistates_
#define _bistates_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
//#include "heap_record.h"

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
   4. Created 7/21/17 by modifying c:\sewell\research\15puzzle\15puzzle_code2\bistates.h
*/

typedef  int   State_index;      // This should be a signed intger type.

/*************************************************************************************************/

class bistate {
public:
   void  fill(const unsigned char g1i, const unsigned char h1i, const unsigned char open1i, const int parent1i, const State_index cluster_index1i,
      const unsigned char g2i, const unsigned char h2i, const unsigned char open2i, const int parent2i, const State_index cluster_index2i,
      const int hash_valuei, const unsigned char *seqi, const int n_pancakes)
   {
      g1 = g1i;
      h1 = h1i;
      open1 = open1i;
      parent1 = parent1i;
      cluster_index1 = cluster_index1i;
      g2 = g2i;
      h2 = h2i;
      open2 = open2i;
      parent2 = parent2i;
      cluster_index2 = cluster_index2i;
      hash_value = hash_valuei;
      memcpy(seq, seqi, n_pancakes + 1);
   };

   unsigned char  g1;                  // = objective function value = number of flips that have been made so far in the forward direction
   unsigned char  h1;                  // = lower bound on the number of moves needed to reach the goal postion
   unsigned char  open1;               // = 2 if this subproblem has not yet been generated in the forward direction
                                       // = 1 if this subproblem is open in the forward direction
                                       // = 0 if this subproblem closed in the forward direction
   int            parent1;             // = index (in states) of the parent subproblem in the forward direction
   __int64        cluster_index1 = -1; // = index in forward_clusters[g1][h1][h2]
   unsigned char  g2;                  // = objective function value = number of flips that have been made so far in the reverse direction
   unsigned char  h2;                  // = lower bound on the number of moves needed to reach the source postion
   unsigned char  open2;               // = 2 if this subproblem has not yet been generated in the reverse direction
                                       // = 1 if this subproblem is open in the reverse direction
                                       // = 0 if this subproblem closed in the reverse direction
   int            parent2;             // = index (in states) of the parent subproblem in the reverse direction
   __int64        cluster_index2 = -1; // = index in reverse_clusters[g2][h1][h2]
   int            hash_value;          // = the index in the hash table to begin searching for the state.
   unsigned char  *seq;                // the number of the pancake that is position i (i.e., order of the pancakes)
                                       // seq[0] = the number of pancakes
};

/*************************************************************************************************/

class bistates_array {
public:
   bistates_array() { n_pancakes = 0;  last = -1; max_size = 0; n_stored = 0; bistates = NULL; }
   bistates_array(const bistates_array&); // copy constructor
   ~bistates_array() { delete [] bistates;}
   void  initialize(const int maximum_size, const int n_pancake)  {  assert(maximum_size > 0);
                                                                     bistates = new bistate[maximum_size + 1];
                                                                     if(bistates == NULL) {
                                                                        fprintf(stderr, "Out of space for bistates\n");
                                                                        exit(1);
                                                                     }
                                                                     assert(bistates != NULL);
                                                                     max_size = maximum_size;
                                                                     n_pancakes = n_pancake;
                                                                     last = -1;
                                                                     n_stored = 0;
                                                                     for (int i = 0; i <= maximum_size; i++) {
                                                                        bistates[i].seq = new unsigned char[n_pancakes + 1];
                                                                        if (bistates[i].seq == NULL) {
                                                                           fprintf(stderr, "Out of space for bistates\n");
                                                                           exit(1);
                                                                        }
                                                                     }
                                                                  };
   bistate& operator[] (int i) const {assert((0 <= i) && (i <= last)); return bistates[i];}
   int   n_of_states()  const {return last + 1;}
   void  null()         {last = -1; max_size = 0; n_pancakes = 0; n_stored = 0; bistates = NULL;}
   void  clear()        {  //for(int i = 0; i <= last; i++) delete [] bistates[i].seq;
                           last = -1; 
                           //n_pancakes = 0; 
                           n_stored = 0;
                        }
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
                                                if(last >= max_size) {
                                                   //fprintf (stderr, "Out of space for bistate\n");
                                                   //exit(1);
                                                   return(-1);
                                                }
                                                n_stored++; 
                                                bistates[last].g1 = bistate->g1;
                                                bistates[last].h1 = bistate->h1;
                                                bistates[last].open1 = bistate->open1;
                                                bistates[last].parent1 = bistate->parent1;
                                                bistates[last].g2 = bistate->g2;
                                                bistates[last].h2 = bistate->h2;
                                                bistates[last].open2 = bistate->open2;
                                                bistates[last].parent2 = bistate->parent2;
                                                bistates[last].hash_value = bistate->hash_value;
                                                //bistates[last] = *bistate;
                                                //bistates[last].seq = new unsigned char[n_pancakes + 1];
                                                //if (bistates[last].seq == NULL) {
                                                //   fprintf(stderr, "Out of space for bistates[last].seq\n");
                                                //   //exit(1);
                                                //   return(-1);
                                                //}
                                                memcpy(bistates[last].seq, bistate->seq, n_pancakes + 1);

                                                return(last);
                                             }
   void  replace_bistate(const int index, const bistate *bistate) {
                                                                     assert((0 <= index) && (index <= last));
                                                                     n_stored++;  
                                                                     bistates[index] = *bistate;
                                                                  }
   void  print_bistate(const int index)   {
                                             unsigned char  f1, f2, f1b, f2b, g1, g2, h1, h2, o1, o2;
                                             int            p1, p2;
                                             assert((0 <= index) && (index <= last));
                                             g1 = bistates[index].g1;   h1 = bistates[index].h1;   o1 = bistates[index].open1;   p1 = bistates[index].parent1;
                                             g2 = bistates[index].g2;   h2 = bistates[index].h2;   o2 = bistates[index].open2;   p2 = bistates[index].parent2;
                                             if ((o1 < 2) && (o2 < 2)) {
                                                f1 = g1 + h1;   f1b = 2 * g1 + h1 - h2;   f2 = g2 + h2;   f2b = 2 * g2 + h2 - h1;
                                                printf("%10d %3d %3d %3d %3d %3d %3d %3d %10d %3d %3d %3d %10d %10d: ", index, f1, f1b, f2, f2b, g1, h1, o1, p1, g2, h2, o2, p2, bistates[index].hash_value);
                                             } else {
                                                if (o1 < 2) {
                                                   f1 = g1 + h1;   f1b = 2 * g1 + h1 - h2;
                                                   printf("%10d %3d %3d   *   * %3d %3d %3d %10d %3d %3d %3d %10d %10d: ", index, f1, f1b, g1, h1, o1, p1, g2, h2, o2, p2, bistates[index].hash_value);
                                                }
                                                if (o2 < 2) {
                                                   f2 = g2 + h2;   f2b = 2 * g2 + h2 - h1;
                                                   printf("%10d   *   * %3d %3d %3d %3d %3d %10d %3d %3d %3d %10d %10d: ", index, f2, f2b, g1, h1, o1, p1, g2, h2, o2, p2, bistates[index].hash_value);
                                                }
                                             }
                                             for(int j = 1; j <= bistates[index].seq[0]; j++) printf("%2d ", bistates[index].seq[j]);
                                             printf("\n");
                                          }
private:
   int            n_pancakes;       // = the number of pancakes.
   int            last;             // = last index in states that is being used.
                                    // = number of states currently in states - 1.
   int            max_size;         // = max number of items in the heap.
   __int64        n_stored;         // = total number of states that have been stored, including replacements.
   bistate        *bistates;        // The states are stored in bistates[0], ..., bistates[last].
};

#endif

