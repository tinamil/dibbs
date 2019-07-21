#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
//#include "states.h"
#include "bistates.h"

#define  HASH_SIZE 400000 // Must be a prime number.  Currently using linear
                              // probing, but if quadratic probing is used, then
                              // it must be a prime of the form 4j+3.  4000039=4*1000009+3  40000003=4*10000000+3   400000043=4*100000010+3   800000011=4*200000002+3   1000000007 = 4*250000001+3

int randomi(int n, double *dseed);

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

class hash_record {
public:
   int      state_index;
   hash_record()  {state_index = -1;}     // This assumes all legitimate state indices are >= 0
   void  clear()  {state_index = -1;}
};

/*************************************************************************************************/

class Hash_table {               // Note: I am capitalizing this class because the compiler will not let me declare a variable with the exact same spelling.
public:
   Hash_table() { n = 0; n_pancakes = 0; table = NULL; hash_values = NULL; }
   ~Hash_table()  {delete [] table; for(int i = 1; i <= n_pancakes; i++) delete [] hash_values[i]; delete [] hash_values;}
   void  initialize(int n_pancake)  {  table = new hash_record[HASH_SIZE];
                                       if(table == NULL) {
                                          fprintf(stderr, "Out of space for hash table\n");
                                          exit(1);
                                       }
                                       n = 0;
                                       n_pancakes = n_pancake;
                                       hash_values = new int*[n_pancakes + 1];
                                       if(hash_values == NULL) {
                                          fprintf(stderr, "Out of space for hash_values\n");
                                          exit(1);
                                       }
                                       for(int i = 1; i <= n_pancakes; i++) {
                                          hash_values[i] = new int[n_pancakes + 1];
                                          if(hash_values[i] == NULL) {
                                             fprintf(stderr, "Out of space for hash_values\n");
                                             exit(1);
                                          }
                                       }
                                       initialize_hash_values();
                                       //print_hash_values();
                                    };
   hash_record  operator[] (int i) const {assert((0 <= i) && (i < HASH_SIZE)); return table[i];}
   int   size()         {return n;}
   int   table_size()   {return HASH_SIZE;}
   void  clear() { for (int i = 0; i < HASH_SIZE; i++) { table[i].clear(); } n = 0; n_pancakes = 0; }
   int   hash_seq(unsigned char *seq);
   int   update_hash_value(unsigned char *seq, int i, int hash_value);
   void  initialize_hash_values()   {  double   seed;
                                       seed = 3.1567;
                                       for(int i = 1; i < n_pancakes; i++) {
                                          for(int j = i+1; j <= n_pancakes; j++) {
                                             hash_values[i][j] = randomi(HASH_SIZE, &seed);
                                             hash_values[j][i] = hash_values[i][j];
                                          }
                                       }
                                    }
   int   insert_at_index(int state_index, int hash_index);
   int   replace_at_index(int state_index, int hash_index);
   //void  print();
   void  print_hash_values();
   int   find_bistate(unsigned char *seq, int hash_value, int *hash_index, bistates_array *bistates);
   int   insert_bistate(int state_index, unsigned char *seq, int hash_index, bistates_array *bistates);
private:
   int            n;
   int            n_pancakes;
   int            **hash_values;    // hash_values[i][j] =random hash value associated with pancake i being adjacent to pancake j in seq: U[0,HASH_SIZE).
   //hash_record    table[HASH_SIZE];
   hash_record    *table;           // Use dynamic allocation because the compiler does not permit a declared array with more than 2^31 bytes. 
};
typedef Hash_table* Hash_tablepnt;
