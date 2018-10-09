#ifndef _PDB_
#define _PDB_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <vector>
#include <queue>
#include "states.h"
#include "in_set.h"

using namespace std;

/*
   The following classes and functions construc Disjoint Pattern Databases (DPDBs), which are used
   to construct better lower bounds for the 15-puzzle.
   1. Disjoint pattern databases.  See Disjoint Pattern Database Heuristics by Korf and Felner and 
      Solving the the 24 Puzzle with Instance Dependent Pattern Databases by Felner and Adler.
   2. To be a valid lower bound, the patterns must be disjoint, meaning that they are distinct
      subsets of the tiles.
   3. Written 12/26/11.
*/

/*************************************************************************************************/

class subproblem {
public:
   subproblem()   {  MD = 0; n_moves = 0; n_tiles_in_pattern = 0; index = 0; location = NULL; }
   subproblem(const subproblem&); // copy constructor
   ~subproblem()  {  if(location != NULL) delete [] location;}
   void  initialize(int n) 
               {  assert(n > 0);
                  MD = 0;
                  n_moves = 0;
                  n_tiles_in_pattern = n;
                  index = 0;
                  location = new unsigned char[n_tiles_in_pattern];
                  if(location == NULL) {
                     fprintf(stderr, "Out of space for location\n");
                     exit(1);
                  }
               }
   subproblem& operator= (const subproblem&);
   unsigned char  MD;                  // Manhattan lower bound on the number of moves needed to reach the goal postion
   unsigned char  n_moves;             // # of moves made to reach this configuration
   int            n_tiles_in_pattern;  // = # of tiles in the pattern.
   __int64        index;               // index (in database) of this configuration
   unsigned char  *location;           // location[t] = location of tile t in this configuration
};

//_________________________________________________________________________________________________

class PDB {
public:
   PDB()    {  found = false; solved = false; max_index = 0; n_explored = 0; n_tiles_in_pattern = 0; n_rows = 0; n_cols = 0, pattern = NULL; powers = NULL; database = NULL; }
   ~PDB()   {  if(pattern != NULL) delete [] pattern; if(powers != NULL) delete [] powers; if(database != NULL) delete [] database; }
   void  initialize(const int n_row, const int n_col, const int n_tiles_pattern, const unsigned char *pattern1) 
               {  assert(n_tiles_pattern > 0);
                  found = false;
                  solved = false;
                  n_explored = 0;
                  n_tiles_in_pattern = n_tiles_pattern;
                  n_rows = n_row;
                  n_cols = n_col;
                  max_index = 1;
                  //for(int i = N_LOCATIONS; i > N_LOCATIONS - n_tiles_in_pattern; i--) max_index *= i;
                  for(int i = 1; i <= n_tiles_in_pattern; i++) max_index *= N_LOCATIONS;
                  pattern = new unsigned char[n_tiles_in_pattern];
                  if(pattern == NULL) {
                     fprintf(stderr, "Out of space for pattern\n");
                     exit(1);
                  }
                  for(int i = 0; i < n_tiles_in_pattern; i++) pattern[i] = pattern1[i]; 
                  powers = new __int64[n_tiles_in_pattern];
                  if(powers == NULL) {
                     fprintf(stderr, "Out of space for powers\n");
                     exit(1);
                  }
                  powers[0] = 1;
                  for(int i = 1; i < n_tiles_in_pattern; i++) powers[i] = powers[i-1] * N_LOCATIONS;
                  database = new unsigned char[max_index + 1];
                  if(database == NULL) {
                     fprintf(stderr, "Out of space for database\n");
                     exit(1);
                  }
                  for(__int64 i = 0; i <= max_index; i++) database[i] = UCHAR_MAX;
               }
   unsigned char& operator[] (__int64 i) const {assert((0 <= i) && (i <= max_index)); return database[i];}
   void  build_IDA(unsigned char  **distances, unsigned char **moves);
   void  build_dfs(unsigned char *location, unsigned char empty_location, unsigned char MD, unsigned char n_moves, __int64 index, char *min_moves, bool *occupied, unsigned char bound, unsigned char  **distances, unsigned char **moves);
   void  build_breadth_first_search(unsigned char  **distances, unsigned char **moves);
   void  gen_subproblems(unsigned char *location, unsigned char empty_location, unsigned char level, char *min_moves, bool *occupied, unsigned char  **distances, unsigned char **moves);
   void  accessible_dfs(bool *accessible, unsigned char empty_location, unsigned char *max_location, bool *occupied, unsigned char **moves);
   unsigned char min_moves(unsigned char  **distances, unsigned char **moves, unsigned char *source_location);
   unsigned char pattern_dfs(unsigned char *location, unsigned char LB, unsigned char z,  unsigned char bound, unsigned char  **distances, unsigned char **moves);
   __int64  compute_index0(unsigned char *location, unsigned char empty_location);
   void invert_index0(__int64 index0, unsigned char *location, unsigned char *empty_location);
   __int64  compute_index(unsigned char *location);
   __int64  compute_index2(unsigned char *location_of_all_tiles);
   void  print();
   void  print2();
   void  print3();
   void  print_binary();
   void  print_config(unsigned char *location);
   void  print_config(unsigned char *location, unsigned char empty_location);
   friend class DPDB;
private:
   bool           found;               // Used by build.
   bool           solved;              // Used by min_moves and pattern_dfs.
   __int64        max_index;           // = maximum index of a configuration.
   __int64        n_explored;          // = number of subproblems explored during branch and bound process.
   int            n_tiles_in_pattern;  // = # of tiles in the pattern.
   int            n_rows;              // n_rows = number of rows.
   int            n_cols;              // n_cols = number of columns.
   unsigned char  *pattern;            // contains the list of tiles in the pattern.
   __int64        *powers;             // powers[i] = N_LOCATIONS^i, which is used to compute the index of a configuration.
   unsigned char  *database;           // database[index] = minimum number of moves to reach the goal configuration
                                       //                   starting from the configuration corresponding index.
};

//_________________________________________________________________________________________________

class DPDB {
public:
   DPDB()   {  max_n_PDBs = 0; n_PDBs = 0; n_rows = 0; n_cols = 0; pattern_number = NULL; PDBs = NULL; reflection = NULL;}
   ~DPDB()  {  if(pattern_number != NULL) delete [] pattern_number; if(PDBs != NULL) delete [] PDBs; if(reflection != NULL) delete reflection;}
   void  initialize(const int n_row, const int n_col, const int max_n_PDB) 
               {  assert(max_n_PDB > 0);
                  max_n_PDBs= max_n_PDB;
                  n_PDBs = 0;
                  n_rows = n_row;
                  n_cols = n_col;
                  pattern_number = new int[N_LOCATIONS];
                  if(pattern_number == NULL) {
                     fprintf(stderr, "Out of space for pattern_number\n");
                     exit(1);
                  }
                  for(int t = 0; t < N_LOCATIONS; t++) pattern_number[t] = 0;
                  PDBs = new PDB[max_n_PDBs + 1];
                  if(PDBs == NULL) {
                     fprintf(stderr, "Out of space for PDBs\n");
                     exit(1);
                  }
                  reflection = new unsigned char[N_LOCATIONS];
                  if(reflection == NULL) {
                     fprintf(stderr, "Out of space for reflection\n");
                     exit(1);
                  }
                  switch(N_LOCATIONS) {
		               case 16:
                        reflection[0] = 0; reflection[1] = 4; reflection[2] = 8;   reflection[3] = 12;  reflection[4] = 1;  reflection[5] = 5;  reflection[6] = 9;  reflection[7] = 13;
                        reflection[8] = 2; reflection[9] = 6; reflection[10] = 10; reflection[11] = 14; reflection[12] = 3; reflection[13] = 7; reflection[14] = 11; reflection[15] = 15;
                        break;
		               case 25: 
                        reflection[0] = 0;  reflection[1] = 5;  reflection[2] = 10;  reflection[3] = 15;  reflection[4] = 20;
                        reflection[5] = 1;  reflection[6] = 6;  reflection[7] = 11;  reflection[8] = 16;  reflection[9] = 21;
                        reflection[10] = 2; reflection[11] = 7; reflection[12] = 12; reflection[13] = 17; reflection[14] = 22;
                        reflection[15] = 3; reflection[16] = 8; reflection[17] = 13; reflection[18] = 18; reflection[19] = 23;
                        reflection[20] = 4; reflection[21] = 9; reflection[22] = 14; reflection[23] = 19; reflection[24] = 24;
                        break;
                     default: fprintf(stderr,"Illegal value of N_LOCATIONS in DPDB.initialize\n"); exit(1); break;
                  }

               }
   PDB& operator[] (int i) const {assert((1 <= i) && (i <= max_n_PDBs)); return PDBs[i];}
   void  add(unsigned char  **distances, unsigned char **moves, const int n_tiles_in_pattern, const unsigned char *pattern);
   void  read(const int n_databases, const int *n_tiles_in_patterns, unsigned char **patterns, const char *file_name);
   unsigned char compute_lb(unsigned char  *tile_in_location);
   bool check_lb(unsigned char  **distances, unsigned char **moves, unsigned char  *tile_in_location, unsigned char LB);
   void  print();
private:
   int            max_n_PDBs;          // = maximum # of pattern databases in the disjoint pattern database.
   int            n_PDBs;              // = # of pattern databases in the disjoint pattern database.
   int            n_rows;              // n_rows = number of rows.
   int            n_cols;              // n_cols = number of columns.
   int            *pattern_number;     // pattern_number[t] = the number of the patter to which tile t belongs.
   PDB            *PDBs;               // PDBs[p] = pattern database p.
   unsigned char  *reflection;         // reflection is used to reflect a configuration about the main diagonal.
};

#endif

