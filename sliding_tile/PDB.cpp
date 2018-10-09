#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <vector>
#include <queue>
#include "PDB.h"


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

//_________________________________________________________________________________________________

subproblem::subproblem(const subproblem& sub_prob){
   MD = sub_prob.MD;
   n_moves = sub_prob.n_moves;
   n_tiles_in_pattern = sub_prob.n_tiles_in_pattern;
   index = sub_prob.index;

	location = new unsigned char[n_tiles_in_pattern];
   if(location == NULL) {
      fprintf(stderr, "Out of space for location\n");
      exit(1);
   }
	memcpy(location, sub_prob.location, n_tiles_in_pattern * sizeof(unsigned char));
}

//__________________________________________________________________________________________________

subproblem& subproblem::operator=(const subproblem& rhs)
{
   if(this != & rhs) {  // Do nothing if assigned to self
      MD = rhs.MD;
      n_moves = rhs.n_moves;
      n_tiles_in_pattern = rhs.n_tiles_in_pattern;
      index = rhs.index;
      
	   //delete[] char_array;
      if(location == NULL) {
         location = new unsigned char[n_tiles_in_pattern];
         if(location == NULL) {
            fprintf(stderr, "Out of space for location\n");
            exit(1);
         }
      }
      memcpy(location, rhs.location, n_tiles_in_pattern * sizeof(unsigned char));
   }
   return *this;
}

/*************************************************************************************************/

void PDB::build_IDA(unsigned char  **distances, unsigned char **moves)
/*
   1. This function uses iterative deepening to compute the database.
   2. Input Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   3. Output Variables
      a. database[index] = minimum number of moves to reach the goal configuration
                           starting from the configuration corresponding index.
   4. Written 1/6/12.
*/
{
   bool           occupied[N_LOCATIONS];
   char           *min_moves;
   unsigned char  bound, empty_location, *location, MD, n_moves;
   int            i;
   __int64        index, index0, size;
   double         cpu;
   clock_t        start_time;

   start_time = clock();

   // Create the root problem.

   empty_location = 0;
   MD = 0;
   n_moves = 0;
   location = new unsigned char[n_tiles_in_pattern];
   for(i = 0; i < n_tiles_in_pattern; i++) {
      location[i] = pattern[i];
   }
   index = compute_index(location);
   database[index] = 0;
   for(i = 0; i < N_LOCATIONS; i++) occupied[i] = false;
   for(i = 0; i < n_tiles_in_pattern; i++) occupied[location[i]] = true;

   // Initialize min_moves.
   // Compute the size of min_moves.

   size = 1;
   for(i = N_LOCATIONS; i >= N_LOCATIONS - n_tiles_in_pattern; i--) size *= i;
   min_moves = new char[size + 1];
   for(index0 = 0; index0 <= size; index0++) min_moves[index0] = CHAR_MAX;

   // Perform iterative deepening.

   bound = 0;
   n_explored = 0;
   do {
      bound++;
      cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
      printf("bound = %3d  n_explored = %10I64d  cpu = %8.2f\n", bound, n_explored, cpu);
      found = false;
      n_explored = 0;
      for(index0 = 0; index0 <= size; index0++) {if(min_moves[index0] != CHAR_MAX) min_moves[index0] = -min_moves[index0]; }
      build_dfs(location, empty_location, MD, n_moves, index, min_moves, occupied, bound, distances, moves);
   } while(found == true);


   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   printf("%8.2f\n", cpu);
   //print2();

   delete [] location;
   delete [] min_moves;
}

//_________________________________________________________________________________________________

void PDB::build_dfs(unsigned char *location, unsigned char empty_location, unsigned char MD, unsigned char n_moves, __int64 index, char *min_moves, bool *occupied, unsigned char bound, unsigned char  **distances, unsigned char **moves)
/*
   1. This algorithm performs limited Depth First Search (DFS).
      It is designed to be used within an iterative deepening algorithm to create a pattern database.
   2. Input Variables
      a. location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
      b. empty_location = location of the empty tile.
      c. MD = Manhattan lower bound on the number of moves needed to reach the goal postion.
      d. n_moves = number of moves that have been made so far.
      e. index = the index (in database) corresponding to the configuration represented by location.
      f. min_moves[index0] = minimum number of moves to reach the goal configuration starting from the 
                             configuration corresponding index0, which includes the location of the empty tile.
      g. occupied[i] = true if location i is occupied by a tile in the pattern.
      h. bound = limit the search to subproblems whose number of moves is less than or equal to bound.
      i. distances[i][j] = Manhattan distance between location i and location j.
      j. moves[i] = list of possible ways to move the empty tile from location i.      
   3. Output Variables
      a. found = true if a descendent is found with n_moves = bound.
      b. database[index] = minimum number of moves to reach the goal configuration
                           starting from the configuration corresponding index is set for descendents of this subproblem.
      c. min_moves[index0] = minimum number of moves to reach the goal configuration starting from the 
                             configuration corresponding index0, which includes the location of the empty tile,
                             is set for descendents of this subproblem.
   4. Written 1/6/12.
*/
{
   bool           accessible[N_LOCATIONS];
   unsigned char  loc, max_location, MD_sub, new_location, n_moves1, tile;
   int            i, j;
   __int64        index0, index_sub;

   n_explored++;

   //printf("%3d %3d %10I64d\n", n_moves, MD, index); print_config(location, empty_location);

   // Determine which locations can be reached by the empty tile without moving any of the tiles in the pattern.

   for(i = 0; i < N_LOCATIONS; i++) accessible[i] = false;
   max_location = 0;
   accessible_dfs(accessible, empty_location, &max_location, occupied, moves);

   // Check if this configuration, including the empty location, has been encountered before.
   // If so, this subproblem can be pruned.  Note: This assumes that we are using some variation of breadth first search
   // that ensures that ensures that current path to this configuration is not shorter than the first path that led to this configuration.

   // Compute the index of this configuration, including the empty tile at its maximum location.
   index0 = compute_index0(location, max_location);
   //printf("%10I64d\n", index0);

   // Determine if this subproblem can be pruned.

   if(min_moves[index0] == CHAR_MAX) {
      min_moves[index0] = n_moves;
   } else {
      if(n_moves <= abs(min_moves[index0])) {            // I think that it might be possible to replace these if statements with
         assert(n_moves == abs(min_moves[index0]));      // if(n_moves == -min_moves[index0]) min_moves[index0] = -min_moves[index0] else return;
         if(min_moves[index0] <= 0) {                    // Use <= 0 instead of < 0 so that the root is not skipped.
            min_moves[index0] = -min_moves[index0];
         } else {
            //printf("pruned\n");
            return;
         }
      } else {
         //printf("pruned\n");
         return;
      }
   }

   // Return if the bound on the number of moves has been reached.

   if(n_moves == bound) {
      found = true;
      return;
   }

   // Set some of the fields of the subproblems.

   n_moves1 = n_moves + 1;

   // Generate all the subproblems from this subproblem.

   for(i = 0; i < n_tiles_in_pattern; i++) {             // Generate subproblems for each tile in the pattern.
      tile = pattern[i];
      loc = location[i];
      for(j = 1; j <= moves[loc][0]; j++) {              // Try each possible move from the current location of the tile.
         new_location = moves[loc][j];
         if(accessible[new_location]) {                  // Only permit moves to locations that can be reached by the empty tile.
            index_sub = index + (new_location - loc) * powers[i];
            if(n_moves1 < database[index_sub]) {         // If a shorter path has been found to the subproblem configuration,
               database[index_sub] = n_moves1;           // then update the database.
            }
            MD_sub = MD - distances[loc][tile] + distances[new_location][tile];
            location[i] = new_location;
            occupied[loc] = false;
            occupied[new_location] = true;
            //printf("%3d %3d %10I64d %10I64d\n", n_moves1, MD_sub, index_sub); print_config(location); printf("\n");
            build_dfs(location, loc, MD_sub, n_moves1, index_sub, min_moves, occupied, bound, distances, moves);
            location[i] = loc;
            occupied[loc] = true;
            occupied[new_location] = false;
         }
      }
   }
}

//_________________________________________________________________________________________________

void PDB::build_breadth_first_search(unsigned char  **distances, unsigned char **moves)
/*
   1. This function uses breadth first search to compute the database.
   2. Input Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   3. Output Variables
      a. database[index] = minimum number of moves to reach the goal configuration
                           starting from the configuration corresponding index.
   4. Written 1/6/12.
*/
{
   bool           accessible[N_LOCATIONS], occupied[N_LOCATIONS];
   char           *min_moves;
   unsigned char  empty_location, level, *location, max_location, MD, n_moves;
   int            i;
   __int64        index, index0, size;
   double         cpu;
   clock_t        start_time;

   start_time = clock();

   // Initialize min_moves.
   // Compute the size of min_moves.

   size = 1;
   for(i = N_LOCATIONS; i >= N_LOCATIONS - n_tiles_in_pattern; i--) size *= i;
   min_moves = new char[size + 1];
   for(index0 = 0; index0 <= size; index0++) min_moves[index0] = CHAR_MAX;

   // Create the root problem.

   empty_location = 0;
   MD = 0;
   n_moves = 0;
   location = new unsigned char[n_tiles_in_pattern];
   for(i = 0; i < n_tiles_in_pattern; i++) {
      location[i] = pattern[i];
   }
   index = compute_index(location);
   database[index] = 0;

   // Compute min_moves for the root problem.

   for(i = 0; i < N_LOCATIONS; i++) occupied[i] = false;
   for(i = 0; i < n_tiles_in_pattern; i++) occupied[location[i]] = true;
   for(i = 0; i < N_LOCATIONS; i++) accessible[i] = false;
   max_location = 0;
   accessible_dfs(accessible, empty_location, &max_location, occupied, moves);

   // Compute the index of this configuration, including the empty tile at its maximum location.
   index0 = compute_index0(location, max_location);
   //min_moves[index0] = n_moves - MD;
   
   // Generate subproblems from the root problem.
   // Do not perform this in the main breadth first search loop because it will attempt to generate subproblems from every subproblem with level = 0.
   
   level = 0;
   n_explored = 0;
   gen_subproblems(location, empty_location, level, min_moves, occupied, distances, moves);
   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   //printf("level = %3d  n_explored = %10I64d  cpu = %8.2f\n", level, n_explored, cpu);

   // Perform breadth first search.

   level = 2;
   do {

      // Read through min_moves to find subproblems that should be explored at this level.

      found = false;
      for(index0 = 0; index0 <= size; index0++) {if(min_moves[index0] == level) min_moves[index0] = -min_moves[index0]; }
      for(index0 = 0; index0 <= size; index0++) {
         if(-min_moves[index0] == level) {

            // Compute location from index0.

            invert_index0(index0, location, &empty_location);
            assert(compute_index0(location, empty_location) == index0);

            // Compute occupied from location.

            for(i = 0; i < N_LOCATIONS; i++) occupied[i] = false;
            for(i = 0; i < n_tiles_in_pattern; i++) occupied[location[i]] = true;

            // Generate subproblems.

            //min_moves[index0] = CHAR_MAX;                // Temporarily mark this configuration as unexplored.
            gen_subproblems(location, empty_location, level, min_moves, occupied, distances, moves);
         }
      }

      cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
      //printf("level = %3d  n_explored = %10I64d  cpu = %8.2f\n", level, n_explored, cpu);
      
      level += 2;
   } while(found == true);

   // Compute database from min_moves.

   for(index0 = 0; index0 <= size; index0++) {
      if(min_moves[index0] < CHAR_MAX) {
         invert_index0(index0, location, &empty_location);
         index = compute_index(location);
         MD = 0;
         for(i = 0; i < n_tiles_in_pattern; i++) {
            MD += distances[location[i]][pattern[i]];
         }
         if(min_moves[index0] + MD < database[index]) database[index] = min_moves[index0] + MD;
      }
   }

   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   printf("%8.2f\n", cpu);
   //print2();

   delete [] location;
   delete [] min_moves;
}

//_________________________________________________________________________________________________

void PDB::gen_subproblems(unsigned char *location, unsigned char empty_location, unsigned char level, char *min_moves, bool *occupied, unsigned char  **distances, unsigned char **moves)
/*
   1. This algorithm performs limited Depth First Search (DFS).
      It is designed to be used within an iterative deepening algorithm to create a pattern database.
   2. Input Variables
      a. location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
      b. empty_location = location of the empty tile.
      c. MD = Manhattan lower bound on the number of moves needed to reach the goal postion.
      d. n_moves = number of moves that have been made so far.
      e. index = the index (in database) corresponding to the configuration represented by location.
      f. min_moves[index0] = minimum number of moves to reach the goal configuration starting from the 
                             configuration corresponding index0, which includes the location of the empty tile.
      g. occupied[i] = true if location i is occupied by a tile in the pattern.
      h. bound = limit the search to subproblems whose number of moves is less than or equal to bound.
      i. distances[i][j] = Manhattan distance between location i and location j.
      j. moves[i] = list of possible ways to move the empty tile from location i.      
   3. Output Variables
      a. found = true if a descendent is found with n_moves = bound.
      b. min_moves[index0] = minimum number of moves to reach the goal configuration starting from the 
                             configuration corresponding index0, which includes the location of the empty tile,
                             is set for descendents of this subproblem.
   4. Written 1/6/12.
*/
{
   bool           accessible[N_LOCATIONS], accessible_sub[N_LOCATIONS];
   unsigned char  loc, max_location, max_location_sub, new_location, tile;
   int            delta_MD, i, ii, j;
   __int64        index0, index0_sub;

   n_explored++;
   //if(n_explored % 1000 == 0) printf("%10I64d\n", n_explored);

   // Determine which locations can be reached by the empty tile without moving any of the tiles in the pattern.

   for(i = 0; i < N_LOCATIONS; i++) accessible[i] = false;
   max_location = 0;
   accessible_dfs(accessible, empty_location, &max_location, occupied, moves);

   // Compute the index of this configuration, including the empty tile at its maximum location.
   index0 = compute_index0(location, max_location);
   //printf("\n%3d %3d %10I64d\n", level, min_moves[index0], index0); print_config(location, empty_location);

   // Determine if this subproblem can be pruned.

   if((level < min_moves[index0]) || ((level > 0) && (level == -min_moves[index0]))) {
      min_moves[index0] = level;
   } else {
      return;
   }

   // Generate all the subproblems from this subproblem.

   for(i = 0; i < n_tiles_in_pattern; i++) {             // Generate subproblems for each tile in the pattern.
      tile = pattern[i];
      loc = location[i];
      for(j = 1; j <= moves[loc][0]; j++) {              // Try each possible move from the current location of the tile.
         new_location = moves[loc][j];
         if(accessible[new_location]) {                  // Only permit moves to locations that can be reached by the empty tile.
            location[i] = new_location;
            occupied[loc] = false;
            occupied[new_location] = true;
            delta_MD = -distances[loc][tile] + distances[new_location][tile];
            if(delta_MD > 0) {                           // Only permit moves that do not increase the difference between the # of moves and MD.
               gen_subproblems(location, loc, level, min_moves, occupied, distances, moves);
            } else {
               found = true;                             // A subproblem has been generated at the next level, so set found = true.
               for(ii = 0; ii < N_LOCATIONS; ii++) accessible_sub[ii] = false;
               max_location_sub = 0;
               accessible_dfs(accessible_sub, loc, &max_location_sub, occupied, moves);
               index0_sub = compute_index0(location, max_location_sub);
               if(min_moves[index0_sub] == CHAR_MAX) min_moves[index0_sub] = level + 2;
            }
            location[i] = loc;
            occupied[loc] = true;
            occupied[new_location] = false;
         }
      }
   }
}

//_________________________________________________________________________________________________

void PDB::accessible_dfs(bool *accessible, unsigned char empty_location, unsigned char *max_location, bool *occupied, unsigned char **moves)
/*
   1. Given the locations of the tiles in the pattern as specified by occupied, this function determines which
      locations are accessible from the empty location without moving one of the tiles in the pattern.
   2. Input Variables
      a. accessible[i] = true if location i has been visited during the dfs search.
      b. empty_location = location of the empty tile.
      c. occupied[i] = true if location i is occupied by a tile in the pattern.
      d. moves[i] = list of possible ways to move the empty tile from location i. 
   3. Output Variables
      a. accessible[i] = is set to true for each location visited during this dfs search.
      b. max_location = maximum location visited during the dfs search.  It should be set equal to 0 prior to the root invocation of the dfs.
   4. Written 1/5/12.
*/
{
   unsigned char  new_location, *pnt_move, stop;
   int            i;

   accessible[empty_location] = true;
   if(empty_location > *max_location) *max_location = empty_location;

   stop = moves[empty_location][0];
   for(i = 1,  pnt_move = moves[empty_location]; i <= stop; i++) {
      new_location = *(++pnt_move);
      if(!accessible[new_location] && !occupied[new_location]) {
         accessible_dfs(accessible, new_location, max_location, occupied, moves);
      }
   }

}

//_________________________________________________________________________________________________

unsigned char PDB::min_moves(unsigned char  **distances, unsigned char **moves, unsigned char *source_location)
/*
   1. Given the locations of the tiles in the pattern (in source_location), this function determines the 
      miniumum number of moves to move from the source location to the goal destination.
   2. The purose of this function is to check the values stored in database.
   3. Input Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
      c. source_location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
   4. Written 1/2/12.
   5. This code is incorrect.  It was based on an incorrect understanding of how to compute a PDB.
*/
{
   unsigned char  bound, LB;
   int            i;

   // Compute the Manhattan lower bound.

   LB = 0;
   for(i = 0; i < n_tiles_in_pattern; i++) {
      LB += distances[pattern[i]][source_location[i]];
   }

   // Perform iterative deepening.

   solved = false;
   bound = LB;
   do {
      //printf("bound = %3d\n", bound);
      bound = pattern_dfs(source_location, LB, 0, bound, distances, moves);
   } while(solved == false);

   return(bound);
}

//_________________________________________________________________________________________________

unsigned char PDB::pattern_dfs(unsigned char *location, unsigned char LB, unsigned char z, unsigned char bound, unsigned char  **distances, unsigned char **moves)
/*
   1. This algorithm performs limited Depth First Search (DFS) on a pattern.
      It is designed to be used within an iterative deepening algorithm.
   2. Input Variables
      a. source_location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
      b. LB = Manhattan lower bound on the number of moves needed to reach the goal postion.
      c. z = objective function value = number of moves that have been made so far.
      d. distances[i][j] = Manhattan distance between location i and location j.
      e. moves[i] = list of possible ways to move the empty tile from location i. 
   3. Output Variables
      a. min_bound = minimum bound of subproblems whose lower bound exceeds bound is returned.
      b. solved = true (false) if the goal position was (not) reached during the search.
      c. solution[d] = the location of the empty tile after move d in the optimal solution.
   4. Written 1/2/12.
   5. This code is incorrect.  It was based on an incorrect understanding of how to compute a PDB.
*/
{
   unsigned char  b, LB_sub, loc, min_bound, new_location, tile, z1;
   int            i, j;
   in_set         occupied;

   if(LB == 0) {
      solved = true;
      return(z);
   }

   // Determine which locations are occupied.

   occupied.initialize(N_LOCATIONS);
   occupied.increment();
   for(i = 0; i < n_tiles_in_pattern; i++) occupied.set(location[i]);

   // Set some of the values for the new subproblems.

   z1 = z + 1;

   // Generate the subproblems.
   // Note: This permits a move back to the parent problem.  A better implementation would prevent such moves.

   min_bound = UCHAR_MAX;
   for(i = 0; i < n_tiles_in_pattern; i++) {             // Generate subproblems for each tile in the pattern.
      tile = pattern[i];
      loc = location[i];
      for(j = 1; j <= moves[loc][0]; j++) {              // Try each possible move from the current location of the tile.
         new_location = moves[loc][j];
         if(!occupied[new_location]) {
            LB_sub = LB - distances[loc][tile] + distances[new_location][tile];
            location[i] = new_location;
            if(z1 + LB_sub <= bound) {
               b = pattern_dfs(location, LB_sub, z1, bound, distances, moves);
            } else {
               b = z1 + LB_sub;
            }
            location[i] = loc;

            if(solved) {
               //prn_configuration(tile_in_location);   printf("\n");
               return(b);
            }
            min_bound = min(min_bound, b);
         }
      }
   }

   return(min_bound);
}

//_________________________________________________________________________________________________

__int64 PDB::compute_index0(unsigned char *location, unsigned char empty_location)
/*
   1. This function computes the index corresponding to the configuration represented by location and the empty location.
      The purpose of including the empty location is to be able to prune repeated configurations.
   2. This function uses a more complicated indexing system than the one used in compute_index (for the database).
      a. The reason for the more complicated system is that it uses less space.  Let n = number of locations and
         k = number of tiles in the pattern.  The simpler system would contain n^(k+1) entries.  The more
         complicated system contains n x (n-1) x ... x (n-k+1) entries.  A pattern with 16 locations and 8 tiles
         uses 4,151,347,200 entries, which is going to push the limits of my current computer (16 GB RAM).
      b. Many of these entries will not be used because a given configuration of the tiles in the pattern partitions
         the unoccupied locations into connected subsets.  Only one empty location need be used for each connected component.
         If a larger database is desired (than 7-8 or 6-6-6-6), then could use a data structure such as a judy array
         to store only the entries that are actually used.
      c. The code is based on code written by Robert Hilchie, which I found on the web.
   2. Input Variables
      a. location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
   3. Written 1/6/12.
*/
{
   unsigned char  locations[N_LOCATIONS];
   int         i, j, mult;
   __int64     index0;

   // Copy location into locations.

   for(i = 0; i < n_tiles_in_pattern; i++) locations[i] = location[i];

   // Combine the location numbers into a single index.

   mult = N_LOCATIONS; // Initially, all locations are available.
   index0 = 0;
   for (i = 0; i < n_tiles_in_pattern; i++) {
      index0 += locations[i]; // Accumulate location of 'i'th tile

      // Adjust location numbers of subsequent tiles because fewer positions are available.
      // Note: If this k^2 algorithm consumes too much time, I think it is possible to sort the locations in linear time
      // (using a bucket sort) and use information from the sort to adjust the location numbers prior to entering the main loop.

      for (j = i + 1; j < n_tiles_in_pattern; j++)
         if (locations[i] < locations[j])
            locations[j]--;
      if(locations[i] < empty_location) empty_location--;

      mult--;  // One fewer position is available
      index0 *= mult;
   }
   index0 += empty_location; //Accumulate location number of the emtpy tile.

   return(index0);
}

//_________________________________________________________________________________________________

void PDB::invert_index0(__int64 index0, unsigned char *location, unsigned char *empty_location)
/*
   1. This function computes the configuration corresponding to index0.
   2. Input Variables
      a. index0 = index corresponding to a configuration of the tiles in the pattern and the empty tile.
      b. The code is based on code written by Robert Hilchie, which I found on the web.
   3. Output Variables
      a. location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
      b. empty_location = location of the empty tile.
   3. Written 1/16/12.
*/
{
   int      i, j;
   unsigned div;

   // Break apart the index into position numbers
   div = N_LOCATIONS - n_tiles_in_pattern - 1;

   div++;
   *empty_location = index0 % div;
   index0 /= div;

   for(i = n_tiles_in_pattern - 1; i >= 0; i--) {
      div++;
      location[i] = index0 % div;
      index0 /= div;
   }

   // Compute true, unadjusted position numbers

   for(i = n_tiles_in_pattern - 1; i >= 0; i--) {
      for(j = i + 1; j < n_tiles_in_pattern; j++)
         if(location[i] <= location[j])
            location[j]++;
      if(location[i] <= *empty_location) (*empty_location)++;
   }
}

//_________________________________________________________________________________________________

__int64 PDB::compute_index(unsigned char *location)
/*
   1. This function computes the index (in database) corresponding to the configuration represented by location.
   2. Input Variables
      a. location[i] = location of tile i of the pattern.  
         The elements of location are stored beginning in location[0].
   3. Written 12/27/11.
*/
{
   int         i;
   __int64     index;

   index = 0;
   for(i = 0; i < n_tiles_in_pattern; i++) index += location[i] * powers[i];
   return(index);
}

//_________________________________________________________________________________________________

__int64 PDB::compute_index2(unsigned char *location_of_all_tiles)
/*
   1. This function computes the index (in database) corresponding to the configuration represented by location.
      a. This function differs from compute_index in that the location of all the tiles is an input parameter.
         In compute_index, location only specifies the location of the tiles in the pattern, and it specifies
         them in the order that the tiles appear in the pattern.
   2. Input Variables
      a. location_of_all_tiles[t] = location of tile t.  
         The elements of location are stored beginning in location_of_all_tiles[0].
   3. Written 12/29/11.
*/
{
   int         i;
   __int64     index;

   index = 0;
   for(i = 0; i < n_tiles_in_pattern; i++) index += location_of_all_tiles[pattern[i]] * powers[i];
   return(index);
}


//_________________________________________________________________________________________________

void PDB::print()
{
   __int64  cnt, index;

   cnt = 0;
   for(index = 0; index <= max_index; index++) {
      if(database[index] < UCHAR_MAX) {
         printf("%10I64d %3d\n", index, database[index]);
         cnt++;
      }
   }
   printf("%10I64d\n", cnt);
}

//_________________________________________________________________________________________________

void PDB::print2()
{
   __int64  cnt, index;

   cnt = 0;
   for(index = 0; index <= max_index; index++) {
      if(database[index] < UCHAR_MAX) {
         cnt++;
         printf("%3d%s", database[index], (cnt % 25) == 0 ? "\n":" ");
      }
   }
   if( (cnt % 25) != 0 )  printf("\n");
   printf("%10I64d\n", cnt);
}

//_________________________________________________________________________________________________

void PDB::print3()
{
   __int64  cnt, index;

   cnt = 0;
   for(index = 0; index <= max_index; index++) {
      cnt++;
      printf("%3d%s", database[index], (cnt % 25) == 0 ? "\n":" ");
   }
   if( (cnt % 25) != 0 )  printf("\n");
   printf("%10I64d\n", cnt);
}

//_________________________________________________________________________________________________

void PDB::print_binary()
{
   FILE     *out;
   __int64  index;

   // Open the file.  a = append, + = remove EOF marker prior to appending, b = binary.

   if (fopen_s(&out, "database.bin", "a+b") != 0) {
      fprintf(stderr,"Unable to open database file for output\n");
      exit(1);
   }

   for(index = 0; index <= max_index; index++) {
      fputc(database[index], out);
   }

   fclose(out);
}

//_________________________________________________________________________________________________

void PDB::print_config(unsigned char *location)
{
   unsigned char *tile_in_location;
   int            cnt, i, j;

   tile_in_location = new unsigned char[N_LOCATIONS];
   for(i = 0; i < N_LOCATIONS; i++) tile_in_location[i] = 0;
   for(i = 0; i < n_tiles_in_pattern; i++) tile_in_location[location[i]] = pattern[i];

   cnt = 0;
   for(i = 1; i <= n_rows; i++) {
      for(j = 1; j <= n_cols; j++) {
         printf(" %2d", tile_in_location[cnt]);
         cnt++;
      }
      printf("\n");
   }
   delete [] tile_in_location;
}

//_________________________________________________________________________________________________

void PDB::print_config(unsigned char *location, unsigned char empty_location)
{
   unsigned char *tile_in_location;
   int            cnt, i, j;

   tile_in_location = new unsigned char[N_LOCATIONS];
   for(i = 0; i < N_LOCATIONS; i++) tile_in_location[i] = UCHAR_MAX;
   for(i = 0; i < n_tiles_in_pattern; i++) tile_in_location[location[i]] = pattern[i];
   tile_in_location[empty_location] = 0;

   cnt = 0;
   for(i = 1; i <= n_rows; i++) {
      for(j = 1; j <= n_cols; j++) {
         if(tile_in_location[cnt] != UCHAR_MAX) {
            printf(" %2d", tile_in_location[cnt]);
         } else {
            printf("  .");
         }
         cnt++;
      }
      printf("\n");
   }
   delete [] tile_in_location;
}

/*************************************************************************************************/

void DPDB::add(unsigned char  **distances, unsigned char **moves, const int n_tiles_in_pattern, const unsigned char *pattern)
{  
   int            i;

   n_PDBs++;
   if(n_PDBs > max_n_PDBs) {
      fprintf(stderr, "n_PDBS > max_n_PDBs\n");
      exit(1);
   }

   // Update pattern_number.  Check that a tile does not appear in two different patterns.

   for(i = 0; i < n_tiles_in_pattern; i++) {
      pattern_number[pattern[i]]++;
      if(pattern_number[pattern[i]] > 1) {
         fprintf(stderr, "a tile is in two different patterns\n");
         exit(1);
      }
   }

   // Add the pattern to PDBs.

   PDBs[n_PDBs].initialize(n_rows, n_cols, n_tiles_in_pattern, pattern);
   //PDBs[n_PDBs].build_IDA(distances, moves);
   PDBs[n_PDBs].build_breadth_first_search(distances, moves);
   PDBs[n_PDBs].print_binary();
   //PDBs[n_PDBs].print3();
}

//_________________________________________________________________________________________________

void DPDB::read(const int n_databases, const int *n_tiles_in_patterns, unsigned char **patterns, const char *file_name)
{
   FILE     *in;
   int      d, i;
   __int64  index;

   n_PDBs = n_databases;
   if(n_PDBs > max_n_PDBs) {
      fprintf(stderr, "n_PDBS > max_n_PDBs\n");
      exit(1);
   }

   // Update pattern_number.  Check that a tile does not appear in two different patterns.

   for(d = 1; d <= n_PDBs; d++) {
      for(i = 0; i < n_tiles_in_patterns[d]; i++) {
         pattern_number[patterns[d][i]]++;
         if(pattern_number[patterns[d][i]] > 1) {
            fprintf(stderr, "a tile is in two different patterns\n");
            exit(1);
         }
      }
   }

   // Open the input file.  r = read, b = binary.

   if (fopen_s(&in, file_name, "rb") != 0) {
      fprintf(stderr,"Unable to open database file for input\n");
      exit(1);
   }

   // Add the patterns to PDBs.

   for(d = 1; d <= n_PDBs; d++) {
      PDBs[d].initialize(n_rows, n_cols, n_tiles_in_patterns[d], patterns[d]);
      for(index = 0; index <= PDBs[d].max_index; index++) {
         PDBs[d].database[index] = (unsigned char) fgetc(in);
      }
      //PDBs[d].print3(); printf("\n\n");
   }
}

//_________________________________________________________________________________________________

unsigned char DPDB::compute_lb(unsigned char  *tile_in_location)
{  
   unsigned char  LB, LB_reflection, *location_of_all_tiles;
   int            i, p;
   __int64        index;

   location_of_all_tiles = new unsigned char[N_LOCATIONS];
   for(i = 0; i < N_LOCATIONS; i++) location_of_all_tiles[tile_in_location[i]] = i;

   LB = 0;
   for(p = 1; p <= n_PDBs; p++) {
      index = PDBs[p].compute_index2(location_of_all_tiles);
      assert(PDBs[p][index] != UCHAR_MAX);
      LB += PDBs[p][index];
   }

   // Compute the LB from the reflection.

   for(i = 0; i < N_LOCATIONS; i++) location_of_all_tiles[reflection[tile_in_location[reflection[i]]]] = i;

   LB_reflection = 0;
   for(p = 1; p <= n_PDBs; p++) {
      index = PDBs[p].compute_index2(location_of_all_tiles);
      assert(PDBs[p][index] != UCHAR_MAX);
      LB_reflection += PDBs[p][index];
   }

   LB = max(LB, LB_reflection);

   delete [] location_of_all_tiles;
   return(LB);
}

//_________________________________________________________________________________________________
bool DPDB::check_lb(unsigned char  **distances, unsigned char **moves, unsigned char  *tile_in_location, unsigned char LB)
{  
   unsigned char  LB2, *location_of_all_tiles, *source_location;
   int            i, p;

   location_of_all_tiles = new unsigned char[N_LOCATIONS];
   for(i = 0; i < N_LOCATIONS; i++) location_of_all_tiles[tile_in_location[i]] = i;
   source_location = new unsigned char[N_LOCATIONS];

   LB2 = 0;
   for(p = 1; p <= n_PDBs; p++) {
      for(i = 0; i < PDBs[p].n_tiles_in_pattern; i++) source_location[i] = location_of_all_tiles[PDBs[p].pattern[i]];
      LB2 += PDBs[p].min_moves(distances, moves, source_location);
   }

   delete [] location_of_all_tiles;
   delete [] source_location;
   return(LB == LB2);
}

