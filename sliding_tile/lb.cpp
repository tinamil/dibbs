#include "main.h"
#include <queue>

static __int64          n_explored;
static __int64          n_generated;
static __int64          n_explored_depth[MAX_DEPTH + 1];
static __int64          n_explored_LB[MAX_DEPTH + 1];
static __int64          **n_explored_LB_b;
static __int64          **n_explored_g_h;
static __int64          *n_explored_g_h0;

/*************************************************************************************************/

__int64 difference_LB(unsigned char *source, unsigned char UB, unsigned char max_diff, int direction, DPDB *DPDB)
/*
   1. This algorithm uses a reverse search from the goal in attempt to compute information that
      can be used to improve the lower bounds during the forward search.
      See Bidirectional Heuristic Search Reconsidered by Kaindl and Kainz.
      a. Search in the reverse direction from the goal.
      b. z = number of moves made from the goal to the subproblem.
      c. LBg = lower bound on the distance from the subproblem to the goal.
      d. LBs = lower bound on the distance from the subproblem to the source.
      e. Limit the search in two ways.
         i.  max_LB = limit the search to subproblems whose LBg is less than or equal to max_LB.
         ii. UB = limit the search to subproblems whose z + LBs is less than UB.
      f. Compute z - LBg.  For a given value v of LBg, the min{z - LBg: LBg == v} can be added to the lower of any
         subproblem in the forward search whose LBg equals v.
      g. I used DFS.  Unfortunately, this produced very poor results.  For values up to v = 30, there were subproblems
         with z - LBg = 0.
      h. The paper suggests a slightly different improved bound and suggests using best first search using z - LBg as the best measure.
         Perhaps this would yield better results, but I did not pursue it.
   2. Input Variables
      a. source[i] = the tile that is in location i in the source (initial) configuration.
         The elements of source are stored beginning in source[0].
      b. UB = upper bound on the number of moves from the source to the goal.
      c. max_diff = limit on the difference between the number of moves made and the lower bound from the starting point (i.e., source in forward search, goal in reverse search).
      d. direction = direction of search = 1 for forward, 2 for reverse.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
   5. Written 11/21/11.
   6. Modified 9/3/12 to permit forward search and to accept max_diff as an input parameter.
*/
{
   unsigned char  empty_location, LB_to_source, LB_to_goal, max_LB, *source_location, *tile_in_location;
   int            i;
   __int64        sum;
   double         cpu;
   clock_t        start_time;

   start_time = clock();

   assert((direction == 1) || (direction == 2));

   // Load the goal configuration into tile_in_location.

   tile_in_location = new unsigned char[n_tiles + 1];
   if(direction == 1) {
      for(i = 0; i <= n_tiles; i++) tile_in_location[i] = source[i];
   } else {
      for(i = 0; i <= n_tiles; i++) tile_in_location[i] = i;
   }

   // Compute the Manhattan lower bounds.

   if(dpdb_lb > 0) {
      LB_to_goal = DPDB->compute_lb(tile_in_location);
   } else {
      LB_to_goal = compute_Manhattan_LB(tile_in_location);
   }
   LB_to_source = compute_Manhattan_LB2(source, tile_in_location);

   // Determine the locations of the tiles in the source configuration.

   source_location = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) source_location[source[i]] = i;

   // Set the location of the empty tile.

   if(direction ==1) {
      empty_location = source_location[0];
   } else {
      empty_location = 0;
   }

   // Initialize the counts.

   n_explored = 0;
   n_generated = 0;
   for(i = 0; i <= MAX_DEPTH; i++) {n_explored_depth[i] = 0; n_explored_LB[i] = 0;}
   n_explored_LB_b = new __int64*[UB + 1];
   for(i = 0; i <= UB; i++) {
      n_explored_LB_b[i] = new __int64[max_diff + 1];
      for(int b = 0; b <= max_diff; b++) n_explored_LB_b[i][b] = 0;
   }
   n_explored_g_h = new __int64*[UB + 1];
   if (direction == 1) {
      for(i = 0; i <= UB; i++) {
         n_explored_g_h[i] = new __int64[LB_to_goal + 1];
         for (int h = 0; h <= LB_to_goal; h++) n_explored_g_h[i][h] = 0;
      }
   } else {
      for(i = 0; i <= UB; i++) {
         n_explored_g_h[i] = new __int64[LB_to_source + 1];
         for (int h = 0; h <= LB_to_source; h++) n_explored_g_h[i][h] = 0;
      }
   }
	n_explored_g_h0 = new __int64[UB + 1];
	for (i = 0; i <= UB; i++) n_explored_g_h0[i] = 0;

   // Perform depth first search.

   max_LB = 20;
   if(direction == 1) {
      forward_diff_dfs(source_location, tile_in_location, empty_location, n_tiles + 1, LB_to_source, LB_to_goal, 0, max_LB, UB, max_diff, DPDB);
   } else {
      diff_dfs(source_location, tile_in_location, empty_location, n_tiles + 1, LB_to_source, LB_to_goal, 0, max_LB, UB, max_diff, DPDB);
   }

   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   printf("UB = %3d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f\n", UB, n_explored, n_generated, cpu);
   //for(i = 0; (i <= MAX_DEPTH) && (n_explored_depth[i] > 0); i++) printf("%3d %14I64d\n", i, n_explored_depth[i]); printf("\n");
   if(direction == 1) {
      //for(i = LB_to_goal; i <= UB; i++) printf("%3d %14I64d\n", i, n_explored_LB[i]);
      for(i = LB_to_goal; i <= UB; i+=2) {printf("%3d ", i); for(int b = 0; b <= max_diff; b+=2) printf("%11I64d ", n_explored_LB_b[i][b]); printf("\n");}
      printf("\n"); for(i = LB_to_goal; i <= UB; i+=2) printf("%11d ", i); printf("\n");
      for(i = LB_to_goal; i <= UB; i+=2) {
         sum = 0;
         for(int b = 0; b <= min(max_diff,LB_to_goal); b+=2) sum += n_explored_LB_b[i-b][b]; 
         printf("%11I64d ", sum);
      }
      printf("\n\n");
      printf("    "); for(int h = LB_to_goal; h >= 0; h-=2) printf("%10d ", h); printf("  h1-h2<=0\n");
      for(i = 0; i <= UB; i++) {printf("%3d ", i); for(int h = LB_to_goal; h >= 0; h-=2) printf("%10I64d ", n_explored_g_h[i][h]); printf("%10I64d\n", n_explored_g_h0[i]);}
      printf("\n");
   } else {
      //for(i = LB_to_source; i <= UB; i++) printf("%3d %14I64d\n", i, n_explored_LB[i]);
      for(i = LB_to_source; i <= UB; i+=2) {printf("%3d ", i); for(int b = 0; b <= max_diff; b+=2) printf("%11I64d ", n_explored_LB_b[i][b]); printf("\n");}
      printf("\n"); for(i = LB_to_source; i <= UB; i+=2) printf("%11d ", i); printf("\n");
      for(i = LB_to_source; i <= UB; i+=2) {
         sum = 0;
         for(int b = 0; b <= min(max_diff,LB_to_source); b+=2) sum += n_explored_LB_b[i-b][b]; 
         printf("%11I64d ", sum);
      }
      printf("\n\n");
      printf("    "); for (int h = LB_to_source; h >= 0; h -= 2) printf("%10d ", h); printf("  h2-h1<=0\n");
      for (i = 0; i <= UB; i++) { printf("%3d ", i); for (int h = LB_to_source; h >= 0; h -= 2) printf("%10I64d ", n_explored_g_h[i][h]); printf("%10I64d\n", n_explored_g_h0[i]); }
      printf("\n");
   }

   delete [] source_location;
   delete [] tile_in_location;

   return(n_explored);
}

//_________________________________________________________________________________________________

void diff_dfs(unsigned char *source_location, unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_source, unsigned char LB_to_goal, unsigned char z, unsigned char max_LB, unsigned char UB, unsigned char max_diff, DPDB *DPDB)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an algorithm that attempts to improve the lower bounds during the forward search.
      a. It performs a reverse DFS from the goal.
   2. Input Variables
      a. source_location[t] = location of tile t in the source (initial) configuration.
         The elements of source_location are stored beginning in source_location[0].
      b. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      c. empty_location = location of the empty tile.
      d. prev_location = location of the empty tile in the parent of this subproblem.
      e. LB_to_source = Manhattan lower bound on the number of moves needed to reach the goal.
      f. LB_to_goal = Manhattan lower bound on the number of moves needed to reach the source.
      g. z = objective function value = number of moves that have been made so far.
      h. max_LB = limit the search to subproblems whose lower bound to the goal is less than or equal to max_LB.
      i. UB = limit the search to subproblems whose z + LBs is less than UB.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
   5. Written 11/21/11.
*/
{
   unsigned char  LB_to_source_sub, LB_to_goal_sub, new_location, tile;
   int            i;

   n_explored++;
   n_explored_depth[z]++;
   n_explored_LB[z + LB_to_source]++;
   n_explored_LB_b[z + LB_to_source][z - LB_to_goal]++;
   if (LB_to_source >= LB_to_goal) {
      n_explored_g_h[z][LB_to_source - LB_to_goal]++;
   }
   else {
      n_explored_g_h0[z]++;
   }
   assert(check_dfs_inputs(tile_in_location, empty_location, prev_location, LB_to_goal));
   //if(LB_to_goal == max_LB) printf("z = %3d  LB_to_source = %3d  LB_to_goal = %3d  z - LB_to_goal = %3d\n", z, LB_to_source, LB_to_goal, z - LB_to_goal);
   //prn_configuration(tile_in_location);
   //prn_dfs_subproblem2(tile_in_location, empty_location, prev_location, LB_to_goal, LB_to_source, z);

   // Generate the subproblems.

   for(i = 1; i <= moves[empty_location][0]; i++) {
      new_location = moves[empty_location][i];        //assert((0 <= new_location) && (new_location <= n_tiles));
      if(new_location != prev_location) {
         n_generated++;
         tile = tile_in_location[new_location];       //assert((1 <= tile) && (tile <= n_tiles));
         tile_in_location[empty_location] = tile;
         tile_in_location[new_location] = 0;

         if(dpdb_lb > 0) {
            LB_to_goal_sub = DPDB->compute_lb(tile_in_location);
         } else {
            LB_to_goal_sub = LB_to_goal + distances[empty_location][tile] - distances[new_location][tile];
         }
         LB_to_source_sub = LB_to_source + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];
         assert(LB_to_goal_sub == compute_Manhattan_LB(tile_in_location));

         //if((z + 1 - LB_to_goal_sub <= max_diff) && (z + 1 + LB_to_source_sub <= UB)) {
         if ((2*(z + 1) + LB_to_source_sub - LB_to_goal_sub <= UB)) {
         //if((LB_to_goal_sub <= max_LB) && (z + 1 + LB_to_source_sub < UB)) {
         //if((z + 1 <= max_LB) && (z + 1 + LB_to_source_sub < UB)) {
            diff_dfs(source_location, tile_in_location, new_location, empty_location, LB_to_source_sub, LB_to_goal_sub, z + 1, max_LB, UB, max_diff, DPDB);
         }
         tile_in_location[empty_location] = 0;        //assert((0 <= empty_location) && (empty_location <= n_tiles));
         tile_in_location[new_location] = tile;       //assert((0 <= new_location) && (new_location <= n_tiles));
      }
   }
}

//_________________________________________________________________________________________________

void forward_diff_dfs(unsigned char *source_location, unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_source, unsigned char LB_to_goal, unsigned char z, unsigned char max_LB, unsigned char UB, unsigned char max_diff, DPDB *DPDB)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an algorithm that attempts to improve the lower bounds during the forward search.
      a. It performs a forward DFS from the source.
   2. Input Variables
      a. source_location[t] = location of tile t in the source (initial) configuration.
         The elements of source_location are stored beginning in source_location[0].
      b. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      c. empty_location = location of the empty tile.
      d. prev_location = location of the empty tile in the parent of this subproblem.
      e. LB_to_source = Manhattan lower bound on the number of moves needed to reach the goal.
      f. LB_to_goal = Manhattan lower bound on the number of moves needed to reach the source.
      g. z = objective function value = number of moves that have been made so far.
      h. max_LB = limit the search to subproblems whose lower bound to the goal is less than or equal to max_LB.
      i. UB = limit the search to subproblems whose z + LBs is less than UB.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
   5. Written 8/30/12.
*/
{
   unsigned char  LB_to_source_sub, LB_to_goal_sub, new_location, tile;
   int            i;

   n_explored++;
   n_explored_depth[z]++;
   n_explored_LB[z + LB_to_goal]++;
   n_explored_LB_b[z + LB_to_goal][z - LB_to_source]++;
	if (LB_to_goal >= LB_to_source) {
		 n_explored_g_h[z][LB_to_goal - LB_to_source]++;
	} else {
		 n_explored_g_h0[z]++;
	}
   assert(check_dfs_inputs(tile_in_location, empty_location, prev_location, LB_to_goal));
   //prn_configuration(tile_in_location);
   //prn_dfs_subproblem2(tile_in_location, empty_location, prev_location, LB_to_goal, LB_to_source, z);

   // Generate the subproblems.

   for(i = 1; i <= moves[empty_location][0]; i++) {
      new_location = moves[empty_location][i];        //assert((0 <= new_location) && (new_location <= n_tiles));
      if(new_location != prev_location) {
         n_generated++;
         tile = tile_in_location[new_location];       //assert((1 <= tile) && (tile <= n_tiles));
         tile_in_location[empty_location] = tile;
         tile_in_location[new_location] = 0;

         if(dpdb_lb > 0) {
            LB_to_goal_sub = DPDB->compute_lb(tile_in_location);
         } else {
            LB_to_goal_sub = LB_to_goal + distances[empty_location][tile] - distances[new_location][tile];
            assert(LB_to_goal_sub == compute_Manhattan_LB(tile_in_location));
         }
         LB_to_source_sub = LB_to_source + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];

         //if((z + 1 - LB_to_source_sub <= max_diff) && (z + 1 + LB_to_goal_sub <= UB)) {
         if(z + 1 + LB_to_goal_sub + z + 1 - LB_to_source_sub <= UB) {
            forward_diff_dfs(source_location, tile_in_location, new_location, empty_location, LB_to_source_sub, LB_to_goal_sub, z + 1, max_LB, UB, max_diff, DPDB);
         }
         tile_in_location[empty_location] = 0;        //assert((0 <= empty_location) && (empty_location <= n_tiles));
         tile_in_location[new_location] = tile;       //assert((0 <= new_location) && (new_location <= n_tiles));
      }
   }
}
