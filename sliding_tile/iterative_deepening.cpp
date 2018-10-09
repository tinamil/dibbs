#include "main.h"
#include <queue>

static bool             solved;
static unsigned char    bound;
static unsigned char    goal[16] = {0, 1, 2, 3, 4, 5,  6, 7,  8,  9, 10,  11, 12, 13, 14,  15};   // Goal configuration
static unsigned char    source[16];          // Source configuration
static __int64          n_explored;
static __int64          n_generated;
static __int64          n_explored_depth[MAX_DEPTH + 1];
static __int64          n_explored_LB[MAX_DEPTH + 1];
static __int64          min_h1_h2[MAX_DEPTH + 1];
static __int64          n_min_h1_h2[MAX_DEPTH + 1];

/*************************************************************************************************/

unsigned char iterative_deepening(unsigned char *tile_in_location, DPDB *DPDB)
/*
   1. This algorithm performs Iterative Deepening A* (IDA*) on the 15-puzzle.
      See Depth-First Iterative-Deepening: An Optimal Admissible Tree Search by  Richard Korf.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      b. empty_location = location of the empty tile.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
      c. solved = 
      d. bound = limit the search to subproblems whose lower bound is less than or equal to bound.
   4. Output Variables
      a. bound = the minimum number of moves required to solve the puzzle is returned.
   5. Written 11/11/11.
*/
{
   unsigned char  empty_location, LB, solution[MAX_DEPTH + 1];
   int            i;
   double         cpu;
   clock_t        start_time;

   start_time = clock();

   // Save the source configuration.

   for(i = 0; i <= n_tiles; i++) source[i] = tile_in_location[i];

   // Compute the Manhattan lower bound.

   if(dpdb_lb > 0) {
      LB = DPDB->compute_lb(tile_in_location);
   } else {
      LB = compute_Manhattan_LB(tile_in_location);
   }

   // Find the location of the empty tile.

   for(i = 0; i <= n_tiles; i++) {
      if(tile_in_location[i] == 0) {
         empty_location = i;
         break;
      }
   }
   solution[0] = empty_location;

   // Perform iterative deepening.

   solved = false;
   bound = LB;
   n_explored = 0;
   n_generated = 0;
   for(i = 0; i <= MAX_DEPTH; i++) {n_explored_depth[i] = 0; n_explored_LB[i] = 0;}
   do {
      cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
      //printf("bound = %3d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f\n", bound, n_explored, n_generated, cpu);
      //for(i = 0; (i <= MAX_DEPTH) && (n_explored_depth[i] > 0); i++) printf("%3d %14I64d\n", i, n_explored_depth[i]); printf("\n");
      //for(i = LB; i <= bound; i++) printf("%3d %14I64d\n", i, n_explored_LB[i]);
      //for(i = LB; i <= bound; i++) printf("%3d %14I64d %3d %14I64d\n", i, n_explored_LB[i], min_h1_h2[i], n_min_h1_h2[i]);
      for(i = 0; i <= MAX_DEPTH; i++) {n_explored_depth[i] = 0; n_explored_LB[i] = 0; /*min_h1_h2[i] = UCHAR_MAX; n_min_h1_h2[0] = 0;*/}
      bound = dfs(tile_in_location, empty_location, n_tiles + 1, LB, 0, solution, DPDB);
//   } while(solved == false);
   } while ((solved == false) && (bound < UB));

   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   printf("%3d  %14I64d  %14I64d  %8.2f\n", bound, n_explored, n_generated, cpu);
   //printf("bound = %3d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f\n", bound, n_explored, n_generated, cpu);
   //for(i = 0; (i <= MAX_DEPTH) && (n_explored_depth[i] > 0); i++) printf("%3d %14I64d\n", i, n_explored_depth[i]); printf("\n");
   //for(i = LB; i <= bound; i++) printf("%3d %14I64d\n", i, n_explored_LB[i]);
   //for(i = LB; i <= bound; i++) printf("%3d %14I64d %3d %14I64d\n", i, n_explored_LB[i], min_h1_h2[i], n_min_h1_h2[i]);
   //prn_solution(tile_in_location, solution, bound, DPDB);
   return(bound);
}

//_________________________________________________________________________________________________

unsigned char dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z, unsigned char *solution, DPDB *DPDB)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an iterative deepening algorithm.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      b. empty_location = location of the empty tile.
      c. prev_location = location of the empty tile in the parent of this subproblem.
      d. LB = Manhattan lower bound on the number of moves needed to reach the goal postion.
      e. z = objective function value = number of moves that have been made so far.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
      c. solved = 
      d. bound = limit the search to subproblems whose lower bound is less than or equal to bound.
   4. Output Variables
      a. min_bound = minimum bound of subproblems whose lower bound exceeds bound is returned.
      b. solved = true (false) if the goal position was (not) reached during the search.
      c. solution[d] = the location of the empty tile after move d in the optimal solution.
   5. Written 11/11/11.
*/
{
   unsigned char  b, LB_sub, min_bound, new_location, *pnt_move, stop, tile, z_sub;
   int            i;

   n_explored++;
   n_explored_depth[z]++;
   n_explored_LB[z + LB]++;

   // Remove this block of code to speed up the iterative deepening.

   //h2 = compute_Manhattan_LB2(tile_in_location, source);
   //if((int) LB - (int) h2 < min_h1_h2[z + LB]) {
   //   min_h1_h2[z + LB] = LB - h2;
   //   n_min_h1_h2[z + LB] = 1;
   //} else if((int) LB - (int) h2 == min_h1_h2[z + LB]) {
   //   n_min_h1_h2[z + LB]++;
   //}

   assert(check_dfs_inputs(tile_in_location, empty_location, prev_location, LB));
   //prn_dfs_subproblem(tile_in_location, empty_location, prev_location, LB, z);
   //printf("%23d %3d %3d %3d", z, LB, z + LB, bound); for(i = 0; i < N_LOCATIONS; i++) printf(" %2d", tile_in_location[i]); printf("\n");

   if(LB == 0) {
      //prn_configuration(tile_in_location);
      solved = true;
      return(z);
   }

   // Generate the subproblems.

   min_bound = UCHAR_MAX;
   stop = moves[empty_location][0];
   z_sub = z + 1;
   for(i = 1,  pnt_move = moves[empty_location]; i <= stop; i++) {
      //new_location = moves[empty_location][i];
      new_location = *(++pnt_move);
      if(new_location != prev_location) {
         n_generated++;
         tile = tile_in_location[new_location];
         tile_in_location[empty_location] = tile;
         tile_in_location[new_location] = 0;
         if(dpdb_lb > 0) {
            LB_sub = DPDB->compute_lb(tile_in_location);
         } else {
            LB_sub = LB + distances[empty_location][tile] - distances[new_location][tile];
            assert(LB_sub == compute_Manhattan_LB(tile_in_location));
         }
         //if(!DPDB->check_lb(distances, moves, tile_in_location, LB_sub)) {
         //   fprintf(stderr, "LB is incorrect\n"); 
         //   prn_dfs_subproblem(tile_in_location, new_location, empty_location, LB_sub, z + 1);
         //   exit(1); 
         //}
         
         if(z_sub + LB_sub <= bound) {
            b = dfs(tile_in_location, new_location, empty_location, LB_sub, z_sub, solution, DPDB);
         } else {
            b = z_sub + LB_sub;
         }
         tile_in_location[empty_location] = 0;
         tile_in_location[new_location] = tile;
         
         if(solved) {
            //prn_configuration(tile_in_location);   printf("\n");
            solution[z + 1] = new_location;
            return(b);
         }
         min_bound = min(min_bound, b);
      }
   }

   return(min_bound);
}

//_________________________________________________________________________________________________

int check_dfs_inputs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB)
{
   if(check_tile_in_location(tile_in_location) == 0) {
      fprintf(stderr, "tile_in_location is illegal\n"); 
      exit(1); 
   }
   if((empty_location < 0) || (empty_location > n_tiles + 1)) {
      fprintf(stderr, "illegal value for empty_location\n"); 
      exit(1); 
   }
   if((prev_location < 0) || (prev_location > n_tiles + 2)) {     // prev_location should equal n_tiles + 1 at the root.
      fprintf(stderr, "illegal value for prev_location\n"); 
      exit(1); 
   }
   //if(LB != compute_Manhattan_LB(tile_in_location)) {
   //   fprintf(stderr, "LB is incorrect\n"); 
   //   exit(1); 
   //}
   return(1);
}
