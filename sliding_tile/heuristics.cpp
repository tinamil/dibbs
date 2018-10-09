#include "main.h"
#include <queue>

/*************************************************************************************************/

unsigned char look_ahead_UB(unsigned char *tile_in_location, unsigned char look_ahead)
/*
   1. This algorithm computes an upper bound for a 15-puzzle.  It repeatedly performs a limited DFS
      and then makes the first move on the best path found by the DFS.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      b. look_ahead = number of moves to look ahead during the limited DFS.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
      a. z = the number of moves required by the heuristic solution to solve the puzzle is returned.
   5. Written 11/23/11.
*/
{
   bool           search;
   unsigned char  b, b_path, empty_location, LB, min_location, prev_location, solution[MAX_DEPTH + 1], *source, tile, z;
   int            i;
   double         cpu;
   clock_t        start_time;
   look_ahead_UB_parameters   parameters;
   look_ahead_UB_info         info;

   start_time = clock();

   // Compute the Manhattan lower bound.

   LB = compute_Manhattan_LB(tile_in_location);

   // Save the source (initial) configuration.

   source = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) source[i] = tile_in_location[i];

   // Find the location of the empty tile.

   for(i = 0; i <= n_tiles; i++) {
      if(tile_in_location[i] == 0) {
         empty_location = i;
         break;
      }
   }
   solution[0] = empty_location;
   prev_location = n_tiles + 1;

   // Perform iterative deepening.

   //n_explored = 0;
   //n_generated = 0;
   z = 0;
   parameters.bound = UCHAR_MAX;
   search = true;
   do {

      // Perform a limited DFS from the current configuration.
      // Select the move that returns the minimum bound from the DFS.
      
      if(search) {
         parameters.max_z = z + look_ahead;
         info.solved = false;
         info.min_z_plus_LB = UCHAR_MAX;
         b = look_ahead_UB_dfs(tile_in_location, empty_location, prev_location, LB, z, &parameters, &info);

         if(info.solved) {
            for( ; z <= b; z++) solution[z] = info.min_solution[z];
            z--;
            break;
         }
      }

      // Make the move that was found by DFS.

      min_location = info.min_solution[z + 1];
      tile = tile_in_location[min_location];
      tile_in_location[empty_location] = tile;
      tile_in_location[min_location] = 0;
      solution[++z] = min_location;
      prev_location = empty_location;
      empty_location = min_location;
      LB = compute_Manhattan_LB(tile_in_location);
      //LB = min_LB;
      b_path = examine_current_path(tile_in_location, info.min_solution, LB, z, look_ahead, &parameters, &info);

      if(info.solved) {
         for( ; z <= b; z++) solution[z] = info.min_solution[z];
         z--;
         break;
      }

      if(b_path == b) {
         search = false;
      } else {
         search = true;
         parameters.bound = b_path;
      }
      //parameters.bound = b + 3;

      cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
      printf("z = %3d  LB = %3d  z + LB = %3d  b = %3d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f\n", z, LB, z + LB, b, info.n_explored, info.n_generated, cpu);
      for(i = 0; i <= z; i++) printf("%2d ", solution[i]); printf("\n");
      for(i = 0; i <= z; i++) printf("   "); for(i = z + 1; i <= parameters.max_z + 1; i++) printf("%2d ", info.min_solution[i]); printf("\n");
   } while(LB > 0);

   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   printf("z = %3d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f\n", z, info.n_explored, info.n_generated, cpu);
   prn_solution(source, solution, z, NULL);

   // Restore the source (initial) configuration.

   for(i = 0; i <= n_tiles; i++) tile_in_location[i] = source[i];
   delete [] source;

   return(z);
}

//_________________________________________________________________________________________________

unsigned char look_ahead_UB_dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, 
                                unsigned char LB, unsigned char z, look_ahead_UB_parameters *parameters, look_ahead_UB_info *info)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within look ahead upper bound heuristic.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      b. empty_location = location of the empty tile.
      c. prev_location = location of the empty tile in the parent of this subproblem.
      d. LB = Manhattan lower bound on the number of moves needed to reach the goal postion.
      e. z = objective function value = number of moves that have been made so far.
      f. parameters
         parameters->bound: limit the search to subproblems whose total lower bound is <= bound.
         parameters->max_z: limit the search to subproblems whose z values is <= max_z.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
      a. min_bound = minimum bound of subproblems whose z exceeds max_z is returned.
      b. info is used to collect information about the search.
   5. Written 11/23/11.
*/
{
   unsigned char  b, LB_sub, min_bound, new_location, tile;
   int            i;

   info->n_explored++;
   assert(check_dfs_inputs(tile_in_location, empty_location, prev_location, LB));
   //prn_dfs_subproblem(tile_in_location, empty_location, prev_location, LB, z);

   if(LB == 0) {
      //prn_configuration(tile_in_location);
      info->solved = true;
      for(int j = 0; j <= z + 1; j++) info->min_solution[j] = info->current_solution[j];
      return(z);
   }

   // Generate the subproblems.

   min_bound = UCHAR_MAX;
   for(i = 1; i <= moves[empty_location][0]; i++) {
      new_location = moves[empty_location][i];
      if(new_location != prev_location) {
         info->n_generated++;
         tile = tile_in_location[new_location];
         tile_in_location[empty_location] = tile;
         tile_in_location[new_location] = 0;
         LB_sub = LB + distances[empty_location][tile] - distances[new_location][tile];
         assert(LB_sub == compute_Manhattan_LB(tile_in_location));
         info->current_solution[z + 1] = new_location;
         if((z + 1  <= parameters->max_z)  && (z + 1 + LB_sub <= parameters->bound)) {
            b = look_ahead_UB_dfs(tile_in_location, new_location, empty_location, LB_sub, z + 1, parameters, info);
            if(info->solved) return(b);
         } else {
            b = z + 1 + LB_sub;
            if((b < parameters->bound) || ((b == parameters->bound) && (z + LB < info->min_z_plus_LB))) {
               parameters->bound = b;
               info->min_z_plus_LB = z + LB;
               for(int j = 0; j <= z + 1; j++) info->min_solution[j] = info->current_solution[j];
            }
         }
         tile_in_location[empty_location] = 0;
         tile_in_location[new_location] = tile;
     
         if(b < min_bound) {
            min_bound = b;
            //*min_location = new_location;
            //*min_LB = LB_sub;
         }
      }
   }

   return(min_bound);
}


//_________________________________________________________________________________________________

unsigned char examine_current_path(unsigned char *initial_configuration, unsigned char *solution, unsigned char LB, unsigned char z, unsigned char look_ahead, look_ahead_UB_parameters *parameters, look_ahead_UB_info *info)
/*
   1. This function examines a path to see if it can be extended by one move without increasing the total lower bound.
   2. Input Variables
      a. intial_configuration[i] = the tile that is in location i.
         The elements of initial_configuration are stored beginning in initial_configuration[0].
      b. solution[d] = the location of the empty tile after move d in the current path.
      c. LB = Manhattan lower bound on the number of moves needed to reach the goal postion.
      d. z = objective function value = number of moves that have been made so far.
      e. parameters
         parameters->bound: limit the search to subproblems whose total lower bound is <= bound.
         parameters->max_z: limit the search to subproblems whose z values is <= max_z.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
      a. b = minimum bound of subproblems whose depth exceeds the current depth by 2 is returned.
      b. info is used to collect information about the search.
   5. Written 11/23/11.
*/
{
   unsigned char  b, empty_location, LB_sub, new_location, tile, *tile_in_location, z_sub;
   int            i;

   // Copy the intial_configuration into tile_in_location.

   tile_in_location = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) tile_in_location[i] = initial_configuration[i];

   // Construct the configuration obtained from the intial configuration (given in tile_in_location) if the current path (given in solution) is followed.

   //printf("\n");  prn_configuration(tile_in_location);   printf("\n");
   for(i = z + 1; i <= z + look_ahead - 1; i++) {
      info->current_solution[i] = solution[i];
      empty_location = solution[i-1];
      new_location = solution[i];
      tile = tile_in_location[new_location];
      tile_in_location[empty_location] = tile;
      tile_in_location[new_location] = 0;
      //prn_configuration(tile_in_location);   printf("\n");
   }
   z_sub = z + look_ahead - 1;
   LB_sub = compute_Manhattan_LB(tile_in_location);

   // Use look_ahead_UB_dfs to determine the minimum bound of subproblems that are two layers deeper.

   info->solved = false;
   parameters->max_z = z_sub + 1;
   parameters->bound = UCHAR_MAX;
   b = look_ahead_UB_dfs(tile_in_location, new_location, empty_location, LB_sub, z_sub, parameters, info);
   delete [] tile_in_location;

   return(b);
}
