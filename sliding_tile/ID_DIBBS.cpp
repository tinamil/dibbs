#include "main.h"
#include <queue>
#include <limits>

static bool             solved;
static unsigned char    bound1;                       // Limit the forward search to states whose f1_bar is less than or equal to bound1.
static unsigned char    bound2;                       // Limit the reverse search to states whose f2_bar is less than or equal to bound2.
static unsigned char    old_bound1;                   // Limit the forward search to states whose f1_bar is less than or equal to bound1.
static unsigned char    old_bound2;                   // Limit the reverse search to states whose f2_bar is less than or equal to bound2.
static unsigned char    LB;                           // LB = lower bound on optimal objective value.
static unsigned char    **min_g1;                     // min_g1[h1][h2] = minimum value of g1 of a state stored in the forward direction with heuristic values h1 and h2.
static unsigned char    **min_g2;                     // min_g2[h1][h2] = minimum value of g2 of a state stored in the reverse direction with heuristic values h1 and h2.
static __int64          n_explored;                   // # of states expanded.
static __int64          n_generated;                  // # of states expanded.
static __int64          n_explored_forward;           // # of states expanded in the forward direction.
static __int64          n_explored_reverse;           // # of states expanded in the reverse direction.
static __int64          n_generated_forward;          // # of states generated in the forward direction.
static __int64          n_generated_reverse;          // # of states generated in the reverse direction.
static __int64          n_exp_depth[MAX_DEPTH + 1];   // n_explored_depth[d] = number of states expanded at depth d.
static __int64          n_exp_f_level[MAX_DEPTH + 1]; // n_exp_f_level[a] = # of states expanded in forward direction with f1_bar = a.
static __int64          n_exp_r_level[MAX_DEPTH + 1]; // n_exp_r_level[a] = # of states expanded in reverse direction with f2_bar = a.
static __int64          n_gen_f_level[MAX_DEPTH + 1]; // n_gen_f_level[a] = # of states generated in forward direction with f1_bar = a.
static __int64          n_gen_r_level[MAX_DEPTH + 1]; // n_gen_r_level[a] = # of states genarated in reverse direction with f2_bar = a.
//static unsigned char    UB;                           // = objective value of best solution found so far.

/*************************************************************************************************/

unsigned char ID_DIBBS(unsigned char *source, unsigned char initial_UB, searchinfo *info, DPDB *DPDB)
/*
   1. This algorithm performs Iterative Deepening Dynamically Improved Bounds Bidirectional Search on the 15-puzzle.
   2. Input Variables
      a. source[i] = the tile that is in location i in the source (initial) configuration.
         The elements of source are stored beginning in source[0].
      b. initial_UB = the initial upper bound for the search.  Use initial_UB = MAX_DEPTH if no solution is known prior to the search.
      c. info stores information about the search.
      d. DPDB Disjoint Pattern Database.
  3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i.
      c. n_tiles = number of tiles.
      d. UB = objective value of best solution found so far.
      e. Also see the static variables defined at the top of this file.
   4. Output Variables
      a. bound = the minimum number of moves required to solve the puzzle is returned.
   5. Created 7/13/18 by modifying iterative_deepening from c:\sewell\research\15puzzle\15puzzle_code2\iterative_deepening.cpp.
*/
{
   bool           last_iteration;
   unsigned char  empty_location, f1_bar_min, f2_bar_min, *goal, h1_goal, h1_source, h2_goal, h2_source, new_bound1, new_bound2, old_direction, solution[MAX_DEPTH + 1], *source_location;
   int            direction, i;
   __int64        sum1, sum2, sum3, sum4;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL;
   double         best, cpu;
   Hash_table     hash_table;
   bistate        goal_state, source_state;			// states for the source and the goal.
   bistates_array states;
   pair<int, int> status_index;
   clock_t        start_time;

   start_time = clock();
   info->initialize();

   // Create the goal configuration.

   goal = new unsigned char[n_tiles + 1];
   for (i = 0; i <= n_tiles; i++) goal[i] = i;

   // Determine the locations of the tiles in the source configuration.
   // source_location[t] = location of tile t in the source (initial) configuration.
   // The elements of source_location are stored beginning in source_location[0].

   source_location = new unsigned char[n_tiles + 1];
   for (i = 0; i <= n_tiles; i++) source_location[source[i]] = i;

   // Find the location of the empty tile.

   for (i = 0; i <= n_tiles; i++) {
      if (source[i] == 0) {
         empty_location = i;
         break;
      }
   }
   solution[0] = empty_location;

   // Initialize data structures.

   if (states.is_null()) states.initialize(STATE_SPACE); else states.clear();
   hash_table.initialize();
   UB = initial_UB;
   min_g1 = new unsigned char*[MAX_DEPTH + 1];
   for (int h1 = 0; h1 <= MAX_DEPTH; h1++) {
      min_g1[h1] = new unsigned char[MAX_DEPTH + 1];
      for (int h2 = 0; h2 <= MAX_DEPTH; h2++) min_g1[h1][h2] = MAX_DEPTH;
   }
   min_g2 = new unsigned char*[MAX_DEPTH + 1];
   for (int h1 = 0; h1 <= MAX_DEPTH; h1++) {
      min_g2[h1] = new unsigned char[MAX_DEPTH + 1];
      for (int h2 = 0; h2 <= MAX_DEPTH; h2++) min_g2[h1][h2] = MAX_DEPTH;
   }

   // Create the root problem in the forward direction.
   // Compute h1 and h2.

   if(dpdb_lb > 0) {
      h1_source = DPDB->compute_lb(source);
   } else {
      h1_source = compute_Manhattan_LB(source);
   }
   h2_source = 0;

   source_state.g1 = 0;                           // = number of moves that have been made so far in the forward direction
   source_state.h1 = h1_source;                   // = lower bound on the number of moves needed to reach the goal postion
   source_state.open1 = 1;                        // = 2 if this subproblem has not yet been generated in the forward direction
                                                // = 1 if this subproblem is open in the forward direction
                                                // = 0 if this subproblem closed in the forward direction
   source_state.empty_location = empty_location;  // = location of the empty tile
   source_state.prev_location1 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the forward direction
   source_state.parent1 = -1;                     // = index of the parent subproblem in the forward direction
   source_state.g2 = UCHAR_MAX;                   // = number of moves that have been made so far in the reverse direction
   source_state.h2 = h2_source;                   // = lower bound on the number of moves needed to reach the source postion
   source_state.open2 = 2;                        // = 2 if this subproblem has not yet been generated in the reverse direction
                                                // = 1 if this subproblem is open in the reverse direction
                                                // = 0 if this subproblem closed in the reverse direction
   source_state.prev_location2 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the reverse direction
   source_state.parent2 = -1;                     // = index of the parent subproblem in the reverse direction
   for (i = 0; i <= n_tiles; i++) source_state.tile_in_location[i] = source[i];  // tile_in_location[i] = the tile that is in location i
   source_state.hash_value = hash_table.hash_configuration(source_state.tile_in_location);

   best = 2 * source_state.g1 + source_state.h1 - source_state.h2;
   f1_bar_min = 2 * source_state.g1 + source_state.h1 - source_state.h2;

   // Add the forward root problem to the list of states and the set of unexplored states.

   status_index = find_or_insert2(&source_state, 1, &states, &hash_table, source_state.hash_value);

   // Create the root problem in the reverse direction.
   // Compute h1 and h2.

   h1_goal = 0;
   h2_goal = compute_Manhattan_LB(source);     // This relies on the fact that the Manhattan distance from the source to the goal is the same as from the goal to the source

   goal_state.g1 = UCHAR_MAX;                   // = number of moves that have been made so far in the forward direction
   goal_state.h1 = h1_goal;                     // = lower bound on the number of moves needed to reach the goal postion
   goal_state.open1 = 2;                        // = 2 if this subproblem has not yet been generated in the forward direction
                                                // = 1 if this subproblem is open in the forward direction
                                                // = 0 if this subproblem closed in the forward direction
   goal_state.empty_location = 0;               // = location of the empty tile
   goal_state.prev_location1 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the forward direction
   goal_state.parent1 = -1;                     // = index of the parent subproblem in the forward direction
   goal_state.g2 = 0;                           // = number of moves that have been made so far in the reverse direction
   goal_state.h2 = h2_goal;                     // = lower bound on the number of moves needed to reach the source postion
   goal_state.open2 = 1;                        // = 2 if this subproblem has not yet been generated in the reverse direction
                                                // = 1 if this subproblem is open in the reverse direction
                                                // = 0 if this subproblem closed in the reverse direction
   goal_state.prev_location2 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the reverse direction
   goal_state.parent2 = -1;                     // = index of the parent subproblem in the reverse direction
   for (i = 0; i <= n_tiles; i++) goal_state.tile_in_location[i] = i;  // tile_in_location[i] = the tile that is in location i
   goal_state.hash_value = hash_table.hash_configuration(goal_state.tile_in_location);

   best = 2 * goal_state.g2 + goal_state.h2 - goal_state.h1;
   f2_bar_min = 2 * goal_state.g2 + goal_state.h2 - goal_state.h1;

   // Add the reverse root problem to the list of states and the set of unexplored states.

   status_index = find_or_insert2(&goal_state, 2, &states, &hash_table, goal_state.hash_value);

   // Perform iterative deepening.

   direction = 1;   old_direction = 0;
   last_iteration = false;
   solved = false;
   bound1 = f1_bar_min;   old_bound1 = bound1;
   bound2 = f2_bar_min;   old_bound2 = bound2;
   LB = max(source_state.h1, goal_state.h2);
   n_explored = 0;   n_generated = 0;   n_explored_forward = 0;   n_explored_reverse = 0;   n_generated_forward = 0;   n_generated_reverse = 0;
   do {
      //for (i = 0; i <= MAX_DEPTH; i++) { n_exp_depth[i] = 0; n_exp_f_level[i] = 0; n_exp_r_level[i] = 0; n_gen_f_level[i] = 0; n_gen_r_level[i] = 0; }

      //if (n_gen_f_level[bound1] <= n_gen_r_level[bound2]) direction = 1; else direction = 2;
      if (2*UB - old_bound1 - old_bound2 <= 6) {
         last_iteration = true;
         if (direction == 1) {
            direction = 2;
            bound2 = old_bound2;
         } else {
            direction = 1;
            bound1 = old_bound1;
         }
      } else {
         if (n_explored_forward <= n_explored_reverse) direction = 1; else direction = 2;
      }

      if(direction == 1) {
         for (i = 0; i <= MAX_DEPTH; i++) { n_exp_depth[i] = 0; n_exp_f_level[i] = 0; n_gen_f_level[i] = 0;}
         new_bound1 = forward_dfs(source_location, &source_state, solution, &states, &hash_table, info, DPDB);
         old_bound1 = bound1;
         bound1 = new_bound1;
      } else {
         for (i = 0; i <= MAX_DEPTH; i++) { n_exp_depth[i] = 0; n_exp_r_level[i] = 0; n_gen_r_level[i] = 0; }
         new_bound2 = reverse_dfs(source_location, &goal_state, solution, &states, &hash_table, info, DPDB);
         old_bound2 = bound2;
         bound2 = new_bound2;
      }

      // If the direction has switched, then update LB.

      if (direction != old_direction) {
         LB = ((old_bound1 + old_bound2) / 2) + 1;
         LB += abs((LB - source_state.h1) % 2);       // This LB depends on the parity property of the sliding tile puzzle.
         old_direction = direction;
      }
      //if (direction == 1) direction = 2; else direction = 1;
      if (LB >= UB) solved = true;
      if (last_iteration) solved = true;


      if (prn_info > 0) {
         cpu = (double)(clock() - start_time) / CLOCKS_PER_SEC;
         if (direction == 1) printf("Forward\n"); else printf("Reverse\n");
         printf("UB = %3d old_bound1 = %3d  new_bound1 = %3d   old_bound2 = %3d  new_bound2 = %3d  n_exp = %12I64d  n_exp_f = %12I64d  n_exp_r = %12I64d  n_gen = %12I64d  n_gen_f = %12I64d  n_gen_r = %12I64d  cpu = %7.2f\n",
                 UB, old_bound1, bound1, old_bound2, bound2, n_explored, n_explored_forward, n_explored_reverse, n_generated, n_generated_forward, n_generated_reverse, cpu);
         if (prn_info > 1) {
            //for (i = 0, sum1 = 0; (i <= MAX_DEPTH) && (n_exp_depth[i] > 0); i++) {printf("%3d %14I64d\n", i, n_exp_depth[i]); sum1 += n_exp_depth[i];}
            //printf("      ------------\n");
            //printf("    %14I64d\n\n", sum1);
            for (i = min(h1_source, h2_goal), sum1 = sum2 = sum3 = sum4 = 0; i <= max(bound1, bound2) + 2; i++) { printf("%3d %14I64d %14I64d %14I64d %14I64d\n", i, n_exp_f_level[i], n_exp_r_level[i], n_gen_f_level[i], n_gen_r_level[i]); sum1 += n_exp_f_level[i]; sum2 += n_exp_r_level[i], sum3 += n_gen_f_level[i], sum4 += n_gen_r_level[i]; }
            printf("      ------------   ------------   ------------   ------------\n");
            printf("    %14I64d %14I64d %14I64d %14I64d\n\n", sum1, sum2, sum3, sum4);
         }
      }
   } while(solved == false);

   cpu = (double) (clock() - start_time) / CLOCKS_PER_SEC;
   info->cpu = cpu;
   info->n_explored = n_explored;
   info->n_explored_forward = n_explored_forward;
   info->n_explored_reverse = n_explored_reverse;
   info->n_generated = n_generated;
   info->n_generated_forward = n_generated_forward;
   info->n_generated_reverse = n_generated_reverse;
   for (i = 0; i <= MAX_DEPTH; i++) info->n_explored_depth[i] = n_exp_depth[i];

   printf("%3d %3d %3d %7.2f %7.2f %9I64d %9I64d %9I64d %9I64d %9I64d %9I64d %9I64d %9I64d\n",
          info->best_z, bound1, bound2, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored());
   if (prn_info > 0) {
      printf("z = %3d bound1 = %3d bound2 = %3d cpu = %7.2f best_cpu = %7.2f best_branch = %9I64d n_exp = %9I64d n_exp_f = %9I64d n_exp_r = %9I64d n_gen = %9I64d n_gen_f = %9I64d n_gen_r = %9I64d n_stored = = %9I64d\n",
             info->best_z, bound1, bound2, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored());
      //for(i = 0; i <= UB; i++) printf("%3d %14I64d\n", i, info->n_explored_depth[i]);
   }
   //prn_solution(source, solution, bound1);

   //search_states(&states, UB);
   analyze_states(&states, UB + 8, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h);

   delete[] goal;
   delete[] source_location;
   for (int h1 = 0; h1 <= MAX_DEPTH; h1++) delete[] min_g1[h1];
   delete[] min_g1;
   for (int h1 = 0; h1 <= MAX_DEPTH; h1++) delete[] min_g2[h1];
   delete[] min_g2;

   return(info->best_z);
}

//_________________________________________________________________________________________________

unsigned char forward_dfs(unsigned char *source_location, bistate *state, unsigned char *solution, bistates_array *states, Hash_table *hash_table, searchinfo *info, DPDB *DPDB)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an iterative deepening algorithm.
   2. Input Variables
      a. source_location[t] = location of tile t in the source (initial) configuration.
         The elements of source_location are stored beginning in source_location[0].
      b. state is the state which is to be expanded.
      c. states = store the states.
      d. hash_table = hash table used to find states.
      e. info stores information about the search.
      f. DPDB Disjoint Pattern Database.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i.
      c. n_tiles = number of tiles.
      d. dpdb_lb: 1 = use disjoint LB, 0 = do not use
      e. UB = objective value of best solution found so far.
      f. Also see the static variables defined at the top of this file.
   4. Output Variables
      a. min_bound = minimum bound of subproblems whose lower bound exceeds bound is returned.
      b. solved = true (false) if the goal position was (not) reached during the search.
      c. solution[d] = the location of the empty tile after move d in the optimal solution.
   5. Created 7/13/18 by modifying dfs from c:\sewell\research\15puzzle\15puzzle_code2\iterative_deepening.cpp.
*/
{
   unsigned char  b, empty_location, f1_bar, f1_sub, f1_bar_sub, g1, g1_sub, h1, h1_sub, h2, h2_sub, min_bound, new_location, prev_location, *pnt_move, stop, tile;
   int            hash_index_sub, hash_value, hash_value_sub, i, state_index, status;
   bistate        new_state;
   pair<int, int>    status_index;

   // Store pertinent variables from state to local variables.

   g1 = state->g1;   h1 = state->h1;   h2 = state->h2;
   empty_location = state->empty_location;   prev_location = state->prev_location1;
   hash_value = state->hash_value;

   f1_bar = 2 * g1 + h1 - h2;
   n_explored++;   n_explored_forward++;   n_exp_depth[g1]++;   n_exp_f_level[f1_bar]++;

   assert(check_dfs_inputs(state->tile_in_location, empty_location, prev_location, h1));
   //prn_forward_dfs_subproblem(tile_in_location, bound1, g1, h1, h2, empty_location, prev_location, 1);

   //if(h1 == 0) {              // This depends on the fact that h1 == 0 only for the goal configuration.
   //   //prn_configuration(tile_in_location);
   //   solved = true;
   //   return(g1);
   //}

   // Create new state and fill in values that will be the same for all subproblems.

   g1_sub = g1 + 1;
   new_state.fill(g1_sub, 0, 1, n_tiles + 1, empty_location, -1, UCHAR_MAX, 0, 2, n_tiles + 1, -1, 0, state->tile_in_location, n_tiles);

   // Generate the subproblems.

   min_bound = UCHAR_MAX;
   stop = moves[empty_location][0];
   for(i = 1,  pnt_move = moves[empty_location]; i <= stop; i++) {
      //new_location = moves[empty_location][i];
      new_location = *(++pnt_move);
      if(new_location != prev_location) {
         n_generated++;   n_generated_forward++;

         // Update the hash value and make the move.

         hash_value_sub = hash_table->update_hash_value(new_state.tile_in_location, empty_location, new_location, hash_value);
         new_state.hash_value = hash_value_sub;
         tile = new_state.tile_in_location[new_location];
         new_state.empty_location = new_location;
         new_state.tile_in_location[empty_location] = tile;
         new_state.tile_in_location[new_location] = 0;
         assert(hash_value_sub == hash_table->hash_configuration(new_state.tile_in_location));

         // Compute the change in h1 and h2 for the subproblem.

         if(dpdb_lb > 0) {
            h1_sub = DPDB->compute_lb(new_state.tile_in_location);
         } else {
            h1_sub = h1 + distances[empty_location][tile] - distances[new_location][tile];
            assert(h1_sub == compute_Manhattan_LB(new_state.tile_in_location));
         }
         new_state.h1 = h1_sub;
         h2_sub = h2 + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];
         new_state.h2 = h2_sub;
         //if(!DPDB->check_lb(distances, moves, tile_in_location, LB_sub)) {
         //   fprintf(stderr, "LB is incorrect\n");
         //   prn_dfs_subproblem(tile_in_location, new_location, empty_location, LB_sub, z + 1);
         //   exit(1);
         //}
         f1_sub = g1_sub + h1_sub;
         f1_bar_sub = 2 * g1_sub + h1_sub - h2_sub;
         //if (f1_bar_sub - f1_bar >= 8) {
         //   printf("Forward: g1 = %3d  h1 = %3d  h1_sub = %3d  h2 = %3d  h2_sub = %3d  f1_bar = %3d  f1_bar_sub = %3d\n", g1, h1, h1_sub, h2, h2_sub, f1_bar, f1_bar_sub);
         //   prn_configuration(state->tile_in_location);   printf("\n");   prn_configuration(new_state.tile_in_location);
         //}
         n_gen_f_level[f1_bar_sub]++;

         // Search for this state if ...

         if ((h2_sub <= h1_sub) && (g1_sub + min_g2[h1_sub][h2_sub] < UB)) {
            status = hash_table->find_bistate(new_state.tile_in_location, hash_value_sub, &hash_index_sub, states);
            if (status == 1) {

               // If a better solution has been found, record it.

               state_index = (*hash_table)[hash_index_sub].state_index;
               if (g1_sub + (*states)[state_index].g2 < UB) {
                  UB = g1_sub + (*states)[state_index].g2;
                  info->best_z = UB;
                  info->best_branch = n_explored;
                  info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
                  //printf("UB = %3d  bound1 = %3d  bound2 = %4d  %4d  n_exp = %14I64d  n_exp_f = %14I64d  n_exp_r = %14I64d  n_gen = %14I64d  n_gen_f = %14I64d  n_gen_r = %14I64d  cpu = %8.2f\n",
                  //        UB, bound1, bound2, 2 * UB - bound1 - bound2, n_explored, n_explored_forward, n_explored_reverse, n_generated, n_generated_forward, n_generated_reverse, info->best_cpu);
                  //LB = ((old_bound1 + old_bound2) / 2) + 1;
                  //LB += (UB - LB) % 2;          // This termination criteria depends on the parity property of the sliding tile puzzle.
                  if (LB >= UB) {
                        solved = true;
                     return(bound1);
                  }
               }
            }
         }

         // Store this state if h1 - h2 <= 0.

         if ((h1_sub <= h2_sub) && (f1_bar_sub >= bound1)) {
            status_index = find_or_insert2(&new_state, 1, states, hash_table, hash_value_sub);
            state_index = status_index.second;
            if(state_index == -1) {
               fprintf(stderr, "Ran out of states. n_exp = %14I64d  n_gen = %14I64d  Press ENTER to continue.\n", n_explored, n_generated);
               cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');    // Keep the console window open.
               exit(1);
            }
            if (g1_sub < min_g1[h1_sub][h2_sub]) min_g1[h1_sub][h2_sub] = g1_sub;
         }

         //if((f1_bar_sub <= bound1) || ((f1_bar_sub <= bound1 + 2)  && (h1_sub <= h2_sub))) {
         //if (f1_bar_sub <= bound1) {
         //   b = forward_dfs(source_location, &new_state, solution, states, hash_table, info, DPDB);
         //} else {
         //   b = f1_bar_sub;
         //}
         if ((f1_bar_sub <= bound1) && (f1_sub < UB)) {
            b = forward_dfs(source_location, &new_state, solution, states, hash_table, info, DPDB);
         } else {
            if (f1_sub >= UB)
               b = UCHAR_MAX;    // Do not use f1_bar_sub in this case because it may equal bound1.
            else
               b = f1_bar_sub;
         }


         // Undo the move.

         new_state.tile_in_location[empty_location] = 0;
         new_state.tile_in_location[new_location] = tile;

         if(solved) {
            //prn_configuration(tile_in_location);   printf("\n");
            solution[g1 + 1] = new_location;
            return(bound1);
         }
         min_bound = min(min_bound, b);
      }
   }

   return(min_bound);
}

//_________________________________________________________________________________________________

unsigned char reverse_dfs(unsigned char *source_location, bistate *state, unsigned char *solution, bistates_array *states, Hash_table *hash_table, searchinfo *info, DPDB *DPDB)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an iterative deepening algorithm.
   2. Input Variables
      a. source_location[t] = location of tile t in the source (initial) configuration.
         The elements of source_location are stored beginning in source_location[0].
      b. state is the state which is to be expanded.
      c. states = store the states.
      d. hash_table = hash table used to find states.
      e. info stores information about the search.
      f. DPDB Disjoint Pattern Database.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i.
      c. n_tiles = number of tiles.
      d. dpdb_lb: 1 = use disjoint LB, 0 = do not use
      e. UB = objective value of best solution found so far.
      f. Also see the static variables defined at the top of this file.
   4. Output Variables
      a. min_bound = minimum bound of subproblems whose lower bound exceeds bound is returned.
      b. solved = true (false) if the goal position was (not) reached during the search.
      c. solution[d] = the location of the empty tile after move d in the optimal solution.
   5. Created 7/14/18 by modifying forward_dfs from this file.
*/
{
   unsigned char  b, empty_location, f2_sub, f2_bar, f2_bar_sub, g2, g2_sub, h1, h1_sub, h2, h2_sub, min_bound, new_location, prev_location, *pnt_move, stop, tile;
   int            hash_index_sub, hash_value, hash_value_sub, i, state_index, status;
   bistate        new_state;
   pair<int, int>    status_index;

   // Store pertinent variables from state to local variables.

   g2 = state->g2;   h1 = state->h1;   h2 = state->h2;
   empty_location = state->empty_location;   prev_location = state->prev_location2;
   hash_value = state->hash_value;

   f2_bar = 2 * g2 + h2 - h1;
   n_explored++;   n_explored_reverse++;   n_exp_depth[g2]++;   n_exp_r_level[f2_bar]++;

   assert(check_dfs_inputs(state->tile_in_location, empty_location, prev_location, h1));
   //prn_reverse_dfs_subproblem(tile_in_location, bound2, g2, h1, h2, empty_location, prev_location, 1);

   //if (h2 == 0) {             // This depends on the fact that h2 == 0 only for the source configuration.
   //   //prn_configuration(tile_in_location);
   //   solved = true;
   //   return(g2);
   //}

   // Create new state and fill in values that will be the same for all subproblems.

   g2_sub = g2 + 1;
   new_state.fill(UCHAR_MAX, 0, 2, n_tiles + 1, n_tiles + 1, -1, g2_sub, 0, 1, empty_location, -1, 0, state->tile_in_location, n_tiles);

   // Generate the subproblems.

   min_bound = UCHAR_MAX;
   stop = moves[empty_location][0];
   g2_sub = g2 + 1;
   for (i = 1, pnt_move = moves[empty_location]; i <= stop; i++) {
      //new_location = moves[empty_location][i];
      new_location = *(++pnt_move);
      if (new_location != prev_location) {
         n_generated++;         n_generated_reverse++;

         // Update the hash value and make the move.

         hash_value_sub = hash_table->update_hash_value(new_state.tile_in_location, empty_location, new_location, hash_value);
         new_state.hash_value = hash_value_sub;
         tile = new_state.tile_in_location[new_location];
         new_state.empty_location = new_location;
         new_state.tile_in_location[empty_location] = tile;
         new_state.tile_in_location[new_location] = 0;
         assert(hash_value_sub == hash_table->hash_configuration(new_state.tile_in_location));

         // Compute the change in h1 and h2 for the subproblem.

         if (dpdb_lb > 0) {
            h1_sub = DPDB->compute_lb(new_state.tile_in_location);
         } else {
            h1_sub = h1 + distances[empty_location][tile] - distances[new_location][tile];
            assert(h1_sub == compute_Manhattan_LB(new_state.tile_in_location));
         }
         new_state.h1 = h1_sub;
         h2_sub = h2 + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];
         new_state.h2 = h2_sub;
         //if(!DPDB->check_lb(distances, moves, tile_in_location, LB_sub)) {
         //   fprintf(stderr, "LB is incorrect\n");
         //   prn_dfs_subproblem(tile_in_location, new_location, empty_location, LB_sub, z + 1);
         //   exit(1);
         //}
         f2_sub = g2_sub + h2_sub;
         f2_bar_sub = 2 * g2_sub + h2_sub - h1_sub;
         n_gen_r_level[f2_bar_sub]++;

         // Search for this state if ...

         if ((h1_sub <= h2_sub) && (min_g1[h1_sub][h2_sub] + g2_sub < UB)) {
            status = hash_table->find_bistate(new_state.tile_in_location, hash_value_sub, &hash_index_sub, states);
            if (status == 1) {

               // If a better solution has been found, record it.

               state_index = (*hash_table)[hash_index_sub].state_index;
               if ((*states)[state_index].g1 + g2_sub < UB) {
                  UB = (*states)[state_index].g1 + g2_sub;
                  info->best_z = UB;
                  info->best_branch = n_explored;
                  info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
                  //printf("UB = %3d  bound1 = %3d  bound2 = %4d  %4d  n_exp = %14I64d  n_exp_f = %14I64d  n_exp_r = %14I64d  n_gen = %14I64d  n_gen_f = %14I64d  n_gen_r = %14I64d  cpu = %8.2f\n",
                  //        UB, bound1, bound2, 2 * UB - bound1 - bound2, n_explored, n_explored_forward, n_explored_reverse, n_generated, n_generated_forward, n_generated_reverse, info->best_cpu);

                  //LB = ((old_bound1 + old_bound2) / 2) + 1;
                  //LB += (UB - LB) % 2;          // This termination criteria depends on the parity property of the sliding tile puzzle.
                  if (LB >= UB) {
                     solved = true;
                     return(bound2);
                  }
               }
            }
         }

         // Store this state if h2 - h1 <= 0.

         if ((h2_sub <= h1_sub) && (f2_bar_sub >= bound2)) {
            status_index = find_or_insert2(&new_state, 2, states, hash_table, hash_value_sub);
            state_index = status_index.second;
            if (state_index == -1) {
               fprintf(stderr, "Ran out of states. n_exp = %14I64d  n_gen = %14I64d  Press ENTER to continue.\n", n_explored, n_generated);
               cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');    // Keep the console window open.
               exit(1);
            }
            if (g2_sub < min_g2[h1_sub][h2_sub]) min_g2[h1_sub][h2_sub] = g2_sub;
         }

         //if ((f2_bar_sub <= bound2) || ((f2_bar_sub <= bound2 + 2) && (h2_sub <= h1_sub))) {
         //if (f2_bar_sub <= bound2) {
         //   b = reverse_dfs(source_location, &new_state, solution, states, hash_table, info, DPDB);
         //}  else {
         //   b = f2_bar_sub;
         //}
         if ((f2_bar_sub <= bound2) && (f2_sub < UB)) {
            b = reverse_dfs(source_location, &new_state, solution, states, hash_table, info, DPDB);
         } else {
            if (f2_sub >= UB)
               b = UCHAR_MAX;    // Do not use f2_bar_sub in this case because it may equal bound2.
            else
               b = f2_bar_sub;
         }

         // Undo the move.

         new_state.tile_in_location[empty_location] = 0;
         new_state.tile_in_location[new_location] = tile;

         if (solved) {
            //prn_configuration(tile_in_location);   printf("\n");
            solution[g2 + 1] = new_location;
            return(bound2);
         }
         min_bound = min(min_bound, b);
      }
   }

   return(min_bound);
}

//_________________________________________________________________________________________________

void search_states(bistates_array *states, unsigned char bound)
/*
   1. This function searches through the list of states to see if there are any with g1 + g2 <= bound.
   2. Input Variables
      a. states = store the states.
      b. bound = search for states with g1 + g2 <= bound.
   3. Global Variables
   4. Output Variables
   5. Written 7/15/18.
*/
{
   int               i;
   unsigned char     g1, g2, h1, h2;
   //unsigned char     tile_in_location[16] = { 4,  6,  7,  0,  5,  9, 14,  1,  8, 10, 15,  2, 12, 13, 11,  3 };
   unsigned char     tile_in_location[16] = { 4,  2, 11,  6,  0,  5,  1, 15, 8, 13,  3,  7, 12, 10,  9, 14};

   for (i = 0; i <= (*states).n_of_states() - 1; i++) {
      g1 = (*states)[i].g1;   h1 = (*states)[i].h1;   g2 = (*states)[i].g2;   h2 = (*states)[i].h2;
      //if (g1 + g2 <= bound) {
      //if ((30 <= g1) && (g1 <= bound)) {
      if (memcmp((*states)[i].tile_in_location, tile_in_location, N_LOCATIONS) == 0) {
         printf("%3d %3d %3d %3d %3d %3d %3d\n", g1, h1, g2, h2, g1 + g2, h1 - h2, 2*g1 + h1 - h2);
      }
   }
}
