#include "main.h"
#include <queue>

unsigned char           goal[16] = {0, 1, 2, 3, 4, 5,  6, 7,  8,  9, 10,  11, 12, 13, 14,  15};   // Goal configuration

static double           f1_bar_min;       // = min {f1_bar(v): v is in open set of nodes in the forward direction}.
static double           f2_bar_min;       // = min {f2_bar(v): v is in open set of nodes in the reverse direction}.
static __int64          n_exp_f_level[MAX_DEPTH + 1]; // n_exp_f_level[a] = # of states expanded in forward direction with f1_bar = a.
static __int64          n_exp_r_level[MAX_DEPTH + 1]; // n_exp_r_level[a] = # of states expanded in reverse direction with f2_bar = a.
static __int64          n_gen_f_level[MAX_DEPTH + 1]; // n_gen_f_level[a] = # of states generated in forward direction with f1_bar = a.
static __int64          n_gen_r_level[MAX_DEPTH + 1]; // n_gen_r_level[a] = # of states genarated in reverse direction with f2_bar = a.
                                          
//static unsigned char    UB;               // = objective value of best solution found so far.

/*************************************************************************************************/

unsigned char bidirectional(unsigned char source[N_LOCATIONS], searchparameters *parameters, searchinfo *info, DPDB *DPDB)
/*
   These functions implement a bidirectional search for the 15-puzzle.
   1. Algorithm Description
      a. It uses bidirectional branch and bound to attempt to minimize the objective function.
      b. The depth of a subproblem equals the number of moves that have been made from the source or goal configuration.
      c. direction = 1 for forward direction
                   = 2 for reverse direction
   2. Conceptually, a subproblem consists of a configuration of the tiles plus the moves that were made to reach this configuration.  
      To reduce memory requirements, do not store the moves that were made for each subproblem.  
      Instead, use pointers to reconstruct the moves.  The same data structure (class) is used to store both forward and reverse subproblems.
      A subproblem consists of:
      a. g1 = number of moves that have been made so far in the forward direction
      b. h1 = lower bound on the number of moves needed to reach the goal postion
      c. open1 = 2 if this subproblem has not yet been generated in the forward direction
               = 1 if this subproblem is open in the forward direction
               = 0 if this subproblem closed in the forward direction
      d. empty_location = location of the empty tile
      e. prev_location1 = location of the empty tile in the parent of this subproblem in the forward direction
      f. parent1 = index of the parent subproblem in the forward direction
      g. g2 = number of moves that have been made so far in the reverse direction
      h. h2 = lower bound on the number of moves needed to reach the source postion
      i. open2 = 2 if this subproblem has not yet been generated in the reverse direction
               = 1 if this subproblem is open in the reverse direction
               = 0 if this subproblem closed in the reverse direction
      j. prev_location2 = location of the empty tile in the parent of this subproblem in the reverse direction
      k. parent2 = index of the parent subproblem in the reverse direction
      l. tile_in_location[i] = the tile that is in location i
   3. Input Variables
      a. source[i] = the tile that is in location i.
         The elements of source are stored beginning in source[0].
      b. parameters
         algorithm;        -a option: algorithm
                           1 = depth first search
                           2 = breadth first search
                           3 = best first search
                           4 = cyclic best first search
                           5 = cyclic best first search using min_max_stacks 
                           6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps
						  Note: Only best first search has been implemented for bidirectional search.
         best_measure:     -b option: best_measure
                           1 = LB = g + h
                           2 = g + 1.5 h (AWA*)
                           3 = g - h2
                           4 = best = z + LB - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                           5 = -z - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
         cpu_limit:        cpu limit for search process
         dpdb_lb;          -d option: 1 = use disjoint LB, 0 = do not use (def = 0)
         gen_skip;         -g option: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0).
         prn_info:         Controls level of printed info (def=0)
   4. Global Variables
      a. n_tiles =  number of tiles.
      b. UB = objective value of the best solution found so far.
      c. distances[i][j] = Manhattan distance between location i and location j.
      d. moves[i] = list of possible ways to move the empty tile from location i.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_bar_min = min {f1_bar(v): v is in open set of nodes in the forward direction}.
      i. f2_bar_min = min {f2_bar(v): v is in open set of nodes in the reverse direction}.
   5. Output Variables
      a. best_z = the best objective value that was found.
      b. info is used to collect information about the search.
         best_solution[d] = the location of the empty tile after move d in the best solution that was found.
   6. Written 2/5/15.
      a. Restarted work on this code on 6/30/18.
*/
{
   unsigned char  empty_location, h1, h2, *source_location;
   int            direction, dominance, forward_index, reverse_index, status, hash_index, i;
   __int64        cnt;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL;
   double         best, cpu, prev_min;
   Hash_table     hash_table;
   min_max_stacks cbfs_stacks;
   bistate        goal_state, source_state;			// states for the source and the goal.
   bistates_array states;
   pair<int, int> status_index;

   assert(check_tile_in_location(source));

   cnt = 0;

   // Find the location of the empty tile.

   for(i = 0; i <= n_tiles; i++) {
      if(source[i] == 0) {
         empty_location = i;
         break;
      }
   }
   info->best_solution[0] = empty_location;

   // Determine the locations of the tiles in the source configuration.  
   // source_location[t] = location of tile t in the source (initial) configuration.
   // The elements of source_location are stored beginning in source_location[0].


   source_location = new unsigned char[n_tiles + 1];
   for (i = 0; i <= n_tiles; i++) source_location[source[i]] = i;

   // Initialize data structures.

   info->initialize();
   initialize_bisearch(parameters, &states, &cbfs_stacks, &hash_table);
   //UB = MAX_DEPTH;
   for (i = 0; i <= MAX_DEPTH; i++) {n_exp_f_level[i] = 0; n_exp_r_level[i] = 0; n_gen_f_level[i] = 0; n_gen_r_level[i] = 0; }


   // Create the root problem in the forward direction.
   // Compute h1 and h2.

   if(dpdb_lb > 0) {
      h1 = DPDB->compute_lb(source);
   } else {
      h1 = compute_Manhattan_LB(source);
   }
   h2 = 0;
   info->root_LB = h1;

   source_state.g1 = 0;                            // = number of moves that have been made so far in the forward direction
   source_state.h1 = h1;                           // = lower bound on the number of moves needed to reach the goal postion
   source_state.open1 = 1;                         // = 2 if this subproblem has not yet been generated in the forward direction
                                                   // = 1 if this subproblem is open in the forward direction
                                                   // = 0 if this subproblem closed in the forward direction
   source_state.empty_location = empty_location;   // = location of the empty tile
   source_state.prev_location1 = n_tiles + 1;      // = location of the empty tile in the parent of this subproblem in the forward direction
   source_state.parent1 = -1;                      // = index of the parent subproblem in the forward direction
   source_state.g2 = UCHAR_MAX;                    // = number of moves that have been made so far in the reverse direction
   source_state.h2 = h2;                           // = lower bound on the number of moves needed to reach the source postion
   source_state.open2 = 2;                         // = 2 if this subproblem has not yet been generated in the reverse direction
                                                   // = 1 if this subproblem is open in the reverse direction
                                                   // = 0 if this subproblem closed in the reverse direction
   source_state.prev_location2 = n_tiles + 1;      // = location of the empty tile in the parent of this subproblem in the reverse direction
   source_state.parent2 = -1;                      // = index of the parent subproblem in the reverse direction
   for(i = 0; i <= n_tiles; i++) source_state.tile_in_location[i] = source[i];  // tile_in_location[i] = the tile that is in location i
   source_state.hash_value = hash_table.hash_configuration(source_state.tile_in_location);

   best = 2 * source_state.g1 + source_state.h1 - source_state.h2;
   f1_bar_min = 2 * source_state.g1 + source_state.h1 - source_state.h2;

   // Add the forward root problem to the list of states and the set of unexplored states.

   status_index = find_or_insert(&source_state, best, 0, 1, &states, parameters, info, &cbfs_stacks, &hash_table, source_state.hash_value);

   // Create the root problem in the reverse direction.
   // Compute h1 and h2.

   h1 = 0;
   h2 = compute_Manhattan_LB(source);          // This relies on the fact that the Manhattan distance from the source to the goal is the same as from the goal to the source

   goal_state.g1 = UCHAR_MAX;                   // = number of moves that have been made so far in the forward direction
   goal_state.h1 = h1;                          // = lower bound on the number of moves needed to reach the goal postion
   goal_state.open1 = 2;                        // = 2 if this subproblem has not yet been generated in the forward direction
                                                // = 1 if this subproblem is open in the forward direction
                                                // = 0 if this subproblem closed in the forward direction
   goal_state.empty_location = 0;               // = location of the empty tile
   goal_state.prev_location1 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the forward direction
   goal_state.parent1 = -1;                     // = index of the parent subproblem in the forward direction
   goal_state.g2 = 0;                           // = number of moves that have been made so far in the reverse direction
   goal_state.h2 = h2;                          // = lower bound on the number of moves needed to reach the source postion
   goal_state.open2 = 1;                        // = 2 if this subproblem has not yet been generated in the reverse direction
                                                // = 1 if this subproblem is open in the reverse direction
                                                // = 0 if this subproblem closed in the reverse direction
   goal_state.prev_location2 = n_tiles + 1;     // = location of the empty tile in the parent of this subproblem in the reverse direction
   goal_state.parent2 = -1;                     // = index of the parent subproblem in the reverse direction
   for(i = 0; i <= n_tiles; i++) goal_state.tile_in_location[i] = i;  // tile_in_location[i] = the tile that is in location i
   goal_state.hash_value = hash_table.hash_configuration(goal_state.tile_in_location);

   best = 2 * goal_state.g2 + goal_state.h2 - goal_state.h1;
   f2_bar_min = 2 * goal_state.g2 + goal_state.h2 - goal_state.h1;

   // Add the reverse root problem to the list of states and the set of unexplored states.

   status_index = find_or_insert(&goal_state, best, 0, 2, &states, parameters, info, &cbfs_stacks, &hash_table, goal_state.hash_value);

   // Main loop

   direction = -1;
   prev_min = min(f1_bar_min, f2_bar_min) - 1;
   //while ((UB > ceil((f1_bar_min + f2_bar_min) / 2)) && (info->optimal >= 0)) {
   while ((2*UB > f1_bar_min + f2_bar_min + 2) && (info->optimal >= 0)) {        // This termination criteria depends on the parity of MD.
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if (cpu > CPU_LIMIT) {
         info->optimal = 0;
         break;
      }

      direction = choose_direction(direction, parameters, info);        // Choose the direction.

      if (direction == 1) {
         status = expand_forward(source, source_location, &states, parameters, info, &cbfs_stacks, DPDB, &hash_table);
         if(status == -1) {
            info->optimal = 0;
            break;
         }
      } else {
         status = expand_reverse(source, source_location, &states, parameters, info, &cbfs_stacks, DPDB, &hash_table);
         if (status == -1) {
            info->optimal = 0;
            break;
         }
      }
   }

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;

   info->n_explored = info->n_explored_forward + info->n_explored_reverse;
   info->n_generated = info->n_generated_forward + info->n_generated_reverse;

   if (prn_info == 0)  {
      printf("%3d %8.2f %8.2f %10I64d %9I64d %10I64d %10I64d %10I64d %10I64d %10I64d %10I64d %2d\n",
             info->best_z, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, 
             info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored(),
             info->optimal);
   }
   if (prn_info > 0)  {
      printf("z = %3d cpu = %8.2f best_cpu = %8.2f best_branch = %9I64d n_exp = %9I64d n_exp_f = %9I64d n_exp_r = %9I64d n_gen = %9I64d n_gen_f = %9I64d n_gen_r = %9I64d n_stored = = %9I64d optimal = %2d states_cpu = %8.2f %9I64d\n",
         info->best_z, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored(), info->optimal, info->states_cpu, info->cnt);
      //for(i = 0; i <= UB; i++) printf("%3d %14I64d\n", i, info->n_explored_depth[i]);
      //cbfs_stacks.print_stats();
   }
   if(prn_info > 2) prn_solution(source, info->best_solution, info->best_z, DPDB);
   //printf("UB = %2d  f1_bar_min = %4.1f  f2_bar_min = %4.1f  %4.0f\n", UB, f1_bar_min, f2_bar_min, 2 * UB - f1_bar_min - f2_bar_min);
   //analyze_states(&states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h);

   //free_bimemory();

   return(info->best_z);
}

//_________________________________________________________________________________________________

int expand_forward(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  g1, g2, h1, h2;
   int            index, status;
   bistate        *state;
   heap_record    item;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(1, info, cbfs_stacks);
   while ((index != -1) && ((*states)[index].open1 == 0)) {
      index = get_bistate(1, info, cbfs_stacks);
   }
   if (index == -1) {
      f1_bar_min = UCHAR_MAX;
      return(1);
   }  else {
      (*states)[index].open1 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   assert(check_bistate(source, state, 1));

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   if (g1 + h1 >= UB) return(1);        // Could apply additional pruning methods here(see summary.pdf), but it does seem likely that they will help
                                        // much for the pancake problem because the GAP LB is so strong.  g1 - h2 + f2_min >= UB and f1 + min{ g2(w) - h1(w) : w in C2_star }

   if(prn_info > 2) {
      prn_bi_subproblem(state, 1, UB, info, 1);
      //if(prn_info > 3) {prn_configuration(state->tile_in_location); printf("\n");}
   }
   if ((prn_info > 1) && (info->n_explored_forward + info->n_explored_reverse) % 10000 == 0) printf("UB = %3d  f1_bar_min = %4.1f  f2_bar_min = %4.1f  n_exp = %12I64d n_exp_f = %12I64d n_exp_r = %12I64d n_gen_f = %12I64d n_gen_r = %12I64d\n", UB, f1_bar_min, f2_bar_min, info->n_explored_forward + info->n_explored_reverse, info->n_explored_forward, info->n_explored_reverse, info->n_generated_forward, info->n_generated_reverse);

   if (state->open2 > 0) {
      info->n_explored_forward++;
      info->n_explored_depth[g1]++;
      n_exp_f_level[2*g1 + h1 - h2]++;

      // Generate all the subproblems from this state.

      status = gen_forward_subproblems(source, source_location, states, index, parameters, info, cbfs_stacks, DPDB, hash_table);
   }  else {
      fprintf(stderr, "Exploring a node that is closed in the opposite direction.\n"); info->optimal = 0; return(-1);
   }

   item = forward_bfs_heap.get_min();
   if (item.key == -1)
      f1_bar_min = UCHAR_MAX;
   else
      f1_bar_min = item.key;
   f1_bar_min = ceil(f1_bar_min);

   return(1);
}

//_________________________________________________________________________________________________

int gen_forward_subproblems(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  empty_location, f1_sub, f1_bar_sub, g1, g2, h1, h2, new_location, *pnt_move, prev_location, stop, tile;
   int            i, index_sub, hash_index, hash_value, hash_value_sub, state_index, status;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL;
   double         best, cpu;
   bistate        *existing_state, new_state, *state;
   pair<int, int>    status_index;

   state = &(*states)[index];

   // Store pertinent variables from state to local variables.

   empty_location = state->empty_location;
   prev_location = state->prev_location1;
   g1 = state->g1;
   g2 = state->g2;
   h1 = state->h1;
   h2 = state->h2;
   hash_value = state->hash_value;

   // Create new state and fill in values that will be the same for all subproblems.

   new_state.g1 = state->g1 + 1;
   new_state.open1 = 1;
   new_state.prev_location1 = empty_location;
   new_state.parent1 = index;
   //for(i = 0; i <= n_tiles; i++) new_state.tile_in_location[i] = state->tile_in_location[i];
   memcpy(new_state.tile_in_location, state->tile_in_location, n_tiles + 1);
   new_state.g2 = UCHAR_MAX;                          
   new_state.open2 = 2;                        
   new_state.prev_location2 = n_tiles + 1;     
   new_state.parent2 = -1;                     

   // Generate all the subproblems from this state.

   switch(parameters->gen_skip) {
      case 0:
         stop = moves[empty_location][0];
         for(i = 1, pnt_move = moves[empty_location]; i <= stop; i++) {
            //new_location = moves[empty_location][i];
            new_location = *(++pnt_move);
            if(new_location != prev_location) {
               info->n_generated_forward++;

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
                  new_state.h1 = DPDB->compute_lb(new_state.tile_in_location);
               } else {
                  new_state.h1 = h1 + distances[empty_location][tile] - distances[new_location][tile];
                  assert(new_state.h1 == compute_Manhattan_LB(new_state.tile_in_location));
               }
               new_state.h2 = h2 + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];
               assert(new_state.h2 == compute_Manhattan_LB2(source, new_state.tile_in_location));

               f1_sub = new_state.g1 + new_state.h1;
               f1_bar_sub = 2 * new_state.g1 + new_state.h1 - new_state.h2;
               n_gen_f_level[f1_bar_sub]++;

               if ((f1_sub < UB) && (f1_bar_sub + f2_bar_min + 2 < 2 * UB)) {
                  best = compute_bibest(1, new_state.g1, new_state.g2, new_state.h1, new_state.h2, parameters);
                  status_index = find_or_insert(&new_state, best, new_state.g1, 1, states, parameters, info, cbfs_stacks, hash_table, new_state.hash_value);
                  state_index = status_index.second;

                  // If there was insufficient memory for the new state, then return -1.
                  if (state_index == -1) {
                     return(-1);
                  }

                  if (prn_info > 3) prn_bi_subproblem2(&new_state, 1, UB, info, 1);

                  // If a better solution has been found, record it.

                  if ((*states)[state_index].g1 + (*states)[state_index].g2 < UB) {
                     UB = (*states)[state_index].g1 + (*states)[state_index].g2;
                     info->best_z = UB;
                     info->best_branch = info->n_explored_forward + info->n_explored_reverse;
                     info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
                     status = bibacktrack(source, states, state_index, info->best_solution);
                     if (status == -1) {
                        return(-1);
                     }
                     //printf("UB = %2d  f1_bar_min = %4.1f  f2_bar_min = %4.1f  %4.0f\n", UB, f1_bar_min, f2_bar_min, 2 * UB - f1_bar_min - f2_bar_min);
                     //analyze_states(states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h);
                  }
               }

               // Undo the move.

               new_state.tile_in_location[empty_location] = 0;
               new_state.tile_in_location[new_location] = tile;
            }
         }
         break;
      default:
         info->n_explored--;                    // Decrement n_explored to avoid double counting.
         info->n_explored_depth[state->g1]--;
         //gen_dfs(source, state->z + state->LB + gen_skip, empty_location, prev_location, state->LB, state->z, states, parameters, info, cbfs_stacks, &new_state, DPDB, hash_table, hash_value);
         fprintf(stderr, "gen_skip and gen_dfs not yet implemented for bidirectiona search\n");
         exit(1);
         break;
   }
   return(1);
}

//_________________________________________________________________________________________________

int expand_reverse(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  g1, g2, h1, h2;
   int            index, status;
   bistate        *state;
   heap_record    item;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(2, info, cbfs_stacks);
   while ((index != -1) && ((*states)[index].open2 == 0)) {
      index = get_bistate(2, info, cbfs_stacks);
   }
   if (index == -1) {
      f2_bar_min = UCHAR_MAX;
      return(1);
   }
   else {
      (*states)[index].open2 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   assert(check_bistate(source, state, 1));

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   if (g2 + h2 >= UB) return(1);        // Could apply additional pruning methods here(see summary.pdf), but it does seem likely that they will help
                                        // much for the pancake problem because the GAP LB is so strong.  g1 - h2 + f2_min >= UB and f1 + min{ g2(w) - h1(w) : w in C2_star }

   if (prn_info > 2) {
      prn_bi_subproblem(state, 2, UB, info, 1);
      //if(prn_info > 3) {prn_configuration(state->tile_in_location); printf("\n");}
   }
   if ((prn_info > 1) && (info->n_explored_forward + info->n_explored_reverse) % 10000 == 0) printf("UB = %3d  f1_bar_min = %4.1f  f2_bar_min = %4.1f  n_exp = %12I64d n_exp_f = %12I64d n_exp_r = %12I64d n_gen_f = %12I64d n_gen_r = %12I64d\n", UB, f1_bar_min, f2_bar_min, info->n_explored_forward + info->n_explored_reverse, info->n_explored_forward, info->n_explored_reverse, info->n_generated_forward, info->n_generated_reverse);

   if (state->open1 > 0) {
      info->n_explored_reverse++;
      info->n_explored_depth[g2]++;
      n_exp_r_level[2 * g2 + h2 - h1]++;

      // Generate all the subproblems from this state.

      status = gen_reverse_subproblems(source, source_location, states, index, parameters, info, cbfs_stacks, DPDB, hash_table);
   }  else {
      fprintf(stderr, "Exploring a node that is closed in the opposite direction.\n"); info->optimal = 0; return(-1);
   }

   item = reverse_bfs_heap.get_min();
   if (item.key == -1)
      f2_bar_min = UCHAR_MAX;
   else
      f2_bar_min = item.key;
   f2_bar_min = ceil(f2_bar_min);

   return(1);
}

//_________________________________________________________________________________________________

int gen_reverse_subproblems(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  empty_location, f2_sub, f2_bar_sub, g1, g2, h1, h2, new_location, *pnt_move, prev_location, stop, tile;
   int            i, index_sub, hash_index, hash_value, hash_value_sub, state_index, status;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL;
   double         best, cpu;
   bistate        *existing_state, new_state, *state;
   pair<int, int>    status_index;

   state = &(*states)[index];

   // Store pertinent variables from state to local variables.

   empty_location = state->empty_location;
   prev_location = state->prev_location2;
   g1 = state->g1;
   g2 = state->g2;
   h1 = state->h1;
   h2 = state->h2;
   hash_value = state->hash_value;

   // Create new state and fill in values that will be the same for all subproblems.

   new_state.g1 = UCHAR_MAX;
   new_state.open1 = 2;
   new_state.prev_location1 = n_tiles + 1;
   new_state.parent1 = -1;
   //for(i = 0; i <= n_tiles; i++) new_state.tile_in_location[i] = state->tile_in_location[i];
   memcpy(new_state.tile_in_location, state->tile_in_location, n_tiles + 1);
   new_state.g2 = state->g2 + 1;
   new_state.open2 = 1;
   new_state.prev_location2 = empty_location;
   new_state.parent2 = index;

   // Generate all the subproblems from this state.

   switch (parameters->gen_skip) {
   case 0:
      stop = moves[empty_location][0];
      for (i = 1, pnt_move = moves[empty_location]; i <= stop; i++) {
         //new_location = moves[empty_location][i];
         new_location = *(++pnt_move);
         if (new_location != prev_location) {
            info->n_generated_reverse++;

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
               new_state.h1 = DPDB->compute_lb(new_state.tile_in_location);
            }
            else {
               new_state.h1 = h1 + distances[empty_location][tile] - distances[new_location][tile];
               assert(new_state.h1 == compute_Manhattan_LB(new_state.tile_in_location));
            }
            new_state.h2 = h2 + distances[empty_location][source_location[tile]] - distances[new_location][source_location[tile]];
            assert(new_state.h2 == compute_Manhattan_LB2(source, new_state.tile_in_location));

            f2_sub = new_state.g2 + new_state.h2;
            f2_bar_sub = 2 * new_state.g2 + new_state.h2 - new_state.h1;
            n_gen_r_level[f2_bar_sub]++;

            if ((f2_sub < UB) && (f1_bar_min + f2_bar_sub + 2 < 2*UB)) {
               best = compute_bibest(2, new_state.g1, new_state.g2, new_state.h1, new_state.h2, parameters);
               status_index = find_or_insert(&new_state, best, new_state.g1, 2, states, parameters, info, cbfs_stacks, hash_table, new_state.hash_value);
               state_index = status_index.second;

               // If there was insufficient memory for the new state, then return -1.
               if (state_index == -1) {
                  return(-1);
               }

               if (prn_info > 3) prn_bi_subproblem2(&new_state, 2, UB, info, 1);

               // If a better solution has been found, record it.

               if ((*states)[state_index].g1 + (*states)[state_index].g2 < UB) {
                  UB = (*states)[state_index].g1 + (*states)[state_index].g2;
                  info->best_z = UB;
                  info->best_branch = info->n_explored_forward + info->n_explored_reverse;
                  info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
                  status = bibacktrack(source, states, state_index, info->best_solution);
                  if (status == -1) {
                     return(-1);
                  }
                  //printf("UB = %2d  f1_bar_min = %4.1f  f2_bar_min = %4.1f  %4.0f\n", UB, f1_bar_min, f2_bar_min, 2 * UB - f1_bar_min - f2_bar_min);
                  //analyze_states(states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h);
               }
            }
            // Undo the move.

            new_state.tile_in_location[empty_location] = 0;
            new_state.tile_in_location[new_location] = tile;
         }
      }
      break;
   default:
      info->n_explored--;                    // Decrement n_explored to avoid double counting.
      info->n_explored_depth[state->g1]--;
      //gen_dfs(source, state->z + state->LB + gen_skip, empty_location, prev_location, state->LB, state->z, states, parameters, info, cbfs_stacks, &new_state, DPDB, hash_table, hash_value);
      fprintf(stderr, "gen_skip and gen_dfs not yet implemented for bidirectiona search\n");
      exit(1);
      break;
   }
   return(1);
}

//_________________________________________________________________________________________________

double compute_bibest(int direction, unsigned char g1, unsigned char g2, unsigned char h1, unsigned char h2, searchparameters *parameters)
/*
   1. This function computes the best measure for a configuration.
   2. Input Variables
      a. direction = 1 to compute best in the forward direction
                   = 2 to compute best in the reverse direction.
      b. g1 = number of moves that have been made so far in the forward direction.
      c. g1 = number of moves that have been made so far in the forward direction.
      d. h1 = lower bound on the number of moves needed to reach the goal postion.
      e. h2 = lower bound on the number of moves needed to reach the source postion.
      f. parameters contains the search parameters.
   3. Output Variables
      a. best = the best measure for this assignment.
   4. Written 1/25/17.
*/
{
   unsigned char  max_n_moves, MD_to_source;
   double         best;

   switch(parameters->best_measure) {		
		case 1:  // best = f_d = g_d + h_d
         if (direction == 1) {
            best = g1 + h1;
         }
         else {
            best = g2 + h2;
         }
			break;
		case 6:  // best = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'.
         if (direction == 1) {
            best = 2 * g1 + h1 - h2;
         } else {
            best = 2 * g2 + h2 - h1;
         }
			break;
      case 7:  // best = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
         if (direction == 1) {
            best = g1 + h1 - (double)g1 / (double)(MAX_DEPTH + 1);
         }
         else {
            best = g2 + h2 - (double)g2 / (double)(MAX_DEPTH + 1);
         }
         break;
      case 8:  // best = f_bar_d - (g_d /(MAX_DEPTH + 1) Break ties in fbar_d in favor of states with larger g_d.
         if (direction == 1) {
            best = 2 * g1 + h1 - h2 - (double)g1 / (double)(MAX_DEPTH + 1);
         }
         else {
            best = 2 * g2 + h2 - h1 - (double)g2 / (double)(MAX_DEPTH + 1);
         }
         break;
      default:
         fprintf(stderr,"Unknown best measure\n"); 
         exit(1); 
         break;
	}

   return(best);
}

//_________________________________________________________________________________________________

int choose_direction(int prev_direction, searchparameters *parameters, searchinfo *info)
/*
   1. This function chooses the direction for the next state to be expanded.
   2. Input Variables
   3. Global Variables
      a. f1_bar_min = min {f1_bar(v): v is in open set of nodes in the forward direction}.
      b. f2_bar_min = min {f2_bar(v): v is in open set of nodes in the reverse direction}.
      c. forward_bfs_heap = heap for the forward best first search.
      d. reverse_bfs_heap = heap for the reverse best first search.
      e. n_exp_f_level[a] = # of states expanded in forward direction with f1_bar = a.
      f. n_exp_r_level[a] = # of states expanded in reverse direction with f2_bar = a.
      g. n_gen_f_level[a] = # of states generated in forward direction with f1_bar = a.
      h. n_gen_r_level[a] = # of states genarated in reverse direction with f2_bar = a.      
      i. UB = objective value of the best solution found so far.
   4. Output Variables
      a. direction = 1 = forward direction
                   = 2 = reverse direction.
   5. Written 7/11/18.
*/
{
   unsigned char  max_n_moves, MD_to_source;
   int            direction;
   double         best;
   static double  prev_min1 = -1, prev_min2 = -1;     // prev_min1 = f1_bar_min in previous iteration.

   if (prev_direction == -1) { prev_min1 = -1; prev_min2 = -1;}     // This is necessary if the bidirectional algorithm is run multiple time.

   switch (parameters->direction_rule) {
      case 0:  // open cardinality rule: |O1| <= |O2| => forward
         if (forward_bfs_heap.n_of_items() <= reverse_bfs_heap.n_of_items()) direction = 1; else direction = 2;
         break;
      case 1:  // closed cardinality rule: |C1| <= |C2| => forward
         if (info->n_explored_forward <= info->n_explored_reverse) direction = 1; else direction = 2; 
         break;
      case 2:  // Best First Direction (BFD): f1_bar_min <= f2_bar_min => forward
         if (f1_bar_min <= f2_bar_min) direction = 1; else direction = 2;
         break;
      case 3:  // Best First Direction (BFD) with fbar_leveling.  Break ties using |O1| <= |O2| => forward
         if (f1_bar_min < f2_bar_min) {
            direction = 1;
         } else {
            if (f1_bar_min > f2_bar_min) {
               direction = 2;
            } else {
               if ((f1_bar_min > prev_min1) || (f2_bar_min > prev_min2)) {
                  if (forward_bfs_heap.n_of_items() <= reverse_bfs_heap.n_of_items()) {
                     direction = 1;
                  } else {
                     direction = 2;
                  }
                  prev_min1 = f1_bar_min;
                  prev_min2 = f2_bar_min;
               } else {
                  direction = prev_direction;
               }
            }
         }
         break;
      case 4:  // open cardinality rule with fbar_leveling. |O1| <= |O2| = > forward
         if ((f1_bar_min > prev_min1) || (f2_bar_min > prev_min2)) {
            if (forward_bfs_heap.n_of_items() <= reverse_bfs_heap.n_of_items()) direction = 1; else direction = 2;
         } else {
            direction = prev_direction;
         }
         prev_min1 = f1_bar_min;
         prev_min2 = f2_bar_min;
         break;
      default:
         fprintf(stderr, "Unknown direction rule\n");
         exit(1);
         break;
   }

   return(direction);
}

//_________________________________________________________________________________________________

int bibacktrack(unsigned char source[N_LOCATIONS], bistates_array *states, int index, unsigned char solution[MAX_DEPTH + 1])
/*
   1. BIBACKTRACK constructs a solution by backtracking through the states.
   2. Input Variables
      a. source[i] = the tile that is in location i.
         The elements of source are stored beginning in source[0].
      b. states = array where the states are stored.
      c. index = the index of the state (in states) from which to begin backtracking.
   3. Global Variables
   4. Output Variables
      a. The number of moves is returned.
         -1 is returned if an error occurs.
      b. solution = array containing the moves that were made in this solution.
   5. WARNING: This function will only work if all the states are stored.  If states are
      skipped in order to save space, then this function will not work.
   6. Written 1/25/17.
*/
{
   unsigned char  empty_location;
   int            d, depth, n_moves1, n_moves2, parent, original_index, status;

   original_index = index;

   // Backtrack from the state to the source.

   depth = (*states)[index].g1;                           assert((0 <= depth) && (depth <= MAX_DEPTH + 1));
   d = depth;
   n_moves1 = 0;
   while(index >= 0) {
      parent = (*states)[index].parent1;
      //prn_configuration((*states)[index].tile_in_location);
      empty_location = (*states)[index].empty_location;  assert((0 <= empty_location) && (empty_location <= N_LOCATIONS));
      solution[d] = empty_location;
      if(parent >= 0) n_moves1++;
      d--;                                               assert(-1 <= d);
      index = parent;
   }
   assert(depth == n_moves1);
   assert(n_moves1 == (*states)[original_index].g1);

   // Backtrack from the state to the goal.

   depth = (*states)[original_index].g2;                           assert((0 <= depth) && (depth <= MAX_DEPTH + 1));
   index = (*states)[original_index].parent2;
   d = n_moves1 + 1;
   n_moves2 = 0;
   while (index >= 0) {
      parent = (*states)[index].parent2;
      empty_location = (*states)[index].empty_location;  assert((0 <= empty_location) && (empty_location <= N_LOCATIONS));
      solution[d] = empty_location;
      n_moves2++;
      d++;                                               assert(-1 <= d);
      index = parent;
   }
   assert(depth == n_moves2);
   assert(n_moves2 == (*states)[original_index].g2);

   status = check_solution(source, solution, n_moves1 + n_moves2);
   if (status == 1) {
      return(n_moves1 + n_moves2);
   }
   else {
      fprintf(stderr, "solution is incorrect\n");
      return(-1);
   }
}

//_________________________________________________________________________________________________

void analyze_states(bistates_array *states, int max_f, int max_e, int max_g, __int64 **Fexp, __int64 **Rexp, __int64 **Fstored, __int64 **Rstored, __int64 **Fexp_g_h, __int64 **Rexp_g_h, __int64 **Fstored_g_h, __int64 **Rstored_g_h)
/*
   1. This function analyzes the list of states.
   2. Input Variables
      a. states = store the states.
      b. max_f = number of rows to allocate to F and R.
      c. max_e = number of columns to allocate to F and R.
      d. max_g = number of rows and columns to allocate to Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h.
   3. Ouptut Variables
      a. Fexp(l,e+1) = |{v expanded in forward direction: f1(v) = l, g1(v)-h2(v) = e}|.
      b. Rexp(l,e+1) = |{v expanded in reverse direction: f2(v) = l, g2(v)-h1(v) = e}|.
      c. Fstored(l,e+1) = |{v stored in forward direction: f1(v) = l, g1(v)-h2(v) = e}|.
      d. Rstored(l,e+1) = |{v stored in reverse direction: f2(v) = l, g2(v)-h1(v) = e}|.
      e. Fexp_g_h(g,h)  = |{v expanded in forward direction: g1(v) = g, h1(v)-h2(v) = h}|.
      f. Rexp_g_h(g,h)  = |{v expanded in reverse direction: g2(v) = g, h2(v)-h1(v) = h}|.
      e. Fstored_g_h(g,h)  = |{v stored in forward direction: g1(v) = g, h1(v)-h2(v) = h}|.
      f. Rstored_g_h(g,h)  = |{v stored in reverse direction: g2(v) = g, h2(v)-h1(v) = h}|.
   4. Created 7/9/18 by modifying c:\sewell\research\pancake\pancake_code\analyze_states.m.
*/
{
   int               i, e, e1, e2, f1, f2, g, h, max_e1, max_e2, max_g1, max_g2, max_h_diff1, max_h_diff2, max_l1, max_l2, min_h_diff1, min_h_diff2, min_l1, min_l2;
   unsigned char     g1, g2, h1, h2;

   Fexp    = new __int64*[max_f + 1];
   Fstored = new __int64*[max_f + 1];
   Rexp    = new __int64*[max_f + 1];
   Rstored = new __int64*[max_f + 1];
   for (i = 0; i <= max_f; i++) {
      Fexp[i]    = new __int64[max_e + 1];
      Fstored[i] = new __int64[max_e + 1];
      for (e = 0; e <= max_e; e++) { Fexp[i][e] = 0; Fstored[i][e] = 0; }
      Rexp[i]    = new __int64[max_e + 1];
      Rstored[i] = new __int64[max_e + 1];
      for (e = 0; e <= max_e; e++) { Rexp[i][e] = 0; Rstored[i][e] = 0; }
   }
   Fexp_g_h    = new __int64*[max_g + 1];
   Fstored_g_h = new __int64*[max_g + 1];
   Rexp_g_h    = new __int64*[max_g + 1];
   Rstored_g_h = new __int64*[max_g + 1];
   for (i = 0; i <= max_g; i++) {
      Fexp_g_h[i]    = new __int64[2 * max_g + 1];
      Fstored_g_h[i] = new __int64[2 * max_g + 1];
      for (h = 0; h <= 2 * max_g; h++) { Fexp_g_h[i][h] = 0; Fstored_g_h[i][h] = 0; }
      Rexp_g_h[i]    = new __int64[2 * max_g + 1];
      Rstored_g_h[i] = new __int64[2 * max_g + 1];
      for (h = 0; h <= 2 * max_g; h++) { Rexp_g_h[i][h] = 0; Rstored_g_h[i][h] = 0; }
   }
   max_e1 = 0;
   max_e2 = 0;
   max_g1 = 0;
   max_g2 = 0;
   max_h_diff1 = -max_g;
   max_h_diff2 = -max_g;
   max_l1 = 0;
   max_l2 = 0;
   min_h_diff1 = max_g;
   min_h_diff2 = max_g;
   min_l1 = INT_MAX;
   min_l2 = INT_MAX;

   for (i = 0; i <= (*states).n_of_states() - 1; i++) {
      g1 = (*states)[i].g1;   h1 = (*states)[i].h1;   g2 = (*states)[i].g2;   h2 = (*states)[i].h2;
      if (((*states)[i].open1 == 0) || ((*states)[i].open1 == 1)) {
         f1 = g1 + h1;  e1 = g1 - h2;
         Fstored[f1][e1]++;
         if ((*states)[i].open1 == 0) Fexp[f1][e1]++;
         Fstored_g_h[g1][h1 - h2 + max_g + 1]++;
         if ((*states)[i].open1 == 0) Fexp_g_h[g1][h1 - h2 + max_g + 1]++;
         max_e1 = max(max_e1, e1);  max_g1 = max(max_g1, (int)g1);  max_h_diff1 = max(max_h_diff1, (int)h1 - (int)h2);  max_l1 = max(max_l1, f1);  min_h_diff1 = min(min_h_diff1, (int)h1 - (int)h2);  min_l1 = min(min_l1, f1);
      }
      if (((*states)[i].open2 == 0) || ((*states)[i].open2 == 1)) {
         f2 = g2 + h2;  e2 = g2 - h1;
         Rstored[f2][e2]++;
         if ((*states)[i].open2 == 0) Rexp[f2][e2]++;
         Rstored_g_h[g2][h2 - h1 + max_g + 1]++;
         if ((*states)[i].open2 == 0) Rexp_g_h[g2][h2 - h1 + max_g + 1]++;
         max_e2 = max(max_e2, e2);  max_g2 = max(max_g2, (int)g2);  max_h_diff2 = max(max_h_diff2, (int)h2 - (int)h1);  max_l2 = max(max_l2, f2);  min_h_diff2 = min(min_h_diff2, (int)h2 - (int)h1);  min_l2 = min(min_l2, f2);
      }
   }

   // Print Fexp.

   if (min_l1 <= max_l1) {
      printf("Forward Expanded: Fexp(l,e) = |{v: f1(v) = l, g1(v)-h2(v) = e}|\n");
      printf("    "); for (i = 0; i <= max_e1; i++) printf("%8d ", i); printf("\n");
      for (i = min_l1; i <= max_l1; i++) {
         printf("%2d: ", i);
         for (e = 0; e <= max_e1; e++) printf("%8I64d ", Fexp[i][e]);
         printf("\n");
      }
   } else {
      printf("Forward Expanded: Fexp(l,e) = |{v: f1(v) = l, g1(v)-h2(v) = e}| is empty\n");
   }

   // Print Rexp

   if (min_l2 <= max_l2) {
      printf("Reverse Expanded: Rexp(l,e) = |{v: f2(v) = l, g2(v)-h1(v) = e}|\n");
      printf("    "); for (i = 0; i <= max_e2; i++) printf("%8d ", i); printf("\n");
      for (i = min_l2; i <= max_l2; i++) {
         printf("%2d: ", i);
         for (e = 0; e <= max_e2; e++) printf("%8I64d ", Rexp[i][e]);
         printf("\n");
      }
   } else {
      printf("Reverse Expanded: Rexp(l,e) = |{v: f2(v) = l, g2(v)-h1(v) = e}| is empty\n");
   }

   // Print Fstored.

   if (min_l1 <= max_l1) {
      printf("Forward Stored: Fstored(l,e) = |{v: f1(v) = l, g1(v)-h2(v) = e}|\n");
      printf("    "); for (i = 0; i <= max_e1; i++) printf("%8d ", i); printf("\n");
      for (i = min_l1; i <= max_l1; i++) {
         printf("%2d: ", i);
         for (e = 0; e <= max_e1; e++) printf("%8I64d ", Fstored[i][e]);
         printf("\n");
      }
   } else {
      printf("Forward Stored: Fstored(l,e) = |{v: f1(v) = l, g1(v)-h2(v) = e}| is empty\n");
   }

   // Print Rstored.

   if (min_l2 <= max_l2) {
      printf("Reverse Stored: Rstored(l,e) = |{v: f2(v) = l, g2(v)-h1(v) = e}|\n");
      printf("    "); for (i = 0; i <= max_e2; i++) printf("%8d ", i); printf("\n");
      for (i = min_l2; i <= max_l2; i++) {
         printf("%2d: ", i);
         for (e = 0; e <= max_e2; e++) printf("%8I64d ", Rstored[i][e]);
         printf("\n");
      }
   } else {
      printf("Reverse Stored: Rstored(l,e) = |{v: f2(v) = l, g2(v)-h1(v) = e}| is empty\n");
   }

   // Print Fexp_g_h.

   if (min_h_diff1 <= max_h_diff1) {
      printf("Forward Expanded: Fexp_g_h(g,h) = |{v: g1(v) = g, h1(v)-h2(v) = h}|\n");
      printf("    "); for (h = max_h_diff1; h >= min_h_diff1; h--) printf("%8d ", h); printf("\n");
      for (g = 0; g <= max_g1; g++) {
         printf("%2d: ", g);
         for (h = max_h_diff1; h >= min_h_diff1; h--) printf("%8I64d ", Fexp_g_h[g][h + max_g + 1]);
         printf("\n");
      }
   } else {
      printf("Forward Expanded: Fexp_g_h(g,h) = |{v: g1(v) = g, h1(v)-h2(v) = h}| is empty\n");
   }

   // Print Rexp_g_h.

   if (min_h_diff2 <= max_h_diff2) {
      printf("Reverse Expanded: Rexp_g_h(g,h) = |{v: g2(v) = g, h2(v)-h1(v) = h}|\n");
      printf("    "); for (h = max_h_diff2; h >= min_h_diff2; h--) printf("%8d ", h); printf("\n");
      for (g = 0; g <= max_g2; g++) {
         printf("%2d: ", g);
         for (h = max_h_diff2; h >= min_h_diff2; h--) printf("%8I64d ", Rexp_g_h[g][h + max_g + 1]);
         printf("\n");
      }
   } else {
      printf("Reverse Expanded: Rexp_g_h(g,h) = |{v: g2(v) = g, h2(v)-h1(v) = h}| is empty\n");
   }

   // Print Fstored_g_h.

   if (min_h_diff1 <= max_h_diff1) {
      printf("Forward Stored: Fstored_g_h(g,h) = |{v: g1(v) = g, h1(v)-h2(v) = h}|\n");
      printf("    "); for (h = max_h_diff1; h >= min_h_diff1; h--) printf("%8d ", h); printf("\n");
      for (g = 0; g <= max_g1; g++) {
         printf("%2d: ", g);
         for (h = max_h_diff1; h >= min_h_diff1; h--) printf("%8I64d ", Fstored_g_h[g][h + max_g + 1]);
         printf("\n");
      }
   } else {
      printf("Forward Stored: Fstored_g_h(g,h) = |{v: g1(v) = g, h1(v)-h2(v) = h}| is empty\n");
   }

   // Print Rstored_g_h.

   if (min_h_diff2 <= max_h_diff2) {
      printf("Reverse Stored: Rstored_g_h(g,h) = |{v: g2(v) = g, h2(v)-h1(v) = h}|\n");
      printf("    "); for (h = max_h_diff2; h >= min_h_diff2; h--) printf("%8d ", h); printf("\n");
      for (g = 0; g <= max_g2; g++) {
         printf("%2d: ", g);
         for (h = max_h_diff2; h >= min_h_diff2; h--) printf("%8I64d ", Rstored_g_h[g][h + max_g + 1]);
         printf("\n");
      }
   } else {
      printf("Reverse Stored: Rstored_g_h(g,h) = |{v: g2(v) = g, h2(v)-h1(v) = h}| is empty\n");
   }
}

//_________________________________________________________________________________________________

void initialize_bisearch(searchparameters *parameters, bistates_array *states, min_max_stacks *cbfs_stacks, Hash_table *hash_table)
{
   int      i, size;

   // Initialize the heaps

   switch (parameters->algorithm) {
      case 3:  // best fs
         if ((parameters->search_direction == 1) || (parameters->search_direction == 3)) {
            if (forward_bfs_heap.is_null()) {
               forward_bfs_heap.initialize(BFS_HEAP_SIZE);
            }
            else {
               forward_bfs_heap.clear();
            }
         }
         if ((parameters->search_direction == 2) || (parameters->algorithm == 3)) {
            if (reverse_bfs_heap.is_null()) {
               reverse_bfs_heap.initialize(BFS_HEAP_SIZE);
            }
            else {
               reverse_bfs_heap.clear();
            }
         }
		   break;
      default:
	      fprintf(stderr, "Unknown algorithm: Only best fs has been implemented for bidirectional search.\n");
		   exit(1);
		   break;
   }

   if(states->is_null())
      states->initialize(STATE_SPACE);
   else
      states->clear();

   //  If a hash table is used, it needs to be initialized here.

   hash_table->initialize();

}

//_________________________________________________________________________________________________

void reinitialize_bisearch(searchparameters *parameters, searchinfo *info, bistates_array *states, Hash_table *hash_table)
{
   int      i;

   info->initialize();
   states->clear();

   switch(parameters->algorithm) {		
		case 3:  // best fs
         bfs_heap.clear();
			break;
      default:
         fprintf(stderr, "Unknown algorithm: Only best fs has been implemented for bidirectional search.\n");
         exit(1);
         break;
   }

   // If a hash table is used, it needs to be emptied or deleted here.

   hash_table->clear();
}

//_________________________________________________________________________________________________

void free_bimemory()
{
	switch(algorithm) {		
		case 3:  // best fs
         //states->clear();      
			break;
      default:
         fprintf(stderr, "Unknown algorithm: Only best fs has been implemented for bidirectional search.\n");
         exit(1);
         break;
   }

   // If a hash table is used, it may need to be emptied or deleted here.
   // Right now, the hash table is declared within bidirectional (and a_star), so it is automatically deleted when 
   // bidirection (or a_star) is exited.
}

//_________________________________________________________________________________________________

int check_bistate(unsigned char source[N_LOCATIONS], bistate *state, int direction)
{
   if (check_tile_in_location(state->tile_in_location) == 0) {
      fprintf(stderr, "tile_in_location is illegal\n");
      exit(1);
   }
   if ((state->empty_location < 0) || (state->empty_location > n_tiles + 1)) {
      fprintf(stderr, "illegal value for empty_location\n");
      exit(1);
   }

   if (direction == 1) {
      if ((state->prev_location1 < 0) || (state->prev_location1 > n_tiles + 2)) {     // prev_location1 should equal n_tiles + 1 at the root.
         fprintf(stderr, "illegal value for prev_location1\n");
         exit(1);
      }
      if ((dpdb_lb == 0) && (state->h1 != compute_Manhattan_LB(state->tile_in_location))) {
         fprintf(stderr, "h1 is incorrect\n");
         exit(1);
      }
   } else {
      if ((state->prev_location2 < 0) || (state->prev_location2 > n_tiles + 2)) {     // prev_location2 should equal n_tiles + 1 at the root.
         fprintf(stderr, "illegal value for prev_location2\n");
         exit(1);
      }
      if (state->h2 != compute_Manhattan_LB2(source, state->tile_in_location)) {
         fprintf(stderr, "h2 is incorrect\n");
         exit(1);
      }
   }
   
   return(1);
}

//_________________________________________________________________________________________________

void prn_bi_subproblem(bistate *state, int direction, unsigned char UB, searchinfo *info, int prn_config)
{
   unsigned char  g1, h1, g2, h2;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if (direction == 1) {
      printf("Forward  UB  f1 f1b  g1  h1  g2  h2  e1  e2    n_exp_f    n_gen_f\n");
      if (state->open2 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d:\n", UB, g1 + h1, 2 * g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("        %3d %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d:\n", UB, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      if (prn_config > 0) prn_configuration(state->tile_in_location);
   }
   else {
      printf("Reverse  UB  f2 f2b  g1  h1  g2  h2  e1  e2    n_exp_r    n_gen_r\n");
      if (state->open1 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d:\n", UB, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("        %3d %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d:\n", UB, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      if (prn_config > 0) prn_configuration(state->tile_in_location);
   }
}

//_________________________________________________________________________________________________

void prn_bi_subproblem2(bistate *state, int direction, unsigned char UB, searchinfo *info, int prn_config)
{
   unsigned char  g1, h1, g2, h2;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if (direction == 1) {
      if (state->open2 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d:\n", UB, g1 + h1, 2 * g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("        %3d %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d:\n", UB, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      if(prn_config > 0) prn_configuration(state->tile_in_location);
   }
   else {
      printf("Reverse  UB  f2 f2b  g1  h1  g2  h2  e1  e2    n_exp_r    n_gen_r\n");
      if (state->open1 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d:\n", UB, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("        %3d %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d:\n", UB, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      if (prn_config > 0) prn_configuration(state->tile_in_location);
   }
}
