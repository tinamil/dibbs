#include "main.h"
#include <queue>

/*************************************************************************************************/

unsigned char search(unsigned char source[N_LOCATIONS], searchparameters *parameters, searchinfo *info, DPDB *DPDB)
/*
   These functions implement a Cyclic Best First Search (CBFS) for the 15-puzzle.
   The aglorithm is exact, given sufficient computer time and memory.  Given the large solution space and the poor bounds, this method may
   be primarily used as a heuristic (especially for puzzles with more than 15 tiles).
   1. Algorithm Description
      a. It uses branch and bound to attempt to minimize the objective function.
      b. It can perform Depth First Search (DFS), Breadth First Search (BrFS), Best First Search (BFS), or Cyclic Best First Search (CBFS).
      c. The depth of a subproblem equals the number of moves that have been made from the source configuration.
      d. Need to implement a method to eliminate duplicate configurations.
   2. Conceptually, a subproblem consists of a configuration of the tiles plus the moves that were made to reach this configuration.  
      To reduce memory requirements, do not store the moves that were made for each subproblem.  
      Instead, use pointers to reconstruct the moves.  A subproblem consists of:
      a. z = objective function value = number of moves that have been made so far.
      b. LB = lower bound on the number of moves needed to reach the goal postion.
      c. empty_location = location of the empty tile.
      d. prev_location = location of the empty tile in the parent of this subproblem.
      e. parent = index of the parent subproblem.
      f. tile_in_location[i] = the tile that is in location i.
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
         best_measure:     -b option: best_measure
                           1 = LB = g + h
                           2 = g + 1.5 h (AWA*)
                           3 = g - h2
                           4 = best = z + LB - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                           5 = -z - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                           6 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'
         cpu_limit:        cpu limit for search process
         dpdb_lb;          -d option: 1 = use disjoint LB, 0 = do not use (def = 0)
         gen_skip;         -g option: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0).
         prn_info:         Controls level of printed info (def=0)
   4. Global Variables
      a. n_tiles =  number of tiles.
      b. UB = objective value of the best solution found so far.
      c. distances[i][j] = Manhattan distance between location i and location j.
      d. moves[i] = list of possible ways to move the empty tile from location i. 
   5. Output Variables
      a. best_z = the best objective value that was found.
      b. info is used to collect information about the search.
         best_solution[d] = the location of the empty tile after move d in the best solution that was found.
   6. Written 12/2/11.
*/
{
   unsigned char  empty_location, LB;
   int            dominance, hash_index, hash_value, i, index;
   __int64        cnt;
   double         cpu;
   Hash_table     hash_table;
   min_max_stacks cbfs_stacks;
   state          root_state;			// state for the root problem
   states_array   states;

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

   // Compute the lower bound.

   if(dpdb_lb > 0) {
      LB = DPDB->compute_lb(source);
   } else {
      LB = compute_Manhattan_LB(source);
   }
   //UB = MAX_DEPTH;

   // Initialize data structures.

   initialize_search(parameters, &states, &cbfs_stacks, &hash_table);

   // Create the root problem.

   root_state.z = 0;                            // = objective function value = number of moves that have been made so far
   root_state.LB = LB;                          // = lower bound on the number of moves needed to reach the goal postion
   root_state.empty_location = empty_location;  // = location of the empty tile
   root_state.prev_location = n_tiles + 1;      // = location of the empty tile in the parent of this subproblem
   root_state.parent = -1;                      // = index of the parent subproblem
   for(i = 0; i <= n_tiles; i++) root_state.tile_in_location[i] = source[i];  // tile_in_location[i] = the tile that is in location i
   info->root_LB = LB;
 
   // Need to add the root problem to the list of states and the set of unexplored states.

   hash_value = hash_table.hash_configuration(root_state.tile_in_location);
   dominance = search_memory(&root_state, &states, &hash_table, hash_value, &hash_index);
   index = add_to_memory(&root_state, LB, 0, &states, info, &cbfs_stacks, &hash_table, hash_index, dominance);

   // Main loop

   index = get_state(info, &cbfs_stacks);
   while ((index >= 0) && (info->optimal >= 0)) {
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if (cpu > CPU_LIMIT) {
         info->optimal = 0;
         break;
      }

      explore_state(source, &states, index, parameters, info, &cbfs_stacks, DPDB, &hash_table);
      //cnt++; if(cnt % 100000 == 0) prn_heap_info();
      index = get_state(info, &cbfs_stacks);
   }

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;

   if (prn_info == 0)  {
      printf("%3d %8.2f %12I64d %12I64d %12I64d %8.2f %12I64d %2d %8.2f %12I64d\n",
             info->best_z, info->cpu, info->n_explored, info->n_generated, states.stored(), info->best_cpu, 
             info->best_branch, info->optimal, info->states_cpu, info->cnt);
   }
   if (prn_info > 0)  {
      printf("z = %10d cpu = %8.2f n_explored = %12I64d n_generated = %12I64d n_stored = = %12I64d best_cpu = %8.2f best_branch = %12I64d optimal = %2d states_cpu = %8.2f %12I64d\n",
         info->best_z, info->cpu, info->n_explored, info->n_generated, states.stored(), info->best_cpu, info->best_branch, info->optimal, info->states_cpu, info->cnt);
      for(i = 0; i <= MAX_DEPTH; i++) printf("%3d %14I64d\n", i, info->n_explored_depth[i]);
      //cbfs_stacks.print_stats();
   }
   if(prn_info > 1) prn_solution(source, info->best_solution, info->best_z, DPDB);

   //states.check_for_dominated_states();
   free_search_memory();

   return(info->best_z);
}

//_________________________________________________________________________________________________

void explore_state(unsigned char source[N_LOCATIONS], states_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  LB, z;
   state          *state;

   state = &(*states)[index];
   assert(check_state(state));

   z = state->z;
   LB = state->LB;

   if(z + LB >= UB) return;

   info->n_explored++;
   info->n_explored_depth[z]++;

   if(prn_info > 2) {
      printf("z = %3d  LB = %3d  z+LB = %3d  UB = %3d  n_explored = %10I64d  n_generated = %10I64d \n", z, LB, z + LB, UB, info->n_explored, info->n_generated); 
      if(prn_info > 3) {prn_configuration(state->tile_in_location); printf("\n");}
   }
   //if(info->n_explored % 10000 == 0) { printf("z = %3d  UB = %3d  LB = %3d  n_explored = %10I64d  n_generated = %10I64d \n", z, UB, LB, info->n_explored, info->n_generated);}

   // Generate all the subproblems from this state.

   gen_subproblems(source, states, index, parameters, info, cbfs_stacks, DPDB, hash_table);
}

//_________________________________________________________________________________________________

void gen_subproblems(unsigned char source[N_LOCATIONS], states_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table)
/*
*/
{
   unsigned char  empty_location, LB, new_location, *pnt_move, prev_location, stop, tile;
   int            dominance, i, hash_index, hash_value, hash_value_sub;
   double         best, cpu;
   state          new_state, *state;

   state = &(*states)[index];

   // Generate all the subproblems from this state.

   empty_location = state->empty_location;
   prev_location = state->prev_location;
   LB = state->LB;
   hash_value = hash_table->hash_configuration(state->tile_in_location);

   new_state.z = state->z + 1;
   new_state.prev_location = empty_location;
   new_state.parent = index;
   //for(i = 0; i <= n_tiles; i++) new_state.tile_in_location[i] = state->tile_in_location[i];
   memcpy(new_state.tile_in_location, state->tile_in_location, n_tiles + 1);

   switch(parameters->gen_skip) {
      case 0:
         stop = moves[empty_location][0];
         for(i = 1, pnt_move = moves[empty_location]; i <= stop; i++) {
            //new_location = moves[empty_location][i];
            new_location = *(++pnt_move);
            if(new_location != prev_location) {
               info->n_generated++;
               hash_value_sub = hash_table->update_hash_value(new_state.tile_in_location, empty_location, new_location, hash_value);
               tile = new_state.tile_in_location[new_location];
               new_state.empty_location = new_location;
               new_state.tile_in_location[empty_location] = tile;
               new_state.tile_in_location[new_location] = 0;
               assert(hash_value_sub == hash_table->hash_configuration(new_state.tile_in_location));
               dominance = search_memory(&new_state, states, hash_table, hash_value_sub, &hash_index);
               //dominance = 0;
               if(dominance != -1) {
                  if(dpdb_lb > 0) {
                     new_state.LB = DPDB->compute_lb(new_state.tile_in_location);
                  } else {
                     new_state.LB = LB + distances[empty_location][tile] - distances[new_location][tile];
                     assert(new_state.LB == compute_Manhattan_LB(new_state.tile_in_location));
                  }

                  if(new_state.z + new_state.LB < UB) {

                     // If a better solution has been found, update the best solution.

                     if(new_state.LB == 0) {
                        //UB = new_state.z;
                        info->best_branch = info->n_explored;
                        info->best_cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC; 
                        UB = backtrack(states, index, info->best_solution) + 1;
                        new_state.z = UB;
                        info->best_z = new_state.z;
                        info->best_solution[UB] = new_location;
                        check_solution(source, info->best_solution, new_state.z);
                        if(parameters->algorithm == 5) cbfs_stacks->new_UB_prune(UB);
                        cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
                        if(prn_info > 0) printf("Better solution found UB = %3d  n_explored =  %10I64d  n_generated = %10I64d  cpu = %8.2f\n", UB, info->n_explored, info->n_generated, cpu);
                     } else {
                        best = compute_best(source, new_state.tile_in_location, new_state.empty_location, new_state.prev_location, new_state.z, new_state.LB, parameters, info);
                        add_to_memory(&new_state, best, new_state.z, states, info, cbfs_stacks, hash_table, hash_index, dominance);
                        //add_to_memory(&new_state, new_state.z + 1.5 * new_state.LB, new_state.z, states, info, cbfs_stacks);
                     }
                  }
               }
               new_state.tile_in_location[empty_location] = 0;
               new_state.tile_in_location[new_location] = tile;
            }
         }
         break;
      default:
         info->n_explored--;                    // Decrement n_explored to avoid double counting.
         info->n_explored_depth[state->z]--;
         gen_dfs(source, state->z + state->LB + gen_skip, empty_location, prev_location, state->LB, state->z, states, parameters, info, cbfs_stacks, &new_state, DPDB, hash_table, hash_value);
         break;
   }
}

//_________________________________________________________________________________________________

void gen_dfs(unsigned char source[N_LOCATIONS], unsigned char bound, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z, states_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, state *new_state, DPDB *DPDB, Hash_table *hash_table, int hash_value)
/*
*/
{
   unsigned char  LB_sub, new_location, *pnt_move, stop, tile;
   int            dominance, i, hash_index, hash_value_sub;
   double         best, cpu;

   info->n_explored++;
   info->n_explored_depth[z]++;

   //printf("z = %3d  UB = %3d  LB = %3d  n_explored = %10I64d  n_generated = %10I64d \n", z, UB, LB, info->n_explored, info->n_generated); 
   //if(prn_info > 2) {prn_configuration(new_state->tile_in_location); printf("\n");}

   // Generate all the subproblems from this state.

   stop = moves[empty_location][0];
   for(i = 1, pnt_move = moves[empty_location]; i <= stop; i++) {
      //new_location = moves[empty_location][i];
      new_location = *(++pnt_move);
      if(new_location != prev_location) {
         info->n_generated++;
         hash_value_sub = hash_table->update_hash_value(new_state->tile_in_location, empty_location, new_location, hash_value);
         tile = new_state->tile_in_location[new_location];
         new_state->tile_in_location[empty_location] = tile;
         new_state->tile_in_location[new_location] = 0;
         if(dpdb_lb > 0) {
            LB_sub = DPDB->compute_lb(new_state->tile_in_location);
         } else {
            LB_sub = LB + distances[empty_location][tile] - distances[new_location][tile];
            assert(LB_sub == compute_Manhattan_LB(new_state->tile_in_location));
         }

         if(z + 1 + LB_sub < UB) {

            // If a better solution has been found, update the best solution.

            if(LB_sub == 0) {
               UB = z + 1;
               info->best_z = z + 1;
               info->best_branch = info->n_explored;
               info->best_cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC; 
               //backtrack(states, new_state->parent, info->best_solution);
               info->best_solution[UB] = new_location;
               //check_solution(source, info->best_solution, new_state.z);
               if(parameters->algorithm == 5) cbfs_stacks->new_UB_prune(UB);
               cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
               if(prn_info > 0) printf("Better solution found UB = %3d  n_explored =  %10I64d  n_generated = %10I64d  cpu = %8.2f\n", UB, info->n_explored, info->n_generated, cpu);
               if(prn_info == 0) printf("%3d  %10I64d  %10I64d  %8.2f\n", UB, info->n_explored, info->n_generated, cpu);
            } else {
               if(z + 1 + LB_sub < bound) {
                  gen_dfs(source, bound, new_location, empty_location, LB_sub, z + 1, states, parameters, info, cbfs_stacks, new_state, DPDB, hash_table, hash_value_sub);
               } else {
                  new_state->z = z + 1;
                  new_state->empty_location = new_location;
                  new_state->prev_location = empty_location;
                  new_state->LB = LB_sub;
                  best = compute_best(source, new_state->tile_in_location, new_state->empty_location,  new_state->prev_location, z + 1, LB_sub, parameters, info);
                  dominance = 0;
                  hash_index = 0;
                  add_to_memory(new_state, best, new_state->z, states, info, cbfs_stacks, hash_table, hash_index, dominance);
                  //add_to_memory(new_state, new_state->z + 1.5 *  new_state->LB, new_state->z, states, info, cbfs_stacks);
               }
            }
         }
         new_state->tile_in_location[empty_location] = 0;
         new_state->tile_in_location[new_location] = tile;
      }
   }
}

//_________________________________________________________________________________________________

double compute_best(unsigned char source[N_LOCATIONS], unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char z, unsigned char LB, searchparameters *parameters, searchinfo *info)
/*
   1. This function computes the best measure for a configuration.
   2. Input Variables
      a. source[i] = the tile that is in location i.
         The elements of source are stored beginning in source[0].
      b. z = number of moves that have been made so far.
      c. LB = lower bound on the number of moves needed to reach the goal postion.
      d. parameters
   3. Output Variables
      a. best = the best measure for this assignment.
   4. Written 2/25/12.
*/
{
   unsigned char  max_n_moves, MD_to_source;
   double         best;

   switch(parameters->best_measure) {		
		case 1:  // best = z + LB = g + h
         best = z + LB;
			break;
		case 2:  // best = g + 1.5 h (AWA*)
         best = z + 1.5 * LB;
			break;
		case 3:  // best = g - h2, where h2 = lower bound on the number of moves to reach the source (initial) configuration.
         MD_to_source = compute_Manhattan_LB2(source, tile_in_location);
         if(z < 5) 
            best = z + LB - MD_to_source;
         else
            best = z + LB;
         //best = z + 1.5 * LB - 0.1 * MD_to_source;
			break;
		case 4:  // best = z + LB - maximum number of moves that can be made without exceeding a bound on the number of uphill moves.
         //printf("---------------\n");
         max_n_moves = look_ahead_z_dfs(tile_in_location, empty_location, prev_location, 0);
         best = z + LB - (double) max_n_moves / MAX_DEPTH;     // Essentially use max_n_moves as a tie-breaker.
         best = z + LB - (double) max_n_moves / 150;
         //if(z + max_n_moves > UB) printf("z +_max_n_moves = %3d  UB = %3d\n", z + max_n_moves, UB);
			break;
		case 5:  // best = -z - maximum number of moves that can be made without exceeding a bound on the number of uphill moves.
         //if(LB < UB - 10) {
         if(LB <= info->root_LB + 6) {
            max_n_moves = look_ahead_z_dfs(tile_in_location, empty_location, prev_location, 2);
         } else {
            max_n_moves = look_ahead_z_dfs(tile_in_location, empty_location, prev_location, 0);
         }
         best = -z - max_n_moves;
			break;
      default:
         fprintf(stderr,"Unknown best measure\n"); 
         exit(1); 
         break;
	}

   return(best);
}

//_________________________________________________________________________________________________

unsigned char look_ahead_z_dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, char bound_uphill_moves)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      Define an uphill move to be a move that increases the Manhattan distance.
      It determines the maximum number of moves that can be made without exceeding a bound on the number of uphill moves.
      It is designed to be used as part of a best measure.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
      b. empty_location = location of the empty tile.
      c. prev_location = location of the empty tile in the parent of this subproblem.
      d. bound_uphill_moves = maximum permissible number of uphill moves.
   3. Global Variables
      a. distances[i][j] = Manhattan distance between location i and location j.
      b. moves[i] = list of possible ways to move the empty tile from location i. 
   4. Output Variables
      a. max_n_moves = maximum number of moves that can be made without exceeding the bound on the number of uphill moves.
   5. Written 2/27/12.
*/
{
   char           bound_uphill_moves_sub;
   unsigned char  new_location, max_n_moves, max_n_moves_sub, *pnt_move, stop, tile;
   int            i;

   //printf("\n%3d %3d\n", bound_uphill_moves, prev_location); prn_configuration(tile_in_location);

   // Generate the subproblems.
   
   max_n_moves = 0;
   stop = moves[empty_location][0];
   for(i = 1,  pnt_move = moves[empty_location]; i <= stop; i++) {
      new_location = *(++pnt_move);
      if(new_location != prev_location) {
         tile = tile_in_location[new_location];
         if(distances[empty_location][tile] > distances[new_location][tile]) {
            bound_uphill_moves_sub = bound_uphill_moves - 1;
         } else {
            bound_uphill_moves_sub = bound_uphill_moves;
         }

         if(bound_uphill_moves_sub >= 0) {
            tile_in_location[empty_location] = tile;
            tile_in_location[new_location] = 0;

            max_n_moves_sub = look_ahead_z_dfs(tile_in_location, new_location, empty_location, bound_uphill_moves_sub);

            tile_in_location[empty_location] = 0;
            tile_in_location[new_location] = tile;
         
            if(max_n_moves_sub + 1 > max_n_moves) max_n_moves = max_n_moves_sub + 1;
         }
      }
   }

   return(max_n_moves);
}

//_________________________________________________________________________________________________

int backtrack(states_array *states, int index, unsigned char solution[MAX_DEPTH + 1])
/*
   1. BACKTRACK constructs a solution by backtracking through the states.
   2. Input Variables
      a. index = the index of the state (in states) from which to begin backtracking.
      b. best_assignment = the best assignment found so far.
   3. Global Variables
      a. n_sites = number of sites.
      b. distances[s][t] = distance between sites s and t.
      c. flow[i][j] = flow between facilities i and j.
   4. Output Variables
      a. assignment = the current assignment.
      b. depth = the depth of this subproblem, which equals the number of swaps that have been made from the best solution.
   5. Written 12/2/11.
   6. Modified 12/14/12.  Due to replacing dominated states, the number of moves on the backtrack path may not equal
      the number of moves stored in (*states)[index].z.
      a. Eliminate assert(depth == n_moves - 1).
      b. Relocate the set of moves to the front of solution.
      c. Return n_moves - 1 instead of depth.
*/
{
   unsigned char  empty_location;
   int            d, depth, n_moves;

   depth = (*states)[index].z;                           assert((0 <= depth) && (depth <= MAX_DEPTH + 1));
   d = depth;
   n_moves = 0;
   while(index >= 0) {
      empty_location = (*states)[index].empty_location;  assert((0 <= empty_location) && (empty_location <= N_LOCATIONS));
      solution[d] = empty_location;  
      n_moves++;
      d--;                                               assert(-1 <= d);
      index = (*states)[index].parent;
   }
   //assert(depth == n_moves - 1);

   // Relocate the set of moves to the front of solution, if necessary.

   if(depth > n_moves - 1) {
      for(int i = 0; i <= n_moves; i++) solution[i] = solution[i + d + 1];
   }

   return(n_moves - 1);
}

//_________________________________________________________________________________________________

void initialize_search(searchparameters *parameters, states_array *states, min_max_stacks *cbfs_stacks, Hash_table *hash_table)
{
   int      i, size;

   // Initialize the heaps

   if(parameters->algorithm == 3) {
      if(bfs_heap.is_null()) {
         bfs_heap.initialize(BFS_HEAP_SIZE);
      } else {
         bfs_heap.clear();
      }
   }

   if((parameters->algorithm == 4) || (parameters->algorithm == 6)){ 
      cbfs_heaps = new min_max_heap[MAX_DEPTH + 1];
      size = 1;
      for(i = 0; i <= MAX_DEPTH; i++) {
         cbfs_heaps[i].initialize(min(size + 3, HEAP_SIZE));
         size = min(3 * size, HEAP_SIZE);   // Use min to avoid overflow when 3^i exceed INT_MAX.
      }
   }

   if(parameters->algorithm == 5){ 
      cbfs_stacks->initialize(MAX_DEPTH, UB);
   }

   states->initialize(STATE_SPACE);

   //  If a hash table is used, it needs to be initialized here.

   hash_table->initialize();

   // Initialize the hash values.  hash_values[t][i] = random value for tile t in location i: U[0,HASH_SIZE].

   //hash_values = new int*[N_LOCATIONS];
   //for(int t = 0; t < N_LOCATIONS; t++) {
   //   hash_values[t] = new int[N_LOCATIONS];
   //   for(i = 0; i < N_LOCATIONS; i++) {
   //      hash_values[t][i] = randomi(HASH_SIZE, &seed);
   //   }
   //}
}

//_________________________________________________________________________________________________

void reinitialize_search(searchparameters *parameters, searchinfo *info, states_array *states, Hash_table *hash_table)
{
   int      i;

   info->initialize();
   states->clear();

   switch(parameters->algorithm) {		
		case 1:  // dfs
			while(!stack_dfs.empty()) stack_dfs.pop();
			break;
		case 2:  // breadth fs
			while(!queue_bfs.empty()) queue_bfs.pop();
			break;
		case 3:  // best fs
         bfs_heap.clear();
			break;
		case 4:  //cbfs using min-max heaps
         for(i = 0; i <= MAX_DEPTH; i++) cbfs_heaps[i].clear();
			break;
	}

   // If a hash table is used, it needs to be emptied or deleted here.

   hash_table->clear();
}

//_________________________________________________________________________________________________

void free_search_memory()
{
	switch(algorithm) {		
		case 1:  // dfs
			while(!stack_dfs.empty()) stack_dfs.pop();
			break;
		case 2:  // breadth fs
			while(!queue_bfs.empty()) queue_bfs.pop();
			break;
		case 3:  // best fs
			break;
		case 4:  //cbfs using min-max heaps			
			delete [] cbfs_heaps;
			break;
	}

   // If a hash table is used, it needs to be emptied or deleted here.

}

//_________________________________________________________________________________________________

int check_state(state *state)
{
   if(check_tile_in_location(state->tile_in_location) == 0) {
      fprintf(stderr, "tile_in_location is illegal\n"); 
      exit(1); 
   }
   if((state->empty_location < 0) || (state->empty_location > n_tiles + 1)) {
      fprintf(stderr, "illegal value for empty_location\n"); 
      exit(1); 
   }
   if((state->prev_location < 0) || (state->prev_location > n_tiles + 2)) {     // prev_location should equal n_tiles + 1 at the root.
      fprintf(stderr, "illegal value for prev_location\n"); 
      exit(1); 
   }
   if((dpdb_lb == 0) && (state->LB != compute_Manhattan_LB(state->tile_in_location))) {
      fprintf(stderr, "LB is incorrect\n"); 
      exit(1); 
   }
   return(1);
}
