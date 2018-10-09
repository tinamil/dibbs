#include "main.h"
#include <queue>

static double           f1_min;           // = min {f1(v): v is in open set of nodes in the forward direction}.
static double           f2_min;           // = min {f2(v): v is in open set of nodes in the reverse direction}.
static int              gap_x;            // = value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
static unsigned char    UB;               // = objective value of best solution found so far.

/*************************************************************************************************/

unsigned char a_star(unsigned char *seq, int direction, unsigned char initial_UB, searchparameters *parameters, searchinfo *info)
/*
   These functions implement the A* algorithm for the pancake problem.
   1. Algorithm Description
      a. It uses branch and bound to attempt to minimize the objective function.
      b. The depth of a subproblem equals the number of moves that have been made from the source or goal configuration.
      c. direction = 1 for forward direction
                   = 2 for reverse direction
   2. Conceptually, a subproblem consists of a sequence of the pancakes.
      To reduce memory requirements, do not store the moves that were made for each subproblem.
      Instead, use pointers to reconstruct the moves.  A subproblem consists of:
      a. g1 = objective function value = number of flips that have been made so far in the forward direction.
      b. h1 = lower bound on the number of moves needed to reach the goal postion.
      c. open1 = 2 if this subproblem has not yet been generated in the forward direction
               = 1 if this subproblem is open in the forward direction
               = 0 if this subproblem closed in the forward direction
      d. parent1 = index (in states) of the parent subproblem in the forward direction.
      e. g2 = objective function value = number of flips that have been made so far in the reverse direction.
      f. h2 = lower bound on the number of flips needed to reach the source postion.
      g. open2 = 2 if this subproblem has not yet been generated in the reverse direction
               = 1 if this subproblem is open in the reverse direction
               = 0 if this subproblem closed in the reverse direction
      h. parent2 = index (in states) of the parent subproblem in the reverse direction.
      i. hash_value = the index in the hash table to begin searching for the state.
      j. seq[i] = the number of the pancake that is position i (i.e., order of the pancakes).
         seq[0] = the number of pancakes.
   3. Input Variables
      a. seq = the order of the pancakes.
         The elements of seq are stored beginning in seq[1].
      b. direction = 1 for forward direction
                   = 2 for reverse direction
      c. initial_UB = the initial upper bound for the search.  Use initial_UB = MAX_DEPTH if no solution is known prior to the search.
      d. parameters
         algorithm         -a option: algorithm
                           1 = iterative deepening
                           2 = forward best first search
                           3 = reverse best first search
                           4 = bidirectional

                    
         best_measure:     -b option: best_measure
                           1 = f = g + h
                           2 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'
                           3 = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
         search_strategy;  -e option: search (exploration) strategy
                           1 = depth first search
                           2 = breadth first search
                           3 = best first search
                           4 = cyclic best first search
                           5 = cyclic best first search using min_max_stacks
                           6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps.
                           Note: Only best first search has been implemented for the search.
         cpu_limit:        cpu limit for search process
         prn_info:         Controls level of printed info (def=0)
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_min = min {f1(v): v is in open set of nodes in the forward direction}.
      i. f2_min = min {f2(v): v is in open set of nodes in the reverse direction}.
      j. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      k. UB = objective value of best solution found so far.
   5. Output Variables
      a. best_z = the best objective value that was found.
            -1 is returned if an error occurs, such as running out of memory.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 7/22/17 by modifying bidirectional from c:\sewell\research\15puzzle\15puzzle_code2\bidirectional.cpp.
      a. Used c:\sewell\research\pancake\matlab\a_star.m as a guide.
*/
{
   bool           finished;
   int            hash_value, i, status;
   __int64        **F = NULL, **R = NULL, sum1, sum2;
   double         best, cpu;
   Hash_table     hash_table;
   bistate        root_state;			// state for the root problem
   //bistates_array states;
   pair<int, int> status_index;

   assert(check_inputs(seq));

   // Initialize data structures.

   info->initialize();
   initialize_search(parameters, &states, &hash_table);
   gap_x = parameters->gap_x;
   UB = initial_UB;

   // Create the root problem in the desired direction.
   
   root_state.seq = new unsigned char[n + 1];   if(root_state.seq == NULL) {fprintf(stderr, "Out of space for root_state.seq\n"); info->optimal = 0; return(-1);}

   if (direction == 1) {
      root_state.g1 = 0;                           // = objective function value = number of flips that have been made so far in the forward direction
      root_state.h1 = gap_lb(seq, 1, gap_x);       // = lower bound on the number of moves needed to reach the goal postion
      root_state.open1 = 1;                        // = 2 if this subproblem has not yet been generated in the forward direction
                                                   // = 1 if this subproblem is open in the forward direction
                                                   // = 0 if this subproblem closed in the forward direction
      root_state.parent1 = -1;                     // = index (in states) of the parent subproblem in the forward direction
      root_state.g2 = UCHAR_MAX;                   // = objective function value = number of flips that have been made so far in the reverse direction
      root_state.h2 = gap_lb(seq, 2, gap_x);       // = lower bound on the number of moves needed to reach the source postion
      root_state.open2 = 2;                        // = 2 if this subproblem has not yet been generated in the reverse direction
                                                   // = 1 if this subproblem is open in the reverse direction
                                                   // = 0 if this subproblem closed in the reverse direction
      root_state.parent2 = -1;                     // = index (in states) of the parent subproblem in the reverse direction
      root_state.hash_value = hash_table.hash_seq(seq);     // = the index in the hash table to begin searching for the state.
      for(i = 1; i <= n; i++) root_state.seq[i] = seq[i];   // the number of the pancake that is position i (i.e., order of the pancakes)
      root_state.seq[0] = n;                                // seq[0] = the number of pancakes

      info->h1_root = root_state.h1;
      best = root_state.h1;

      // Need to add the forward root problem to the list of states and the set of unexplored states.

      hash_value = root_state.hash_value;
      status_index = find_or_insert(&root_state, best, 0, direction, &states, parameters, info, &hash_table, hash_value);

   } else {

      root_state.g1 = UCHAR_MAX;                   // = objective function value = number of flips that have been made so far in the forward direction
      root_state.h1 = gap_lb(goal, 1, gap_x);      // = lower bound on the number of moves needed to reach the goal postion
      root_state.open1 = 2;                        // = 2 if this subproblem has not yet been generated in the forward direction
                                                   // = 1 if this subproblem is open in the forward direction
                                                   // = 0 if this subproblem closed in the forward direction
      root_state.parent1 = -1;                     // = index (in states) of the parent subproblem in the forward direction
      root_state.g2 = 0;                           // = objective function value = number of flips that have been made so far in the reverse direction
      root_state.h2 = gap_lb(goal, 2, gap_x);      // = lower bound on the number of moves needed to reach the source postion
      root_state.open2 = 1;                        // = 2 if this subproblem has not yet been generated in the reverse direction
                                                   // = 1 if this subproblem is open in the reverse direction
                                                   // = 0 if this subproblem closed in the reverse direction
      root_state.parent2 = -1;                     // = index (in states) of the parent subproblem in the reverse direction
      root_state.hash_value = hash_table.hash_seq(goal);    // = the index in the hash table to begin searching for the state.
      for (i = 1; i <= n; i++) root_state.seq[i] = goal[i]; // the number of the pancake that is position i (i.e., order of the pancakes)
      root_state.seq[0] = n;                                // seq[0] = the number of pancakes

      info->h1_root = root_state.h2;
      best = root_state.h2;

      // Need to add the reverse root problem to the list of states and the set of unexplored states.

      hash_value = root_state.hash_value;
      status_index = find_or_insert(&root_state, best, 0, direction, &states, parameters, info, &hash_table, hash_value);
   }

   // Main loop

   finished = false;
   while ((!finished) && (info->optimal >= 0)) {
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if(cpu > CPU_LIMIT) {
         info->optimal = 0;
         break;
      }

      if(direction == 1) {
         status = explore_forward(&states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0;
            //return(-1);
            break;
         }
         if(UB <= ceil(f1_min)) finished = true;
      } else {
         status = explore_reverse(&states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0;
            //return(-1);
            break;
         }
         if(UB <= ceil(f2_min)) finished = true;
      }
   }

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
   if (info->optimal == 1) {
      for (i = 0, sum1 = 0; i <= floor(info->best_z / 2); i++) sum1 += info->n_explored_depth[i];
      for (i = 0, sum2 = 0; i <= info->best_z; i++) sum2 += info->n_explored_depth[i];
   } else {
      sum1 = -1;  sum2 = 1;
   }

   if (prn_info == 0)  {
      if (direction == 1) {
         printf("%3d %2d %7.2f %7.2f %12I64d %12I64d %12I64d %12I64d %2d %6.3f\n",
            info->best_z, info->h1_root, info->cpu, info->best_cpu, info->best_branch, info->n_explored_forward, info->n_generated_forward, states.stored(), info->optimal, (double)sum1 / (double)sum2);
      } else {
         printf("%3d %2d %7.2f %7.2f %12I64d %12I64d %12I64d %12I64d %2d %6.3f\n",
            info->best_z, info->h1_root, info->cpu, info->best_cpu, info->best_branch, info->n_explored_reverse, info->n_generated_reverse, states.stored(), info->optimal, (double)sum1 / (double)sum2);
      }
   }
   if (prn_info > 0)  {
      if(direction == 1) {
         printf("z = %3d cpu = %7.2f best_cpu = %7.2f best_branch = %12I64d n_exp_f = %12I64d n_gen_f = %12I64d n_stored = %12I64d opt = %2d %6.3f\n",
            info->best_z, info->cpu, info->best_cpu, info->best_branch, info->n_explored_forward, info->n_generated_forward, states.stored(), info->optimal, (double)sum1 / (double)sum2);
         //for(i = 0; i <= info->best_z; i++) printf("%3d %14I64d\n", i, info->n_explored_depth[i]);
      } else {
         printf("z = %3d cpu = %7.2f best_cpu = %7.2f best_branch = %12I64d n_exp_r = %12I64d n_gen_r = %12I64d n_stored = %12I64d opt = %2d %6.3f\n",
            info->best_z, info->cpu, info->best_cpu, info->best_branch, info->n_explored_reverse, info->n_generated_reverse, states.stored(), info->optimal, (double)sum1 / (double)sum2);
      }
   }
   //if(prn_info > 1) prn_data(info->best_solution, info->best_z);

   //analyze_states(&states, UB + 5, UB, F, R);

   free_search_memory(parameters, &states);


   return(info->best_z);
}

//_________________________________________________________________________________________________

int explore_forward(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function selects the next state to be explored in the forward direction and then explores it.
   2. Input Variables
      a. states = store the states.
      b. parameters = controls the search.  See a_star for details.
      c. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_min = min {f1(v): v is in open set of nodes in the forward direction}.
      i. f2_min = min {f2(v): v is in open set of nodes in the reverse direction}.
      j. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      k. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 8/1/17 by modifying explore_forward_state from c:\sewell\research\15puzzle\15puzzle_code2\bidirectional.cpp.
      a. Used expand_forward in c:\sewell\research\pancake\matlab\a_star.m as a guide.
*/
{
   unsigned char     *cur_seq, f1_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, index, state_index, status;
   double            best;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(1, parameters, info);
   while((index != -1) && ((*states)[index].open1 == 0)) {
      index = get_bistate(1, parameters, info);
   }
   if(index == -1) {
      f1_min = UCHAR_MAX;
      return(1);
   } else {
      (*states)[index].open1 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   //assert(check_bistate(state, gap_x, states, hash_table));

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   if(g1 + h1 >= UB) return(1);
   hash_value = state->hash_value;

   info->n_explored_forward++;
   info->n_explored_depth[g1]++;

   if(prn_info > 2) prn_a_star_subproblem(state, 1, UB, info);
   if((prn_info > 1) && (info->n_explored_forward % 10000 == 0)) prn_a_star_subproblem(state, 1, UB, info); 

   // Create new state and fill in values that will be the same for all subproblems.

   g1_sub = g1 + 1;
   g2_sub = UCHAR_MAX;
   new_state.g1 = g1_sub;
   new_state.open1 = 1;
   new_state.parent1 = index;
   new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); info->optimal = 0; return(-1); }
   memcpy(new_state.seq, state->seq, n + 1);
   new_state.g2 = UCHAR_MAX;
   new_state.open2 = 2;
   new_state.parent2 = -1;
   depth1 = g1_sub;

   // Generate all the subproblems from this state.

   for(i = 2; i <= n; i++) {
      info->n_generated_forward++;

      // Compute the change in h1 and h2 for the subproblem.

      h1_sub = update_gap_lb(cur_seq, 1, i, h1, gap_x);
      h2_sub = update_gap_lb(cur_seq, 2, i, h2, gap_x);

      f1_sub = g1_sub + h1_sub;
      if (f1_sub < UB) {
         reverse_vector2(i, n, cur_seq, new_state.seq);
         hash_value_sub = hash_table->update_hash_value(cur_seq, i, hash_value);
         new_state.h1 = h1_sub;
         new_state.h2 = h2_sub;
         new_state.hash_value = hash_value_sub;
         //assert(hash_value_sub == hash_table->hash_seq(new_state.seq)); assert(h1_sub == gap_lb(new_state.seq, 1, gap_x)); assert(h2_sub == gap_lb(new_state.seq, 2, gap_x));
         best = compute_bibest(1, g1_sub, g2_sub, h1_sub, h2_sub, parameters);
         status_index = find_or_insert(&new_state, best, depth1, 1, states, parameters, info, hash_table, hash_value_sub);
         find_status = status_index.first;
         state_index = status_index.second;

         // If there was insufficient memory for the new state, then return -1.
         if(state_index == -1) {
            delete[] new_state.seq;
            return(-1);
         }

         if(parameters->prn_info > 3) prn_a_star_subproblem2(&new_state, 1, find_status, info);

         // If a better solution has been found, record it.

         if(h1_sub == 0) {
            // Verify that we have found a solution.  The GAP-0 LB guarantees that when h1 = 0, then it is a solution, but other LB's may not.
            if (memcmp(new_state.seq, goal, n + 1) == 0) {
               UB = g1_sub;
               info->best_z = UB;
               info->best_branch = info->n_explored_forward;
               info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
               status = backtrack(1, states, state_index, info->best_solution);
               if(status == -1) {
                  delete[] new_state.seq;
                  return(-1);
               }
            }
         }
      }
   }
   
   item = forward_bfs_heap.get_min();
   if (item.key == -1)
      f1_min = UCHAR_MAX;
   else
      f1_min = item.key;

   delete[] new_state.seq;
   return(1);
}

//_________________________________________________________________________________________________

int explore_reverse(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function selects the next state to be explored in the reverse direction and then explores it.
   2. Input Variables
      a. states = store the states.
      b. parameters = controls the search.  See a_star for details.
      c. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_min = min {f1(v): v is in open set of nodes in the forward direction}.
      i. f2_min = min {f2(v): v is in open set of nodes in the reverse direction}.
      j. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 8/2/17 by modifying explore_forward from this file.
      a. Used expand_reverse in c:\sewell\research\pancake\matlab\a_star.m as a guide.
*/
{
   unsigned char     *cur_seq, f2_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, index, state_index, status;
   double            best;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(2, parameters, info);
   while ((index != -1) && ((*states)[index].open2 == 0)) {
      index = get_bistate(2, parameters, info);
   }
   if (index == -1) {
      f2_min = UCHAR_MAX;
      return(1);
   }
   else {
      (*states)[index].open2 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   //assert(check_bistate(state, gap_x, states, hash_table));

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   if (g2 + h2 >= UB) return(1);
   hash_value = state->hash_value;

   info->n_explored_reverse++;
   info->n_explored_depth[g2]++;

   if (prn_info > 2) prn_a_star_subproblem(state, 2, UB, info);
   if ((prn_info > 1) && (info->n_explored_reverse % 10000 == 0)) prn_a_star_subproblem(state, 2, UB, info);

   // Create new state and fill in values that will be the same for all subproblems.

   g1_sub = UCHAR_MAX;
   g2_sub = g2 + 1;
   new_state.g1 = UCHAR_MAX;
   new_state.open1 = 2;
   new_state.parent1 = -1;
   new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); info->optimal = 0; return(-1); }
   memcpy(new_state.seq, state->seq, n + 1);
   new_state.g2 = g2_sub;
   new_state.open2 = 1;
   new_state.parent2 = index;
   depth1 = g2_sub;

   // Generate all the subproblems from this state.

   for (i = 2; i <= n; i++) {
      info->n_generated_reverse++;

      // Compute the change in h1 and h2 for the subproblem.

      h1_sub = update_gap_lb(cur_seq, 1, i, h1, gap_x);
      h2_sub = update_gap_lb(cur_seq, 2, i, h2, gap_x);

      f2_sub = g2_sub + h2_sub;
      if (f2_sub < UB) {
         reverse_vector2(i, n, cur_seq, new_state.seq);
         hash_value_sub = hash_table->update_hash_value(cur_seq, i, hash_value);
         new_state.h1 = h1_sub;
         new_state.h2 = h2_sub;
         new_state.hash_value = hash_value_sub;
         //assert(hash_value_sub == hash_table->hash_seq(new_state.seq)); assert(h1_sub == gap_lb(new_state.seq, 1, gap_x)); assert(h2_sub == gap_lb(new_state.seq, 2, gap_x));
         best = compute_bibest(2, g1_sub, g2_sub, h1_sub, h2_sub, parameters);
         status_index = find_or_insert(&new_state, best, depth1, 2, states, parameters, info, hash_table, hash_value_sub);
         find_status = status_index.first;
         state_index = status_index.second;

         // If there was insufficient memory for the new state, then return -1.
         if (state_index == -1) {
            delete[] new_state.seq;
            return(-1);
         }

         if (parameters->prn_info > 3) prn_a_star_subproblem2(&new_state, 2, find_status, info);

         // If a better solution has been found, record it.

         if (h2_sub == 0) {
            // Verify that we have found a solution.  The GAP-0 LB guarantees that when h2 = 0, then it is a solution, but other LB's may not.
            if (memcmp(new_state.seq, source, n + 1) == 0) {
               UB = g2_sub;
               info->best_z = UB;
               info->best_branch = info->n_explored_reverse;
               info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
               status = backtrack(2, states, state_index, info->best_solution);
               if(status == -1) {
                  delete[] new_state.seq;
                  return(-1);
               }
            }
         }
      }
   }

   item = reverse_bfs_heap.get_min();
   if (item.key == -1)
      f2_min = UCHAR_MAX;
   else
      f2_min = item.key;

   delete[] new_state.seq;
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
		case 2:  // best = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'.
         if (direction == 1) {
            best = 2 * g1 + h1 - h2;
         } else {
            best = 2 * g2 + h2 - h1;
         }
			break;
      case 3:  // best = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
         if (direction == 1) {
            best = g1 + h1 - (double) g1 / (double) (MAX_DEPTH + 1);
         }
         else {
            best = g2 + h2 - (double) g2 / (double)(MAX_DEPTH + 1);
         }
         break;
      case 4:  // best = f_bar_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
         if (direction == 1) {
            best = 2 * g1 + h1 - h2 - (double) g1 / (double) (MAX_DEPTH + 1);
         }
         else {
            best = 2 * g2 + h2 - h1 - (double) g2 / (double) (MAX_DEPTH + 1);
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

int backtrack(int direction, bistates_array *states, int index, unsigned char solution[MAX_DEPTH + 1])
/*
   1. BACKTRACK constructs a solution by backtracking through the states.
   2. Input Variables
      a. direction = 1 for forward direction
                   = 2 for reverse direction
      b. states = array where the states are stored.
      c. index = the index of the state (in states) from which to begin backtracking.
   3. Global Variables
      a. n = number of pancakes.
      b. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
      a. The number of flips is returned.
         -1 is returned if an error occurs.
      b. solution = array containing the moves that were made in this solution.
   5. Created 8/1/17 by modifying bibacktrack from c:\sewell\research\15puzzle\15puzzle_code2\bidirectional.cpp.
      a. Used backtrack in c:\sewell\research\pancake\matlab\a_star.m as a guide.
   */
{
   int            d, i, n_flips, original_index, parent, status;

   original_index = index;

   if(direction ==1)
      d = (*states)[index].g1;
   else
      d = 1;

   // Backtrack from the state to the source or from the state to the goal.

   n_flips = 0;
   while(index >= 0) {
      if(direction == 1)
         parent = (*states)[index].parent1;
      else
         parent = (*states)[index].parent2;

      // Find the location of the flip.

      if(parent >= 0) {
         i = n;
         while ((*states)[index].seq[i] == (*states)[parent].seq[i]) i--;
         solution[d] = i;
         n_flips++;
      }

      if(direction == 1) {
         d--;                       assert(-1 <= d);
      } else {
         d++;                       assert(d <= (*states)[original_index].g2 + 2);
      }
      index = parent;
   }
   assert(((direction == 1) && (n_flips == (*states)[original_index].g1)) || ((direction ==2) && (n_flips == (*states)[original_index].g2)));
   
   status = check_solution(source, solution, n_flips);
   if (status == 1) {
      return(n_flips);
   } else {
      fprintf(stderr, "solution is incorrect\n");
      return(-1);
   }
}

//_________________________________________________________________________________________________

void initialize_search(searchparameters *parameters, bistates_array *states, Hash_table *hash_table)
{

   // Initialize the heaps

   switch (parameters->search_strategy) {
      case 3:  // best fs
         if((parameters->algorithm == 2) || (parameters->algorithm == 4)) {
            if(forward_bfs_heap.is_null()) {
               forward_bfs_heap.initialize(HEAP_SIZE);
            } else {
               forward_bfs_heap.clear();
            }
         }
         if((parameters->algorithm == 3) || (parameters->algorithm == 4)) {
            if (reverse_bfs_heap.is_null()) {
               reverse_bfs_heap.initialize(HEAP_SIZE);
            } else {
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
      states->initialize(STATE_SPACE, n);
   else
      states->clear();

   //  If a hash table is used, it needs to be initialized here.

   hash_table->initialize(n);

}

//_________________________________________________________________________________________________

void reinitialize_search(searchparameters *parameters, searchinfo *info, bistates_array *states, Hash_table *hash_table)
{
   info->initialize();
   states->clear();

   switch(parameters->search_strategy) {		
		case 3:  // best fs
         if((parameters->algorithm == 2) || (parameters->algorithm == 4)) forward_bfs_heap.clear();
         if((parameters->algorithm == 3) || (parameters->algorithm == 4)) reverse_bfs_heap.clear();
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

void free_search_memory(searchparameters *parameters, bistates_array *states)
{
	switch(parameters->search_strategy) {		
		case 3:  // best fs
         states->clear();
			break;
      default:
         fprintf(stderr, "Unknown algorithm: Only best fs has been implemented for bidirectional search.\n");
         exit(1);
         break;
   }

   // If a hash table is used, it may need to be emptied or deleted here.
   // Right now, the hash table is declared within a_star, so it is automatically deleted when a_star is exited.

}

//_________________________________________________________________________________________________

int check_bistate(bistate *state, int gap_x, bistates_array *states, Hash_table *hash_table)
{
   if(check_inputs(state->seq) == 0) {
      fprintf(stderr, "seq is not legitimate\n"); 
      return(0);
   }

   if(state->h1 != gap_lb(state->seq, 1, gap_x)) {
      fprintf(stderr, "h1 is incorrect\n"); 
      return(0);
   }
   if(state->h2 != gap_lb(state->seq, 2, gap_x)) { 
      fprintf(stderr, "h2 is incorrect\n"); 
      return(0);
   }

   if((state->open1 != 0) && (state->open1 != 1)  && (state->open1 != 2)) {
      fprintf(stderr, "open1 is incorrect\n"); 
      return(0);
   }
   if((state->open2 != 0) && (state->open2 != 1) && (state->open2 != 2)) { 
      fprintf(stderr, "open2 is incorrect\n"); 
      return(0);
   }

   if(state->open1 == 2) {
      if(state->g1 != UCHAR_MAX) {
         fprintf(stderr, "illegal value for g1\n"); 
         return(0); 
      }
      if(state->parent1 != -1) {
         fprintf(stderr, "illegal value for parent1\n"); 
         return(0); 
      }
   } else {
      if((state->g1 < 0) || (state->g1 > MAX_DEPTH)) {
         fprintf(stderr, "illegal value for g1\n"); 
         return(0);
      }
      if((state->parent1 < -1) || (state->parent1 > states->n_of_states())) {
         fprintf(stderr, "illegal value for parent1\n"); 
         return(0);
      }
   }
   if (state->open2 == 2) {
      if (state->g2 != UCHAR_MAX) { 
         fprintf(stderr, "illegal value for g2\n"); 
         return(0); 
      }
      if (state->parent2 != -1) { 
         fprintf(stderr, "illegal value for parent2\n"); 
         return(0); 
      }
   } else {
      if ((state->g2 < 0) || (state->g2 > MAX_DEPTH)) { 
         fprintf(stderr, "illegal value for g2\n"); 
         return(0); 
      }
      if ((state->parent2 < -1) || (state->parent2 > states->n_of_states())) { 
         fprintf(stderr, "illegal value for parent2\n"); 
         return(0); 
      }
   }

   if(state->hash_value != hash_table->hash_seq(state->seq)) {
      fprintf(stderr, "hash_value is incorrect\n"); 
      return(0);
   }
   
   return(1);
}
