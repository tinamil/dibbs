#include "main.h"
#include <queue>

static double           f1_min;        // = min {f1(v): v is in open set of nodes in the forward direction}.
static double           f2_min;        // = min {f2(v): v is in open set of nodes in the reverse direction}.
static double           g1_min;        // = min {g1(v): v is in open set of nodes in the forward direction}.
static double           g2_min;        // = min {g2(v): v is in open set of nodes in the reverse direction}.
static bool             path_found;    // = true (false) if a path has (not) been found.
static double           p1_min;        // = min {p1(v): v is in open set of nodes in the forward direction}, where p1 is the priority function in the forward direction.
static double           p2_min;        // = min {p2(v): v is in open set of nodes in the reverse direction}, where p2 is the priority function in the reverse direction.
static int              gap_x;         // = value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."

static unsigned char    UB;            // = objective value of best solution found so far.

/*************************************************************************************************/

unsigned char MM(unsigned char *seq, unsigned char initial_UB, searchparameters *parameters, searchinfo *info)
/*
   These functions implements the meet in the middle (MM) bidirectional search algorithm.
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
      b. initial_UB = the initial upper bound for the search.  Use initial_UB = MAX_DEPTH if no solution is known prior to the search.
      c. parameters
         algorithm         -a option: algorithm
                           1 = iterative deepening
                           2 = forward best first search
                           3 = reverse best first search
                           4 = bidirectional

                    
         best_measure:     -b option: best_measure
                           1 = f = g + h
                           2 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'
                           3 = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
                           4 = f_bar_d - (g_d /(MAX_DEPTH + 1) Break ties in fbar_d in favor of states with larger g_d.
                           5 = max(2*g_d, f_d) MM priority function.
                           6 = max(2*g_d + 1, f_d) MMe priority function.
                           7 = max(2*g_d, f_d) + (g_d /(MAX_DEPTH + 1) MM priority function.  Break ties in favor of states with smaller g_d.
                           8 = max(2*g_d + 1, f_d) + (g_d /(MAX_DEPTH + 1) MMe priority function.  Break ties in favor of states with smaller g_d.
                           9 = max(2*g_d + 1, f_d) - (g_d /(MAX_DEPTH + 1) MMe priority function.  Break ties in favor of states with larger g_d.
         search_strategy;  -e option: search (exploration) strategy
                           1 = depth first search
                           2 = breadth first search
                           3 = best first search
                           4 = best first search using clusters
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
      j. g1_min = min {g1(v): v is in open set of nodes in the forward direction}.
      k. g2_min = min {g2(v): v is in open set of nodes in the reverse direction}.
      l. p1_min = min {p1(v): v is in open set of nodes in the forward direction} where p1 is the priority function in the forward direction.
      m. p2_min = min {p2(v): v is in open set of nodes in the reverse direction} where p2 is the priority function in the reverse direction.
      n. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      o. open_g1_values This is a multiset representation of the g1 values of the open nodes in the forward direction.  It is designed for keeping track of the minimum value of g among the open nodes.
      p. open_g2_values This is a multiset representation of the g2 values of the open nodes in the reverse direction.  It is designed for keeping track of the minimum value of g among the open nodes.
      q. open_f1_values This is a multiset representation of the f1 values of the open nodes in the forward direction.  It is designed for keeping track of the minimum value of f among the open nodes.
      r. open_f2_values This is a multiset representation of the f2 values of the open nodes in the reverse direction.  It is designed for keeping track of the minimum value of f among the open nodes.
      s. UB = objective value of best solution found so far.
   5. Output Variables
      a. best_z = the best objective value that was found.
            -1 is returned if an error occurs, such as running out of memory.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 1/10/19 by modifying bidirectional2 from c:\sewell\research\pancake\pancake_code\bidirectional2.cpp.
*/
{
   int            direction, hash_value, i, start, status, stop, LB;
   double         best, cpu;
   static double  prev_min = -1, prev_min1 = -1, prev_min2 = -1;     // prev_min = min(p1_min, p2_min) in previous iteration, prev_min1 = p1_min in previous iteration, prev_min2 = p2_min in previous iteration.
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   Hash_table     hash_table;
   bistate        root_state;			// state for the root problem
   pair<int, int> status_index;

   assert(check_inputs(seq));

   // Initialize data structures.

   info->initialize();

   initialize_search(parameters, &states, &hash_table, NULL);
   gap_x = parameters->gap_x;
   UB = initial_UB;

   // Create the root problem in the desired direction.
   
   root_state.seq = new unsigned char[n + 1];   if(root_state.seq == NULL) {fprintf(stderr, "Out of space for root_state.seq\n"); info->optimal = 0; return(-1);}


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
   best = compute_bibest(1, root_state.g1, root_state.g2, root_state.h1, root_state.h2, parameters);
   p1_min = best;
   if ((parameters->best_measure == 8) || (parameters->best_measure == 10) || (parameters->best_measure == 11)) p1_min = floor(p1_min);
   if (parameters->best_measure == 9) p1_min = ceil(p1_min);


   // Need to add the forward root problem to the list of states and the set of unexplored states.

   hash_value = root_state.hash_value;
   status_index = find_or_insert(&root_state, best, 0, 1, &states, parameters, info, &hash_table, hash_value, NULL, { 0,0,0 });

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

   //info->h1_root = root_state.h1;
   best = compute_bibest(2, root_state.g1, root_state.g2, root_state.h1, root_state.h2, parameters);
   p2_min = best;
   if ((parameters->best_measure == 8) || (parameters->best_measure == 10) || (parameters->best_measure == 10)) p2_min = floor(p2_min);
   if (parameters->best_measure == 9) p2_min = ceil(p2_min);

   // Need to add the reverse root problem to the list of states and the set of unexplored states.

   hash_value = root_state.hash_value;
   status_index = find_or_insert(&root_state, best, 0, 2, &states, parameters, info, &hash_table, hash_value, NULL, { 0,0,0 });

   // Main loop

   direction = -1;
   path_found = false;
   g1_min = g2_min = 0;
   LB = min((int)p1_min, (int)p2_min); if (open_f1_values.get_min() > LB) LB = open_f1_values.get_min(); if (open_f2_values.get_min() > LB) LB = open_f2_values.get_min();
   //while ((UB > ceil(max(p1_min,p2_min))) && (info->optimal >= 0)) {
   //while ((UB > max( max(min((int) p1_min, (int) p2_min), open_g1_values.get_min() + open_g2_values.get_min() + 1), max(open_f1_values.get_min(), open_f2_values.get_min()))) && (info->optimal >= 0)) {
   while ((UB > LB) && (info->optimal >= 0)) {
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if(cpu > CPU_LIMIT) {
         info->optimal = 0;
         break;
      }

      if(prev_min < min(p1_min, p2_min)) {
         printf("UB = %2d  LB = %2d  p1_min = %4.1f  p2_min = %4.1f  n_exp_f = %12I64d  n_exp_r = %12I64d  |O1| = %12d  |O2| = %12d\n", UB, LB, p1_min, p2_min, info->n_explored_forward, info->n_explored_reverse, forward_bfs_heap.n_of_items(), reverse_bfs_heap.n_of_items());
         //analyze_states(&states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);
         prn_open_g_h1_h2_values2(1); printf("\n"); prn_open_g_h1_h2_values2(2);
         compute_front_to_front_LB();
         prev_min = min(p1_min, p2_min);
      }

      direction = mm_choose_direction(direction, 2);          // Choose the direction.

      if(direction == 1) {
         status = mm_expand_forward(&states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0;
            //return(-1);
            break;
         }
      } else {
         status = mm_expand_reverse(&states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0; 
            //return(-1);
            break;
         }
      }
      //check_open_value(&states);
      start = open_f1_values.get_min();   stop = min(UB - 1, open_f1_values.get_max());
      g1_min = MAX_DEPTH;  for (i = start; i <= stop; i++) if (!open_g1_values[i].empty() && (open_g1_values[i].get_min() < g1_min)) g1_min = open_g1_values[i].get_min();
      start = open_f2_values.get_min();   stop = min(UB - 1, open_f2_values.get_max());
      g2_min = MAX_DEPTH;  for (i = start; i <= stop; i++) if (!open_g2_values[i].empty() && (open_g2_values[i].get_min() < g2_min)) g2_min = open_g2_values[i].get_min();
      LB = min((int)p1_min, (int)p2_min); if (open_f1_values.get_min() > LB) LB = open_f1_values.get_min(); if (open_f2_values.get_min() > LB) LB = open_f2_values.get_min(); if (g1_min + g2_min + 1 > LB) LB = g1_min + g2_min + 1;
   }

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;

   info->n_explored = info->n_explored_forward + info->n_explored_reverse;
   info->n_generated = info->n_generated_forward + info->n_generated_reverse;

   if (prn_info == 0)  {
      printf("%3d %2d %7.2f %7.2f %12I64d %12I64d %12I64d %12I64d %12I64d %12I64d %12I64d %12I64d %2d\n",
         info->best_z, info->h1_root, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored(), info->optimal);
   }
   if (prn_info > 0)  {
      printf("z = %3d cpu = %7.2f best_cpu = %7.2f best_branch = %12I64d n_exp = %12I64d n_exp_f = %12I64d n_exp_r = %12I64d n_gen = %12I64d n_gen_f = %12I64d n_gen_r = %12I64d n_stored = %12I64d opt = %2d\n",
         info->best_z, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_explored_forward, info->n_explored_reverse, info->n_generated, info->n_generated_forward, info->n_generated_reverse, states.stored(), info->optimal);
      //for(i = 0; i <= info->best_z; i++) printf("%3d %14I64d\n", i, info->n_explored_depth[i]);
   }
   //if(prn_info > 1) prn_data(info->best_solution, info->best_z);

   //printf("UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, p1_min, p2_min); 
   //analyze_states(&states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);

   free_search_memory(parameters, &states);


   return(info->best_z);
}

//_________________________________________________________________________________________________

int mm_expand_forward(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function selects the next state to be explored in the forward direction and then explores it.
   2. Input Variables
      a. states = store the states.
      b. parameters = controls the search.  See bidirectional for details.
      c. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_bar_min = min {f1_bar(v): v is in open set of nodes in the forward direction}.
      i. f2_bar_min = min {f2_bar(v): v is in open set of nodes in the reverse direction}.
      j. g1_min = min {g1(v): v is in open set of nodes in the forward direction}.
      k. g2_min = min {g2(v): v is in open set of nodes in the reverse direction}.
      l. p1_min = min {p1(v): v is in open set of nodes in the forward direction} where p1 is the priority function in the forward direction.
      m. p2_min = min {p2(v): v is in open set of nodes in the reverse direction} where p2 is the priority function in the reverse direction.
      n. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      o. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 12/31/18 by modifying explore_forward from c:\sewell\research\pancake\pancake_code\bidirectional.cpp.
*/
{
   unsigned char     *cur_seq, f1_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, index, state_index, status;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   double            best;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(states, 1, parameters, info, NULL, { 0,0,0 });
   while((index != -1) && ((*states)[index].open1 == 0)) {
      index = get_bistate(states, 1, parameters, info, NULL, { 0,0,0 });
   }
   if(index == -1) {
      p1_min = UCHAR_MAX;
      return(1);
   } else {
      (*states)[index].open1 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   //assert(check_bistate(state, gap_x, states, hash_table));
   new_state.seq = NULL;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   if(g1 + h1 >= UB) return(1);
   if (g1 + g2_min + 1 >= UB) return(1);
   hash_value = state->hash_value;

   if(state->open2 > 0) {
      info->n_explored_forward++;
      info->n_explored_depth[g1]++;

      if(prn_info > 2) prn_MM_subproblem(state, 1, UB, parameters, info);
      if((prn_info > 1) && (info->n_explored_forward % 10000 == 0)) prn_MM_subproblem(state, 1, UB, parameters, info); 

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

      for (i = 2; i <= n; i++) {
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
            status_index = find_or_insert(&new_state, best, depth1, 1, states, parameters, info, hash_table, hash_value_sub, NULL, { 0,0,0 });
            find_status = status_index.first;
            state_index = status_index.second;

            // If there was insufficient memory for the new state, then return -1.
            if (state_index == -1) {
               delete[] new_state.seq;
               return(-1);
            }

            if (parameters->prn_info > 3) prn_MM_subproblem2(&new_state, 1, find_status, parameters, info);

            // If a better solution has been found, record it.

            if ((*states)[state_index].g1 + (*states)[state_index].g2 < UB) {
               //if(UB < MAX_DEPTH) printf("Old UB = %2d  New UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, (*states)[state_index].g1 + (*states)[state_index].g2, p1_min, p2_min);
               UB = (*states)[state_index].g1 + (*states)[state_index].g2;
               path_found = true;
               info->best_z = UB;
               info->best_branch = info->n_explored_forward + info->n_explored_reverse;
               info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
               status = bibacktrack(states, state_index, info->best_solution);
               if (status == -1) {
                  delete[] new_state.seq;
                  return(-1);
               }
               //printf("UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, p1_min, p2_min);
               //analyze_states(states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);
            }
         }
      } 
   } 

   item = forward_bfs_heap.get_min();
   if (item.key == -1)
      p1_min = UCHAR_MAX;
   else {
      p1_min = item.key;
      if ((parameters->best_measure == 8) || (parameters->best_measure == 10) || (parameters->best_measure == 11)) p1_min = floor(p1_min);
      if(parameters->best_measure == 9) p1_min = ceil(p1_min);
   }

   if(new_state.seq != NULL) delete[] new_state.seq;
   return(1);
}

//_________________________________________________________________________________________________

int mm_expand_reverse(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function selects the next state to be explored in the reverse direction and then explores it.
   2. Input Variables
      a. states = store the states.
      b. parameters = controls the search.  See bidirectional for details.
      c. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. f1_bar_min = min {f1_bar(v): v is in open set of nodes in the forward direction}.
      i. f2_bar_min = min {f2_bar(v): v is in open set of nodes in the reverse direction}.
      j. g1_min = min {g1(v): v is in open set of nodes in the forward direction}.
      k. g2_min = min {g2(v): v is in open set of nodes in the reverse direction}.
      l. p1_min = min {p1(v): v is in open set of nodes in the forward direction} where p1 is the priority function in the forward direction.
      m. p2_min = min {p2(v): v is in open set of nodes in the reverse direction} where p2 is the priority function in the reverse direction.
      n. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      o. open_g1_values This is a multiset representation of the g1 values of the open nodes in the forward direction.  It is designed for keeping track of the minimum value of g among the open nodes.
      p. open_g2_values This is a multiset representation of the g2 values of the open nodes in the reverse direction.  It is designed for keeping track of the minimum value of g among the open nodes.
      q. open_f1_values This is a multiset representation of the f1 values of the open nodes in the forward direction.  It is designed for keeping track of the minimum value of f among the open nodes.
      r. open_f2_values This is a multiset representation of the f2 values of the open nodes in the reverse direction.  It is designed for keeping track of the minimum value of f among the open nodes.
      s. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 1/1/19 by modifying explore_reverse from c:\sewell\research\pancake\pancake_code\bidirectional.cpp.
*/
{
   unsigned char     *cur_seq, f2_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, index, state_index, status;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   double            best;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   index = get_bistate(states, 2, parameters, info, NULL, { 0,0,0 });
   while ((index != -1) && ((*states)[index].open2 == 0)) {
      index = get_bistate(states, 2, parameters, info, NULL, { 0,0,0 });
   }
   if(index == -1) {
      p2_min = UCHAR_MAX;
      return(1);
   } else {
      (*states)[index].open2 = 0;         // Close this subproblem.
   }
   state = &(*states)[index];
   //assert(check_bistate(state, gap_x, states, hash_table));
   new_state.seq = NULL;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   if (g2 + h2 >= UB) return(1);
   if (g1_min + g2 + 1 >= UB) return(1);
   hash_value = state->hash_value;

   if(state->open1 > 0) {
      info->n_explored_reverse++;
      info->n_explored_depth[g2]++;

      if (prn_info > 2) prn_MM_subproblem(state, 2, UB, parameters, info);
      if ((prn_info > 1) && (info->n_explored_reverse % 10000 == 0)) prn_MM_subproblem(state, 2, UB, parameters, info);

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
            status_index = find_or_insert(&new_state, best, depth1, 2, states, parameters, info, hash_table, hash_value_sub, NULL, { 0,0,0 });
            find_status = status_index.first;
            state_index = status_index.second;

            // If there was insufficient memory for the new state, then return -1.
            if (state_index == -1) {
               delete[] new_state.seq;
               return(-1);
            }

            if (parameters->prn_info > 3) prn_MM_subproblem2(&new_state, 2, find_status, parameters, info);

            // If a better solution has been found, record it.

            if ((*states)[state_index].g1 + (*states)[state_index].g2 < UB) {
               //if (UB < MAX_DEPTH) printf("Old UB = %2d  New UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, (*states)[state_index].g1 + (*states)[state_index].g2, p1_min, p2_min);
               UB = (*states)[state_index].g1 + (*states)[state_index].g2;
               path_found = true;
               info->best_z = UB;
               info->best_branch = info->n_explored_forward + info->n_explored_reverse;
               info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
               status = bibacktrack(states, state_index, info->best_solution);
               if(status == -1) {
                  delete[] new_state.seq;
                  return(-1);
               }
               //printf("UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, p1_min, p2_min);
               //analyze_states(states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);
            }
         }
      } 
   } 

   item = reverse_bfs_heap.get_min();
   if(item.key == -1)
      p2_min = UCHAR_MAX;
   else {
      p2_min = item.key;
      if ((parameters->best_measure == 8) || (parameters->best_measure == 10) || (parameters->best_measure == 10)) p2_min = floor(p2_min);
      if(parameters->best_measure == 9) p2_min = ceil(p2_min);
   }

   if(new_state.seq != NULL) delete[] new_state.seq;
   return(1);
}

//_________________________________________________________________________________________________

int mm_choose_direction(int prev_direction, int rule)
/*
   1. This function chooses the direction for the next state to be expanded.
   2. Input Variables
      a. rule = which rule should be used to choose the direction.
   3. Global Variables
   4. Output Variables
      a. direction = 1 = forward direction
                   = 2 = reverse direction.
   5. Created 1/16/19 by modifying choose_direction from c:\sewell\research\15puzzle\15puzzle_code2\bidirectional.cpp.
*/
{
   int            direction;
   heap_record    item1, item2;
   static double  prev_min1, prev_min2;     // prev_min1 = p1_min in previous iteration.

   if (prev_direction == -1) { prev_min1 = -1; prev_min2 = -1; } 

   switch (rule) {
      case 0:  // Best First Direction (BFD): p1_min <= p2_min => forward
         if (p1_min <= p2_min) direction = 1; else direction = 2;
         break;
      case 1:  // Best First Direction (BFD).  Break ties using tie-breakers built into the priority function.
         if (p1_min < p2_min) {
            direction = 1;
         } else {
            if (p1_min > p2_min) {
               direction = 2;
            } else {
               item1 = forward_bfs_heap.get_min();
               item2 = reverse_bfs_heap.get_min();
               if (item1.key <= item2.key) {             // If this is used with the -b 8 option, I think it will be the same rule as used by Holte et al.
                  direction = 1;
               } else {
                  direction = 2;
               }
            }
         }
         break;
      case 2:  // Two phases.  Phase I: Exploration Phase.  Phase II: Verification Phase.
         if (!path_found) {
            if (p1_min < p2_min) {     // Phase I: If a path has not been found yet, use BFD.  Break ties using tie-breakers built into the priority function.
               direction = 1;
            } else {
               if (p1_min > p2_min) {
                  direction = 2;
               } else {
                  item1 = forward_bfs_heap.get_min();
                  item2 = reverse_bfs_heap.get_min();
                  if (item1.key <= item2.key) {          // If this is used with the -b 8 option, I think it will be the same rule as used by Holte et al.
                     direction = 1;
                  } else {             
                     direction = 2;
                  }
               }
            }
         } else {                      // Phase II: BFD with p_leveling.  Break ties using |O1| <= |O2| => forward.
            if (p1_min < p2_min) {
               direction = 1;
            } else {
               if (p1_min > p2_min) {
                  direction = 2;
               } else {
                  if ((p1_min > prev_min1) || (p2_min > prev_min2)) {
                     if (forward_bfs_heap.n_of_items() <= reverse_bfs_heap.n_of_items()) {
                        direction = 1;
                     } else {
                        direction = 2;
                     }
                     prev_min1 = p1_min;
                     prev_min2 = p2_min;
                  }  else {
                     direction = prev_direction;
                  }
               }
            }
         }
         break;
      default:
         fprintf(stderr, "Unknown direction rule\n");
         exit(1);
         break;
   }

   return(direction);
}

//_________________________________________________________________________________________________

void compute_front_to_front_LB()
{
   int            g, h1, h2, i, j, max1_h1, max1_h2, max2_h1, max2_h2;
   int            LB1_h1_h2[MAX_DEPTH+1][MAX_DEPTH+1], LB2_h1_h2[MAX_DEPTH+1][MAX_DEPTH+1];

   // Compute the largest h1,h2 values in both directions.

   max1_h1 = 0;
   max1_h2 = 0;
   for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
      for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
         if (!open_g1_h1_h2_values[h1][h2].empty()) {
            if (h1 > max1_h1) max1_h1 = h1;
            if (h2 > max1_h2) max1_h2 = h2;
         }
         LB1_h1_h2[h1][h2] = MAX_DEPTH;
      }
   }
   max2_h1 = 0;
   max2_h2 = 0;
   for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
      for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
         if (!open_g2_h1_h2_values[h1][h2].empty()) {
            if (h1 > max2_h1) max2_h1 = h1;
            if (h2 > max2_h2) max2_h2 = h2;
         }
         LB2_h1_h2[h1][h2] = MAX_DEPTH;
      }
   }

   // Compute the front-to-front LB for the forward clusters.

   for (h1 = 0; h1 <= max1_h1; h1++) {
      for (h2 = 0; h2 <= max1_h2; h2++) {
         if (!open_g1_h1_h2_values[h1][h2].empty()) {
            g = open_g1_h1_h2_values[h1][h2].get_min();
            for (i = 0; i <= max2_h1; i++) {
               for (j = 0; j <= max2_h2; j++) {
                  if (!open_g2_h1_h2_values[i][j].empty()) {
                     LB1_h1_h2[h1][h2] = min(LB1_h1_h2[h1][h2], g + max(max(abs(h1 - i), abs(h2 - j)), 1) + open_g2_h1_h2_values[i][j].get_min());
                  }
               }
            }
         }
      }
   }

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max1_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max1_h2; h2++) {
         if (LB1_h1_h2[h1][h2] < MAX_DEPTH) {
            printf(" %2d", LB1_h1_h2[h1][h2]);
         } else {
            printf("  *");
         }
      }
      printf("\n");
   }

   // Compute the front-to-front LB for the reverse clusters.

   for (h1 = 0; h1 <= max2_h1; h1++) {
      for (h2 = 0; h2 <= max2_h2; h2++) {
         if (!open_g2_h1_h2_values[h1][h2].empty()) {
            g = open_g2_h1_h2_values[h1][h2].get_min();
            for (i = 0; i <= max1_h1; i++) {
               for (j = 0; j <= max1_h2; j++) {
                  if (!open_g1_h1_h2_values[i][j].empty()) {
                     LB2_h1_h2[h1][h2] = min(LB2_h1_h2[h1][h2], open_g1_h1_h2_values[i][j].get_min() + max(max(abs(h1 - i), abs(h2 - j)), 1) + g);
                  }
               }
            }
         }
      }
   }

   printf("   ");  for (h2 = 0; h2 <= max2_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max2_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max2_h2; h2++) {
         if (LB2_h1_h2[h1][h2] < MAX_DEPTH) {
            printf(" %2d", LB2_h1_h2[h1][h2]);
         }
         else {
            printf("  *");
         }
      }
      printf("\n");
   }
}

//_________________________________________________________________________________________________

void prn_MM_subproblem(bistate *state, int direction, unsigned char UB, searchparameters *parameters, searchinfo *info)
{
   unsigned char  g1, h1, g2, h2;
   double         p;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if (direction == 1) {
      p = compute_bibest(1, g1, g2, h1, h2, parameters);
      printf("Forward  UB    p1  f1 f1b  g1  h1  g2  h2  e1  e2    n_exp_f    n_gen_f   p1_min = %5.2f  p2_min = %5.2f   g1_min = %5.2f  g2_min = %5.2f  f1_min = %3d  f2_min = %3d  %5d\n", p1_min, p2_min, g1_min, g2_min, open_f1_values.get_min(), open_f2_values.get_min(), forward_bfs_heap.n_of_items());
      if (state->open2 != 2)
         printf("        %3d %5.2f %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", UB, p, g1 + h1, 2 * g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("        %3d %5.2f %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d: ", UB, p, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      prn_sequence(state->seq, n);
   }
   else {
      p = compute_bibest(2, g1, g2, h1, h2, parameters);
      printf("Reverse  UB    p2  f2 f2b  g1  h1  g2  h2  e1  e2    n_exp_r    n_gen_r   p1_min = %5.2f  p2_min = %5.2f   g1_min = %5.2f  g2_min = %5.2f  f1_min = %3d  f2_min = %3d  %5d\n", p1_min, p2_min, g1_min, g2_min, open_f1_values.get_min(), open_f2_values.get_min(), reverse_bfs_heap.n_of_items());
      if (state->open1 != 2)
         printf("        %3d %5.2f %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", UB, p, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("        %3d %5.2f %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d: ", UB, p, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      prn_sequence(state->seq, n);
   }
}

//_________________________________________________________________________________________________

void prn_MM_subproblem2(bistate *state, int direction, int status, searchparameters *parameters, searchinfo *info)
{
   unsigned char  g1, h1, g2, h2;
   double         p;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if (direction == 1) {
      p = compute_bibest(1, g1, g2, h1, h2, parameters);
      if (state->open2 != 2)
         printf("         %2d %5.2f %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", status, p, g1 + h1, 2 * g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("         %2d %5.2f %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d: ", status, p, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      prn_sequence(state->seq, n);
   }
   else {
      p = compute_bibest(2, g1, g2, h1, h2, parameters);
      if (state->open1 != 2)
         printf("         %2d %5.2f %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", status, p, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("         %2d %5.2f %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d: ", status, p, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      prn_sequence(state->seq, n);
   }
}

//_________________________________________________________________________________________________

int check_open_value(bistates_array *states)
/*
1. This function searches through the list of states to check if open_f1_values and open_f2_values are correct.
2. Input Variables
   a. states = store the states.
3. Global Variables
   a. open_f1_values;   This is a multiset representation of the f1 values of the open nodes in the forward direction.
   b. open_f2_values;   This is a multiset representation of the f2 values of the open nodes in the reverse direction.
4. Output Variables
   a. 1 is returned if the values are defined correctly.
   o.w., the program is terminated.
5. Written 3/22/19.
*/
{
   unsigned char     f1, f1_min, f1_max, f2, f2_min, f2_max, g1, g2, h1, h2;
   int               f1_min_g1[MAX_DEPTH + 1], f1_max_g1[MAX_DEPTH + 1], f2_min_g2[MAX_DEPTH + 1], f2_max_g2[MAX_DEPTH + 1];
   int               min_g1_h1_h2[MAX_DEPTH + 1][MAX_DEPTH + 1], max_g1_h1_h2[MAX_DEPTH + 1][MAX_DEPTH + 1], min_g2_h1_h2[MAX_DEPTH + 1][MAX_DEPTH + 1], max_g2_h1_h2[MAX_DEPTH + 1][MAX_DEPTH + 1];
   int               i, j;

   for (i = 0; i <= MAX_DEPTH; i++) {
      f1_min_g1[i] = f2_min_g2[i] = UCHAR_MAX;
      f1_max_g1[i] = f2_max_g2[i] = -1;
   }
   f1_min = UCHAR_MAX; f1_max = 0; f2_min = UCHAR_MAX; f2_max = 0;
   for (i = 0; i <= MAX_DEPTH; i++) {
      for (j = 0; j <= MAX_DEPTH; j++) {
         min_g1_h1_h2[i][j] = min_g2_h1_h2[i][j] = UCHAR_MAX;
         max_g1_h1_h2[i][j] = max_g2_h1_h2[i][j] = -1;
      }
   }
   for (i = 0; i <= (*states).n_of_states() - 1; i++) {
      if ((*states)[i].open1 == 1) {
         g1 = (*states)[i].g1;   h1 = (*states)[i].h1;   h2 = (*states)[i].h2;   f1 = g1 + h1;
         if (f1 < f1_min) f1_min = f1;
         if (f1 > f1_max) f1_max = f1;
         if (g1 < f1_min_g1[f1]) f1_min_g1[f1] = g1;
         if (g1 > f1_max_g1[f1]) f1_max_g1[f1] = g1;
         if (g1 < min_g1_h1_h2[h1][h2]) min_g1_h1_h2[h1][h2] = g1;
         if (g1 > max_g1_h1_h2[h1][h2]) max_g1_h1_h2[h1][h2] = g1;
      }
      if ((*states)[i].open2 == 1) {
         g2 = (*states)[i].g2;   h1 = (*states)[i].h1;   h2 = (*states)[i].h2;   f2 = g2 + h2;
         if (g2 + h2 < f2_min) f2_min = g2 + h2;
         if (g2 + h2 > f2_max) f2_max = g2 + h2;
         if (g2 < f2_min_g2[f2]) f2_min_g2[f2] = g2;
         if (g2 > f2_max_g2[f2]) f2_max_g2[f2] = g2;
         if (g2 < min_g2_h1_h2[h1][h2]) min_g2_h1_h2[h1][h2] = g2;
         if (g2 > max_g2_h1_h2[h1][h2]) max_g2_h1_h2[h1][h2] = g2;
      }
   }
   if (f1_min != open_f1_values.get_min()) {
      fprintf(stderr, "f1_min != open_f1_values.get_min()  Press ENTER to continue\n");
      cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }
   if (f1_max != open_f1_values.get_max()) {
      fprintf(stderr, "f1_max != open_f1_values.get_max()  Press ENTER to continue\n");
      cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }
   if (f2_min != open_f2_values.get_min()) {
      fprintf(stderr, "f2_min != open_f2_values.get_min()  Press ENTER to continue\n");
      cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }
   if (f2_max != open_f2_values.get_max()) {
      fprintf(stderr, "f2_max != open_f2_values.get_max()  Press ENTER to continue\n");
      cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }
   for (i = 0; i <= MAX_DEPTH; i++) {
      if ((f1_min_g1[i] < UCHAR_MAX) && (f1_min_g1[i] != open_g1_values[i].get_min())) {
         fprintf(stderr, "f1_min_g1[i] != open_g1_values[i].get_min()  Press ENTER to continue\n");
         cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
      if (f1_max_g1[i] != open_g1_values[i].get_max()) {
         fprintf(stderr, "f1_max_g1[i] != open_g1_values[i].get_max()  Press ENTER to continue\n");
         cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
      if ((f2_min_g2[i] < UCHAR_MAX) && (f2_min_g2[i] != open_g2_values[i].get_min())) {
         fprintf(stderr, "f2_min_g2[i] != open_g2_values[i].get_min()  Press ENTER to continue\n");
         cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
      if (f2_max_g2[i] != open_g2_values[i].get_max()) {
         fprintf(stderr, "f2_max_g2[i] != open_g2_values[i].get_max()  Press ENTER to continue\n");
         cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
   }
   for (i = 0; i <= MAX_DEPTH; i++) {
      for (j = 0; j <= MAX_DEPTH; j++) {
         if ((min_g1_h1_h2[i][j] < UCHAR_MAX) && (min_g1_h1_h2[i][j] != open_g1_h1_h2_values[i][j].get_min())) {
            fprintf(stderr, "min_g1_h1_h2[i][j] != open_g1_h1_h2_values[i][j].get_min()  Press ENTER to continue\n");
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
         if (max_g1_h1_h2[i][j] != open_g1_h1_h2_values[i][j].get_max()) {
            fprintf(stderr, "max_g1_h1_h2[i][j] != open_g1_h1_h2_values[i][j].get_max()  Press ENTER to continue\n");
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
         if ((min_g2_h1_h2[i][j] < UCHAR_MAX) && (min_g2_h1_h2[i][j] != open_g2_h1_h2_values[i][j].get_min())) {
            fprintf(stderr, "min_g2_h1_h2[i] != open_g2_h1_h2_values[i].get_min()  Press ENTER to continue\n");
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
         if (max_g2_h1_h2[i][j] != open_g2_h1_h2_values[i][j].get_max()) {
            fprintf(stderr, "max_g2_h1_h2[i][j] != open_g2_h1_h2_values[i][j].get_max()  Press ENTER to continue\n");
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
      }
   }
}
