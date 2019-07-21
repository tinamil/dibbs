#include "main.h"
#include <queue>

static bool             path_found;    // = true (false) if a path has (not) been found.
static int              gap_x;         // = value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."

static unsigned char    LB;            // = lower bound on the objective value.
static unsigned char    UB;            // = objective value of best solution found so far.

/*************************************************************************************************/

unsigned char FtF(unsigned char *seq, unsigned char epsilon, unsigned char initial_UB, searchparameters *parameters, searchinfo *info)
/*
   These functions implements a Front-to-Front (FtF) bidirectional search algorithm.
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
      b. epsilon = length of the longest edge in the graph.
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
      e. states = stores the states.
      f. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      g. LB = lower bound on the objective value.
      h. UB = objective value of best solution found so far.
   5. Output Variables
      a. best_z = the best objective value that was found.
            -1 is returned if an error occurs, such as running out of memory.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 5/29/19 by modifying MM from c:\sewell\research\pancake\pancake_code\meet_in_middle.cpp.
*/
{
   unsigned char     h1, h2;
   int               direction, hash_value, i, start, status, stop;
   double            best, cpu;
   static double     prev_min = -1, prev_min1 = -1, prev_min2 = -1;     // prev_min = min(p1_min, p2_min) in previous iteration, prev_min1 = p1_min in previous iteration, prev_min2 = p2_min in previous iteration.
   __int64           **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   Hash_table        hash_table;
   bistate           root_state;			// state for the root problem
   pair<int, int>    status_index;
   tuple<int, Cluster_indices, Min_values>   direction_indices_min_values;
   Clusters          clusters;
   Cluster_indices   indices;
   Min_values        min_vals;

   assert(check_inputs(seq));

   // Initialize data structures.

   info->initialize();
   initialize_search(parameters, &states, &hash_table, &clusters);
   clusters.set_eps(epsilon);
   gap_x = parameters->gap_x;
   UB = initial_UB;

   // Create the forward root problem and add it to the list of states and the set of unexplored states.

   root_state.seq = new unsigned char[n + 1];   if(root_state.seq == NULL) {fprintf(stderr, "Out of space for root_state.seq\n"); info->optimal = 0; return(-1);}
   h1 = gap_lb(seq, 1, gap_x);                  // = lower bound on the number of moves needed to reach the goal postion
   h2 = gap_lb(seq, 2, gap_x);                  // = lower bound on the number of moves needed to reach the source postion
   hash_value = hash_table.hash_seq(seq);       // = the index in the hash table to begin searching for the state.
   root_state.fill(0, h1, 1, -1, -1, UCHAR_MAX, h2, 2, -1, -1, hash_value, seq, n);
   info->h1_root = h1;
   best = compute_bibest(1, root_state.g1, root_state.g2, root_state.h1, root_state.h2, parameters);
   status_index = find_or_insert(&root_state, best, 0, 1, &states, parameters, info, &hash_table, hash_value, &clusters, Cluster_indices(0, h1, h2));

   // Create the reverse root problem and add it to the list of states and the set of unexplored states.

   h1 = gap_lb(goal, 1, gap_x);      // = lower bound on the number of moves needed to reach the goal postion
   h2 = gap_lb(goal, 2, gap_x);      // = lower bound on the number of moves needed to reach the source postion
   hash_value = hash_table.hash_seq(goal);      // = the index in the hash table to begin searching for the state.
   root_state.fill(UCHAR_MAX, h1, 2, -1, -1, 0, h2, 1, -1, -1, hash_value, goal, n);
   best = compute_bibest(2, root_state.g1, root_state.g2, root_state.h1, root_state.h2, parameters);
   status_index = find_or_insert(&root_state, best, 0, 2, &states, parameters, info, &hash_table, hash_value, &clusters, Cluster_indices(0, h1, h2));

   // Main loop

   direction = -1;
   path_found = false;
   LB = max(h1, h2);
   while ((UB > LB) && (info->optimal >= 0)) {
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if(cpu > CPU_LIMIT) {
         info->optimal = 0;
         break;
      }

      //if(prev_min < min(p1_min, p2_min)) {
      //   printf("UB = %2d  LB = %2d  p1_min = %4.1f  p2_min = %4.1f  n_exp_f = %12I64d  n_exp_r = %12I64d  |O1| = %12d  |O2| = %12d\n", UB, LB, p1_min, p2_min, info->n_explored_forward, info->n_explored_reverse, forward_bfs_heap.n_of_items(), reverse_bfs_heap.n_of_items());
      //   //analyze_states(&states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);
      //   prn_open_g_h1_h2_values2(1); printf("\n"); prn_open_g_h1_h2_values2(2);
      //   compute_front_to_front_LB();
      //   prev_min = min(p1_min, p2_min);
      //}

      direction_indices_min_values = clusters.choose_cluster(parameters->best_measure);
      direction = get<0>(direction_indices_min_values);   indices = get<1>(direction_indices_min_values);   min_vals = get<2>(direction_indices_min_values);
      clusters.check_clusters();
      clusters.print_nonempty_clusters();

      if(direction == 1) {
         status = expand_forward_cluster(&clusters, indices, &states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0;
            break;
         }
      } else {
         status = expand_reverse_cluster(&clusters, indices, &states, parameters, info, &hash_table);
         if(status == -1) {
            info->optimal = 0; 
            break;
         }
      }

      LB = max(min_vals.f1_hat_min, min_vals.f2_hat_min);      // Note: f1_hat_min should equal f2_hat_min, so we shouldn't need to take the maximum of the two.  
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

int expand_forward_cluster(Clusters *clusters, Cluster_indices indices, bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function expands the nodes in the cluster specified by indices in the forward direction.
   2. Input Variables
      a. clusters = stores the clusters of open nodes.
      b. indices = the indices of the cluster of nodes to be expanded.
      c. states = store the states.
      d. parameters = controls the search.  See bidirectional for details.
      e. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. states = stores the states.
      f. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      g. LB = lower bound on the objective value.
      h. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 5/30/19 by modifying mm_expand_forward from c:\sewell\research\pancake\pancake_code\meet_in_middle.cpp.
*/
{
   unsigned char     *cur_seq, f1_hat, f1_hat_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, state_index, status;
   __int64           **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   double            best;
   State_index       index;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); info->optimal = 0; return(-1); }

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   //while (!(*clusters)(1, indices.g, indices.h1, indices.h2).empty()) {
   while (!clusters->empty(1, indices.g, indices.h1, indices.h2)) {

      // Get the next node to expand.

      index = get_bistate(states, 1, parameters, info, clusters, indices);
      //index = get_bistate_from_cluster(clusters, indices, states, 1);
      assert(index != -1);
      state = &(*states)[index];
      assert(check_bistate(state, gap_x, states, hash_table));
      assert(state->open1 == 1);
      state->open1 = 0;         // Close this subproblem.

      g1 = state->g1;
      h1 = state->h1;
      g2 = state->g2;
      h2 = state->h2;
      hash_value = state->hash_value;
      assert(indices.g == g1);   assert(indices.h1 == h1);   assert(indices.h2 == h2);
      f1_hat = g1 + clusters->LB1(h1, h2);
      if (f1_hat >= UB) { fprintf(stderr, "f1_hat >= UB in expand_forward_cluster.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
      cur_seq = state->seq;

      if (state->open2 > 0) {
         info->n_explored_forward++;
         info->n_explored_depth[g1]++;

         if (prn_info > 2) prn_MM_subproblem(state, 1, UB, parameters, info);
         if ((prn_info > 1) && (info->n_explored_forward % 10000 == 0)) prn_MM_subproblem(state, 1, UB, parameters, info);

         // Create new state and fill in values that will be the same for all subproblems.

         g1_sub = g1 + 1;
         g2_sub = UCHAR_MAX;
         new_state.g1 = g1_sub;
         new_state.open1 = 1;
         new_state.parent1 = index;
         memcpy(new_state.seq, state->seq, n + 1);
         new_state.g2 = UCHAR_MAX;
         new_state.open2 = 2;
         new_state.parent2 = -1;
         new_state.cluster_index2 = -1;
         depth1 = g1_sub;

         // Generate all the subproblems from this state.

         for (i = 2; i <= n; i++) {
            info->n_generated_forward++;

            // Compute the change in h1 and h2 for the subproblem.

            h1_sub = update_gap_lb(cur_seq, 1, i, h1, gap_x);
            h2_sub = update_gap_lb(cur_seq, 2, i, h2, gap_x);

            f1_hat_sub = g1_sub + clusters->LB1(h1_sub, h2_sub);
            if (f1_hat_sub < UB) {
               reverse_vector2(i, n, cur_seq, new_state.seq);
               hash_value_sub = hash_table->update_hash_value(cur_seq, i, hash_value);
               new_state.h1 = h1_sub;
               new_state.h2 = h2_sub;
               new_state.hash_value = hash_value_sub;
               assert(hash_value_sub == hash_table->hash_seq(new_state.seq)); assert(h1_sub == gap_lb(new_state.seq, 1, gap_x)); assert(h2_sub == gap_lb(new_state.seq, 2, gap_x));
               best = compute_bibest(1, g1_sub, g2_sub, h1_sub, h2_sub, parameters);
               status_index = find_or_insert(&new_state, best, depth1, 1, states, parameters, info, hash_table, hash_value_sub, clusters, Cluster_indices(g1_sub, h1_sub, h2_sub));
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
                  if (LB >= UB) return(1);
               }
            }
         }
      }  else {
         fprintf(stderr, "Exploring a node that is closed in the opposite direction.\n"); info->optimal = 0; return(-1);
      }
   }

   delete[] new_state.seq;
   return(1);
}

//_________________________________________________________________________________________________

int expand_reverse_cluster(Clusters *clusters, Cluster_indices indices, bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
/*
   1. This function expands the nodes in the cluster specified by indices in the reverse direction.
   2. Input Variables
      a. clusters = stores the clusters of open nodes.'=
      b. indices = the indices of the cluster of nodes to be expanded.
      c. states = store the states.
      d. parameters = controls the search.  See bidirectional for details.
      e. hash_table = hash table used to find states.
   4. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. states = stores the states.
      f. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      g. LB = lower bound on the objective value.
      h. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
         best_solution = the optimal sequence of flips.
   6. Created 5/30/19 by modifying mm_expand_reverse from c:\sewell\research\pancake\pancake_code\meet_in_middle.cpp.
*/
{
   unsigned char     *cur_seq, f2_hat, f2_hat_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
   int               depth1, find_status, hash_value, hash_value_sub, i, state_index, status;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   double            best;
   State_index       index;
   bistate           new_state, *state;
   heap_record       item;
   pair<int, int>    status_index;

   new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); info->optimal = 0; return(-1); }

   // Find the next node to explore.  Nodes may be in the heaps more than once, so delete the node with minimum key
   // until an open node has been found.

   while (!clusters->empty(2, indices.g, indices.h1, indices.h2)) {

      // Get the next node to expand.

      index = get_bistate(states, 2, parameters, info, clusters, indices);
      //index = get_bistate_from_cluster(clusters, indices, states, 2);
      assert(index != -1);
      state = &(*states)[index];
      assert(check_bistate(state, gap_x, states, hash_table));
      assert(state->open2 == 1);
      state->open2 = 0;         // Close this subproblem.

      g1 = state->g1;
      h1 = state->h1;
      g2 = state->g2;
      h2 = state->h2;
      hash_value = state->hash_value;
      assert(indices.g == g2);   assert(indices.h1 == h1);   assert(indices.h2 == h2);
      f2_hat = g2 + clusters->LB2(h1, h2);
      if (f2_hat >= UB) { fprintf(stderr, "f2_hat >= UB in expand_reverse_cluster.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
      cur_seq = state->seq;

      if (state->open1 > 0) {
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
         new_state.cluster_index1 = -1;
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

            f2_hat_sub = g2_sub + clusters->LB2(h1_sub, h2_sub);
            if (f2_hat_sub < UB) {
               reverse_vector2(i, n, cur_seq, new_state.seq);
               hash_value_sub = hash_table->update_hash_value(cur_seq, i, hash_value);
               new_state.h1 = h1_sub;
               new_state.h2 = h2_sub;
               new_state.hash_value = hash_value_sub;
               assert(hash_value_sub == hash_table->hash_seq(new_state.seq)); assert(h1_sub == gap_lb(new_state.seq, 1, gap_x)); assert(h2_sub == gap_lb(new_state.seq, 2, gap_x));
               best = compute_bibest(2, g1_sub, g2_sub, h1_sub, h2_sub, parameters);
               status_index = find_or_insert(&new_state, best, depth1, 2, states, parameters, info, hash_table, hash_value_sub, clusters, Cluster_indices(g2_sub, h1_sub, h2_sub));
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
                  if (status == -1) {
                     delete[] new_state.seq;
                     return(-1);
                  }
                  //printf("UB = %2d  p1_min = %4.1f  p2_min = %4.1f\n", UB, p1_min, p2_min);
                  //analyze_states(states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);
                  if (LB >= UB) return(1);
               }
            }
         }
      } else {
         fprintf(stderr, "Exploring a node that is closed in the opposite direction.\n"); info->optimal = 0; return(-1);
      }
   }

   delete[] new_state.seq;
   return(1);
}

