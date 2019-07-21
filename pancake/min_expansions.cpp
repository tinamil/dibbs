#include "main.h"
#include <queue>

static double           p1_min;        // = min {p1(v): v is in open set of nodes in the forward direction}.
static double           p2_min;        // = min {p2(v): v is in open set of nodes in the reverse direction}.
static int              gap_x;         // = value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
static unsigned char    UB;            // = objective value of best solution found so far.

/*************************************************************************************************/

void min_expansions(unsigned char *seq, unsigned char C_star, unsigned char f_lim, unsigned char f_bar_lim, unsigned char g_lim, unsigned char h_lim, searchparameters *parameters, searchinfo *info)
/*
   1. This function analyzes the miniumum number of states that must be expanded by a bidirectional heuristic search algorithm.  See
      a. Sufficient Conditions for Node Expansion in Bidirectional Heuristic Search, Eckerle et al., 2017.
      b. Front-to-End Bidirectional Heuristic Search with Near-Optimal Node Expansions, Chen et al., 2017.
      c. The Minimal Set of States That Must be Expanded in a Front-to-End Bidirectional Search, Shaham et al. 2017.
      d. Minimizing Node Expansions in Bidirectional Search with Consistent Heuristics, Shaham et al., 2018.
   2. Input Variables
      a. seq = the order of the pancakes.
         The elements of seq are stored beginning in seq[1].
      b. C_star = length of the shortest path.
      c. f_lim = Only expand nodes with f < f_lim.
      d. f_bar_lim = Only expand nodes with f_bar <= f_bar_lim.
      e. g_lim = Only expand nodes with g < g_lim.
      f. h_lim = Estimate of the largest h-value that will be encoutered.  h_lim is used to allocate space for several arrays.
      g. parameters = controls the search.  See bidirectional for details.
      h. info = collects information about the search.
   3. Ouptut Variables
   4. Written 4/11/19.
*/
{
   int               i, j, k, status, stop, t1, t2;
   unsigned char     f1, f2, f1_bar, f2_bar, g1, g2, h1, h2, max_f1_bar, max_f2_bar, max_g1, max_g2, min_C, min_g1, min_g2, min_f1_bar, min_f2_bar;
   __int64           *cum_sum_f1_bar, *cum_sum_f2_bar, *cum_sum_g1, *cum_sum_g2, min_sum, *n_f1_bar, *n_f2_bar, *n_g1, *n_g2, ***n_g1_h1_h2,  ***n_g2_h1_h2, sum;

   status = bi_expansions(seq, f_lim, f_bar_lim, g_lim, parameters, info);
   if (status == -1) { fprintf(stderr, "bi_expansions status == -1\n"); return; }

   // n_f1_bar[a] = # of states with f1_bar = a.  n_f2_bar[a] = # of states with f2_bar = a.
   // n_g1[a] = # of states with g1 = a.  n_g2[a] = # of states with g2 = a.
   // n_g1_h1_h2[a][b][c] = # of states with g1 = a, h1 = b, and h2 = c.  n_g1_h1_h2[a][b][c] = # of states with g2 = a, h1 = b, and h2 = c.

   n_f1_bar = new __int64[f_bar_lim + 1];       if (n_f1_bar == NULL) { fprintf(stderr, "Out of space for n_f1_bar  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   n_f2_bar = new __int64[f_bar_lim + 1];       if (n_f2_bar == NULL) { fprintf(stderr, "Out of space for n_f2_bar  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   for (i = 0; i <= f_bar_lim; i++) { n_f1_bar[i] = 0; n_f2_bar[i] = 0; }
   n_g1 = new __int64[g_lim + 1];               if (n_g1 == NULL) { fprintf(stderr, "Out of space for n_g1  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   n_g2 = new __int64[g_lim + 1];               if (n_g2 == NULL) { fprintf(stderr, "Out of space for n_g2  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   for (i = 0; i <= g_lim; i++) { n_g1[i] = 0; n_g2[i] = 0; }
   n_g1_h1_h2 = new __int64**[g_lim + 1];       if (n_g1_h1_h2 == NULL) { fprintf(stderr, "Out of space for n_g1_h1_h2  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   n_g2_h1_h2 = new __int64**[g_lim + 1];       if (n_g2_h1_h2 == NULL) { fprintf(stderr, "Out of space for n_g2_h1_h2  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   for (i = 0; i <= g_lim; i++) {
      n_g1_h1_h2[i] = new __int64*[h_lim + 1];  if (n_g1_h1_h2[i] == NULL) { fprintf(stderr, "Out of space for n_g1_h1_h2[i]  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
      n_g2_h1_h2[i] = new __int64*[h_lim + 1];  if (n_g2_h1_h2[i] == NULL) { fprintf(stderr, "Out of space for n_g2_h1_h2[i]  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
      for (j = 0; j <= h_lim; j++) {
         n_g1_h1_h2[i][j] = new __int64[h_lim + 1];   if (n_g1_h1_h2[i][j] == NULL) { fprintf(stderr, "Out of space for n_g1_h1_h2[i][j]  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
         n_g2_h1_h2[i][j] = new __int64[h_lim + 1];   if (n_g2_h1_h2[i][j] == NULL) { fprintf(stderr, "Out of space for n_g2_h1_h2[i][j]  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
         for (k = 0; k <= h_lim; k++) {
            n_g1_h1_h2[i][j][k] = 0;
            n_g2_h1_h2[i][j][k] = 0;
         }
      }
   }

   min_C = UCHAR_MAX;
   max_g1 = 0;
   max_g2 = 0;
   min_g1 = INT_MAX;
   min_g2 = INT_MAX;
   max_f1_bar = 0;
   max_f2_bar = 0;
   min_f1_bar = INT_MAX;
   min_f2_bar = INT_MAX;

   //printf("     index  f1 f1b  f2 f2b  g1  h1  o1         p1  g2  h2  o2         p2  hash_value\n");
   for (i = 0; i <= states.n_of_states() - 1; i++) {
      g1 = states[i].g1;   h1 = states[i].h1;   g2 = states[i].g2;   h2 = states[i].h2;
      assert(h1 <= h_lim); assert(h2 <= h_lim);
      if (g1 + g2 < min_C) min_C = min_C = g1 + g2;
      //(*states).print_bistate(i);
      if ((states[i].open1 == 0) || (states[i].open1 == 1)) {
         f1 = g1 + h1;
         f1_bar = 2 * g1 + h1 - h2;
         assert(g1 <= g_lim); assert(f1_bar <= f_bar_lim);
         n_f1_bar[f1_bar]++;
         n_g1[g1]++;
         n_g1_h1_h2[g1][h1][h2]++;
         max_g1 = max(max_g1, g1);  max_f1_bar = max(max_f1_bar, f1_bar);  min_g1 = min(min_g1, g1);  min_f1_bar = min(min_f1_bar, f1_bar);
      }
      if ((states[i].open2 == 0) || (states[i].open2 == 1)) {
         f2 = g2 + h2;
         f2_bar = 2 * g2 + h2 - h1;
         assert(g2 <= g_lim); assert(f2_bar <= f_bar_lim);
         n_f2_bar[f2_bar]++;
         n_g2[g2]++;
         n_g2_h1_h2[g2][h1][h2]++;
         max_g2 = max(max_g2, g2);  max_f2_bar = max(max_f2_bar, f2_bar);  min_g2 = min(min_g2, g2);  min_f2_bar = min(min_f2_bar, f2_bar);
      }
   }
   //if (C_star != min_C) { fprintf(stderr, "C_star != min_C in bi_expansions\n"); return; } // This test may fail because I am not storing states with f(v) = C_star.

   cum_sum_f1_bar = new __int64[f_bar_lim + 1];    if (cum_sum_f1_bar == NULL) { fprintf(stderr, "Out of space for cum_sum_f1_bar  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   cum_sum_f2_bar = new __int64[f_bar_lim + 1];    if (cum_sum_f2_bar == NULL) { fprintf(stderr, "Out of space for cum_sum_f2_bar  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   cum_sum_f1_bar[0] = n_f1_bar[0];
   cum_sum_f2_bar[0] = n_f2_bar[0];
   for (i = 1; i <= f_bar_lim; i++) {
      cum_sum_f1_bar[i] = cum_sum_f1_bar[i - 1] + n_f1_bar[i]; 
      cum_sum_f2_bar[i] = cum_sum_f2_bar[i - 1] + n_f2_bar[i];
   }
   cum_sum_g1 = new __int64[g_lim + 1];               if (cum_sum_g1 == NULL) { fprintf(stderr, "Out of space for cum_sum_g1  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   cum_sum_g2 = new __int64[g_lim + 1];               if (cum_sum_g2 == NULL) { fprintf(stderr, "Out of space for cum_sum_g2  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); return; }
   cum_sum_g1[0] = n_g1[0];
   cum_sum_g2[0] = n_g2[0];
   for (i = 1; i <= g_lim; i++) {
      cum_sum_g1[i] = cum_sum_g1[i - 1] + n_g1[i];
      cum_sum_g2[i] = cum_sum_g2[i - 1] + n_g2[i];
   }

   // Compute the minimum number of nodes that must be expanded using the method defined in "The Minimal Set of States That Must be Expanded in a Front-to-End Bidirectional Search."
   // lb(u,v) = max(f1(u), f2(v), g1(u)+g2(v)).  (u,v) is a must-expand pair if lb(u,v) < C*.
   // For this to work correctly, 1. set f_lim = C*, g_lim = C*; 2. do not use f_bar_lim; 3. expand nodes even if they are closed in the opposite direction; 4. use best measure 1.
   // Note: I have defined t1 and t2 slightly differently than in the paper.

   assert(C_star == g_lim);
   sum = cum_sum_g2[C_star - 1];
   min_sum = sum;
   t1 = -1; t2 = C_star - 1;
   for (i = 0; i < C_star - 2; i++) {
      j = C_star - i - 2;
      if (cum_sum_g1[i] + cum_sum_g2[j] < min_sum) {
         min_sum = cum_sum_g1[i] + cum_sum_g2[j];
         t1 = i;
         t2 = j;
      }
   }
   if (cum_sum_g1[C_star - 1] < min_sum) { min_sum = cum_sum_g1[C_star - 1]; t1 = C_star - 1; t2 = -1; }


   // Print number of states by g values.

   //printf(" g1       n_g1     cum_sum   g2       n_g2     cum_sum\n");
   //for (i = 0; i <= g_lim; i++) {
   //   printf("%3d %10I64d  %10I64d  ", i, n_g1[i], cum_sum_g1[i]);
   //   j = g_lim - i;
   //   printf("%3d %10I64d  %10I64d\n", j, n_g2[j], cum_sum_g2[j]);
   //}
   //printf("C_star = %3d  n_exp = %10I64d  n_exp_f = %10I64d  n_exp_r = %10I64d  t1 = %3d  t2 = %3d  %10I64d\n", C_star, info->n_explored, info->n_explored_forward, info->n_explored_reverse, t1, t2, min_sum);
   printf("%3d  %10I64d  %10I64d  %10I64d  %3d  %3d  %10I64d\n", C_star, info->n_explored, info->n_explored_forward, info->n_explored_reverse, t1, t2, min_sum);

   // Print number of states by f_bar values.

   //printf("f1_bar  n_f1_bar    cum_sum  f2_bar  n_f2_bar      cum_sum\n");
   //for (i = 0; i <= f_bar_lim; i++) {
   //   printf("   %3d", i);
   //   if ((min_f1_bar <= i) && (i <= max_f1_bar)) printf("%10I64d ", n_f1_bar[i]); else printf("%10I64d ", 0);
   //   printf("%10I64d     ", cum_sum_f1_bar[i]);
   //   j = f_bar_lim - i;
   //   printf("%3d", j);
   //   if ((min_f2_bar <= j) && (j <= max_f2_bar)) printf("%10I64d   ", n_f2_bar[j]); else printf("%10I64d   ", 0);
   //   printf("%10I64d", cum_sum_f2_bar[j]); 
   //   printf("\n");
   //}
   //printf("f1_bar  n_f1_bar   n_f2_bar      cum_sum    cum_sum\n");
   //for (i = min(min_f1_bar,min_f2_bar); i <= f_bar_lim; i++) {
   //   printf("   %3d", i);
   //   if ((min_f1_bar <= i) && (i <= max_f1_bar)) printf("%10I64d ", n_f1_bar[i]); else printf("%10d ", 0);
   //   if ((min_f2_bar <= i) && (i <= max_f2_bar)) printf("%10I64d   ", n_f2_bar[i]); else printf("%10d   ", 0);
   //   printf("%10I64d %10I64d\n", cum_sum_f1_bar[i], cum_sum_f2_bar[i]);
   //}

   // n_g1_h1_h2 and n_g2_h1_h2 have been defined in preparation for computing the minimum number of node expansions as defined in
   // "Minimizing Node Expansions in Bidirectional Search with Consistent Heuristics."  
   // This paper exploits the consistency of the heuristic.  
   // lbfs(u,v) = max(f1(u), f2(v), g1(u)+g2(v)+h_C(u,v), where h_C(u,v) = max(|h1(u)-h1(v)|, |h2(u)-h2(v)|).
   // They prove that a minimum vertex cover of an auxiliary graph corresponds to the minimum number of nodes that must be
   // expanded, but do not provide an efficient algorithm for computing the MVC.
   // This code is incomplete.

   // Print number of states by h1, h2, g values.

   //printf("h1 h2  g n_g1_h1_h2 n_g2_h1_h2\n");
   //for (i = 0; i <= h_lim; i++) {
   //   for (j = 0; j <= h_lim; j++) {
   //      for (k = 0; k <= g_lim; k++) {
   //         if ((n_g1_h1_h2[k][i][j] > 0) || (n_g2_h1_h2[k][i][j])) {
   //            printf("%2d %2d %2d %10I64d %10I64d\n", i, j, k, n_g1_h1_h2[k][i][j], n_g2_h1_h2[k][i][j]);
   //         }
   //      }
   //   }
   //}

   free_search_memory(parameters, &states);
   delete[] cum_sum_f1_bar;
   delete[] cum_sum_f2_bar;
   delete[] n_f1_bar;
   delete[] n_f2_bar;
   delete[] n_g1;
   delete[] n_g2;
   delete[] cum_sum_g1;
   delete[] cum_sum_g2;
   for (i = 0; i <= g_lim; i++) {
      for (j = 0; j <= h_lim; j++) {
         delete[] n_g1_h1_h2[i][j];
         delete[] n_g2_h1_h2[i][j];
      }
      delete[] n_g1_h1_h2[i];
      delete[] n_g2_h1_h2[i];
   }
   delete[] n_g1_h1_h2;
   delete[] n_g2_h1_h2;
}

//_________________________________________________________________________________________________

int bi_expansions(unsigned char *seq, unsigned char f_lim, unsigned char f_bar_lim, unsigned char g_lim, searchparameters *parameters, searchinfo *info)
/*
   1. This function calls exp_forward and exp_reverse to expand nodes in both the forward and reverse direction for the pancake problem.
      a. In the forward direction, it expands all nodes v such that f1(v) < f_lim and g1(v) < g_lim.
         i. The code can be modified to
            a. Include f1_bar(v) <= f_bar_lim in the test.
            b. Not expand nodes that are closed in the reverse direction. 
      b. In the reverse direction, it expands all nodes v such that f2(v) < f_lim and g2(v) < g_lim.
         i. The code can be modified to 
            a. Include f2_bar(v) <= f_bar_lim in the test.
            b. Not expand nodes that are closed in the reverse direction.
      c. To save memory, nodes that are not qualified to be expanded are not stored.
   2. The purpose of the function is to store the expanded nodes in states, so that they can be analyzed to determine the 
      minimum number of node expansions.
   3. Note: The code has been written to use p1_min and p2_min.  You can use different best measures to use 
      use f1_min,f2_min or f1_bar_min,f2_bar_min instead.
   4. Input Variables
      a. seq = the order of the pancakes.
         The elements of seq are stored beginning in seq[1].
      b. upper_bound = the upper bound for the search.
      c. parameters = controls the search.  See bidirectional for details.
   5. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. goal = goal sequence of pancakes.
      e. forward_bfs_heap = min-max heap for the forward best first search.
      f. reverse_bfs_heap = min-max heap for the reverse best first search.
      g. states = stores the states.
      h. p1_min = min {p1(v): v is in open set of nodes in the forward direction}.
      i. p2_min = min {p2(v): v is in open set of nodes in the reverse direction}.
      j. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      k. UB = objective value of best solution found so far.
   6. Output Variables
      a. 0 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
   7. Created 4/11/19 by modifying bidirectional from c:\sewell\research\pancake\pancake_code\bidirectional.cpp.
*/
{
   int            direction, hash_value, i, status;
   double         best, cpu;
   __int64        **Fexp = NULL, **Fstored = NULL, **Rexp = NULL, **Rstored = NULL, **Fexp_g_h = NULL, **Fstored_g_h = NULL, **Rexp_g_h = NULL, **Rstored_g_h = NULL, **Fexp_g_f = NULL, **Fstored_g_f = NULL, **Rexp_g_f = NULL, **Rstored_g_f = NULL;
   Hash_table     hash_table;
   bistate        root_state;			// state for the root problem
   pair<int, int> status_index;

   assert(check_inputs(seq));

   // Initialize data structures.

   info->initialize();
   initialize_search(parameters, &states, &hash_table, NULL);
   gap_x = parameters->gap_x;

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
   best = 2 * root_state.g1 + root_state.h1 - root_state.h2;
   p1_min = 2*root_state.g1 + root_state.h1 - root_state.h2;

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

   best = 2 * root_state.g2 + root_state.h2 - root_state.h1;
   p2_min = 2 * root_state.g2 + root_state.h2 - root_state.h1;

   // Need to add the reverse root problem to the list of states and the set of unexplored states.

   hash_value = root_state.hash_value;
   status_index = find_or_insert(&root_state, best, 0, 2, &states, parameters, info, &hash_table, hash_value, NULL, { 0,0,0 });

   // Main loop

   direction = -1;
   while ((p1_min < UCHAR_MAX) || (p2_min < UCHAR_MAX)) {
      cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
      if(cpu > CPU_LIMIT) return(-1);

      direction = min_expansions_choose_direction(direction, 2, info);          // Choose the direction.

      if(direction == 1) {
         status = exp_forward(f_lim, f_bar_lim, g_lim, &states, parameters, info, &hash_table);
         if(status == -1) return(-1);
      } else {
         status = exp_reverse(f_lim, f_bar_lim, g_lim, &states, parameters, info, &hash_table);
         if(status == -1) return(-1);
      }
   }

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;

   info->n_explored = info->n_explored_forward + info->n_explored_reverse;
   info->n_generated = info->n_generated_forward + info->n_generated_reverse;

   //printf("UB = %2d  p1_min = %4.1f  p1_min = %4.1f\n", UB, p1_min, p2_min); 
   //analyze_states(&states, UB + 5, UB, UB, Fexp, Rexp, Fstored, Rstored, Fexp_g_h, Rexp_g_h, Fstored_g_h, Rstored_g_h, Fexp_g_f, Fstored_g_f, Rexp_g_f, Rstored_g_f);

   //free_search_memory(parameters, &states);
   return(1);
}

//_________________________________________________________________________________________________

int exp_forward(unsigned char f_lim, unsigned char f_bar_lim, unsigned char g_lim, bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
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
      h. p1_min = min {p1(v): v is in open set of nodes in the forward direction}.
      i. p2_min = min {p2(v): v is in open set of nodes in the reverse direction}.
      j. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      k. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
   6. Created 4/11/19 by modifying expand_forward from c:\sewell\research\pancake\pancake_code\bidirectional.cpp.
*/
{
   unsigned char     *cur_seq, f1_sub, f1_bar_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
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

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   hash_value = state->hash_value;

//   if (state->open2 > 0) {             // Do not expand the node if it already is closed in the opposite direction.
      info->n_explored_forward++;
      info->n_explored_depth[g1]++;

      if (prn_info > 2) prn_a_star_subproblem(state, 1, UB, info);
      if ((prn_info > 1) && (info->n_explored_forward % 10000 == 0)) prn_a_star_subproblem(state, 1, UB, info);

      // Create new state and fill in values that will be the same for all subproblems.

      g1_sub = g1 + 1;
      g2_sub = UCHAR_MAX;
      new_state.g1 = g1_sub;
      new_state.open1 = 1;
      new_state.parent1 = index;
      new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); return(-1); }
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
         f1_bar_sub = 2 * g1_sub + h1_sub - h2_sub;
         //if ((f1_sub < f_lim) && (f1_bar_sub <= f_bar_lim) && (g1_sub < g_lim)) {
         if ((f1_sub < f_lim) && (g1_sub < g_lim)) {
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

            if (parameters->prn_info > 3) prn_a_star_subproblem2(&new_state, 1, find_status, info);
         }
      }
      delete[] new_state.seq;
//   }

   item = forward_bfs_heap.get_min();
   if (item.key == -1)
      p1_min = UCHAR_MAX;
   else
      p1_min = item.key;

   return(1);
}

//_________________________________________________________________________________________________

int exp_reverse(unsigned char f_lim, unsigned char f_bar_lim, unsigned char g_lim, bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table)
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
      h. p1_min = min {p1(v): v is in open set of nodes in the forward direction}.
      i. p2_min = min {p2(v): v is in open set of nodes in the reverse direction}.
      j. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
      k. UB = objective value of best solution found so far.
   5. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
         O.w. 1 is returned.
      b. info is used to collect information about the search.
   6. Created 4/11/19 by modifying expand_reverse from c:\sewell\research\pancake\pancake_code\bidirectional.cpp.
*/
{
   unsigned char     *cur_seq, f2_sub, f2_bar_sub, g1, g2, g1_sub, g2_sub, h1, h2, h1_sub, h2_sub;
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

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;
   cur_seq = state->seq;
   hash_value = state->hash_value;

//   if (state->open1 > 0) {             // Do not expand the node if it already is closed in the opposite direction.
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
      new_state.seq = new unsigned char[n + 1];   if (new_state.seq == NULL) { fprintf(stderr, "Out of space for new_state.seq\n"); return(-1); }
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
         f2_bar_sub = 2 * g2_sub + h2_sub - h1_sub;
         //if ((f2_sub < f_lim) && (f2_bar_sub <= f_bar_lim) && (g2_sub < g_lim)) {
         if ((f2_sub < f_lim) && (g2_sub < g_lim)) {
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

            if (parameters->prn_info > 3) prn_a_star_subproblem2(&new_state, 2, find_status, info);
         }
      }
      delete[] new_state.seq;
//   }

   item = reverse_bfs_heap.get_min();
   if(item.key == -1)
      p2_min = UCHAR_MAX;
   else
      p2_min = item.key;

   return(1);
}

//_________________________________________________________________________________________________

int min_expansions_choose_direction(int prev_direction, int rule, searchinfo *info)
/*
   1. This function chooses the direction for the next state to be expanded.
   2. Input Variables
      a. prev_direction = direction expanded in previous iteration.  Set to -1 for the first iteration.
      a. rule = which rule should be used to choose the direction.
   3. Global Variables
   4. Output Variables
      a. direction = 1 = forward direction
                   = 2 = reverse direction.
   5. Created 5/9/19 by modifying choose_direction from c:\sewell\research\pancake\pancake_code\meet_in_middle.cpp.
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
      case 1:  // Open cardinality rule with p_leveling.  |O1| <= |O2| => forward.
         if (forward_bfs_heap.empty()) return(2);
         if (reverse_bfs_heap.empty()) return(1);
         if ((p1_min > prev_min1) || (p2_min > prev_min2)) {
            if (forward_bfs_heap.n_of_items() <= reverse_bfs_heap.n_of_items()) {
               direction = 1;
            } else {
               direction = 2;
            }
            prev_min1 = p1_min;
            prev_min2 = p2_min;
         } else {
            direction = prev_direction;
         }
         break;
      case 2:  // Closed cardinality rule with p_leveling.  |C1| <= |C2| => forward.
         if (forward_bfs_heap.empty()) return(2);
         if (reverse_bfs_heap.empty()) return(1);
         if ((p1_min > prev_min1) || (p2_min > prev_min2)) {
            if (info->n_explored_forward <= info->n_explored_reverse) {
               direction = 1;
            }
            else {
               direction = 2;
            }
            prev_min1 = p1_min;
            prev_min2 = p2_min;
         }
         else {
            direction = prev_direction;
         }
         break;
      default:
         fprintf(stderr, "Unknown direction rule\n");
         exit(1);
         break;
   }

   return(direction);
}

