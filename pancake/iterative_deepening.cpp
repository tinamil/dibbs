
static bool             solved;
static unsigned char    bound;
static unsigned char    *cur_seq;
static unsigned char    cur_sol[MAX_DEPTH + 1];
static int              gap_x;
static __int64          n_explored_depth[MAX_DEPTH + 1];

/*************************************************************************************************/

unsigned char iterative_deepening(unsigned char *seq, searchparameters *parameters, searchinfo *info)
/*
   1. This algorithm performs Iterative Deepening A* (IDA*) to find the minimum number of flips for a pancake problem.
      See Depth-First Iterative-Deepening: An Optimal Admissible Tree Search by  Richard Korf.
   2. Input Variables
      a. seq = the order of the pancakes.
         The elements of seq are stored beginning in seq[1].
      b. info stores information about the search.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. solved = true (false) if the goal position was (not) reached during the search.
      e. bound = limit the search to subproblems whose lower bound is less than or equal to bound.
      f. cur_seq = the current sequence of the pancakes.
      g. cur_sol = flips in the current solution.
      h. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
   4. Output Variables
      a. bound = the minimum number of flips needed to sort the pancakes.
   5. Created 7/18/17 by modifying iterative deepening from c:\sewell\research\15puzzle\15puzzle_code2\iterative_deepening.cpp.
      a. Used c:\sewell\research\pancake\matlab\ida.m as a guide.
*/
{
   unsigned char  f1, g1, h1, old_bound;
   int            i;
   __int64        sum1, sum2;


   info->initialize();
   gap_x = parameters->gap_x;

   // Copy the sequence into cur_seq.
   
   cur_seq = new unsigned char[n + 1];
   cur_seq[0] = n;
   for(i = 1; i <= n; i++) cur_seq[i] = seq[i];
   for(i = 1; i <= MAX_DEPTH; i++) cur_sol[i] = 0;

   // Compute the GAP lower bound.
   
   g1 = 0;
   h1 = gap_lb(cur_seq, 1, gap_x);
   info->h1_root = h1;
   f1 = g1 + h1;
    
   // Perform iterative deepening.

   solved = false;
   bound = f1;
   for(i = 0; i <= MAX_DEPTH; i++) n_explored_depth[i] = 0;
   do {
      for (i = 0; i <= MAX_DEPTH; i++) n_explored_depth[i] = 0;
      old_bound = bound;
      bound = dfs(g1, h1, info);
      info->cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
      //printf("old_bound = %3d   new_bound = %3d  n_explored = %14I64d  n_generated = %14I64d  best_branch = %14I64d  cpu = %8.2f\n", old_bound, bound, info->n_explored, info->n_generated, info->best_branch, info->cpu);
      //for (i = 0; (i <= MAX_DEPTH) && (n_explored_depth[i] > 0); i++) printf("%3d %14I64d\n", i, n_explored_depth[i]); printf("\n");
   } while(solved == false);

   check_solution(seq, info->best_solution, info->best_z);

   info->cpu = (double) (clock() - info->start_time) / CLOCKS_PER_SEC;
   for (i = 0, sum1 = 0; i <= floor(bound/2); i++) sum1 += n_explored_depth[i];
   for (i = 0, sum2 = 0; i <= bound; i++) sum2 += n_explored_depth[i];
   //if (prn_info == 0) {
   //   printf("%2d %2d %7.2f %7.2f %12I64d %12I64d %12I64d  %6.3f\n", bound, info->h1_root, info->cpu, info->best_cpu, info->best_branch, info->n_explored, info->n_generated, (double)sum1 / (double)sum2);
   //}
   if (prn_info > 0) {
      printf("bound = %3d  n_explored = %14I64d  n_generated = %14I64d  best_branch = %14I64d  cpu = %8.2f  %6.3f\n", bound, info->n_explored, info->n_generated, info->best_branch, info->cpu, (double)sum1 / (double)sum2);
   }
   //for(i = 0; (i <= MAX_DEPTH) && (n_explored_depth[i] > 0); i++) printf("%3d %14I64d\n", i, n_explored_depth[i]); printf("\n");
   //prn_solution(tile_in_location, solution, bound);
   return(bound);
}

//_________________________________________________________________________________________________

unsigned char dfs(unsigned char g1, unsigned char h1, searchinfo *info)
/*
   1. This algorithm performs limited Depth First Search (DFS) on the 15-puzzle.
      It is designed to be used within an iterative deepening algorithm.
   2. Input Variables
      a. g1 = number of moves that have been made so far in the forward direction.
      b. h1 = lower bound on the number of moves needed to reach the goal postion.
      c. info stores information about the search.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
      d. solved = true (false) if the goal position was (not) reached during the search.
      e. bound = limit the search to subproblems whose lower bound is less than or equal to bound.
      f. cur_seq = the current sequence of the pancakes.
      g. cur_sol = flips in the current solution.
      h. gap_x = Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
   4. Output Variables
      a. min_bound = minimum bound of subproblems whose lower bound exceeds bound is returned.
   5. Created 7/18/17 by modifying dfs from c:\sewell\research\15puzzle\15puzzle_code2\iterative_deepening.cpp.
      a. Used c:\sewell\research\pancake\matlab\ida_dfs.m as a guide.
*/
{
   unsigned char  b, f1_sub, g1_sub, h1_sub, min_bound;
   int            depth_sub, i;

   info->n_explored++;
   n_explored_depth[g1]++;

   //prn_dfs_subproblem(bound, g1, h1, cur_seq);

   // If a solution has been found, record it and return.

   if(h1 == 0) {
      // Verify that we have found a solution.  The GAP-0 LB guarantees that when h1 = 0, then it is a solution, but other LB's may not.
      if (memcmp(cur_seq, goal, n + 1) == 0) {
         info->best_z = g1;
         for (i = 1; i <= info->best_z; i++) info->best_solution[i] = cur_sol[i];
         info->best_branch = info->n_explored;
         info->best_cpu = (double)(clock() - info->start_time) / CLOCKS_PER_SEC;
         min_bound = g1;
         solved = true;
         return(min_bound);
      }
   }

   // Generate the subproblems.

   min_bound = UCHAR_MAX;
   g1_sub = g1 + 1;
   depth_sub = g1_sub;
   for(i = 2; i <= n; i++) {
      info->n_generated++;

      // Compute the change in h1 for the subproblem.

      h1_sub = update_gap_lb(cur_seq, 1, i, h1, gap_x);
      f1_sub = g1_sub + h1_sub;  // Note: For GAP-x (x > 0), it is possible to generate a solution with f1_sub > bound.  Currently, this solution will not be "found" until that state is explored at a deeper iteration of IDA.
                                 // Could check here if a solution has been found (only check when h1_sub = 0).  If found, it may not be optimal.  But if it is optimal, it would prevent a deeper iteration of IDA.  2/15/18.
      if(f1_sub <= bound) {
         reverse_vector(1, i, n, cur_seq);
         //prn_dfs_subproblem2(g1_sub, h1_sub, cur_seq);
         assert(h1_sub == gap_lb(cur_seq, 1, gap_x));
         cur_sol[depth_sub] = i;
         b = dfs(g1_sub, h1_sub, info);
         reverse_vector(1, i, n, cur_seq);
      } else {
         b = f1_sub;
      }

      if(solved) {
         min_bound = b;
         return(min_bound);
      }
      min_bound = min(min_bound, b);
   }

   return(min_bound);
}
