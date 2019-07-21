/*
   1. This project was created on 7/17/17.
      a. I copied the following files from c:\sewell\research\15puzzle\15puzzle_code2:
         heap_record.h, main.h, io.cpp, main.cpp, memory.cpp.
         These files (together with several files that were not needed for this project) implemented a CBFS for the sliding tile puzzle.
   2. This project implements various branch and bound algorithms for the pancake problem.
*/

#include "main.h"

bistates_array states;                 // Stores the states
unsigned char  *goal;                  // goal sequence of pancakes
unsigned char  *inv_source;            // invserse of sequence of pancakes
unsigned char  *source;                // source sequence of pancakes
char           *prob_file;             // problem file
int            n;                      // n = number of pancakes.

int      algorithm;           /* -a option: algorithm
                                 1 = iterative deepening
                                 2 = forward best first search
                                 3 = reverse best first search
                                 4 = bidirectional */
int      best_measure = 1;    /* -b option: best_measure
                                 1 = f = g + h
                                 2 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'
                                 3 = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.
                                 4 = f_bar_d - (g_d /(MAX_DEPTH + 1) Break ties in fbar_d in favor of states with larger g_d.
                                 5 = max(2*g_d, f_d) MM priority function.
                                 6 = max(2*g_d + 1, f_d) MMe priority function.
                                 7 = max(2*g_d, f_d) + (g_d /(MAX_DEPTH + 1) MM priority function.  Break ties in favor of states with smaller g_d.
                                 8 = max(2*g_d + 1, f_d) + (g_d /(MAX_DEPTH + 1) MMe priority function.  Break ties in favor of states with smaller g_d.
                                 9 = max(2*g_d + 1, f_d) - (g_d /(MAX_DEPTH + 1) MMe priority function.  Break ties in favor of states with larger g_d.*/
int      gap_x = 0;           // -g option: Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
int      search_strategy = 3; /* -e option: search (exploration) strategy
                                 1 = depth first search
                                 2 = breadth first search
                                 3 = best first search
                                 4 = best first search using clusters */
int      prn_info;            // -p option: controls level of printed info
double   seed = 3.1567;       // -s option: random seed (def = 3.1567)

// Data structures for selecting the next unexplored state.

min_max_heap   forward_bfs_heap;          // min-max heap for the forward best first search
min_max_heap   reverse_bfs_heap;          // min-max heap for the reverse best first search

// Data strutures for keeping track of g1_min, g2_min, f1_min, f2_min.

min_max_multiset  open_g1_values[MAX_DEPTH + 1];   // open_g1_values[f] is a multiset representation of the g1 values of the open nodes in the forward direction with f1(v) = f.  It is designed for keeping track of the minimum value of g among the open nodes.
min_max_multiset  open_g2_values[MAX_DEPTH + 1];   // open_g2_values[f] is a multiset representation of the g2 values of the open nodes in the reverse direction with f2(v) = f.  It is designed for keeping track of the minimum value of g among the open nodes.
min_max_multiset  open_f1_values;                  // This is a multiset representation of the f1 values of the open nodes in the forward direction.  It is designed for keeping track of the minimum value of f among the open nodes.
min_max_multiset  open_f2_values;                  // This is a multiset representation of the f2 values of the open nodes in the reverse direction.  It is designed for keeping track of the minimum value of f among the open nodes.
min_max_multiset  open_g1_h1_h2_values[MAX_DEPTH + 1][MAX_DEPTH + 1];   // open_g1_h1_h2_values[a][b] is a multiset representation of the g1 values of the open nodes in the forward direction with h1(v) = a and h2(v) = b.  It is designed for keeping track of the minimum value of g among the open nodes.
min_max_multiset  open_g2_h1_h2_values[MAX_DEPTH + 1][MAX_DEPTH + 1];   // open_g2_h1_h2_values[a][b] is a multiset representation of the g2 values of the open nodes in the forward direction with h1(v) = a and h2(v) = b.  It is designed for keeping track of the minimum value of g among the open nodes.


int main(int ac, char **av)
{
   int               direction, status;
   searchparameters  parameters;
   searchinfo        a_star_info, bidirectional_info, info;
   unsigned char     f_lim, f_bar_lim, g_lim, h_lim, UB, z;
   //unsigned char     seq[6] = {5, 1, 3, 2, 5, 4};   // Simple test problem.
   //unsigned char     seq[6] = {5, 2, 4, 3, 5, 1};   // Simple test problem.
   //unsigned char     seq[11] = {10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10};  // Allocate a sequence with 10 pancakes.
   unsigned char     seq[11] = {10,  9,  1, 10,  7,  8,  2,  5,  6,  3,  4};  // A sequence that caused an error in MM.
   //unsigned char     seq[21] = { 20,  4,  5,  2,  6, 13, 20, 19, 14, 17, 12,  3, 11, 15,  1, 18,  7,  9,  8, 10, 16 };  // A randomly generated instance that is hard for reverse A*.
   //unsigned char     seq[31] = {30, 20, 13,  2, 24, 28, 26,  1,  6, 29, 23, 10, 22,  3, 12,  9,  5,  7, 30, 27,  4, 11, 19, 25, 21, 18, 16, 17, 14, 15,  8}; // A randomly generated instance that is hard for forward A*.

   parseargs(ac, av);

   //read_data(prob_file);
   //define_problems(20, 0, 75, seq);

   n = seq[0];
   assert(check_inputs(seq));
   //prn_data(seq, n);

   initialize(seq, seq[0]);

   parameters.algorithm = algorithm;
   parameters.best_measure = best_measure;
   parameters.gap_x = gap_x;
   parameters.search_strategy = search_strategy;
   parameters.cpu_limit = CPU_LIMIT;
   parameters.prn_info = prn_info;
   UB = MAX_DEPTH;
   //UB = 17;
   direction = 1;
   //z = iterative_deepening(seq, &parameters, &info);
   //parameters.best_measure = 1;
   //parameters.algorithm = 2;
   //z = a_star(seq, 1, UB, &parameters, &a_star_info);
   //parameters.best_measure = 1;
   //parameters.algorithm = 3;
   //z = a_star(seq, 2, UB, &parameters, &a_star_info);
   //parameters.best_measure = 2;
   //parameters.algorithm = 4;
   //parameters.gap_x = 5;
   //UB = z;
   //z = bidirectional(seq, UB, &parameters, &bidirectional_info);
   //z = bidirectional2(seq, UB, &parameters, &bidirectional_info);
   //z = MM(seq, UB, &parameters, &bidirectional_info);
   //UB = z;
   //status = bi_expansions(seq, UB, &parameters, &bidirectional_info);
   //f_lim = z; f_bar_lim = z + 5; g_lim = z; h_lim = 2 * z;
   //min_expansions(seq, z, f_lim, f_bar_lim, g_lim, h_lim, &parameters, &bidirectional_info);
   z = FtF(seq, 0, UB, &parameters, &bidirectional_info);

   random_instances();
   //problems();

   //test_vec();
   //test_clusters();

   // Keep the console window open.
   printf("Press ENTER to continue");
   cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   return(0);
}

//_________________________________________________________________________________________________

void parseargs(int ac, char **av)
{
   int c, cnt;

   cnt = 0;
   while (++cnt < ac && av[cnt][0] == '-') {
      c = av[cnt][1];
      switch (c) {
      case 'a':
         algorithm = atoi(av[++cnt]);
         break;
      case 'b':
         best_measure = atoi(av[++cnt]);
         break;
      case 'e':
         search_strategy = atoi(av[++cnt]);
         break;
      case 'g':
         gap_x = atoi(av[++cnt]);
         break;
      case 'p':
         prn_info = atoi(av[++cnt]);
         break;
      case 's':
         seed = atof(av[++cnt]);
         break;
      default:
         usage(*av);
         break;
      }
   }
   if (cnt > ac) usage(*av);
   //prob_file = av[cnt++];
   //if (cnt < ac) usage (*av);

}

//_________________________________________________________________________________________________

void usage(char *prog)
{
   fprintf(stderr, "Usage: %s probfile\n", prog);
   fprintf(stderr, "    -a: algorithm to use\n");
   fprintf(stderr, "    -b: best measure to use (def = 1 = f = g + h)\n");
   fprintf(stderr, "    -e: search (exploration) strategy (def = 3 = Best First Search)\n");
   fprintf(stderr, "    -g: value of X for the GAP-X heuristic. (def = 0)\n");
   fprintf(stderr, "    -p: controls level of printed information (def=0)\n");
   fprintf(stderr, "    -s: seed for random number generation\n");
   exit(1);
}

/*************************************************************************************************/

unsigned char gap_lb(unsigned char *cur_seq, int direction, int x)
/*
   1. This function computes the GAP LB for a sequence of pancakes.
   2. Input Variables
      a. cur_seq = the order of the pancakes.
      b. direction = = 1(2) for forward (reverse) search.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
      a. LB = GAP LB for the sequence of pancakes is returned.
   5. Created 7/17/17 by modifying c:\sewell\research\pancake\matlab\gap_lb.m.
   6. Modified 8/5/17 to compute the GAP-X LB.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
*/
{
   unsigned char  LB;
   int            i;

   LB = 0;
   if (x == -1) return(LB);

   if (direction == 1) {
      for (i = 2; i <= n; i++) {
         if ((cur_seq[i] <= x) || (cur_seq[i - 1] <= x)) continue;
         if (abs(cur_seq[i] - cur_seq[i - 1]) > 1) LB = LB + 1;
      }
      if ((abs(n + 1 - cur_seq[n]) > 1) && (cur_seq[n] > x)) LB = LB + 1;
   } else {
      for (i = 2; i <= n; i++) {
         if ((cur_seq[i] <= x) || (cur_seq[i - 1] <= x)) continue;
         if (abs(inv_source[cur_seq[i]] - inv_source[cur_seq[i - 1]]) > 1) LB = LB + 1;
      }
      if ((abs(n + 1 - inv_source[cur_seq[n]]) > 1) && (cur_seq[n] > x)) LB = LB + 1;
   }

   return(LB);
}

//_________________________________________________________________________________________________

unsigned char update_gap_lb(unsigned char *cur_seq, int direction, int i, unsigned char LB, int x)
/*
   1. This function updates the GAP lower bound when a flip is made at position i.
   2. Input Variables
      a. cur_seq = the order of the pancakes.
      b. direction = = 1(2) for forward (reverse) search.
      c. i = position where the flip is to be made.  I.e., the new sequence is obtained by reversing the sequence
             of pancakes in positions 1 through i.
      d. LB = the GAP LB for cur_seq before the flip.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
      a. LB = GAP LB for the sequence after the flip has been made is returned.
   5. Created 7/17/17 by modifying c:\sewell\research\pancake\matlab\update_gap_lb.m.
   6. Modified 8/7/17 to compute the GAP-X LB.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
   */
{
   int            inv_p1, inv_pi, inv_pi1, p1, pi, pi1;

   if (x == -1) return(0);

   assert((1 <= i) && (i <= n));

   if (direction == 1) {
      p1 = cur_seq[1];
      pi = cur_seq[i];
      if (i < n)
         pi1 = cur_seq[i + 1];
      else
         pi1 = n + 1;

      if ((pi <= x) || (pi1 <= x) || (abs(pi1 - pi) <= 1)) LB = LB + 1;
      if ((p1 <= x) || (pi1 <= x) || (abs(pi1 - p1) <= 1)) LB = LB - 1;
   } else {
      p1 = cur_seq[1];
      pi = cur_seq[i];
      inv_p1 = inv_source[p1];
      inv_pi = inv_source[pi];
      if (i < n) {
         pi1 = cur_seq[i + 1];
         inv_pi1 = inv_source[cur_seq[i + 1]];
      } else {
         pi1 = n + 1;
         inv_pi1 = n + 1;
      }
      if ((pi <= x) || (pi1 <= x) || (abs(inv_pi1 - inv_pi) <= 1)) LB = LB + 1;
      if ((p1 <= x) || (pi1 <= x) || (abs(inv_pi1 - inv_p1) <= 1)) LB = LB - 1;
   }

   return(LB);
}

/*************************************************************************************************/

void initialize(unsigned char *seq, int n_seq)
/*
   1. This function initializes some of the data.
   2. Input Variables
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
   5. Written 7/17/17.
*/
{
   int            i;

   n = n_seq;

   // Save the source sequence.

   source = new unsigned char[n + 1];
   for (i = 1; i <= n; i++) source[i] = seq[i];
   source[0] = n;

   // Create the inverse of the sequence.

   inv_source = new unsigned char[n + 1];
   for (i = 1; i <= n; i++) inv_source[source[i]] = i;

   // Create the goal sequence.

   goal = new unsigned char[n + 1];
   for (i = 1; i <= n; i++) goal[i] = i;
   goal[0] = n;
}

//_________________________________________________________________________________________________

void random_instances()
{
   int            cnt, h1_root_sum_ida, i, n_reps, rep, z_a_star_forward, z_sum_a_star_forward, z_a_star_reverse, z_sum_a_star_reverse, z_b, z_sum_b, z_ida, z_sum_ida;
   __int64        n_explored_sum_a_star_forward, n_generated_sum_a_star_forward;
   __int64        n_explored_sum_a_star_reverse, n_generated_sum_a_star_reverse;
   __int64        n_explored_sum_b, n_generated_sum_b, n_explored_sum_ida, n_generated_sum_ida;
   double         cpu_sum_a_star_forward, cpu_sum_a_star_reverse, cpu_sum_b, cpu_sum_ida;
   unsigned char  f_lim, f_bar_lim, g_lim, h_lim, *seq, UB;
   searchparameters  a_star_parameters, b_parameters, ida_parameters;
   searchinfo        a_star_forward_info, a_star_reverse_info, b_info, ida_info;

   n = 20;
   n_reps = 1;
   seq = new unsigned char[n + 1];

   ida_parameters.algorithm = 1;
   ida_parameters.best_measure = 1;
   ida_parameters.gap_x = 0;
   ida_parameters.search_strategy = search_strategy;
   ida_parameters.cpu_limit = CPU_LIMIT;
   ida_parameters.prn_info = prn_info;
   cpu_sum_ida = 0;
   h1_root_sum_ida = 0;
   n_explored_sum_ida = 0;
   n_generated_sum_ida = 0;
   z_sum_ida = 0;
   cnt = 0;

   a_star_parameters.algorithm = 2;
   a_star_parameters.best_measure = 1;
   a_star_parameters.gap_x = 1;
   a_star_parameters.search_strategy = search_strategy;
   a_star_parameters.cpu_limit = CPU_LIMIT;
   a_star_parameters.prn_info = prn_info;
   UB = MAX_DEPTH;
   cpu_sum_a_star_forward = 0;
   n_explored_sum_a_star_forward = 0;
   n_generated_sum_a_star_forward = 0;
   z_sum_a_star_forward = 0;

   cpu_sum_a_star_reverse = 0;
   n_explored_sum_a_star_reverse = 0;
   n_generated_sum_a_star_reverse = 0;
   z_sum_a_star_reverse = 0;

   b_parameters.algorithm = 4;
   b_parameters.best_measure = 6;
   b_parameters.gap_x = 1;
   b_parameters.search_strategy = search_strategy;
   b_parameters.cpu_limit = CPU_LIMIT;
   b_parameters.prn_info = prn_info;
   cpu_sum_b = 0;
   n_explored_sum_b = 0;
   n_generated_sum_b = 0;
   z_sum_b = 0;

   // Solve a trivial instance using bidirectional search so that the cpu for initializing the data structures is not included in any of the instances solved during the loop.

   seq[0] = n;
   for (i = 1; i <= n; i++) seq[i] = i;
   initialize(seq, n);
   z_b = bidirectional(seq, UB, &b_parameters, &b_info);

   //while(cnt < 100) {
   for (rep = 1; rep <= n_reps; rep++) {
      //printf("rep = %5d\n", rep);
      printf("%5d ", rep);

      // Randomly generate the next problem.

      for (i = 1; i <= n; i++) seq[i] = i;
      random_permutation2(n, seq, &seed);
      initialize(seq, n);
      //if (rep <= 2) continue;

      z_ida = iterative_deepening(seq, &ida_parameters, &ida_info);
      //if(ida_info.best_z == ida_info.h1_root + 0) {
      //   //printf("z = %2d  h1_root = %2d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f: ", ida_info.best_z, ida_info.h1_root, ida_info.n_explored, ida_info.n_generated, ida_info.cpu);
      //   prn_sequence2(seq, n);
      //   cnt++;
      //}
      cpu_sum_ida += ida_info.cpu;
      h1_root_sum_ida += ida_info.h1_root;
      n_explored_sum_ida += ida_info.n_explored;
      n_generated_sum_ida += ida_info.n_generated;
      z_sum_ida += ida_info.best_z;

      //a_star_parameters.algorithm = 2;
      //UB = z_ida;
      //z_a_star_forward = a_star(seq, 1, UB, &a_star_parameters, &a_star_forward_info);
      ////if((z_a_star_forward == -1) || (a_star_forward_info.optimal == 0)) printf("Warning: z_a_star_forward == -1 or a_star_forward_info.optmial == 0\n");
      ////if(z_a_star_forward != z_ida) printf("Error: z_a_star_forward != z_ida\n");
      //cpu_sum_a_star_forward += a_star_forward_info.cpu;
      //n_explored_sum_a_star_forward += a_star_forward_info.n_explored_forward;
      //n_generated_sum_a_star_forward += a_star_forward_info.n_generated_forward;
      //z_sum_a_star_forward += a_star_forward_info.best_z;

      //a_star_parameters.algorithm = 3;
      //UB = z_ida;
      //z_a_star_reverse = a_star(seq, 2, UB, &a_star_parameters, &a_star_reverse_info);
      ////if ((z_a_star_reverse == -1) || (a_star_reverse_info.optimal == 0)) printf("Warning: z_a_star_reverse == -1 or a_star_reverse_info.optmial == 0\n");
      ////if (z_a_star_reverse != z_ida) printf("Error: z_a_star_reverse != z_ida\n");
      //cpu_sum_a_star_reverse += a_star_reverse_info.cpu;
      //n_explored_sum_a_star_reverse += a_star_reverse_info.n_explored_reverse;
      //n_generated_sum_a_star_reverse += a_star_reverse_info.n_generated_reverse;
      //z_sum_a_star_reverse += a_star_reverse_info.best_z;

      //UB = z_ida;
      //z_b = bidirectional(seq, UB, &b_parameters, &b_info);
      //z_b = bidirectional2(seq, UB, &b_parameters, &b_info);
      //prn_sequence(seq, n);
      z_b = MM(seq, UB, &b_parameters, &b_info);
      if ((z_b == -1) || (b_info.optimal == 0)) printf("Warning: z_b == -1 or b_info.optmial == 0\n");
      if (z_b != z_ida) printf("Error: z_b != z_ida\n");
      cpu_sum_b += b_info.cpu;
      n_explored_sum_b += b_info.n_explored;
      n_generated_sum_b += b_info.n_generated;
      z_sum_b += b_info.best_z;
      f_lim = z_ida; f_bar_lim = 2 * z_ida; g_lim = z_ida; h_lim = 2 * z_ida;
      //min_expansions(seq, z_ida, f_lim, f_bar_lim, g_lim, h_lim, &b_parameters, &b_info);

      ////prn_sequence(seq, n);
   }

   printf("%10.3f %10.3f %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f\n", (double)z_sum_ida / n_reps, (double)h1_root_sum_ida / n_reps, (double)n_explored_sum_ida / n_reps, (double)n_generated_sum_ida / n_reps, cpu_sum_ida / n_reps,
      (double)n_explored_sum_a_star_forward / n_reps, (double)n_generated_sum_a_star_forward / n_reps, cpu_sum_a_star_forward / n_reps,
      (double)n_explored_sum_a_star_reverse / n_reps, (double)n_generated_sum_a_star_reverse / n_reps, cpu_sum_a_star_reverse / n_reps,
      (double)n_explored_sum_b / n_reps, (double)n_generated_sum_b / n_reps, cpu_sum_b / n_reps);

   delete[] seq;
}

//_________________________________________________________________________________________________

void problems()
{
   int            cnt, gap, h1_root_sum_ida, i, n_reps, rep, z_a_star_forward, z_sum_a_star_forward, z_a_star_reverse, z_sum_a_star_reverse, z_b, z_sum_b, z_ida, z_sum_ida;
   __int64        n_explored_sum_a_star_forward, n_generated_sum_a_star_forward;
   __int64        n_explored_sum_a_star_reverse, n_generated_sum_a_star_reverse;
   __int64        n_explored_sum_b, n_generated_sum_b, n_explored_sum_ida, n_generated_sum_ida;
   double         cpu_sum_a_star_forward, cpu_sum_a_star_reverse, cpu_sum_b, cpu_sum_ida;
   unsigned char  f_lim, f_bar_lim, g_lim, h_lim, *seq, UB;
   searchparameters  a_star_parameters, b_parameters, ida_parameters;
   searchinfo        a_star_forward_info, a_star_reverse_info, b_info, ida_info;

   n = 16;
   gap = 0;
   n_reps = 50;
   seq = new unsigned char[n + 1];

   ida_parameters.algorithm = 1;
   ida_parameters.best_measure = 1;
   ida_parameters.gap_x = 0;
   ida_parameters.search_strategy = search_strategy;
   ida_parameters.cpu_limit = CPU_LIMIT;
   ida_parameters.prn_info = prn_info;
   cpu_sum_ida = 0;
   h1_root_sum_ida = 0;
   n_explored_sum_ida = 0;
   n_generated_sum_ida = 0;
   z_sum_ida = 0;
   cnt = 0;

   a_star_parameters.algorithm = 2;
   a_star_parameters.best_measure = 1;
   a_star_parameters.gap_x = 3;
   a_star_parameters.search_strategy = search_strategy;
   a_star_parameters.cpu_limit = CPU_LIMIT;
   a_star_parameters.prn_info = prn_info;
   UB = MAX_DEPTH;
   cpu_sum_a_star_forward = 0;
   n_explored_sum_a_star_forward = 0;
   n_generated_sum_a_star_forward = 0;
   z_sum_a_star_forward = 0;

   cpu_sum_a_star_reverse = 0;
   n_explored_sum_a_star_reverse = 0;
   n_generated_sum_a_star_reverse = 0;
   z_sum_a_star_reverse = 0;

   b_parameters.algorithm = 4;
   b_parameters.best_measure = 1;
   b_parameters.gap_x = 3;
   b_parameters.search_strategy = search_strategy;
   b_parameters.cpu_limit = CPU_LIMIT;
   b_parameters.prn_info = prn_info;
   cpu_sum_b = 0;
   n_explored_sum_b = 0;
   n_generated_sum_b = 0;
   z_sum_b = 0;


   for (rep = 1; rep <= n_reps; rep++) {
      //printf("rep = %3d\n", rep);

      // Get the next problem.

      for (i = 1; i <= n; i++) seq[i] = i;
      define_problems(n, gap, rep, seq);
      initialize(seq, n);

      z_ida = iterative_deepening(seq, &ida_parameters, &ida_info);
      //if (ida_info.best_z == ida_info.h1_root + 1) {
      //   //printf("z = %2d  h1_root = %2d  n_explored = %14I64d  n_generated = %14I64d  cpu = %8.2f: ", ida_info.best_z, ida_info.h1_root, ida_info.n_explored, ida_info.n_generated, ida_info.cpu);
      //   prn_sequence2(seq, n);
      //   cnt++;
      //}
      cpu_sum_ida += ida_info.cpu;
      h1_root_sum_ida += ida_info.h1_root;
      n_explored_sum_ida += ida_info.n_explored;
      n_generated_sum_ida += ida_info.n_generated;
      z_sum_ida += ida_info.best_z;

      //a_star_parameters.algorithm = 2;
      //UB = z_ida;
      //z_a_star_forward = a_star(seq, 1, UB, &a_star_parameters, &a_star_forward_info);
      //if ((z_a_star_forward == -1) || (a_star_forward_info.optimal == 0)) printf("Warning: z_a_star_forward == -1 or a_star_forward_info.optmial == 0\n");
      //if (z_a_star_forward != z_ida) printf("Error: z_a_star_forward != z_ida\n");
      //cpu_sum_a_star_forward += a_star_forward_info.cpu;
      //n_explored_sum_a_star_forward += a_star_forward_info.n_explored_forward;
      //n_generated_sum_a_star_forward += a_star_forward_info.n_generated_forward;
      //z_sum_a_star_forward += a_star_forward_info.best_z;

      //a_star_parameters.algorithm = 3;
      //UB = z_ida;
      //z_a_star_reverse = a_star(seq, 2, UB, &a_star_parameters, &a_star_reverse_info);
      //if ((z_a_star_reverse == -1) || (a_star_reverse_info.optimal == 0)) printf("Warning: z_a_star_reverse == -1 or a_star_reverse_info.optmial == 0\n");
      //if (z_a_star_reverse != z_ida) printf("Error: z_a_star_reverse != z_ida\n");
      //cpu_sum_a_star_reverse += a_star_reverse_info.cpu;
      //n_explored_sum_a_star_reverse += a_star_reverse_info.n_explored_reverse;
      //n_generated_sum_a_star_reverse += a_star_reverse_info.n_generated_reverse;
      //z_sum_a_star_reverse += a_star_reverse_info.best_z;

      //UB = z_ida;
      //z_b = bidirectional(seq, UB, &b_parameters, &b_info);
      //z_b = bidirectional2(seq, UB, &b_parameters, &b_info);
      //z_b = MM(seq, UB, &b_parameters, &b_info);
      //if ((z_b == -1) || (b_info.optimal == 0)) printf("Warning: z_b == -1 or b_info.optmial == 0\n");
      //if (z_b != z_ida) printf("Error: z_b != z_ida\n");
      cpu_sum_b += b_info.cpu;
      n_explored_sum_b += b_info.n_explored;
      n_generated_sum_b += b_info.n_generated;
      z_sum_b += b_info.best_z;
      f_lim = z_ida; f_bar_lim = 2 * z_ida; g_lim = z_ida; h_lim = 2 * z_ida;
      min_expansions(seq, z_ida, f_lim, f_bar_lim, g_lim, h_lim, &b_parameters, &b_info);

      ////prn_sequence(seq, n);
   }

   printf("%10.3f %10.3f %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f : %10.3f %10.3f %10.3f\n", (double)z_sum_ida / n_reps, (double)h1_root_sum_ida / n_reps, (double)n_explored_sum_ida / n_reps, (double)n_generated_sum_ida / n_reps, cpu_sum_ida / n_reps,
      (double)n_explored_sum_a_star_forward / n_reps, (double)n_generated_sum_a_star_forward / n_reps, cpu_sum_a_star_forward / n_reps,
      (double)n_explored_sum_a_star_reverse / n_reps, (double)n_generated_sum_a_star_reverse / n_reps, cpu_sum_a_star_reverse / n_reps,
      (double)n_explored_sum_b / n_reps, (double)n_generated_sum_b / n_reps, cpu_sum_b / n_reps);

   delete[] seq;
}

//_________________________________________________________________________________________________

void analyze_states(bistates_array *states, int max_f, int max_e, int max_g, __int64 **Fexp, __int64 **Rexp, __int64 **Fstored, __int64 **Rstored, __int64 **Fexp_g_h, __int64 **Rexp_g_h, __int64 **Fstored_g_h, __int64 **Rstored_g_h, __int64 **Fexp_g_f, __int64 **Rexp_g_f, __int64 **Fstored_g_f, __int64 **Rstored_g_f)
/*
   1. This function analyzes the list of states.
   2. Input Variables
      a. states = store the states.
      b. max_f = number of rows to allocate to F and R.
      c. max_e = number of columns to allocate to F and R.
   3. Ouptut Variables
      a. Fexp(l,e+1) = |{v expanded in forward direction: f1(v) = l, g1(v)-h2(v) = e}|.
      b. Rexp(l,e+1) = |{v expanded in reverse direction: f2(v) = l, g2(v)-h1(v) = e}|.
      c. Fstored(l,e+1) = |{v stored in forward direction: f1(v) = l, g1(v)-h2(v) = e}|.
      d. Rstored(l,e+1) = |{v stored in reverse direction: f2(v) = l, g2(v)-h1(v) = e}|.
      e. Fexp_g_h(g,h)  = |{v expanded in forward direction: g1(v) = g, h1(v)-h2(v) = h}|.
      f. Rexp_g_h(g,h)  = |{v expanded in reverse direction: g2(v) = g, h2(v)-h1(v) = h}|.
      e. Fstored_g_h(g,h)  = |{v stored in forward direction: g1(v) = g, h1(v)-h2(v) = h}|.
      f. Rstored_g_h(g,h)  = |{v stored in reverse direction: g2(v) = g, h2(v)-h1(v) = h}|.
      g. Fexp_g_f(g,f)  = |{v expanded in forward direction: g1(v) = g, f1(v) = f}|.
      h. Rexp_g_f(g,f)  = |{v expanded in reverse direction: g2(v) = g, f2(v) = f}|.
      i. Fstored_g_f(g,f)  = |{v stored in forward direction: g1(v) = g, f1(v) = f}|.
      j. Rstored_g_f(g,f)  = |{v stored in reverse direction: g2(v) = g, f2(v) = f}|.
   4. Created 8/8/17 by modifying c:\sewell\research\pancake\matlab\analyze_states.m.
   5. Modified 4/30/18 to distinguish between expanded nodes and stored nodes.
   6. Modified 1/24/19 to compute Fexp_g_f, Rexp_g_f, Fstored_g_f, Rstored_g_f.
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
   Fexp_g_f = new __int64*[max_g + 1];
   Fstored_g_f = new __int64*[max_g + 1];
   Rexp_g_f = new __int64*[max_g + 1];
   Rstored_g_f = new __int64*[max_g + 1];
   for (i = 0; i <= max_g; i++) {
      Fexp_g_h[i]    = new __int64[2*max_g + 1];
      Fstored_g_h[i] = new __int64[2*max_g + 1];
      for (h = 0; h <= 2*max_g; h++) { Fexp_g_h[i][h] = 0; Fstored_g_h[i][h] = 0; }
      Rexp_g_h[i]    = new __int64[2*max_g + 1];
      Rstored_g_h[i] = new __int64[2*max_g + 1];
      for (h = 0; h <= 2*max_g; h++) { Rexp_g_h[i][h] = 0; Rstored_g_h[i][h] = 0; }

      Fexp_g_f[i] = new __int64[max_f + 1];
      Fstored_g_f[i] = new __int64[max_f + 1];
      for (h = 0; h <= max_f; h++) { Fexp_g_f[i][h] = 0; Fstored_g_f[i][h] = 0; }
      Rexp_g_f[i] = new __int64[max_f + 1];
      Rstored_g_f[i] = new __int64[max_f + 1];
      for (h = 0; h <= max_f; h++) { Rexp_g_f[i][h] = 0; Rstored_g_f[i][h] = 0; }

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

   //printf("     index  f1 f1b  f2 f2b  g1  h1  o1         p1  g2  h2  o2         p2  hash_value\n");
   for (i = 0; i <= (*states).n_of_states() - 1; i++) {
      g1 = (*states)[i].g1;   h1 = (*states)[i].h1;   g2 = (*states)[i].g2;   h2 = (*states)[i].h2;
      //(*states).print_bistate(i);
      if (((*states)[i].open1 == 0) || ((*states)[i].open1 == 1)) {
         f1 = g1 + h1;  e1 = g1 - h2;
         Fstored[f1][e1]++;
         if ((*states)[i].open1 == 0) Fexp[f1][e1]++;
         Fstored_g_h[g1][h1-h2+max_g+1]++;
         Fstored_g_f[g1][f1]++;
         if ((*states)[i].open1 == 0) { Fexp_g_h[g1][h1 - h2 + max_g + 1]++; Fexp_g_f[g1][f1]++; }
         max_e1 = max(max_e1, e1);  max_g1 = max(max_g1, (int)g1);  max_h_diff1 = max(max_h_diff1, (int)h1 - (int)h2);  max_l1 = max(max_l1, f1);  min_h_diff1 = min(min_h_diff1, (int)h1 - (int)h2);  min_l1 = min(min_l1, f1);
      }
      if (((*states)[i].open2 == 0) || ((*states)[i].open2 == 1)) {
         f2 = g2 + h2;  e2 = g2 - h1;
         Rstored[f2][e2]++;
         if ((*states)[i].open2 == 0) Rexp[f2][e2]++;
         Rstored_g_h[g2][h2 - h1 + max_g + 1]++;
         Rstored_g_f[g2][f2]++;
         if ((*states)[i].open2 == 0) { Rexp_g_h[g2][h2 - h1 + max_g + 1]++; Rexp_g_f[g2][f2]++; }
         max_e2 = max(max_e2, e2);  max_g2 = max(max_g2, (int) g2);  max_h_diff2 = max(max_h_diff2, (int)h2 - (int)h1);  max_l2 = max(max_l2, f2);  min_h_diff2 = min(min_h_diff2, (int) h2 - (int) h1);  min_l2 = min(min_l2, f2);
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
         for (h = max_h_diff1; h >= min_h_diff1; h--) printf("%8I64d ", Fexp_g_h[g][h+max_g+1]);
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
   }
   else {
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

   // Print Fexp_g_f.

   if (min_l1 <= max_l1) {
      printf("Forward Expanded: Fexp_g_f(g,f) = |{v: g1(v) = g, f1(v) = f}|\n");
      printf("    "); for (f1 = min_l1; f1 <= max_l1; f1++) printf("%8d ", f1); printf("\n");
      for (g = 0; g <= max_g1; g++) {
         printf("%2d: ", g);
         for (f1 = min_l1; f1 <= max_l1; f1++) printf("%8I64d ", Fexp_g_f[g][f1]);
         printf("\n");
      }
   } else {
      printf("Forward Expanded: Fexp_g_f(g,f) = |{v: g1(v) = g, f1(v) = f}| is empty\n");
   }

   // Print Rexp_g_f.

   if (min_l2 <= max_l2) {
      printf("Reverse Expanded: Rexp_g_f(g,f) = |{v: g2(v) = g, f2(v) = f}|\n");
      printf("    "); for (f2 = min_l2; f2 <= max_l2; f2++) printf("%8d ", f2); printf("\n");
      for (g = 0; g <= max_g2; g++) {
         printf("%2d: ", g);
         for (f2 = min_l2; f2 <= max_l2; f2++) printf("%8I64d ", Rexp_g_f[g][f2]);
         printf("\n");
      }
   } else {
      printf("Reverse Expanded: Rexp_g_f(g,f) = |{v: g2(v) = g, f2(v) = f}| is empty\n");
   }

   // Print Fstored_g_f.

   if (min_l1 <= max_l1) {
      printf("Forward Stored: Fstored_g_f(g,f) = |{v: g1(v) = g, f1(v) = f}|\n");
      printf("    "); for (f1 = min_l1; f1 <= max_l1; f1++) printf("%8d ", f1); printf("\n");
      for (g = 0; g <= max_g1; g++) {
         printf("%2d: ", g);
         for (f1 = min_l1; f1 <= max_l1; f1++) printf("%8I64d ", Fstored_g_f[g][f1]);
         printf("\n");
      }
   } else {
      printf("Forward Stored: Fstored_g_f(g,f) = |{v: g1(v) = g, f1(v) = f}| is empty\n");
   }

   // Print Rstored_g_f.

   if (min_l2 <= max_l2) {
      printf("Reverse Stored: Rstored_g_f(g,f) = |{v: g2(v) = g, f2(v) = f}|\n");
      printf("    "); for (f2 = min_l2; f2 <= max_l2; f2++) printf("%8d ", f2); printf("\n");
      for (g = 0; g <= max_g2; g++) {
         printf("%2d: ", g);
         for (f2 = min_l2; f2 <= max_l2; f2++) printf("%8I64d ", Rstored_g_f[g][f2]);
         printf("\n");
      }
   } else {
      printf("Reverse Stored: Rstored_g_f(g,f) = |{v: g2(v) = g, f2(v) = f}| is empty\n");
   }

}

//_________________________________________________________________________________________________

void reverse_vector(int i, int j, int n, unsigned char *x)
/*
1. This routine reverses the order of continguous portion of a vector.
2. n = number of entries in x, assuming data is stored in x[1], ..., x[n].
3. i = index where reversal should begin.
4. j = index where reversal should end.
5. Must have 1 <= i <= j <= n;
6. Created 7/18/17 by modifying reverse_vector from c:\sewell\research\nms\tsp\c\nms.c.
*/
{
   unsigned char  temp;

   assert((1 <= i) && (i <= n));
   assert((i <= j) && (j <= n));

   while (i < j) {
      temp = x[i];
      x[i] = x[j];
      x[j] = temp;
      i++;
      j--;
   }
}

//_________________________________________________________________________________________________

void reverse_vector2(int j, int n, unsigned char *x, unsigned char *y)
/*
1. This routine reverses the order of continguous portion of a vector.
   a. It does not change x.
   b. It copies x[j], x[j-1], ..., x[1] to y[1], y[2], ..., y[j].
2. n = number of entries in x and y, assuming data is stored in x[1], ..., x[n] and y[1], ..., y[n].
4. j = index where reversal should end.
5. Must have 1 <= j <= n;
6. Created 7/24/17 by modifying reverse_vector from c:\sewell\research\pancake\pancake_code\main.cpp.
*/
{
   int            i;

   assert((1 <= j) && (j <= n));

   i = 1;
   while (j > 0) {
      y[i] = x[j];
      i++;
      j--;
   }
}

//_________________________________________________________________________________________________

unsigned char max(const unsigned char a, const unsigned char b)
/*
   1. This routine returns the maximum of two unsigned char.
   2. Written 5/30/19.
*/
{
   if (a >= b) return(a); else return(b);
}

//_________________________________________________________________________________________________

unsigned char max(const unsigned char a, const unsigned char b, const unsigned char c)
/*
   1. This routine returns the maximum of three unsigned char.
   2. Written 5/30/19.
*/
{
   if (a >= b) {
      if (a >= c) {
         return(a);
      } else {
         if (b >= c) return(b); else return(c);
      }
   } else {
      if (b >= c) return(b); else return(c);
   }
}

//_________________________________________________________________________________________________

unsigned char min(const unsigned char a, const unsigned char b)
/*
   1. This routine returns the minimum of two unsigned char.
   2. Written 5/30/19.
*/
{
   if (a <= b) return(a); else return(b);
}

//_________________________________________________________________________________________________

unsigned char min(const unsigned char a, const unsigned char b, const unsigned char c)
/*
   1. This routine returns the minimum of three unsigned char.
   2. Written 5/30/19.
*/
{
   if (a <= b) {
      if (a <= c) {
         return(a);
      }
      else {
         if (b <= c) return(b); else return(c);
      }
   }
   else {
      if (b <= c) return(b); else return(c);
   }
}

//_________________________________________________________________________________________________

int check_inputs(unsigned char *seq)
/*
   1. This routine performs some simple checks on seq.
      If an error is found, 0 is returned, otherwise 1 is returned.
   2. Written 7/18/17.
*/
{
   int      i, *used;

   used = new int[n + 1];
   for (i = 0; i <= n; i++) used[i] = 0;

   // Check that all the indices in seq are legitimate and that there are no duplicates.

   for (i = 1; i <= n; i++) {
      if ((seq[i] < 1) || (seq[i] > n)) {
         fprintf(stderr, "illegal number in seq\n");
         delete[] used;
         return(0);
      }
      if (used[seq[i]]) {
         fprintf(stderr, "seq contains the same number twice\n");
         delete[] used;
         return(0);
      }
      used[seq[i]] = 1;
   }

   delete[] used;

   return(1);
}

//_________________________________________________________________________________________________

int check_solution(unsigned char *seq, unsigned char *solution, unsigned char z)
/*
   1. This routine performs some simple checks on solution.
      If an error is found, 0 is returned, otherwise 1 is returned.
   2. Written 7/18/17.
*/
{
   unsigned char  *seq2;
   int            i;

   // Check that each entry in solution is a valid number.

   for (i = 1; i <= z; i++) {
      if ((solution[i] < 1) || (solution[i] > n)) {
         fprintf(stderr, "illegal number in solution\n");
         return(0);
      }
   }

   // Copy seq configuration into seq2.

   seq2 = new unsigned char[n + 1];
   for (i = 1; i <= n; i++) seq2[i] = seq[i];

   // Apply the flips in solution to seq2.

   for (i = 1; i <= z; i++) reverse_vector(1, solution[i], n, seq2);

   // Check that the final sequence is the goal sequence.

   for (i = 1; i <= n; i++) {
      if (seq2[i] != goal[i]) {
         fprintf(stderr, "Error: solution is incorrect\n");
         delete[] seq2;
         return(0);
      }
   }

   delete[] seq2;

   return(1);
}

//_________________________________________________________________________________________________

void test_vec()
{
   __int64              n;
   double               cpu;
   clock_t              start_time;
   Cluster_indices      indices;
   typedef  int         element;
   vec<element>         v;
   //vec<cluster_indices>   v;

   n = 100000000;
   //v.allocate(n);

   start_time = clock();
   for (int i = 1; i <= n; i++) {
      //indices.g = 1;   indices.h1 = 2;   indices.h2 = 3;
      v.push(i);
      if (i % 1000000 == 0) printf("%10d  %10I64d  %10I64d\n", i, v.size(), v.n_alloc());
   }
   cpu = (double)(clock() - start_time) / CLOCKS_PER_SEC;
   printf("%8.2f\n", cpu);
   //v.print();

   //for (__int64 i = 1; i <= n/2; i++) {
   //   v.remove(i);
   //   if (i % 1000000 == 0) printf("%10I64d  %10I64d  %10I64d\n", i, v.size(), v.n_alloc());
   //}
   //v.print();

   start_time = clock();
   for (__int64 i = 1; i <= n; i++) {
      v.pop();
      if (i % 1000000 == 0) printf("%10I64d  %10I64d  %10I64d\n", i, v.size(), v.n_alloc());
   }
   cpu = (double)(clock() - start_time) / CLOCKS_PER_SEC;
   printf("%8.2f\n", cpu);
   //v.print();

   // Keep the console window open.
   printf("Press ENTER to continue");
   cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   v.release_memory();

   // Keep the console window open.
   printf("Press ENTER to continue");
   cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

//_________________________________________________________________________________________________

void test_clusters()
{
   unsigned char        g, h1, h2;
   __int64              n;
   double               cpu;
   clock_t              start_time;
   Cluster_indices      indices;
   set<Cluster_indices> s;
   Clusters             clustrs;

   n = 100;
   //v.allocate(n);

   //start_time = clock();
   //for (int i = 1; i <= n; i++) {
   //   indices.g = i % 17;   indices.h1 = 2 * i % 19;   indices.h2 = 3 * i % 23;
   //   s.insert(indices);
   //   if (i % 1000000 == 0) printf("%10d  %10I64d\n", i, s.size());
   //}
   //cpu = (double)(clock() - start_time) / CLOCKS_PER_SEC;
   //printf("%8.2f\n", cpu);
   ////v.print();

   //for (auto p = s.begin(); p != s.end(); p++) {
   //   printf("%3d  %3d  %3d\n", p->g, p->h1, p->h2);
   //}

   clustrs.initialize(0, MAX_DEPTH, MAX_DEPTH);
   start_time = clock();
   for (int i = 1; i <= n; i++) {
      //g = i % 17;   h1 = 2 * i % 19;   h2 = 3 * i % 23;
      g = i % 11;   h1 = 2 * i % 13;   h2 = 3 * i % 17;
      clustrs.insert(1, g, h1, h2, i, NULL);
      clustrs.insert(2, g, h1, h2, i, NULL);
      if (i % 1000000 == 0) printf("%10d  %10I64d\n", i, s.size());
   }
   cpu = (double)(clock() - start_time) / CLOCKS_PER_SEC;
   printf("%8.2f\n", cpu);
   clustrs.compute_LBs();
   clustrs.print_min_g();
   clustrs.print_LBs();
   //clustrs.print_min_g_n_open();
   //clustrs.print_nonempty_clusters();
   clustrs.check_clusters();

   //for (int i = 1; i <= n / 2; i++) {
   //   //g = i % 17;   h1 = 2 * i % 19;   h2 = 3 * i % 23;
   //   g = i % 11;   h1 = 2 * i % 13;   h2 = 3 * i % 17;
   //   clustrs.pop(1, g, h1, h2);
   //   clustrs.pop(2, g, h1, h2);
   //}
   clustrs.check_clusters();

   // Keep the console window open.
   printf("Press ENTER to continue");
   cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
