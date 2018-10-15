#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <queue>
#include <stack>
#include <math.h>
#include <utility>
#include <cassert>
#include "heap_record.h"
#include "min_max_heap.h"
#include "bistates.h"
//#include "min_max_stacks.h"
#include "hash_table.h"

using namespace std;

#define  ABS(i) ((i < 0) ? -(i) : (i) )
#define  MAX(i,j) ((i < j) ? (j) : (i) )
#define  MIN(i,j) ((i < j) ? (i) : (j) )

#define  CPU_LIMIT   100000
#define  STATE_SPACE 300000000      // allocate space for this many states
#define  HEAP_SIZE   300000000      // Allocate space for this many items per heap
#define  MAX_DEPTH   150            // Maximum depth permitted during a search

class searchparameters {
public:
   int      algorithm;        /* -a option: algorithm
                                 1 = iterative deepening
                                 2 = forward best first search
                                 3 = reverse best first search
                                 4 = bidirectional */
   int      best_measure;     /* -b option: best_measure
                                 1 = f = g + h
                                 2 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d' */
   int      gap_x;            // -g option: Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
   int      search_strategy;  /* -e option: search (exploration) strategy
                                 1 = depth first search
                                 2 = breadth first search
                                 3 = best first search
                                 4 = cyclic best first search
                                 5 = cyclic best first search using min_max_stacks
                                 6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps.*/
   double   cpu_limit;     // cpu limit for search process
   int      prn_info;      // Controls level of printed info (def=0)
};


class searchinfo {
public:
   searchinfo()   {  best_z = UCHAR_MAX; h1_root = 0; best_branch = 0; n_generated = 0; n_generated_forward = 0; n_generated_reverse = 0;
                     n_explored = 0; n_explored_forward = 0; n_explored_reverse = 0; start_time = clock(); best_cpu = 0; cpu = 0; states_cpu = 0; optimal = 1;
                     for(int i = 0; i <= MAX_DEPTH; i++) { best_solution[i] = 0; n_explored_depth[i] = 0;}
                  };
   void    initialize() {  best_z = UCHAR_MAX; h1_root = 0; best_branch = 0; n_generated = 0; n_generated_forward = 0; n_generated_reverse = 0;
                           n_explored = 0; n_explored_forward = 0; n_explored_reverse = 0; start_time = clock(); best_cpu = 0; cpu = 0; states_cpu = 0; optimal = 1;
                           for(int i = 0; i <= MAX_DEPTH; i++) { best_solution[i] = 0; n_explored_depth[i] = 0;}
                        };
   unsigned char best_z;                // objective value of best solution found
   unsigned char h1_root;               // lower bound of the root problem
   __int64  best_branch;                // branch at which best solution set was found
   __int64  n_generated;                // # of states generated during the branch and bound algorithm
   __int64  n_generated_forward;        // # of states generated in the forward direction during the branch and bound algorithm
   __int64  n_generated_reverse;        // # of states generated in the reverse direction during the branch and bound algorithm
   __int64  n_explored;                 // # of states explored during the branch and bound algorithm
   __int64  n_explored_forward;         // # of states explored in the forward direction during the branch and bound algorithm
   __int64  n_explored_reverse;         // # of states explored in the reverse direction during the branch and bound algorithm
   clock_t  start_time;                 // starting cpu time
   clock_t  end_time;                   // ending cpu time
   double   best_cpu;                   // time at which best solution was found
   double   cpu;                        // cpu used during search process
   double   states_cpu;                 // cpu used to find and store states
   int      optimal;                    // = 1 if the search verified the optimality of the solution
   unsigned char  best_solution[MAX_DEPTH + 1];    // the optimal sequence of flips
   __int64  n_explored_depth[MAX_DEPTH + 1]; // n_explored_depth[d] = number of states explored during the branch and bound algorithm
};

class sortCriteriaBest{
public:
	bool operator() (const heap_record &j1, const heap_record &j2)const{
		//return j1.key > j2.key;     //sort by increasing order
      return j1.key < j2.key;       //sort by decreasing order
	}
};

extern   bistates_array states;                // Stores the states
extern   unsigned char  *goal;                  // goal sequence of pancakes
extern   unsigned char  *inv_source;            // invserse of sequence of pancakes
extern   unsigned char  *source;                // source sequence of pancakes
extern   char           *prob_file;             // problem file
extern   int            n;                      // n = number of pancakes.

extern   int      algorithm;        /* -a option: algorithm
                                       1 = iterative deepening
                                       2 = forward best first search
                                       3 = reverse best first search
                                       4 = bidirectional */
extern   int      best_measure;     /* -b option: best_measure
                                       1 = f = g + h
                                       2 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'
                                       3 = f_d - (g_d /(MAX_DEPTH + 1) Break ties in f_d in favor of states with larger g_d.*/
//extern   int      gap_x;            // -g option: Value of X for the GAP-X heuristic.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
extern   int      search_strategy;  /* -e option: search (exploration) strategy
                                       1 = depth first search
                                       2 = breadth first search
                                       3 = best first search
                                       4 = cyclic best first search
                                       5 = cyclic best first search using min_max_stacks
                                       6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps. */
extern   int      prn_info;         // -p option: controls level of printed info
extern   double   seed;             // -s option: random seed (def = 3.1567)

// Data structures for selecting the next unexplored state.

extern   min_max_heap   forward_bfs_heap;         // min-max heap for the forward best first search
extern   min_max_heap   reverse_bfs_heap;         // min-max heap for the reverse best first search

// Functions in main.cpp

void parseargs(int ac, char **av);
void usage (char *prog);
unsigned char gap_lb(unsigned char *cur_seq, int direction, int x);
unsigned char update_gap_lb(unsigned char *cur_seq, int direction, int i, unsigned char LB, int x);
void initialize(unsigned char *seq, int n_seq);
void random_instances();
void problems();
void analyze_states(bistates_array *states, int max_f, int max_e, int max_g, __int64 **Fexp, __int64 **Rexp, __int64 **Fstored, __int64 **Rstored, __int64 **Fexp_g_h, __int64 **Rexp_g_h, __int64 **Fstored_g_h, __int64 **Rstored_g_h);
void reverse_vector(int i, int j, int n, unsigned char *x);
void reverse_vector2(int j, int n, unsigned char *x, unsigned char *y);
int check_inputs(unsigned char *seq);
int check_solution(unsigned char *seq, unsigned char *solution, unsigned char z);

// Functions in a_star.cpp

unsigned char a_star(unsigned char *seq, int direction, unsigned char initial_UB, searchparameters *parameters, searchinfo *info);
int explore_forward(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table);
int explore_reverse(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table);
double compute_bibest(int direction, unsigned char g1, unsigned char g2, unsigned char h1, unsigned char h2, searchparameters *parameters);
int backtrack(int direction, bistates_array *states, int index, unsigned char solution[MAX_DEPTH + 1]);
void initialize_search(searchparameters *parameters, bistates_array *states, Hash_table *hash_table);
void reinitialize_search(searchparameters *parameters, searchinfo *info, bistates_array *states, Hash_table *hash_table);
void free_search_memory(searchparameters *parameters, bistates_array *states);
int check_bistate(bistate *state, int gap_x, bistates_array *states, Hash_table *hash_table);

// Functions in bidirectional.cpp

unsigned char bidirectional(unsigned char *seq, unsigned char initial_UB, searchparameters *parameters, searchinfo *info);
int expand_forward(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table);
int expand_reverse(bistates_array *states, searchparameters *parameters, searchinfo *info, Hash_table *hash_table);
int bibacktrack(bistates_array *states, int index, unsigned char solution[MAX_DEPTH + 1]);

// Functions in bimemory.cpp

int search_bimemory(bistate *new_bistate, bistates_array *bistates, searchparameters *parameters, Hash_table *hash_table, int hash_value);
pair<int, int> find_or_insert(bistate *new_state, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, Hash_table *hash_table, int hash_value);
int add_to_bimemory(bistate *bistate, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, Hash_table *hash_table, int hash_index);
int get_bistate(int direction, searchparameters *parameters, searchinfo *info);
void insert_bidirectional_unexplored(heap_record rec, int depth, int direction, searchparameters *parameters);

// Functions in io.cpp

//void read_data(char *f);
void prn_data(unsigned char *seq, int n_seq);
void prnvec(int n, int *vec);
void prn_double_vec(int n, double *vec);
void prnmatrix( int **matrix, int m, int n);
void prn_sequence(unsigned char *seq, int n_seq);
void prn_sequence2(unsigned char *seq, int n_seq);
void prn_dfs_subproblem(unsigned char bound, unsigned char g1, unsigned char h1, unsigned char *seq);
void prn_dfs_subproblem2(unsigned char g1, unsigned char h1, unsigned char *seq);
void prn_a_star_subproblem(bistate *state, int direction, unsigned char UB, searchinfo *info);
void prn_a_star_subproblem2(bistate *state, int direction, int status, searchinfo *info);
//void prn_heap_info();

// Functions in iterative_deepening.cpp

unsigned char iterative_deepening(unsigned char *seq, searchparameters *parameters, searchinfo *info);
unsigned char dfs(unsigned char g1, unsigned char h1, searchinfo *info);

// Functions in memory.cpp

//int search_memory(state *new_state, states_array *states, Hash_table *hash_table, int hash_value, int *hash_index);
//int add_to_memory(state *state, double best, int depth, states_array *states, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_index, int dominance);
//int get_state(searchinfo *info, min_max_stacks *cbfs_stacks);
//void insert_unexplored(heap_record rec, int depth, unsigned char LB, min_max_stacks *cbfs_stacks);

// Functions in problems.cpp

void define_problems(int n, int gap, int i, unsigned char *seq);
void define_problems_10_0(int i, unsigned char *seq);
void define_problems_10_1(int i, unsigned char  *seq);
void define_problems_10_2(int i, unsigned char *seq);
void define_problems_10_3(int i, unsigned char *seq);
void define_problems_20_0(int i, unsigned char *seq);
void define_problems_20_1(int i, unsigned char *seq);
void define_problems_20_2(int i, unsigned char *seq);
void define_problems_20_3(int i, unsigned char *seq);
void define_problems_20_4(int i, unsigned char *seq);
void define_problems_30_0(int i, unsigned char *seq);
void define_problems_30_1(int i, unsigned char *seq);
void define_problems_30_2(int i, unsigned char *seq);
void define_problems_30_3(int i, unsigned char *seq);
void define_problems_30_3(int i, unsigned char *seq);
void define_problems_40_0(int i, unsigned char *seq);
void define_problems_40_1(int i, unsigned char *seq);
void define_problems_40_2(int i, unsigned char *seq);
void define_problems_40_3(int i, unsigned char *seq);

// Functions in random.cpp

double ggubfs(double *dseed);
int randomi(int n, double *dseed);
int random_int_0n(int n, double *dseed);
void random_permutation(int n_s, int *s, double *dseed);
void random_permutation2(int n_s, unsigned char *s, double *dseed);
