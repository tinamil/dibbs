#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <queue>
#include <stack>
#include <math.h>
#include <utility>
//#define NDEBUG
#include <cassert>
#include "heap_record.h"
#include "min_max_heap.h"
#include "states.h"
#include "bistates.h"
#include "stack_int.h"
#include "min_max_stacks.h"
#include "PDB.h"
#include "hash_table.h"

using namespace std;

#define  ABS(i) ((i < 0) ? -(i) : (i) )
#define  MAX(i,j) ((i < j) ? (j) : (i) )
#define  MIN(i,j) ((i < j) ? (i) : (j) )

#define  CPU_LIMIT   100000
//#define  STATE_SPACE 100000000      // allocate space for this many states
//#define  HEAP_SIZE   5000000        // Allocate space for this many items per heap
#define  STATE_SPACE 100000000      // allocate space for this many states
#define  HEAP_SIZE   5000        // Allocate space for this many items per heap
#define  BFS_HEAP_SIZE  STATE_SPACE // Allocate space for this many items in the BFS heap

#define  MAX_DEPTH   150            // Maximum depth permitted during a search

class look_ahead_UB_parameters {
public:
   unsigned char    bound;          // limit the search to subproblems whose total lower bound is <= bound.
   unsigned char    max_z;          // limit the search to subproblems whose z values is <= max_z.
};

class look_ahead_UB_info {
public:
   look_ahead_UB_info()    {  n_generated = 0; n_explored = 0; start_time = clock(); cpu = 0; solved = false;
                              for(int i = 0; i <= MAX_DEPTH; i++) { current_solution[i] = 0; min_solution[i] = 0;}
                           };
   __int64 n_generated;                // # of states generated during the branch and bound algorithm
   __int64 n_explored;                 // # of states stored during the branch and bound algorithm
   clock_t start_time;                 // starting cpu time
   clock_t end_time;                   // ending cpu time
   double  cpu;                        // cpu used during search process
   bool    solved;                     // = true (false) if the goal position was (not) reached during the search 
   unsigned char  min_z_plus_LB;
   unsigned char  current_solution[MAX_DEPTH + 1]; // current_solution[z] = location of the empty tile after z moves in the current solution
   unsigned char  min_solution[MAX_DEPTH + 1];     // min_solution[z] = location of the empty tile after z moves in the solution with the minimum bound
};

class searchparameters {
public:
   int      algorithm;     /* -a option: algorithm
                              1 = depth first search
                              2 = breadth first search
                              3 = best first search
                              4 = cyclic best first search
                              5 = cyclic best first search using min_max_stacks
                              6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps.*/
   int      best_measure;  /* -b option: best_measure
                           1 = LB = g + h
                           2 = g + 1.5 h (AWA*)
                           3 = g - h2
                           4 = best = z + LB - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                           5 = -z - maximum number of moves that can be made without exceeding a bound on the number of uphill moves 
                           6 = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d' */
   double   cpu_limit;     // cpu limit for search process
   int      dpdb_lb;       // -d option: 1 = ue disjoint LB, 0 = do not use (def = 0)
   int      search_direction; /* -e option: search direction used for A* and bidirectional.
                              1 = forward best first search
                              2 = reverse best first search
                              3 = bidirectional */
   int      gen_skip;      // -g option: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0).
   int      direction_rule;/* -r option: rule for choosing the direction in bidirectional search (def = 0)
                              0 = open cardinality rule: |O1| <= |O2| => forward
                              1 = closed cardinality rule: |C1| <= |C2| => forward
                              2 = Best First Direction (BFD): f1_bar_min <= f2_bar_min => forward
                              3 = Best First Direction (BFD) with fbar_leveling.  Break ties using |O1| <= |O2| => forward
                              4 = open cardinality rule with fbar_leveling. |O1| <= |O2| = > forward */
   int      prn_info;      // Controls level of printed info (def=0)
};


class searchinfo {
public:
   searchinfo()   {  best_z = UCHAR_MAX; root_LB = 0; best_branch = 0; n_generated = 0; n_explored = 0; start_time = clock(); 
                     best_cpu = 0; cpu = 0; states_cpu = 0; optimal = 1; cnt = 0;
                     for(int i = 0; i <= MAX_DEPTH; i++) { best_solution[i] = 0; n_explored_depth[i] = 0;}
                  };
   void    initialize() {  best_z = UCHAR_MAX; root_LB = 0; root_h1 = 0; root_h2 = 0; best_branch = 0;
                           n_generated = 0; n_generated_forward = 0; n_generated_reverse = 0;
                           n_explored = 0; n_explored_forward = 0; n_explored_reverse = 0; start_time = clock();
                           best_cpu = 0; cpu = 0; states_cpu = 0; optimal = 1; cnt = 0;
                           for(int i = 0; i <= MAX_DEPTH; i++) { best_solution[i] = 0; n_explored_depth[i] = 0;}
                        };
   unsigned char best_z;                // objective value of best solution found
   unsigned char root_LB;               // lower bound of the root problem
   unsigned char root_h1;               // lower bound in the forward direction of the root problem
   unsigned char root_h2;               // lower bound in the reverse direction of the root problem
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
   unsigned char  best_solution[MAX_DEPTH + 1];    // best_solution[z] = location of the empty tile after z moves in the best solution found
   __int64  n_explored_depth[MAX_DEPTH + 1]; // n_explored_depth[d] = number of states explored during the branch and bound algorithm
   __int64  cnt;
};

class sortCriteriaBest{
public:
	bool operator() (const heap_record &j1, const heap_record &j2)const{
		//return j1.key > j2.key;     //sort by increasing order
      return j1.key < j2.key;       //sort by decreasing order
	}
};

//extern   searchinfo     search_info;
//extern   int            next_state;             // index (in states) where next state can be stored
//extern   state          states[STATE_SPACE+1];  // Stores states
extern   char           *prob_file;             // problem file
extern   int            n_tiles;                // n_tiles = number of tiles.
extern   int            n_rows;                 // n_rows = number of rows.
extern   int            n_cols;                 // n_cols = number of columns.
extern   int            UB;                     // UB = objective value of the best solution found so far.
extern   unsigned char  **distances;            // distances[i][j] = Manhattan distance between location i and location j.
extern   unsigned char  **moves;                // moves[i] = list of possible ways to move the empty tile from location i.
//extern   int            **hash_values;          // hash_values[t][i] = random value for tile t in location i: U[0,HASH_SIZE].

extern   int      algorithm;     /* -a option: algorithm
                                    1 = depth first search
                                    2 = breadth first search
                                    3 = best first search
                                    4 = cyclic best first search 
                                    5 = cyclic best first search using min_max_stacks
                                    6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps. */
extern   int      best_measure;  // -b option: 1 = LB
extern   int      dpdb_lb;       // -d option: 1 = ue disjoint LB, 0 = do not use (def = 0)
extern   int      gen_skip;      // -g option: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0).
extern   int      prn_info;      // -p option: controls level of printed info
extern   int      direction_rule;/* -r option: rule for choosing the direction in bidirectional search (def = 0)
                                    0 = open cardinality rule: |O1| <= |O2| => forward
                                    1 = closed cardinality rule: |C1| <= |C2| => forward
                                    2 = Best First Direction (BFD): f1_bar_min <= f2_bar_min => forward
                                    3 = Best First Direction (BFD) with fbar_leveling.  Break ties using |O1| <= |O2| => forward
                                    4 = open cardinality rule with fbar_leveling. |O1| <= |O2| = > forward */ 
extern   double   seed;          // -s option: random seed (def = 3.1567)

// Data structures for selecting the next unexplored state.

                                          // I am using a min-max heap instead of a priority queue because priority-queues do not permit replacement of the lowest priority item.
extern   min_max_heap   bfs_heap;         // min-max heap for best first search
extern   stack<heap_record> stack_dfs;
extern   queue<heap_record> queue_bfs;
extern   min_max_heap   *cbfs_heaps;      // Array of min-max heaps for cyclic best first search

extern   min_max_heap   forward_bfs_heap;         // min-max heap for best first search
extern   stack<heap_record> forward_stack_dfs;
extern   queue<heap_record> forward_queue_bfs;
extern   min_max_heap   *forward_cbfs_heaps;      // Array of min-max heaps for cyclic best first search

extern   min_max_heap   reverse_bfs_heap;         // min-max heap for best first search
extern   stack<heap_record> reverse_stack_dfs;
extern   queue<heap_record> reverse_queue_bfs;
extern   min_max_heap   *reverse_cbfs_heaps;      // Array of min-max heaps for cyclic best first search


// Functions in main.cpp

void parseargs(int ac, char **av);
void usage (char *prog);
unsigned char compute_Manhattan_LB(unsigned char *tile_in_location);
unsigned char compute_Manhattan_LB2(unsigned char *tile_in_location1, unsigned char *tile_in_location2);
void initialize(int nrows, int ncols, DPDB *DPDB);
void benchmarks(DPDB *DPDB);
void define_problems15(int i, unsigned char *tile_in_location);
void define_problems24(int i, unsigned char *tile_in_location);
int check_tile_in_location(unsigned char *tile_in_location);
int check_solution(unsigned char source[N_LOCATIONS], unsigned char solution[MAX_DEPTH + 1], unsigned char z);
void reverse_vector(int i, int j, int n, unsigned char *x);

// Functions in bidirectional.cpp

unsigned char bidirectional(unsigned char source[N_LOCATIONS], searchparameters *parameters, searchinfo *info, DPDB *DPDB);
int expand_forward(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
int gen_forward_subproblems(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
int expand_reverse(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
int gen_reverse_subproblems(unsigned char source[N_LOCATIONS], unsigned char *source_location, bistates_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
double compute_bibest(int direction, unsigned char g1, unsigned char g2, unsigned char h1, unsigned char h2, searchparameters *parameters);
int choose_direction(int prev_direction, searchparameters *parameters, searchinfo *info);
int bibacktrack(unsigned char source[N_LOCATIONS], bistates_array *states, int index, unsigned char solution[MAX_DEPTH + 1]);
void analyze_states(bistates_array *states, int max_f, int max_e, int max_g, __int64 **Fexp, __int64 **Rexp, __int64 **Fstored, __int64 **Rstored, __int64 **Fexp_g_h, __int64 **Rexp_g_h, __int64 **Fstored_g_h, __int64 **Rstored_g_h);
void initialize_bisearch(searchparameters *parameters, bistates_array *states, min_max_stacks *cbfs_stacks, Hash_table *hash_table);
void reinitialize_bisearch(searchparameters *parameters, searchinfo *info, bistates_array *states, Hash_table *hash_table);
void free_bimemory();
int check_bistate(unsigned char source[N_LOCATIONS], bistate *state, int direction);
void prn_bi_subproblem(bistate *state, int direction, unsigned char UB, searchinfo *info, int prn_config);
void prn_bi_subproblem2(bistate *state, int direction, unsigned char UB, searchinfo *info, int prn_config);

// Functions in bimemory.cpp

int search_bimemory(bistate *new_bistate, bistates_array *bistates, Hash_table *hash_table, int hash_value);
int add_to_bimemory(bistate *bistate, double best, int depth, int direction, bistates_array *bistates, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_index);
pair<int, int> find_or_insert(bistate *new_state, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_value);
pair<int, int> find_or_insert2(bistate *new_state, int direction, bistates_array *bistates, Hash_table *hash_table, int hash_value);
int get_bistate(int direction, searchinfo *info, min_max_stacks *cbfs_stacks);
void insert_bidirectional_unexplored(heap_record rec, int depth, unsigned char LB, int direction, min_max_stacks *cbfs_stacks);


// Functions in cbfs.cpp

unsigned char search(unsigned char source[N_LOCATIONS], searchparameters *parameters, searchinfo *info, DPDB *DPDB);
void explore_state(unsigned char source[N_LOCATIONS], states_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
void gen_subproblems(unsigned char source[N_LOCATIONS], states_array *states, int index, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, DPDB *DPDB, Hash_table *hash_table);
void gen_dfs(unsigned char source[N_LOCATIONS], unsigned char bound, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z, states_array *states, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, state *new_state, DPDB *DPDB, Hash_table *hash_table, int hash_value);
double compute_best(unsigned char source[N_LOCATIONS], unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char z, unsigned char LB, searchparameters *parameters, searchinfo *info);
unsigned char look_ahead_z_dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, char bound_uphill_moves);
int backtrack(states_array *states, int index, unsigned char solution[MAX_DEPTH + 1]);
void initialize_search(searchparameters *parameters, states_array *states, min_max_stacks *cbfs_stacks, Hash_table *hash_table);
void reinitialize_search(searchparameters *parameters, searchinfo *info, states_array *states, Hash_table *hash_table);
void free_search_memory();
int check_state(state *state);

// Functions in heuristics.cpp

unsigned char look_ahead_UB(unsigned char *tile_in_location, unsigned char look_ahead);
unsigned char look_ahead_UB_dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, 
                             unsigned char LB, unsigned char z, look_ahead_UB_parameters *parameters, look_ahead_UB_info *info);
unsigned char examine_current_path(unsigned char *initial_configuration, unsigned char *solution, unsigned char LB, unsigned char z, unsigned char look_ahead, look_ahead_UB_parameters *parameters, look_ahead_UB_info *info);

// Functions in ID_DIBBS.cpp

unsigned char ID_DIBBS(unsigned char *source, unsigned char initial_UB, searchinfo *info, DPDB *DPDB);
unsigned char forward_dfs(unsigned char *source_location, bistate *state, unsigned char *solution, bistates_array *states, Hash_table *hash_table, searchinfo *info, DPDB *DPDB);
unsigned char reverse_dfs(unsigned char *source_location, bistate *state, unsigned char *solution, bistates_array *states, Hash_table *hash_table, searchinfo *info, DPDB *DPDB);
void search_states(bistates_array *states, unsigned char bound);

// Functions in io.cpp

//void read_data(char *f);
void prn_data(unsigned char *tile_in_location);
void prnvec(int n, int *vec);
void prn_double_vec(int n, double *vec);
void prnmatrix( int **matrix, int m, int n);
void prn_configuration(unsigned char *tile_in_location);
void prn_distances();
void prn_moves();
void prn_dfs_subproblem(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z);
void prn_dfs_subproblem2(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_goal, unsigned char LB_to_source, unsigned char z);
void prn_forward_dfs_subproblem(unsigned char *tile_in_location, unsigned char bound1, unsigned char g1, unsigned char h1, unsigned char h2, unsigned char empty_location, unsigned char prev_location, int prn_config);
void prn_reverse_dfs_subproblem(unsigned char *tile_in_location, unsigned char bound2, unsigned char g2, unsigned char h1, unsigned char h2, unsigned char empty_location, unsigned char prev_location, int prn_config);
void prn_solution(unsigned char *tile_in_location, unsigned char *solution, unsigned char z, DPDB *DPDB);
void prn_heap_info();

// Functions in iterative_deepening.cpp

unsigned char iterative_deepening(unsigned char *tile_in_location, DPDB *DPDB);
unsigned char dfs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z, unsigned char *solution, DPDB *DPDB);
int check_dfs_inputs(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB);

// Functions in lb.cpp

__int64 difference_LB(unsigned char *source, unsigned char UB, unsigned char max_diff, int direction, DPDB *DPDB);
void diff_dfs(unsigned char *source_location, unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_source, unsigned char LB_to_goal, unsigned char z, unsigned char max_LB, unsigned char UB, unsigned char max_diff, DPDB *DPDB);
void forward_diff_dfs(unsigned char *source_location, unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_source, unsigned char LB_to_goal, unsigned char z, unsigned char max_LB, unsigned char UB, unsigned char max_diff, DPDB *DPDB);

// Functions in memory.cpp

int search_memory(state *new_state, states_array *states, Hash_table *hash_table, int hash_value, int *hash_index);
int add_to_memory(state *state, double best, int depth, states_array *states, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_index, int dominance);
int get_state(searchinfo *info, min_max_stacks *cbfs_stacks);
void insert_unexplored(heap_record rec, int depth, unsigned char LB, min_max_stacks *cbfs_stacks);

// Functions in random.cpp

double ggubfs(double *dseed);
int randomi(int n, double *dseed);
int random_int_0n(int n, double *dseed);
void random_permutation(int n_s, int *s, double *dseed);