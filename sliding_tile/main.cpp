/*
   1. This project was created on 11/10/11.  
      a. I copied the following files from c:\sewell\research\facility\facility_cbfns:
         heap_record.h, main.h, min_max_heap.h, io.cpp, main.cpp, memory.cpp, min_max_heap.cpp, neighborhood_search.cpp.
         These files (together with several files that were not needed for this project) implemented a CBFNS for the quadratic assignment problem.
   2. This project implements various branch and bound algorithms for the 15-puzzle.
*/

#include "main.h"

#define  ABS(i) ((i < 0) ? -(i) : (i) )

//searchinfo     search_info;
//int            next_state;             // index (in states) where next state can be stored
//state          states[STATE_SPACE+1];  // Stores states
char           *prob_file;             // problem file
int            n_tiles;                // n_tiles = number of tiles.
int            n_rows;                 // n_rows = number of rows.
int            n_cols;                 // n_cols = number of columns.
int            UB;                     // UB = objective value of the best solution found so far
unsigned char  **distances;            // distances[i][j] = Manhattan distance between location i and location j.
unsigned char  **moves;                // moves[i] = list of possible ways to move the empty tile from location i.
//int            **hash_values;          // hash_values[t][i] = random value for tile t in location i: U[0,HASH_SIZE].


int      algorithm = 4; /* -a option: algorithm
                           1 = depth first search
                           2 = breadth first search
                           3 = best first search
                           4 = cyclic best first search
                           5 = cyclic best first search using min_max_stacks
                           6 = CBFS: Cylce through LB instead of depth.  Use min-max heaps. */
int      best_measure = 1; /* -b option: best_measure
                              1 = LB = g + h
                              2 = g + 1.5 h (AWA*)
                              3 = g - h2
                              4 = best = z + LB - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                              5 = -z - maximum number of moves that can be made without exceeding a bound on the number of uphill moves
                              6   = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d' */
int      search_direction = 1;   /* -e option: search direction used for A* and bidirectional.
                                    1 = forward best first search
                                    2 = reverse best first search
                                    3 = bidirectional */
int      gen_skip = 0;     // -g option: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0).
int      dpdb_lb = 0;      // -d option: 1 = ue disjoint LB, 0 = do not use (def = 0)
int      prn_info;         // -p option: controls level of printed info
int      direction_rule=0; /* -r option: rule for choosing the direction in bidirectional search (def = 0)
                              0 = open cardinality rule: |O1| <= |O2| => forward
                              1 = closed cardinality rule: |C1| <= |C2| => forward
                              2 = Best First Direction (BFD): f1_bar_min <= f2_bar_min => forward
                              3 = Best First Direction (BFD) with fbar_leveling.  Break ties using |O1| <= |O2| => forward
                              4 = open cardinality rule with fbar_leveling. |O1| <= |O2| = > forward */ 
double   seed = 3.1567;    // -s option: random seed (def = 3.1567)

// Data structures for selecting the next unexplored state.

                                          // I am using a min-max heap instead of a priority queue because priority-queues do not permit replacement of the lowest priority item.
min_max_heap   bfs_heap;                  // min-max heap for best first search
stack<heap_record> stack_dfs;
queue<heap_record> queue_bfs;
min_max_heap   *cbfs_heaps;               // Array of min-max heaps for distributed best first search

min_max_heap   forward_bfs_heap;                  // min-max heap for best first search
stack<heap_record> forward_stack_dfs;
queue<heap_record> forward_queue_bfs;
min_max_heap   *forward_cbfs_heaps;               // Array of min-max heaps for distributed best first search

min_max_heap   reverse_bfs_heap;                  // min-max heap for best first search
stack<heap_record> reverse_stack_dfs;
queue<heap_record> reverse_queue_bfs;
min_max_heap   *reverse_cbfs_heaps;               // Array of min-max heaps for distributed best first search

int main(int ac, char **av)
{
   unsigned char  z;
   int            i;
   DPDB           DPDB;
   searchparameters  parameters;
   searchinfo        info;
   unsigned char     pattern[3] = {1, 4, 5};
   //unsigned char  tile_in_location[16] = {1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};   // Simple test problem.
   //unsigned char  tile_in_location[16] = {4, 2, 6, 1, 5, 0, 7, 3, 8, 9, 10, 11, 12, 13, 14, 15};   // Simple test problem.
   unsigned char  tile_in_location[16] = {14, 13, 15,  7, 11, 12,  9,  5,  6,  0,  2,  1,  4,  8, 10,  3};   // Problem 1 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {13,  5,  4, 10,  9, 12,  8, 14,  2,  3,  7,  1,  0, 15, 11,  6};   // Problem 2 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = { 2, 11, 15, 5,  13,  4,  6,  7, 12,  8, 10,  1,  9,  3, 14,  0};   // Problem 7 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = { 5,  9, 13, 14,  6,  3,  7, 12, 10,  8,  4,  0, 15,  2, 11,  1};   // Problem 11 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {14,  1,  9,  6,  4,  8, 12,  5,  7,  2,  3,  0, 10, 11, 13, 15};   // Problem 12 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = { 3,  6,  5,  2, 10,  0, 15, 14,  1,  4, 13, 12,  9,  8, 11,  7 };  // Problem 13 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {15, 14,  0,  4, 11,  1,  6, 13,  7,  5,  8,  9,  3,  2, 10, 12};   // Problem 17 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {12,  8, 15, 13,  1,  0,  5,  4,  6,  3,  2, 11,  9,  7, 14, 10};   // Problem 31 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {14, 10,  2,  1, 13,  9,  8, 11,  7,  3,  6, 12, 15,  5,  4,	0};   // Problem 82 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {14,  7,  8,  2, 13, 11, 10,  4,  9, 12,  5,  0,  3,  6,  1, 15};   // Problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {15,  2, 12, 11, 14, 13,  9,  5,  1,  3,  8,  7,  0, 10,  6,  4};   // Hardest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {};   // Problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {14, 1, 9, 6, 4, 8, 12, 5, 7, 2, 3, 0, 10, 11, 13, 15};   // Easiest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[16] = {15, 2, 12, 11, 14, 13, 9, 5, 1, 3, 8, 7, 0, 10, 6, 4};   // Hardest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
   //unsigned char  tile_in_location[25] = {14,  5,  9,  2, 18,  8, 23, 19, 12, 17, 15,  0, 10, 20,  4,  6, 11, 21,  1,  7, 24,  3, 16, 22, 13};   // First problem from Disjoint Pattern Database Heuristics
   //unsigned char  tile_in_location[25] = {16,  5,  1, 12,  6, 24, 17,  9,  2, 22,  4, 10, 13, 18, 19, 20,  0, 23,  7, 21, 15, 11,  8,  3, 14};   // Second problem from Disjoint Pattern Database Heuristics
   //unsigned char  tile_in_location[25] = { 3, 17,  9,  8, 24,  1, 11, 12, 14,  0,  5,  4, 22, 13, 16, 21, 15,  6,  7, 10, 20, 23,  2, 18, 19};   // Problem 25 from Disjoint Pattern Database Heuristics
   //unsigned char  tile_in_location[25] = { 1, 12, 18, 13, 17, 15,  3,  7, 20,  0, 19, 24,  6,  5, 21, 11,  2,  8,  9, 16, 22, 10,  4, 23, 14};   // Problem 32 from Disjoint Pattern Database Heuristics
   //unsigned char  tile_in_location[25] = {10,  3, 24, 12,  0,  7,  8, 11, 14, 21, 22, 23,  2,  1,  9, 17, 18,  6, 20,  4, 13, 15,  5, 19, 16};   // Easiest problem from Disjoint Pattern Database Heuristics
   //unsigned char  tile_in_location[25] = {23,  1, 12,  6, 16,  2, 20, 10, 21, 18, 14, 13, 17, 19, 22,  0, 15, 24,  3,  7,  4,  8,  5,  9, 11};   // Hardest problem from Disjoint Pattern Database Heuristics

   parseargs (ac, av);

   //read_data(prob_file);
   initialize(4, 4, &DPDB);
   define_problems15(1, tile_in_location);
   //initialize(5, 5, &DPDB);
   //define_problems24(38, tile_in_location);


   //assert(check_tile_in_location(tile_in_location));
   //prn_data(tile_in_location); printf("\n");
   //prn_distances();
   //prn_moves();

   //UB = 57;
   //UB = 65;
   //UB = 113;
   UB = 200;
   //UB = look_ahead_UB(tile_in_location, 40);
   //difference_LB(tile_in_location, UB, 22, 1, &DPDB);
   //z = iterative_deepening(tile_in_location, &DPDB);

   parameters.algorithm = algorithm;
   parameters.best_measure = best_measure;
   parameters.cpu_limit = CPU_LIMIT;
   parameters.dpdb_lb = dpdb_lb;
   parameters.gen_skip = gen_skip;
   parameters.prn_info = prn_info;
   parameters.direction_rule = direction_rule;
   parameters.search_direction = search_direction;
   //UB = 45;
   //z = search(tile_in_location, &parameters, &info, &DPDB);

   //UB = 55;
   //z = bidirectional(tile_in_location, &parameters, &info, &DPDB);

   //z = ID_DIBBS(tile_in_location, MAX_DEPTH, &info, &DPDB);
   
   benchmarks(&DPDB);

   for(i = 0; i <= n_tiles; i++) {
      delete [] moves[i];
      delete [] distances[i];
      //delete [] hash_values[i];
   }

   // Keep the console window open.
   printf("Press ENTER to continue");
   cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

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
	      case 'd':
            dpdb_lb = atoi(av[++cnt]);
	         break;
         case 'e':
            search_direction = atoi(av[++cnt]);
            break;
         case 'g':
            gen_skip = atoi(av[++cnt]);
	         break;
         case 'p':
            prn_info = atoi(av[++cnt]);
            break;
         case 'r':
            direction_rule = atoi(av[++cnt]);
            break;
         case 's':
            seed = atof(av[++cnt]);
	         break;
   	   default:
	         usage (*av);
            break;
      }
    }
    if (cnt > ac) usage (*av);
    //prob_file = av[cnt++];
    //if (cnt < ac) usage (*av);

}

//_________________________________________________________________________________________________

void usage (char *prog)
{
    fprintf (stderr, "Usage: %s probfile\n", prog);
    fprintf (stderr, "    -a: method (algorithm) to use (def = 4 = CBFS)\n");
    fprintf (stderr, "    -b: best measure to use (def = 1 = LB)\n");
    fprintf (stderr, "    -d: 1 = use disjoint LB, 0 = do not use (def = 0)\n");
    fprintf (stderr, "    -e: search direction used for A* and bidirectional(def = 1 = forward)\n"); 
    fprintf (stderr, "    -g: generation rule - do not add a descendent to memory until the bound has increased by gen_skip (def = 0)\n");
    fprintf (stderr, "    -p: controls level of printed information (def=0)\n");
    fprintf (stderr, "    -r: rule for choosing the direction in bidirectional search (def = 0)\n");
    fprintf (stderr, "    -s: seed for random number generation\n");
    exit (1);
}

/*************************************************************************************************/

unsigned char compute_Manhattan_LB(unsigned char *tile_in_location)
/*
   1. This function computes the Manhattan lower bound.
   2. Input Variables
      a. tile_in_location[i] = the tile that is in location i.
         The elements of tile_in_location are stored beginning in tile_in_location[0].
   3. Global Variables
      a. n_tiles = number of tiles.
      a. distances[i][j] = Manhattan distance between location i and location j.
   4. Output Variables
      a. LB = Manhattan lower bound is returned.
   5. Written 11/18/11.
*/
{
   unsigned char  LB;
   int            i;

   LB = 0;
   for(i = 0; i <= n_tiles; i++) {
      if(tile_in_location[i] != 0) {
         LB += distances[i][tile_in_location[i]];
      }
   }
   return(LB);
}

//_________________________________________________________________________________________________

unsigned char compute_Manhattan_LB2(unsigned char *tile_in_location1, unsigned char *tile_in_location2)
/*
   1. This function computes the Manhattan lower bound on the distance between two configurations.
   2. Input Variables
      a. tile_in_location1[i] = the tile that is in location i in the first configuration.
         The elements of tile_in_location1 are stored beginning in tile_in_location1[0].
      a. tile_in_location2[i] = the tile that is in location i in the second configuration.
         The elements of tile_in_location2 are stored beginning in tile_in_location2[0].
   3. Global Variables
      a. n_tiles = number of tiles.
      a. distances[i][j] = Manhattan distance between location i and location j.
   4. Output Variables
      a. LB = Manhattan lower bound is returned.
   5. Written 11/21/11.
*/
{
   unsigned char  LB, *location_of_tile1, *location_of_tile2;
   int            i;

   // Determine the locations of the tiles in both configurations.

   location_of_tile1 = new unsigned char[n_tiles + 1];
   location_of_tile2 = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) {
      location_of_tile1[tile_in_location1[i]] = i;
      location_of_tile2[tile_in_location2[i]] = i;
   }

   // Compute the lower bound.

   LB = 0;
   for(i = 1; i <= n_tiles; i++) {
      LB += distances[location_of_tile1[i]][location_of_tile2[i]];
   }

   delete [] location_of_tile1;
   delete [] location_of_tile2;

   return(LB);
}

/*************************************************************************************************/

void initialize(int nrows, int ncols, DPDB *DPDB)
/*
   1. This function computes the Manhattan lower bound on the distance between two configurations.
   2. Input Variables
      a. tile_in_location1[i] = the tile that is in location i in the first configuration.
         The elements of tile_in_location1 are stored beginning in tile_in_location1[0].
      a. tile_in_location2[i] = the tile that is in location i in the second configuration.
         The elements of tile_in_location2 are stored beginning in tile_in_location2[0].
   3. Global Variables
      a. n_tiles = number of tiles.
      a. distances[i][j] = Manhattan distance between location i and location j.
   4. Output Variables
      a. LB = Manhattan lower bound is returned.
   5. Written 11/21/11.
*/
{
   unsigned char  *pattern, **patterns;
   int            cnt, d, i, i_row, i_col, j, j_row, j_col, n_databases, *n_tiles_in_patterns;

   n_rows = nrows;
   n_cols = ncols;
   n_tiles = n_rows * n_cols - 1;
   if(n_tiles != N_LOCATIONS - 1) {
      fprintf(stderr, "n_tiles != N_LOCATIONS\n");
      exit(1);
   }

   // Intialize moves.
   // moves[i] = list of possible ways to move the empty tile from location i.  
   // Modified 1/4/12 to change the order in which moves are generated.  They were generated (up, right, down, left).
   // Changed to (up, left, right, down) to match the order that Korf used.

   moves = new unsigned char*[n_tiles + 1];

   for(i = 0; i <= n_tiles; i++) {
      moves[i] = new unsigned char[5];
      i_col = 1 + (i % n_rows);
      cnt = 0;
      // Up.
      j = i - n_cols;
      if((0 <= j) && (j <= n_tiles)) moves[i][++cnt] = j;
      // Left
      if(i_col > 1) {
         j = i - 1;
      } else {
         j = -1;
      }
      if((0 <= j) && (j <= n_tiles)) moves[i][++cnt] = j;
      // Right.
      if(i_col < n_cols) {
         j = i + 1;
      } else {
         j = -1;
      }
      if((0 <= j) && (j <= n_tiles)) moves[i][++cnt] = j;
      // Down.
      j = i + n_cols;
      if((0 <= j) && (j <= n_tiles)) moves[i][++cnt] = j;
      moves[i][0] = cnt;
   }

   // Initializes distances.
   // distances[i][j] = Manhattan distance between location i and location j.

   distances = new unsigned char*[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) distances[i] = new unsigned char[n_tiles + 1];

   for(i = 0; i <= n_tiles; i++) {
      i_row = 1 + (i / n_cols);
      i_col = 1 + (i % n_rows);
      for(j = i; j <= n_tiles; j++) {
         j_row = 1 + (j / n_cols);
         j_col = 1 + (j % n_rows);
         distances[i][j] = abs(i_row - j_row) + abs(i_col - j_col);
         distances[j][i] = distances[i][j];
      }
   }

   // Initialize disjoint pattern databases.


   if(dpdb_lb > 0) {
      switch(N_LOCATIONS) {
		   case 16:
            DPDB->initialize(n_rows, n_cols, 4);
            pattern = new unsigned char[N_LOCATIONS]; 
            pattern[0] =  1; pattern[1] =  4; pattern[2] =  5;                  DPDB->add(distances, moves, 3, pattern);
            pattern[0] =  2; pattern[1] =  3; pattern[2] =  6; pattern[3] =  7; DPDB->add(distances, moves, 4, pattern);
            pattern[0] =  8; pattern[1] =  9; pattern[2] = 12; pattern[3] = 13; DPDB->add(distances, moves, 4, pattern);
            pattern[0] = 10; pattern[1] = 11; pattern[2] = 14; pattern[3] = 15; DPDB->add(distances, moves, 4, pattern);
           
            //pattern[0] =  1; pattern[1] =  4; pattern[2] = 5; pattern[3] = 8; pattern[4] = 9; pattern[5] = 12; pattern[6] = 13; 
            //DPDB->add(distances, moves, 7, pattern);
            //pattern[0] =  2; pattern[1] =  3; pattern[2] = 6; pattern[3] = 7; pattern[4] = 10; pattern[5] = 11; pattern[6] = 14; pattern[7] = 15; 
            //DPDB->add(distances, moves, 8, pattern);
/*
            n_databases = 4;
            n_tiles_in_patterns = new int[n_databases + 1];
            n_tiles_in_patterns[1] = 3; n_tiles_in_patterns[2] = 4; n_tiles_in_patterns[3] = 4; n_tiles_in_patterns[4] = 4;
            patterns = new unsigned char*[n_databases + 1];
            for(d = 1; d <= n_databases; d++) patterns[d] = new unsigned char[n_tiles_in_patterns[d] + 1];
            patterns[1][0] =  1; patterns[1][1] =  4; patterns[1][2] =  5;
            patterns[2][0] =  2; patterns[2][1] =  3; patterns[2][2] =  6; patterns[2][3] =  7;
            patterns[3][0] =  8; patterns[3][1] =  9; patterns[3][2] = 12; patterns[3][3] = 13;
            patterns[4][0] = 10; patterns[4][1] = 11; patterns[4][2] = 14; patterns[4][3] = 15;
            DPDB->read(n_databases, n_tiles_in_patterns, patterns, "database3444.bin");
            delete [] n_tiles_in_patterns;
            for(d = 1; d <= n_databases; d++) delete [] patterns[d];
            delete patterns;
*/
            delete [] pattern;
            break;
		   case 25: 
            DPDB->initialize(n_rows, n_cols, 4);
            pattern = new unsigned char[N_LOCATIONS]; 
/*
            pattern[0] =  1;  pattern[1] =  5;  pattern[2] = 6;  pattern[3] = 10; pattern[4] = 11; pattern[5] = 12; 
            DPDB->add(distances, moves, 6, pattern);
            pattern[0] =  2;  pattern[1] =  3;  pattern[2] = 4;  pattern[3] = 7;  pattern[4] = 8;  pattern[5] = 9;
            DPDB->add(distances, moves, 6, pattern);
            pattern[0] =  15; pattern[1] =  16; pattern[2] = 17; pattern[3] = 20; pattern[4] = 21; pattern[5] = 22;
            DPDB->add(distances, moves, 6, pattern);
            pattern[0] =  13; pattern[1] =  14; pattern[2] = 18; pattern[3] = 19; pattern[4] = 23; pattern[5] = 24;
            DPDB->add(distances, moves, 6, pattern);
*/
            n_databases = 4;
            n_tiles_in_patterns = new int[n_databases + 1];
            n_tiles_in_patterns[1] = 6; n_tiles_in_patterns[2] = 6; n_tiles_in_patterns[3] = 6; n_tiles_in_patterns[4] = 6;
            patterns = new unsigned char*[n_databases + 1];
            for(d = 1; d <= n_databases; d++) patterns[d] = new unsigned char[n_tiles_in_patterns[d] + 1];
            patterns[1][0] =  1;  patterns[1][1] =  5;  patterns[1][2] = 6;  patterns[1][3] = 10; patterns[1][4] = 11; patterns[1][5] = 12; 
            patterns[2][0] =  2;  patterns[2][1] =  3;  patterns[2][2] = 4;  patterns[2][3] = 7;  patterns[2][4] = 8;  patterns[2][5] = 9;
            patterns[3][0] =  15; patterns[3][1] =  16; patterns[3][2] = 17; patterns[3][3] = 20; patterns[3][4] = 21; patterns[3][5] = 22;
            patterns[4][0] =  13; patterns[4][1] =  14; patterns[4][2] = 18; patterns[4][3] = 19; patterns[4][4] = 23; patterns[4][5] = 24;
            DPDB->read(n_databases, n_tiles_in_patterns, patterns, "database6666.bin");
            delete [] n_tiles_in_patterns;
            for(d = 1; d <= n_databases; d++) delete [] patterns[d];
            delete patterns;

            delete [] pattern;
            break;
         default: fprintf(stderr,"Illegal value of N_LOCATIONS in initialize\n"); exit(1); break;
      }

   }
}

//_________________________________________________________________________________________________

void benchmarks(DPDB *DPDB)
{
   int            i, n_problems, z;
   __int64        n_explored_forward, n_explored_reverse;
   unsigned char  tile_in_location[N_LOCATIONS], z_bidirectional, z_ID_DIBBS, z_optimal[101];
   searchparameters  parameters;
   searchinfo        info;

   switch(N_LOCATIONS) {
		case 16:
         z_optimal[0]  =  0; z_optimal[1]  = 57; z_optimal[2] =  55; z_optimal[3] =  59; z_optimal[4] =  56;
         z_optimal[5]  = 56; z_optimal[6]  = 52; z_optimal[7] =  52; z_optimal[8] =  50; z_optimal[9] =  46;
         z_optimal[10] = 59; z_optimal[11] = 57; z_optimal[12] = 45; z_optimal[13] = 46; z_optimal[14] = 59;
         z_optimal[15] = 62; z_optimal[16] = 42; z_optimal[17] = 66; z_optimal[18] = 55; z_optimal[19] = 46; 
         z_optimal[20] = 52; z_optimal[21] = 54; z_optimal[22] = 59; z_optimal[23] = 49; z_optimal[24] = 54;
         z_optimal[25] = 52; z_optimal[26] = 58; z_optimal[27] = 53; z_optimal[28] = 52; z_optimal[29] = 54; 
         z_optimal[30] = 47; z_optimal[31] = 50; z_optimal[32] = 59; z_optimal[33] = 60; z_optimal[34] = 52; 
         z_optimal[35] = 55; z_optimal[36] = 52; z_optimal[37] = 58; z_optimal[38] = 53; z_optimal[39] = 49; 
         z_optimal[40] = 54; z_optimal[41] = 54; z_optimal[42] = 42; z_optimal[43] = 64; z_optimal[44] = 50;
         z_optimal[45] = 51; z_optimal[46] = 49; z_optimal[47] = 47; z_optimal[48] = 49; z_optimal[49] = 59;
         z_optimal[50] = 53; z_optimal[51] = 56; z_optimal[52] = 56; z_optimal[53] = 64; z_optimal[54] = 56;
         z_optimal[55] = 41; z_optimal[56] = 55; z_optimal[57] = 50; z_optimal[58] = 51; z_optimal[59] = 57;
         z_optimal[60] = 66; z_optimal[61] = 45; z_optimal[62] = 57; z_optimal[63] = 56; z_optimal[64] = 51; 
         z_optimal[65] = 47; z_optimal[66] = 61; z_optimal[67] = 50; z_optimal[68] = 51; z_optimal[69] = 53;
         z_optimal[70] = 52; z_optimal[71] = 44; z_optimal[72] = 56; z_optimal[73] = 49; z_optimal[74] = 56;
         z_optimal[75] = 48; z_optimal[76] = 57; z_optimal[77] = 54; z_optimal[78] = 53; z_optimal[79] = 42;
         z_optimal[80] = 57; z_optimal[81] = 53; z_optimal[82] = 62; z_optimal[83] = 49; z_optimal[84] = 55;
         z_optimal[85] = 44; z_optimal[86] = 45; z_optimal[87] = 52; z_optimal[88] = 65; z_optimal[89] = 54;
         z_optimal[90] = 50; z_optimal[91] = 57; z_optimal[92] = 57; z_optimal[93] = 46; z_optimal[94] = 53; 
         z_optimal[95] = 50; z_optimal[96] = 49; z_optimal[97] = 44; z_optimal[98] = 54; z_optimal[99] = 57; 
         z_optimal[100]= 54; 
         n_problems = 100; 
         initialize(4, 4, DPDB); 
         break;
		case 25:
         z_optimal[0] =    0; z_optimal[1] =   95; z_optimal[2] =   96; z_optimal[3] =   97; z_optimal[4] =   98; 
         z_optimal[5] =  100; z_optimal[6] =  101; z_optimal[7] =  104; z_optimal[8] =  108; z_optimal[9] =  113;
         z_optimal[10] = 114; z_optimal[11] = 106; z_optimal[12] = 109; z_optimal[13] = 101; z_optimal[14] = 111; 
         z_optimal[15] = 103; z_optimal[16] =  96; z_optimal[17] = 109; z_optimal[18] = 110; z_optimal[19] = 106; 
         z_optimal[20] =  92; z_optimal[21] = 103; z_optimal[22] =  95; z_optimal[23] = 104; z_optimal[24] = 107; 
         z_optimal[25] =  81; z_optimal[26] = 105; z_optimal[27] =  99; z_optimal[28] =  98; z_optimal[29] =  88; 
         z_optimal[30] =  92; z_optimal[31] =  99; z_optimal[32] =  97; z_optimal[33] = 106; z_optimal[34] = 102;
         z_optimal[35] =  98; z_optimal[36] =  90; z_optimal[37] = 100; z_optimal[38] =  96; z_optimal[39] = 104; 
         z_optimal[40] =  82; z_optimal[41] = 106; z_optimal[42] = 108; z_optimal[43] = 104; z_optimal[44] =  93;
         z_optimal[45] = 101; z_optimal[46] = 100; z_optimal[47] =  92; z_optimal[48] = 107; z_optimal[49] = 100; 
         z_optimal[50] = 113;
         n_problems =  50; 
         initialize(5, 5, DPDB); 
         break;
      default: fprintf(stderr,"Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
   }

   for(i = 1; i <= n_problems; i++) {
   //for(i = 80; i <= 85; i++) {

      // Load the next benchmark problem into tile_in_location.

      switch(N_LOCATIONS) {
		   case 16: define_problems15(i, tile_in_location); break;
		   case 25: define_problems24(i, tile_in_location); break;
         default: fprintf(stderr,"Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }
/*
      UB = z_optimal[i];
      n_explored_forward = difference_LB(tile_in_location, UB, 26, 1, DPDB);
      n_explored_reverse = difference_LB(tile_in_location, UB, 26, 2, DPDB);
      printf("%14I64d  %14I64d\n", n_explored_forward, n_explored_reverse);
*/
      UB = z_optimal[i];
      z = iterative_deepening(tile_in_location, DPDB);

      parameters.algorithm = algorithm;
      parameters.best_measure = best_measure;
      parameters.cpu_limit = CPU_LIMIT;
      parameters.dpdb_lb = dpdb_lb;
      parameters.gen_skip = gen_skip;
      parameters.prn_info = prn_info;
      parameters.direction_rule = direction_rule;
      parameters.search_direction = search_direction;
      info.initialize();
      UB = z_optimal[i];
      //z = search(tile_in_location, &parameters, &info, DPDB);
      //z_bidirectional = bidirectional(tile_in_location, &parameters, &info, DPDB);
      //if (z_bidirectional != z_optimal[i]) printf("Error: z_bidirectinal != z_optimal[%d]\n", i);
      //z_ID_DIBBS = ID_DIBBS(tile_in_location, MAX_DEPTH, &info, DPDB);
      //if (z_ID_DIBBS != z_optimal[i]) printf("Error: z_IDD_DIBBS != z_optimal[%d]\n", i);
      //info.initialize();

   }
}

//_________________________________________________________________________________________________

void define_problems15(int i, unsigned char *tile_in_location)
{
   int            j;
   unsigned char  problems[101][16] = 
   {
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {14, 13, 15,  7, 11, 12,  9,  5,  6,  0,  2,  1,  4,  8, 10,  3},
      {13,  5,  4, 10,  9, 12,  8, 14,  2,  3,  7,  1,  0, 15, 11,  6},
      {14,  7,  8,  2, 13, 11, 10,  4,  9, 12,  5,  0,  3,  6,  1, 15},
      { 5, 12, 10,  7, 15, 11, 14,  0,  8,  2,  1, 13,  3,  4,  9,  6},
      { 4,  7, 14, 13, 10,  3,  9, 12, 11,  5,  6, 15,  1,  2,  8,  0},
      {14,  7,  1,  9, 12,  3,  6, 15,  8, 11,  2,  5, 10,  0,  4, 13},
      { 2, 11, 15,  5, 13,  4,  6,  7, 12,  8, 10,  1,  9,  3, 14,  0},
      {12, 11, 15,  3,  8,  0,  4,  2,  6, 13,  9,  5, 14,  1, 10,  7},
      { 3, 14,  9, 11,  5,  4,  8,  2, 13, 12,  6,  7, 10,  1, 15,  0},
      {13, 11,  8,  9,  0, 15,  7, 10,  4,  3,  6, 14,  5, 12,  2,  1},
      { 5,  9, 13, 14,  6,  3,  7, 12, 10,  8,  4,  0, 15,  2, 11,  1},
      {14,  1,  9,  6,  4,  8, 12,  5,  7,  2,  3,  0, 10, 11, 13, 15},
      { 3,  6,  5,  2, 10,  0, 15, 14,  1,  4, 13, 12,  9,  8, 11,  7},
      { 7,  6,  8,  1, 11,  5, 14, 10,  3,  4,  9, 13, 15,  2,  0, 12},
      {13, 11,  4, 12,  1,  8,  9, 15,  6,  5, 14,  2,  7,  3, 10,  0},
      { 1,  3,  2,  5, 10,  9, 15,  6,  8, 14, 13, 11, 12,  4,  7,  0},
      {15, 14,  0,  4, 11,  1,  6, 13,  7,  5,  8,  9,  3,  2, 10, 12},
      { 6,  0, 14, 12,  1, 15,  9, 10, 11,  4,  7,  2,  8,  3,  5, 13},
      { 7, 11,  8,  3, 14,  0,  6, 15,  1,  4, 13,  9,  5, 12,  2, 10},
      { 6, 12, 11,  3, 13,  7,  9, 15,  2, 14,  8, 10,  4,  1,  5,  0},
      {12,  8, 14,  6, 11,  4,  7,  0,  5,  1, 10, 15,  3, 13,  9,  2},
      {14,  3,  9,  1, 15,  8,  4,  5, 11,  7, 10, 13,  0,  2, 12,  6},
      {10,  9,  3, 11,  0, 13,  2, 14,  5,  6,  4,  7,  8, 15,  1, 12},
      { 7,  3, 14, 13,  4,  1, 10,  8,  5, 12,  9, 11,  2, 15,  6,  0},
      {11,  4,  2,  7,  1,  0, 10, 15,  6,  9, 14,  8,  3, 13,  5, 12},
      { 5,  7,  3, 12, 15, 13, 14,  8,  0, 10,  9,  6,  1,  4,  2, 11},
      {14,  1,  8, 15,  2,  6,  0,  3,  9, 12, 10, 13,  4,  7,  5, 11},
      {13, 14,  6, 12,  4,  5,  1,  0,  9,  3, 10,  2, 15, 11,  8,  7},
      { 9,  8,  0,  2, 15,  1,  4, 14,  3, 10,  7,  5, 11, 13,  6, 12},
      {12, 15,  2,  6,  1, 14,  4,  8,  5,  3,  7,  0, 10, 13,  9, 11},
      {12,  8, 15, 13,  1,  0,  5,  4,  6,  3,  2, 11,  9,  7, 14, 10},
      {14, 10,  9,  4, 13,  6,  5,  8,  2, 12,  7,  0,  1,  3, 11, 15},
      {14,  3,  5, 15, 11,  6, 13,  9,  0, 10,  2, 12,  4,  1,  7,  8},
      { 6, 11,  7,  8, 13,  2,  5,  4,  1, 10,  3,  9, 14,  0, 12, 15},
      { 1,  6, 12, 14,  3,  2, 15,  8,  4,  5, 13,  9,  0,  7, 11, 10},
      {12,  6,  0,  4,  7,  3, 15,  1, 13,  9,  8, 11,  2, 14,  5, 10},
      { 8,  1,  7, 12, 11,  0, 10,  5,  9, 15,  6, 13, 14,  2,  3,  4},
      { 7, 15,  8,  2, 13,  6,  3, 12, 11,  0,  4, 10,  9,  5,  1, 14},
      { 9,  0,  4, 10,  1, 14, 15,  3, 12,  6,  5,  7, 11, 13,  8,  2},
      {11,  5,  1, 14,  4, 12, 10,  0,  2,  7, 13,  3,  9, 15,  6,  8},
      { 8, 13, 10,  9, 11,  3, 15,  6,  0,  1,  2, 14, 12,  5,  4,  7},
      { 4,  5,  7,  2,  9, 14, 12, 13,  0,  3,  6, 11,  8,  1, 15, 10},
      {11, 15, 14, 13,  1,  9, 10,  4,  3,  6,  2, 12,  7,  5,  8,  0},
      {12,  9,  0,  6,  8,  3,  5, 14,  2,  4, 11,  7, 10,  1, 15, 13},
      { 3, 14,  9,  7, 12, 15,  0,  4,  1,  8,  5,  6, 11, 10,  2, 13},
      { 8,  4,  6,  1, 14, 12,  2, 15, 13, 10,  9,  5,  3,  7,  0, 11},
      { 6, 10,  1, 14, 15,  8,  3,  5, 13,  0,  2,  7,  4,  9, 11, 12},
      { 8, 11,  4,  6,  7,  3, 10,  9,  2, 12, 15, 13,  0,  1,  5, 14},
      {10,  0,  2,  4,  5,  1,  6, 12, 11, 13,  9,  7, 15,  3, 14,  8},
      {12,  5, 13, 11,  2, 10,  0,  9,  7,  8,  4,  3, 14,  6, 15,  1},
      {10,  2,  8,  4, 15,  0,  1, 14, 11, 13,  3,  6,  9,  7,  5, 12},
      {10,  8,  0, 12,  3,  7,  6,  2,  1, 14,  4, 11, 15, 13,  9,  5},
      {14,  9, 12, 13, 15,  4,  8, 10,  0,  2,  1,  7,  3, 11,  5,  6},
      {12, 11,  0,  8, 10,  2, 13, 15,  5,  4,  7,  3,  6,  9, 14,  1},
      {13,  8, 14,  3,  9,  1,  0,  7, 15,  5,  4, 10, 12,  2,  6, 11},
      { 3, 15,  2,  5, 11,  6,  4,  7, 12,  9,  1,  0, 13, 14, 10,  8},
      { 5, 11,  6,  9,  4, 13, 12,  0,  8,  2, 15, 10,  1,  7,  3, 14},
      { 5,  0, 15,  8,  4,  6,  1, 14, 10, 11,  3,  9,  7, 12,  2, 13},
      {15, 14,  6,  7, 10,  1,  0, 11, 12,  8,  4,  9,  2,  5, 13,  3},
      {11, 14, 13,  1,  2,  3, 12,  4, 15,  7,  9,  5, 10,  6,  8,  0},
      { 6, 13,  3,  2, 11,  9,  5, 10,  1,  7, 12, 14,  8,  4,  0, 15},
      { 4,  6, 12,  0, 14,  2,  9, 13, 11,  8,  3, 15,  7, 10,  1,  5},
      { 8, 10,  9, 11, 14,  1,  7, 15, 13,  4,  0, 12,  6,  2,  5,  3},
      { 5,  2, 14,  0,  7,  8,  6,  3, 11, 12, 13, 15,  4, 10,  9,  1},
      { 7,  8,  3,  2, 10, 12,  4,  6, 11, 13,  5, 15,  0,  1,  9, 14},
      {11,  6, 14, 12,  3,  5,  1, 15,  8,  0, 10, 13,  9,  7,  4,  2},
      { 7,  1,  2,  4,  8,  3,  6, 11, 10, 15,  0,  5, 14, 12, 13,  9},
      { 7,  3,  1, 13, 12, 10,  5,  2,  8,  0,  6, 11, 14, 15,  4,  9},
      { 6,  0,  5, 15,  1, 14,  4,  9,  2, 13,  8, 10, 11, 12,  7,  3},
      {15,  1,  3, 12,  4,  0,  6,  5,  2,  8, 14,  9, 13, 10,  7, 11},
      { 5,  7,  0, 11, 12,  1,  9, 10, 15,  6,  2,  3,  8,  4, 13, 14},
      {12, 15, 11, 10,  4,  5, 14,  0, 13,  7,  1,  2,  9,  8,  3,  6},
      { 6, 14, 10,  5, 15,  8,  7,  1,  3,  4,  2,  0, 12,  9, 11, 13},
      {14, 13,  4, 11, 15,  8,  6,  9,  0,  7,  3,  1,  2, 10, 12,  5},
      {14,  4,  0, 10,  6,  5,  1,  3,  9,  2, 13, 15, 12,  7,  8, 11},
      {15, 10,  8,  3,  0,  6,  9,  5,  1, 14, 13, 11,  7,  2, 12,  4},
      { 0, 13,  2,  4, 12, 14,  6,  9, 15,  1, 10,  3, 11,  5,  8,  7},
      { 3, 14, 13,  6,  4, 15,  8,  9,  5, 12, 10,  0,  2,  7,  1, 11},
      { 0,  1,  9,  7, 11, 13,  5,  3, 14, 12,  4,  2,  8,  6, 10, 15},
      {11,  0, 15,  8, 13, 12,  3,  5, 10,  1,  4,  6, 14,  9,  7,  2},
      {13,  0,  9, 12, 11,  6,  3,  5, 15,  8,  1, 10,  4, 14,  2,  7},
      {14, 10,  2,  1, 13,  9,  8, 11,  7,  3,  6, 12, 15,  5,  4,  0},
      {12,  3,  9,  1,  4,  5, 10,  2,  6, 11, 15,  0, 14,  7, 13,  8},
      {15,  8, 10,  7,  0, 12, 14,  1,  5,  9,  6,  3, 13, 11,  4,  2},
      { 4,  7, 13, 10,  1,  2,  9,  6, 12,  8, 14,  5,  3,  0, 11, 15},
      { 6,  0,  5, 10, 11, 12,  9,  2,  1,  7,  4,  3, 14,  8, 13, 15},
      { 9,  5, 11, 10, 13,  0,  2,  1,  8,  6, 14, 12,  4,  7,  3, 15},
      {15,  2, 12, 11, 14, 13,  9,  5,  1,  3,  8,  7,  0, 10,  6,  4},
      {11,  1,  7,  4, 10, 13,  3,  8,  9, 14,  0, 15,  6,  5,  2, 12},
      { 5,  4,  7,  1, 11, 12, 14, 15, 10, 13,  8,  6,  2,  0,  9,  3},
      { 9,  7,  5,  2, 14, 15, 12, 10, 11,  3,  6,  1,  8, 13,  0,  4},
      { 3,  2,  7,  9,  0, 15, 12,  4,  6, 11,  5, 14,  8, 13, 10,  1},
      {13,  9, 14,  6, 12,  8,  1,  2,  3,  4,  0,  7,  5, 10, 11, 15},
      { 5,  7, 11,  8,  0, 14,  9, 13, 10, 12,  3, 15,  6,  1,  4,  2},
      { 4,  3,  6, 13,  7, 15,  9,  0, 10,  5,  8, 11,  2, 12,  1, 14},
      { 1,  7, 15, 14,  2,  6,  4,  9, 12, 11, 13,  3,  0,  8,  5, 10},
      { 9, 14,  5,  7,  8, 15,  1,  2, 10,  4, 13,  6, 12,  0, 11,  3},
      { 0, 11,  3, 12,  5,  2,  1,  9,  8, 10, 14, 15,  7,  4, 13,  6},
      { 7, 15,  4,  0, 10,  9,  2,  5, 12, 11, 13,  6,  1,  3, 14,  8},
      {11,  4,  0,  8,  6, 10,  5, 13, 12,  7, 14,  3,  1,  2,  9, 15}
   };

   assert((1 <= i) && (i <= 100));
   for(j = 0; j <= 15; j++) tile_in_location[j] = problems[i][j];
}

//_________________________________________________________________________________________________

void define_problems24(int i, unsigned char *tile_in_location)
{
   int            j;
   unsigned char  problems[51][25] = 
   {
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {14,  5,  9,  2, 18,  8, 23, 19, 12, 17, 15,  0, 10, 20,  4,  6, 11, 21,  1,  7, 24,  3, 16, 22, 13},
      {16,  5,  1, 12,  6, 24, 17,  9,  2, 22,  4, 10, 13, 18, 19, 20,  0, 23,  7, 21, 15, 11,  8,  3, 14},
      { 6,  0, 24, 14,  8,  5, 21, 19,  9, 17, 16, 20, 10, 13,  2, 15, 11, 22,  1,  3,  7, 23,  4, 18, 12},
      {18, 14,  0,  9,  8,  3,  7, 19,  2, 15,  5, 12,  1, 13, 24, 23,  4, 21, 10, 20, 16, 22, 11,  6, 17},
      {17,  1, 20,  9, 16,  2, 22, 19, 14,  5, 15, 21,  0,  3, 24, 23, 18, 13, 12,  7, 10,  8,  6,  4, 11},
      { 2,  0, 10, 19,  1,  4, 16,  3, 15, 20, 22,  9,  6, 18,  5, 13, 12, 21,  8, 17, 23, 11, 24,  7, 14},
      {21, 22, 15,  9, 24, 12, 16, 23,  2,  8,  5, 18, 17,  7, 10, 14, 13,  4,  0,  6, 20, 11,  3,  1, 19},
      { 7, 13, 11, 22, 12, 20,  1, 18, 21,  5,  0,  8, 14, 24, 19,  9,  4, 17, 16, 10, 23, 15,  3,  2,  6},
      { 3,  2, 17,  0, 14, 18, 22, 19, 15, 20,  9,  7, 10, 21, 16,  6, 24, 23,  8,  5,  1,  4, 11, 12, 13},
      {23, 14,  0, 24, 17,  9, 20, 21,  2, 18, 10, 13, 22,  1,  3, 11,  4, 16,  6,  5,  7, 12,  8, 15, 19},
      {15, 11,  8, 18, 14,  3, 19, 16, 20,  5, 24,  2, 17,  4, 22, 10,  1, 13,  9, 21, 23,  7,  6, 12,  0},
      {12, 23,  9, 18, 24, 22,  4,  0, 16, 13, 20,  3, 15,  6, 17,  8,  7, 11, 19,  1, 10,  2, 14,  5, 21},
      {21, 24,  8,  1, 19, 22, 12,  9,  7, 18,  4,  0, 23, 14, 10,  6,  3, 11, 16,  5, 15,  2, 20, 13, 17},
      {24,  1, 17, 10, 15, 14,  3, 13,  8,  0, 22, 16, 20,  7, 21,  4, 12,  9,  2, 11,  5, 23,  6, 18, 19},
      {24, 10, 15,  9, 16,  6,  3, 22, 17, 13, 19, 23, 21, 11, 18,  0,  1,  2,  7,  8, 20,  5, 12,  4, 14},
      {18, 24, 17, 11, 12, 10, 19, 15,  6,  1,  5, 21, 22,  9,  7,  3,  2, 16, 14,  4, 20, 23,  0,  8, 13},
      {23, 16, 13, 24,  5, 18, 22, 11, 17,  0,  6,  9, 20,  7,  3,  2, 10, 14, 12, 21,  1, 19, 15,  8,  4},
      { 0, 12, 24, 10, 13,  5,  2,  4, 19, 21, 23, 18,  8, 17,  9, 22, 16, 11,  6, 15,  7,  3, 14,  1, 20},
      {16, 13,  6, 23,  9,  8,  3,  5, 24, 15, 22, 12, 21, 17,  1, 19, 10,  7, 11,  4, 18,  2, 14, 20,  0},
      { 4,  5,  1, 23, 21, 13,  2, 10, 18, 17, 15,  7,  0,  9,  3, 14, 11, 12, 19,  8,  6, 20, 24, 22, 16},
      {24,  8, 14,  5, 16,  4, 13,  6, 22, 19,  1, 10,  9, 12,  3,  0, 18, 21, 20, 23, 15, 17, 11,  7,  2},
      { 7,  6,  3, 22, 15, 19, 21,  2, 13,  0,  8, 10,  9,  4, 18, 16, 11, 24,  5, 12, 17,  1, 23, 14, 20},
      {24, 11, 18,  7,  3, 17,  5,  1, 23, 15, 21,  8,  2,  4, 19, 14,  0, 16, 22,  6,  9, 13, 20, 12, 10},
      {14, 24, 18, 12, 22, 15,  5,  1, 23, 11,  6, 19, 10, 13,  7,  0,  3,  9,  4, 17,  2, 21, 16, 20,  8},
      { 3, 17,  9,  8, 24,  1, 11, 12, 14,  0,  5,  4, 22, 13, 16, 21, 15,  6,  7, 10, 20, 23,  2, 18, 19},
      {22, 21, 15,  3, 14, 13,  9, 19, 24, 23, 16,  0,  7, 10, 18,  4, 11, 20,  8,  2,  1,  6,  5, 17, 12},
      { 9, 19,  8, 20,  2,  3, 14,  1, 24,  6, 13, 18,  7, 10, 17,  5, 22, 12, 21, 16, 15,  0, 23, 11,  4},
      {17, 15,  7, 12,  8,  3,  4,  9, 21,  5, 16,  6, 19, 20,  1, 22, 24, 18, 11, 14, 23, 10,  2, 13,  0},
      {10,  3,  6, 13,  1,  2, 20, 14, 18, 11, 15,  7,  5, 12,  9, 24, 17, 22,  4,  8, 21, 23, 19, 16,  0},
      { 8, 19,  7, 16, 12,  2, 13, 22, 14,  9, 11,  5,  6,  3, 18, 24,  0, 15, 10, 23,  1, 20,  4, 17, 21},
      {19, 20, 12, 21,  7,  0, 16, 10,  5,  9, 14, 23,  3, 11,  4,  2,  6,  1,  8, 15, 17, 13, 22, 24, 18},
      { 1, 12, 18, 13, 17, 15,  3,  7, 20,  0, 19, 24,  6,  5, 21, 11,  2,  8,  9, 16, 22, 10,  4, 23, 14},
      {11, 22,  6, 21,  8, 13, 20, 23,  0,  2, 15,  7, 12, 18, 16,  3,  1, 17,  5,  4,  9, 14, 24, 10, 19},
      { 5, 18,  3, 21, 22, 17, 13, 24,  0,  7, 15, 14, 11,  2,  9, 10,  1,  8,  6, 16, 19,  4, 20, 23, 12},
      { 2, 10, 24, 11, 22, 19,  0,  3,  8, 17, 15, 16,  6,  4, 23, 20, 18,  7,  9, 14, 13,  5, 12,  1, 21},
      { 2, 10,  1,  7, 16,  9,  0,  6, 12, 11,  3, 18, 22,  4, 13, 24, 20, 15,  8, 14, 21, 23, 17, 19,  5},
      {23, 22,  5,  3,  9,  6, 18, 15, 10,  2, 21, 13, 19, 12, 20,  7,  0,  1, 16, 24, 17,  4, 14,  8, 11},
      {10,  3, 24, 12,  0,  7,  8, 11, 14, 21, 22, 23,  2,  1,  9, 17, 18,  6, 20,  4, 13, 15,  5, 19, 16},
      {16, 24,  3, 14,  5, 18,  7,  6,  4,  2,  0, 15,  8, 10, 20, 13, 19,  9, 21, 11, 17, 12, 22, 23,  1},
      { 2, 17,  4, 13,  7, 12, 10,  3,  0, 16, 21, 24,  8,  5, 18, 20, 15, 19, 14,  9, 22, 11,  6,  1, 23},
      {13, 19,  9, 10, 14, 15, 23, 21, 24, 16, 12, 11,  0,  5, 22, 20,  4, 18,  3,  1,  6,  2,  7, 17,  8},
      {16,  6, 20, 18, 23, 19,  7, 11, 13, 17, 12,  9,  1, 24,  3, 22,  2, 21, 10,  4,  8, 15, 14,  5,  0},
      { 7,  4, 19, 12, 16, 20, 15, 23,  8, 10,  1, 18,  2, 17, 14, 24,  9,  5,  0, 21,  6,  3, 11, 13, 22},
      { 8, 12, 18,  3,  2, 11, 10, 22, 24, 17,  1, 13, 23,  4, 20, 16,  6, 15,  9, 21, 19,  5, 14,  0,  7},
      { 9,  7, 16, 18, 12,  1, 23,  8, 22,  0,  6, 19,  4, 13,  2, 24, 11, 15, 21, 17, 20,  3, 10, 14,  5},
      { 1, 16, 10, 14, 17, 13,  0,  3,  5,  7,  4, 15, 19,  2, 21,  9, 23,  8, 12,  6, 11, 24, 22, 20, 18},
      {21, 11, 10,  4, 16,  6, 13, 24,  7, 14,  1, 20,  9, 17,  0, 15,  2,  5,  8, 22,  3, 12, 18, 19, 23},
      { 2, 22, 21,  0, 23,  8, 14, 20, 12,  7, 16, 11,  3,  5,  1, 15,  4,  9, 24, 10, 13,  6, 19, 17, 18},
      { 2, 21,  3,  7,  0,  8,  5, 14, 18,  6, 12, 11, 23, 20, 10, 15, 17,  4,  9, 16, 13, 19, 24, 22,  1},
      {23,  1, 12,  6, 16,  2, 20, 10, 21, 18, 14, 13, 17, 19, 22,  0, 15, 24,  3,  7,  4,  8,  5,  9, 11}
   };

   assert((1 <= i) && (i <= 50));
   for(j = 0; j <= 24; j++) tile_in_location[j] = problems[i][j];
}

/*************************************************************************************************/

int check_tile_in_location(unsigned char *tile_in_location)
/*
   1. This routine performs some simple checks on tile_in_location.
      If an error is found, 0 is returned, otherwise 1 is returned.
   2. Written 11/19/11.
*/
{
   int      i, *tile_used;

   tile_used = new int[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) tile_used[i] = 0;

   // Check that all the indices in tile_in_location are legitimate and that there are no duplicates.

   for(i = 0; i <= n_tiles; i++) {
      if((tile_in_location[i] < 0) || (tile_in_location[i] > n_tiles)) {
         fprintf(stderr, "illegal tile in tile_in_location\n"); 
         delete [] tile_used;
         return(0); 
      }
      if(tile_used[tile_in_location[i]]) {
         fprintf(stderr, "in_tile_location contains the same tile twice\n"); 
         delete [] tile_used;
         return(0); 
      }
      tile_used[tile_in_location[i]] = 1;
   }

   delete [] tile_used;

   return(1);
}

//_________________________________________________________________________________________________

int check_solution(unsigned char source[N_LOCATIONS], unsigned char solution[MAX_DEPTH + 1], unsigned char z)
/*
   1. This routine performs some simple checks on solution.
      If an error is found, 0 is returned, otherwise 1 is returned.
   2. Written 11/19/11.
*/
{
   bool           found;
   unsigned char  empty_location, new_location, prev_location, tile, *tile_in_location;
   int            i, j;

   // Check that each entry in solution is a valid location.

   for(i = 0; i <= z; i++) {
      if((solution[i] < 0) || (solution[i] > N_LOCATIONS)) {
         fprintf(stderr, "illegal location in solution\n"); 
         return(0); 
      }
   }

   // Check that consecutive entries in solution represent a valid move.

   prev_location = solution[0];
   for(i = 1; i <= z; i++) {
      new_location = solution[i];
      found = false;
      for(j = 1; j <= moves[prev_location][0]; j++) {
         if(new_location == moves[prev_location][j]) {
            found = true;
            break;
         }
      }
      if(!found) {
         fprintf(stderr, "solution[%d] is not a neighbor of solution[%d]\n", i-1, i); 
         return(0); 
      }
      prev_location = new_location;
   }

   // Copy the source configuration into tile_in_location.

   tile_in_location = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) tile_in_location[i] = source[i];

   // Construct the configurations obtained from the intial configuration along the path given in solution.
   // Check that each configuration is valid.

   for(i = 1; i <= z; i++) {
      empty_location = solution[i-1];
      new_location = solution[i];
      tile = tile_in_location[new_location];
      tile_in_location[empty_location] = tile;
      tile_in_location[new_location] = 0;
      //prn_configuration(tile_in_location);
      if(check_tile_in_location(tile_in_location) == 0) {
         fprintf(stderr, "solution produced an illegal configuration\n", i-1, i); 
         return(0); 
      }
   }

   // Check that the final configuration is the goal configuration.

   if(compute_Manhattan_LB(tile_in_location) != 0) {
      fprintf(stderr, "solution did not reach the goal configuration\n", i-1, i); 
      return(0); 
   }

   delete [] tile_in_location;

   return(1);
}

//_________________________________________________________________________________________________

void reverse_vector(int i, int j, int n, unsigned char *x)
/*
   1. This routine reverses the order of continguous portion of a vector.
   2. n = number of entries in x, assuming data is stored in x[0], ..., x[n-1].
   3. i = index where reversal should begin.
   4. j = index where reversal should end.
   5. Must have 0 <= i <= j <= n-1;
   6. Created 7/5/18 by modifiying reverse_vector from c:\sewell\research\pancake\pancake_code\main.cpp.
*/
{
   unsigned char  temp;

   assert((0 <= i) && (i <= n-1));
   assert((0 <= j) && (j <= n-1));

   while (i < j) {
      temp = x[i];
      x[i] = x[j];
      x[j] = temp;
      i++;
      j--;
   }
}

