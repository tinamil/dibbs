#ifndef _min_max_stacks_
#define _min_max_stacks_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <vector>
#include <stack>
#include "stack_int.h"
//#include "heap_record.h"

using namespace std;

/*
   The following class and functions implement a matrix of min-max stacks, which can be used for implementing a 
   cyclic best bound first search.  
   1. The following assumptions must be satisfied in order to use this data structure.
      a. The best measure for each subproblem is a nonnegative integer.
      b. The maximum search depth can be bounded prior to the search.
      c. The maximum best value can be bounded prior to the search.
   2. Min-max stacks.  The stacks support the following operations in constant time*:
      a. Insert an item into one of the stacks.  (*This requires linear time if additional space must be allocated.)
      b. Delete the minimum item from the stacks for depth d.
      c. Delete the maximum item from the stacks for depth d.
   3. A subproblem at depth d with best measure b is stored in stacks[b][d].
   4. Comparison with min-max heaps.
      a. The min-max stacks are designed to be used when the best measure is integral and there are relatively
         few distinct values of best measure.
      b. Min-max stacks essentially sort the subproblems by putting them into buckets (stacks[d][b] acts 
         like a bucket for subproblems with depth d and best measure b).  Thus the sorting time
         is constant.  In contrast, the sorting time for min-max heaps is  O(log n).
   5. This class also provides a method for getting the next subproblem to be explored using the CBFS strategy.
      It also includes a variation of CBFS that skips a level if there is a deeper subproblem that
      has a better best measure.
   6. An item in a stack is defined by an integral index, which is the index of the subproblem (in states, or some other data structure).
   7. The stacks consists of the following.
      a. depth = depth of the last subproblem chosen for exploration.
      b. max_size_depth = max depth of the search tree.
      c. max_size_best = max value of best.
      d. min_best[d] = min value of best among subproblems at depth d.
      e. max_best[d] = max value of best among subproblems at depth d.
      f. min_deeper[d] = arg min {min_best[d*]: d* > d}.  In case of ties, choose the deepest one.  Set to -1 if there are no deeper subproblems.
      g. stacks[d][b] contains the indices of the subproblems at depth d with best measure b.
   8. Written 12/5/11.
*/

/*************************************************************************************************/

class best_index {
public:
   int      best;
   int      state_index;
};

class min_max_stacks {
public:
   min_max_stacks()  {  depth = 0; max_size_depth = 0; max_size_best = 0; stacks = NULL; }
   ~min_max_stacks() {  if(stacks != NULL) {for(int i = 0; i <= max_size_depth; i++) delete [] stacks[i]; delete [] stacks;}}
   void  initialize(const int maximum_depth, const int maximum_best) 
               {  assert(maximum_depth > 0);
                  assert(maximum_best > 0);
                  depth = 0;
                  max_size_depth = maximum_depth;
                  max_size_best = maximum_best;
                  /*stacks.reserve(max_size_depth + 1);
                  for(int i = 0; i <= max_size_depth; i++) {
                     vector<stack<int>,std::vector<int>> temp;
                     stacks[i].push_back(;
                     stacks[i].reserve(max_size_best + 1);
                  }*/
                  //stacks = new stack<int>*[max_size_depth + 1];
                  //stacks = new stack<int,std::vector<int>>*[max_size_depth + 1];
                  stacks = new stack_int*[max_size_depth + 1];
                  if(stacks == NULL) {
                     fprintf(stderr, "Out of space for stacks\n");
                     exit(1);
                  }
                  for(int i = 0; i <= max_size_depth; i++) {
                     //stacks[i] = new stack<int>[max_size_best + 1];
                     //stacks[i] = new stack<int,std::vector<int>>[max_size_best + 1];
                     stacks[i] = new stack_int[max_size_best + 1];
                     if(stacks[i] == NULL) {
                        fprintf(stderr, "Out of space for stacks\n");
                        exit(1);
                     }
                  }
                  min_best = new int[max_size_depth + 1];
                  if(min_best == NULL) {
                     fprintf(stderr, "Out of space for min_best\n");
                     exit(1);
                  }
                  max_best = new int[max_size_depth + 1];
                  if(max_best == NULL) {
                     fprintf(stderr, "Out of space for min_best\n");
                     exit(1);
                  }
                  for(int d = 0; d <= max_size_depth; d++) {
                     min_best[d] = INT_MAX;
                     max_best[d] = -1;
                  }
                  min_deeper = new int[max_size_depth + 1];
                  if(min_deeper == NULL) {
                     fprintf(stderr, "Out of space for min_deeper\n");
                     exit(1);
                  }
                  for(int d = 0; d <= max_size_depth; d++) min_deeper[d] = -1;
               }
   int   n_of_items(int d, int b)   const {return (int) stacks[d][b].size();}
   bool  empty(int d)               const {return(min_best[d] == INT_MAX);}
   bool  empty(int d, int b)        const {return(stacks[d][b].empty());}
   void  print();
   void  print_stats();
   void  insert(int d, int b, int index);
   best_index  get_min(int d);
   best_index  delete_min(int d);
   void  update_min(int d, int b);
   best_index  get_max(int d);
   best_index  delete_max(int d);
   void  update_max(int d, int b);
   int   replace_min(int d, int b, int index);
   int   replace_max(int d, int b, int index);
   void  update_min_deeper_decrease(int d);
   void  update_min_deeper_increase(int d);
   int   get_state(int UB);
   void  new_UB_prune(int UB);
   void  check_stacks();
private:
   int            depth;               // = depth of the last subproblem chosen for exploration.
   int            max_size_depth;      // = max depth of the search tree.
   int            max_size_best;       // = max value of best.
   int            *min_best;           // min_best[d] = min value of best among subproblems at depth d.
   int            *max_best;           // max_best[d] = max value of best among subproblems at depth d.
   int            *min_deeper;         // min_deeper[d] = arg min {min_best[d*]: d* > d}.  In case of ties, choose the deepest one.  Set to -1 if there are no deeper subproblems.
   //stack<int>     **stacks;            // heap[1] is a dummy item.
   //stack<int,std::vector<int>>     **stacks;            // heap[1] is a dummy item.
   //vector<vector<stack<int>>> stacks;
   //vector<vector<stack<int,std::vector<int>>>> stacks;   // This specifies to use vector to implement the stack.  The default is dequeue.  vector may be more efficient than deque since dequeue permits inserting at the front and the back of the list.
   stack_int      **stacks;
};

#endif

