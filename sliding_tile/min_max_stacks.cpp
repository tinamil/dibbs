#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <math.h>
#include <string.h>
#include <vector>
#include <stack>
#include "min_max_stacks.h"


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

//_________________________________________________________________________________________________

void min_max_stacks::insert(int d, int b, int index)
// INSERT inserts the index of a subproblem at depth d and best b into the stacks.
{
   // I do not know how to check if there is sufficient space to add to the stack, or whether
   
   assert((0 <= d) && (d <= max_size_depth));
   assert((0 <= b) && (b <= max_size_best));

   stacks[d][b].push(index);

   if(b < min_best[d]) {
      min_best[d] = b;
      update_min_deeper_decrease(d);
   }
   if(b > max_best[d]) max_best[d] = b;
}

//_________________________________________________________________________________________________

best_index min_max_stacks::get_min(int d)
/*
   1. GET_MIN returns the minimum best value and the index of the subproblem with minimum best at depth d (without deleting it).
   2. If there are no subproblems at depth d, then -1 is returned as the index.
   3. Written 12/7/11.
*/
{
   best_index  best_index;

   assert((0 <= d) && (d <= max_size_depth));
   best_index.best = min_best[d];
   if(min_best[d] < INT_MAX) {
      best_index.state_index = stacks[d][min_best[d]].top();
      return(best_index);
   } else {
      best_index.state_index = -1;
      return(best_index);                      // If there are no subproblems at depth d, return -1.
   }
}

//_________________________________________________________________________________________________

best_index min_max_stacks::delete_min(int d)
/*
   1. DELETE_MIN deletes the index of the subproblem with minimum best at depth d.   
   2. The minimum best value and the index of the subproblem with minimum best at depth d is returned.
   3. If there are no subproblems at depth d, then -1 is returned as the index.
   4. Written 12/8/11.
*/
{
   int         min_b;
   best_index  best_index;

   assert((0 <= d) && (d <= max_size_depth));
   min_b = min_best[d];
   best_index.best = min_b;
   if(min_b < INT_MAX) {
      best_index.state_index = stacks[d][min_b].pop();      // top returns the top element, but does not delete it.
      //stacks[d][min_b].pop();                               // Delete the top element.

      // If the stack is now empty, then update min_best (and max_best if there are no more subproblems at depth d).

      if(stacks[d][min_b].empty()) {
         update_min(d, min_b);
      }
      return(best_index);
   } else {
      best_index.state_index = -1;
      return(best_index);                                   // If there are no subproblems at depth d, return -1.
   }
}

//_________________________________________________________________________________________________

void min_max_stacks::update_min(int d, int b)
/*
   1. This function updates the minimum value of best among subproblems at depth d.
      It also updates the maximum value of best among subproblems at depth d if there are no subproblems at depth d.
   2. It assumes that b is the previous minimum value of best among subproblems at depth d and stacks[d][b] is empty.
      It starts the search for the new minimum value from b + 1.
      If the previous minimum value is not known, then call this funtion with b = -1.
   3. Input Variables
      a. d = depth in the search tree.
      b. b = the previous minimum value of best among subproblems at depth d.
   4. Written 12/8/11.
*/
{
   int      i;

   assert((0 <= d) && (d <= max_size_depth));
   assert((-1 <= b) && (b <= max_size_best));

   // If b equals max_best[d], then there are no subproblems at depth d.
   // Set min_best[d] and max_best[d] to their default values.

   if(b == max_best[d]) {
      min_best[d] = INT_MAX;
      max_best[d] = -1;
      update_min_deeper_increase(d);
      return;
   }

   // Starting from b + 1, search for the first nonempty stack at depth d.

   for(i = b + 1; i <= max_size_best; i++) {
      if(!stacks[d][i].empty()) {
         min_best[d] = i;
         update_min_deeper_increase(d);
         return;
      }
   }
   if(i > max_size_best) {
      min_best[d] = INT_MAX;
      update_min_deeper_increase(d);
   }
}

//_________________________________________________________________________________________________

best_index min_max_stacks::get_max(int d)
/*
   1. GET_MAX returns the maximum best and the index of the subproblem with maximum best at depth d (without deleting it).
   2. If there are no subproblems at depth d, then -1 is returned as the index.
   3. Written 12/7/11.
*/
{
   best_index  best_index;

   assert((0 <= d) && (d <= max_size_depth));
   best_index.best = max_best[d];
   if(max_best[d] > -1) {
      best_index.state_index = stacks[d][max_best[d]].top();
      return(best_index);
   } else {
      best_index.state_index = -1;
      return(best_index);                      // If there are no subproblems at depth d, return -1.
   }
}

//_________________________________________________________________________________________________

best_index min_max_stacks::delete_max(int d)
/*
   1. DELETE_MAX deletes the index of the subproblem with maximum best at depth d.   
   2. The maximum best value and the index of the subproblem with maximum best at depth d is returned.
   3. If there are no subproblems at depth d, then -1 is returned as the index.
   4. Written 12/8/11.
*/
{
   int         max_b;
   best_index  best_index;

   assert((0 <= d) && (d <= max_size_depth));
   max_b = max_best[d];
   best_index.best = max_b;
   if(max_b > -1) {
      best_index.state_index = stacks[d][max_b].pop();      // top returns the top element, but does not delete it.
      //stacks[d][max_b].pop();                               // Delete the top element.

      // If the stack is now empty, then update max_best (and min_best if there are no more subproblems at depth d).

      if(stacks[d][max_b].empty()) {
         update_max(d, max_b);
      }
      return(best_index);
   } else {
      best_index.state_index = -1;
      return(best_index);                                   // If there are no subproblems at depth d, return -1.
   }
}

//_________________________________________________________________________________________________

void min_max_stacks::update_max(int d, int b)
/*
   1. This function updates the maximum value of best among subproblems at depth d.
      It also updates the minimum value of best among subproblems at depth d if there are no subproblems at depth d.
   2. It assumes that b is the previous maximum value of best among subproblems at depth d and stacks[d][b] is empty.
      It starts the search for the new maximum value from b - 1.
      If the previous minimum value is not known, then call this funtion with b = max_size_best + 1.
   3. Input Variables
      a. d = depth in the search tree.
      b. b = the previous maximum value of best among subproblems at depth d.
   4. Written 12/8/11.
*/
{
   int      i;

   assert((0 <= d) && (d <= max_size_depth));
   assert((0 <= b) && (b <= max_size_best + 1));

   // If b equals min_best[d], then there are no subproblems at depth d.
   // Set min_best[d] and max_best[d] to their default values.

   if(b == min_best[d]) {
      min_best[d] = INT_MAX;
      max_best[d] = -1;
      update_min_deeper_increase(d);
      return;
   }

   // Starting from b - 1, search for the first nonempty stack at depth d.

   for(i = b - 1; i >= 0; i--) {
      if(!stacks[d][i].empty()) {
         max_best[d] = i;
         return;
      }
   }
   if(i < 0) max_best[d] = -1;
}

//_________________________________________________________________________________________________

int min_max_stacks::replace_min(int d, int b, int index)
/*
   1. REPLACE_MIN replaces the subproblem with minimum best at depth d with the new subproblem supplied in the input parameters.
      If there are no subproblems at depth d, then the new subproblem is simply inserted.
   2. If there are no subproblems at depth d, then -1 is returned.
   3. Written 12/8/11.
*/
{
   best_index  best_index;

   assert((0 <= d) && (d <= max_size_depth));
   assert((0 <= b) && (b <= max_size_best));

   if(min_best[d] == INT_MAX) {        // If there are no subproblems at depth d, then insert the new subproblem
      insert(d, b, index);             // and return -1.
      return(-1);
   }

   // If there are subproblems at depth d, then delete the subproblem with the minimum best and insert the new subproblem.
   // Return the index of the deleted subproblem.

   best_index = delete_min(d);
   insert(d, b, index);
   return(best_index.state_index);
}

//_________________________________________________________________________________________________

int min_max_stacks::replace_max(int d, int b, int index)
/*
   1. REPLACE_MAX replaces the subproblem with maximum best at depth d with the new subproblem supplied in the input parameters.
      If there are no subproblems at depth d, then the new subproblem is simply inserted.
   2. If there are no subproblems at depth d, then -1 is returned.
   3. Written 12/8/11.
*/
{
   best_index best_index;

   assert((0 <= d) && (d <= max_size_depth));
   assert((0 <= b) && (b <= max_size_best));

   if(max_best[d] == -1) {             // If there are no subproblems at depth d, then insert the new subproblem
      insert(d, b, index);             // and return -1.
      return(-1);
   }

   // If there are subproblems at depth d, then delete the subproblem with the maximum best and insert the new subproblem.
   // Return the index of the deleted subproblem.

   best_index = delete_max(d);
   insert(d, b, index);
   return(best_index.state_index);
}

//_________________________________________________________________________________________________

void min_max_stacks::update_min_deeper_decrease(int d)
/*
   1. This function updates min_deeper whenever min_best[d] has decreased.
   2. min_deeper[d] = arg min {min_best[d*]: d* > d}.  In case of ties, choose the deepest one.  Set to -1 if there are no deeper subproblems.
   3. Input Variables
      a. d = depth in the search tree.
   4. Written 12/13/11.
*/
{
   int      d2, min_best_d;

   assert((0 <= d) && (d <= max_size_depth));

   // Starting from depth d - 1, replace min_deeper at smaller depths if min_best[d] is better.
   // The search can be halted as soon as a depth is encountered where min_best[d] is not better.

   min_best_d = min_best[d];
   d2 = d - 1;
   while(d2 >= 0) {
      if((min_deeper[d2] == -1) || (min_best_d < min_best[min_deeper[d2]]) || ((min_best_d == min_best[min_deeper[d2]]) && (d > min_deeper[d2]))) {
         min_deeper[d2] = d;
         d2--;
      } else {
         if(min_best_d < min_best[min_deeper[d2]]) {
            break;
         } else {
            d2--;
         }
      }
   }
}

//_________________________________________________________________________________________________

void min_max_stacks::update_min_deeper_increase(int d)
/*
   1. This function updates min_deeper whenever min_best[d] has increased.
   2. min_deeper[d] = arg min {min_best[d*]: d* > d}.  In case of ties, choose the deepest one.  Set to -1 if there are no deeper subproblems.
   3. Input Variables
      a. d = depth in the search tree.
   4. Written 12/13/11.
*/
{
   int      b, d1, d2;

   assert((0 <= d) && (d <= max_size_depth));

   // Determine the minimum value of best among subproblems at depth d or greater.  In case of ties, choose the deepest one.
   // Special cases to consider.
   // 1. There are no deeper subproblems, as signified by min_deeper[d] == -1.
   // 2. There are no subproblems at depth d, as signified by min_best[d] == INT_MAX.

   if(min_deeper[d] == -1) {
      if(min_best[d] == INT_MAX) {           // There are no subproblems at depth d now.
         b = INT_MAX;
         d1 = -1;
      } else {
         b = min_best[d];
         d1 = d;
      }
   } else {
      if(min_best[d] < min_best[min_deeper[d]]) {
         b = min_best[d];
         d1 = d;
      } else {
         b = min_best[min_deeper[d]];
         d1 = min_deeper[d];
      }
   }

   // Starting from depth d - 1, replace min_deeper at smaller depths if min_best[d] is better.
   // The search can be halted as soon as a depth is encountered where min_deeper is not equal to d.
   // Note that min_deeper should be defined for all depths less than d because depth d was not empty (although it might be empty now).

   d2 = d - 1;
   while((d2 >= 0) && (min_deeper[d2] == d)) {
      min_deeper[d2] = d1;
      if(min_best[d2] < b) {
         b = min_best[d2];
         d1 = d2;
      }
      d2--;
   }
}

//_________________________________________________________________________________________________

int min_max_stacks::get_state(int UB)
/*
   1. This function tries to obtain an unexplored subproblem from the stacks.
   2. Written 12/16/11.
*/
{
   int         max_depth;
   best_index  best_index;

/*
   max_depth = min(UB, max_size_depth);
	for(int i= 0; i <= max_depth; i++) {
		depth = (depth + 1) % (max_depth + 1);
      if(!empty(depth)) {
         best_index = delete_min(depth);
			return(best_index.state_index);
		}
      if(min_deeper[depth] == -1) depth = -1;
	}
	return -1; //went through all levels and everything is empty
*/

   max_depth = min(UB, max_size_depth);
   depth = (depth + 1) % (max_depth + 1);
   if((min_deeper[depth] == -1) && (min_best[depth] == INT_MAX)) depth = 0;   // If there are no subproblems at this depth or lower, then reset dept to 0.
   if(min_deeper[depth] == -1) {
      if(min_best[depth] == INT_MAX) {
         return(-1);                         // There are no subproblems at any depth.
      } 
   } else {
      while(min_best[depth] == INT_MAX) depth++;            // Move to a nonempty level.
      if((min_deeper[depth] != -1) && (min_best[depth] > min_best[min_deeper[depth]])) {
         depth = min_deeper[depth];
      }
   }
   best_index = delete_min(depth);
   return(best_index.state_index);
}

//_________________________________________________________________________________________________

void min_max_stacks::new_UB_prune(int UB)
/*
   1. This function clears all stacks whose best measure is greater than or equal to UB.
   2. Written 12/19/11.
*/
{
   int         b, d;

	for(d = 0; d <= max_size_depth; d++) {
      b = max_best[d];
      while(b >= UB) {
         stacks[d][b].reset();
         update_max(d, b);
         b = max_best[d];
      }
	}

}

//_________________________________________________________________________________________________

void min_max_stacks::check_stacks()
/*
   1. CHECK_STACKS performs some elementary checks the stacks.
   2. Written 6/15/07.
*/
{
   int      b, d, max_b, min_b, min_d;

   for(d = 0; d <= max_size_depth; d++) {

      // Check if min_best[d] is correct.

      min_b = INT_MAX;
      for(b = 0; b <= max_size_best; b++) {
         if(!stacks[d][b].empty()) {
            if(b != min_best[d]) {
               fprintf(stderr, "Error in check_stacks: b != min_best[d]\n");
               exit(1);
            }
            break;
         }
      }
      if((b > max_size_best) && (min_best[d] != INT_MAX)) {
         fprintf(stderr, "Error in check_stacks: min_best[d] != INT_MAX\n");
         exit(1);
      }

      // Check if max_best[d] is correct.

      max_b = -1;
      for(b = max_size_best; b >= 0; b--) {
         if(!stacks[d][b].empty()) {
            if(b != max_best[d]) {
               fprintf(stderr, "Error in check_stacks: b != max_best[d]\n");
               exit(1);
            }
            break;
         }
      }
      if((b < 0) && (max_best[d] != -1)) {
         fprintf(stderr, "Error in check_stacks: max_best[d] != -1\n");
         exit(1);
      }
   }  

   // Check if min_deeper is defined correctly.

   d = max_size_depth;
   min_d = -1;
   min_b = INT_MAX;
   while(d >= 0) {
      if(min_deeper[d] != min_d) {
         fprintf(stderr, "Error in check_stacks: min_deeper is incorrect\n");
         print();
         exit(1);
      }
      if(min_best[d] < min_b) {
         min_b = min_best[d];
         min_d = d;
      }
      d--;
   }
}

//_________________________________________________________________________________________________

void min_max_stacks::print()
{
   int      b, d;

   for(d = 0; d <= max_size_depth; d++) {
      for(b = 0; b <= max_size_best; b++) {
         if(!stacks[d][b].empty()) {
            printf("%3d %3d:", d, b);
            //for each(int index in stacks[d][b]._Get_container()) {
            for(int index = 1; index <= stacks[d][b].size(); index++) {
               //printf(" %5d", index);
               printf(" %5d", stacks[d][b][index]);
            }
            printf("\n");
         }
      }
   }

   for(d = 0; d <= max_size_depth; d++) {
      printf("%3d %12d %3d\n", d, min_best[d], min_deeper[d]);
   }
}

//_________________________________________________________________________________________________

void min_max_stacks::print_stats()
{
   int      b, d;

   for(d = 0; d <= max_size_depth; d++) {
      for(b = 0; b <= max_size_best; b++) {
         if(!stacks[d][b].empty()) {
            printf("%3d %3d %10d\n", d, b, stacks[d][b].size());
         }
      }
   }
}