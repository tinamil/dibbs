#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <math.h>
#include <string.h>
#include <vector>
#include <stack>
#include "min_max_multiset.h"


/*
   The following class and functions implement a vector for storing a multiset of nonnegative integer values.
   It keeps track of how many times each value is in the multiset and the minimum and maximum value in the multiset.
   It is designed to be used for keeping track of the minimum value of g or f among the open nodes in a bidirectional search algorithm.
   1. The following assumptions must be satisfied in order to use this data structure.
      a. Each value in the multiset is a nonnegative integer.
      b. The maximum maximum value in the multiset can be bounded when the multiset is initialized.
   2. The stacks consists of the following.
      a. max_size_value = max value permitted in the multiset.
      b. min_value = min value currently in the multiset.
      c. max_value = max value currently in the multiset.
      d. n_elements[v] = the number of elements in the multiset with value v.
      e. total_n_elements = total number of elements currently in the multiset.
   3. Written 1/10/19.
*/

/*************************************************************************************************/

//_________________________________________________________________________________________________

void min_max_multiset::insert(int v)
// INSERT inserts the value v into the multiset.
{
   assert((0 <= v) && (v <= max_size_value));
   n_elements[v]++;
   total_n_elements++;

   if(v < min_value) min_value = v;
   if(v > max_value) max_value = v;
}

//_________________________________________________________________________________________________

void min_max_multiset::delete_element(int v)
// DELETE_ELEMENT deleles an element whose value is v.
{
   assert((0 <= v) && (v <= max_size_value));
   assert(n_elements[v] > 0);
   n_elements[v]--;
   total_n_elements--;

   // If there are no other elements with value, then check if min_value or max_value need to be updated.

   if(n_elements[v] == 0) {
      if(v == min_value) update_min(v);
      if(v == max_value) update_max(v);
   }
}

//_________________________________________________________________________________________________

int min_max_multiset::get_min()
/*
   1. GET_MIN returns the minimum value in the multiset (without deleting the corresponding element in the multiset).
   2. If the multiset is empty, then -1 is returned.
   3. Written 1/10/19.
*/
{
   if(min_value < INT_MAX) {
      return(min_value);
   } else {
      return(-1);
   }
}

//_________________________________________________________________________________________________

int min_max_multiset::delete_min()
/*
   1. DELETE_MIN deletes an element that equals the minimum value in the multiset.   
   2. The minimum value is returned.
   3. If the multiset is empty, then -1 is returned.
   4. Written 1/10/19.
*/
{
   int         min_v;

   min_v = min_value;
   if(min_v < INT_MAX) {
      n_elements[min_v]--;                                  // Delete an element that equals the minimum value in the multiset.
      total_n_elements--;

      // If there are no other elements that equal the minimum value, then update min_value (and max_value if the multiset is now empty).

      if(n_elements[min_v] == 0) {
         update_min(min_v);
      }
      return(min_v);
   } else {
      return(-1);
   }
}

//_________________________________________________________________________________________________

void min_max_multiset::update_min(int v)
/*
   1. This function updates the minimum value of the elements currently in the multiset.
      It also updates the maximum value if the multiset is empty.
   2. It assumes that v is the previous minimum value of the elements in the multiset and n_elements[v] = 0.
      It starts the search for the new minimum value from v.
      If the previous minimum value is not known, then call this funtion with v = -1.
   3. Input Variables
      a. v = the previous minimum value of the elements in the multiset.
   4. Written 1/10/19.
*/
{
   int      i;

   assert((-1 <= v) && (v <= max_size_value));

   // If the multiset is empty, then set min_value and max_value to their default values.

   if(total_n_elements == 0) {
      min_value = INT_MAX;
      max_value = -1;
      return;
   }

   // Starting from v, search for the smallest value in the multiset.

   for(i = v; i <= max_size_value; i++) {
      if(n_elements[i] > 0) {
         min_value = i;
         return;
      }
   }
   if(i > max_size_value) {fprintf(stderr, "Error while updating min_value\n"); exit(1);}
}

//_________________________________________________________________________________________________

int min_max_multiset::get_max()
/*
   1. GET_MAX returns the maximum value in the multiset (without deleting the corresponding element in the multiset).
   2. If the multiset is empty, then -1 is returned.
   3. Written 1/10/19.
*/
{
   return(max_value);
}

//_________________________________________________________________________________________________

int min_max_multiset::delete_max()
/*
   1. DELETE_MAX deletes an element that equals the maximum value in the multiset.
   2. The maximum value is returned.
   3. If the multiset is empty, then -1 is returned.
   4. Written 1/10/19.
*/
{
   int         max_v;

   max_v = max_value;
   if (max_v > -1) {
      n_elements[max_v]--;                                  // Delete an element that equals the maximum value in the multiset.
      total_n_elements--;

      // If there are no other elements that equal the maximum value, then update max_value (and min_value if the multiset is now empty).

      if (n_elements[max_v] == 0) {
         update_max(max_v);
      }
      return(max_v);
   }
   else {
      return(-1);
   }
}

//_________________________________________________________________________________________________

void min_max_multiset::update_max(int v)
/*
   1. This function updates the maximum value of the elements currently in the multiset.
      It also updates the minimum value if the multiset is empty.
   2. It assumes that v is the previous maximum value of the elements in the multiset and n_elements[v] = 0.
      It starts the search for the new maximum value from v.
      If the previous maximum value is not known, then call this funtion with v = max_size_value.
   3. Input Variables
      a. v = the previous maximum value of the elements in the multiset.
   4. Written 1/10/19.
*/
{
   int      i;

   assert((0 <= v) && (v <= max_size_value));

   // If the multiset is empty, then set min_value and max_value to their default values.

   if(total_n_elements == 0) {
      min_value = INT_MAX;
      max_value = -1;
      return;
   }

   // Starting from v, search for the first nonempty stack at depth d.

   for(i = v; i >= 0; i--) {
      if(n_elements[i] > 0) {
         max_value = i;
         return;
      }
   }
   if(i < 0) { fprintf(stderr, "Error while updating max_value\n"); exit(1); }
}

//_________________________________________________________________________________________________

void min_max_multiset::check_multiset()
/*
   1. CHECK_MULTISET performs some elementary checks on the multiset.
   2. Written 1/10/19.
*/
{
   int      max_v, min_v, v;
   __int64  total;

   // Check if total_n_elements is correct.

   total = 0;
   for (v = 0; v <= max_size_value; v++) total += n_elements[v];
   if (total != total_n_elements) {
      fprintf(stderr, "Error in check_multiset: total != total_n_elements\n");
      exit(1);
   }


   // Check if min_value is correct.

   min_v = INT_MAX;
   for(v = 0; v <= max_size_value; v++) {
      if(n_elements[v] > 0) {
         if(v != min_value) {
            fprintf(stderr, "Error in check_multiset: v != min_value\n");
            exit(1);
         }
         break;
      }
   }
   if((v > max_size_value) && (min_value != INT_MAX)) {
      fprintf(stderr, "Error in check_multiset: min_value != INT_MAX\n");
      exit(1);
   }

   // Check if max_value is correct.

   max_v = -1;
   for(v = max_size_value; v >= 0; v--) {
      if(n_elements[v] > 0) {
         if(v != max_value) {
            fprintf(stderr, "Error in check_multiset: v != max_value\n");
            exit(1);
         }
         break;
      }
   }
   if((v < 0) && (max_value != -1)) {
      fprintf(stderr, "Error in check_multiset: max_value != -1\n");
      exit(1);
   }
}

//_________________________________________________________________________________________________

void min_max_multiset::print()
{
   int      v;

   printf("min_value = %3d  max_value = %3d  total_n_elements = %12I64d\n\n", min_value, max_value, total_n_elements);
   for(v = 0; v <= max_size_value; v++) printf("%3d %10I64d\n", v, n_elements[v]);
   printf("\n");
}

//_________________________________________________________________________________________________

void min_max_multiset::print_stats()
{
   int      b, d;

   printf("min_value = %3d  max_value = %3d  total_n_elements = %12I64d\n", min_value, max_value, total_n_elements);
}