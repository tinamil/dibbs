#ifndef _stack_int_
#define _stack_int_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <algorithm>

using namespace std;

/*
   The following class and functions implement a stack of integers.
   1. This list supports the following operations in constant time*:
      a. Push an item onto the top of the stack.  (*This requires linear time if additional space must be allocated.)
      b. Pop the item from the top of the stack.
   2. The stack is stored in an array of integers. 
   3. The items are stored in int_array[1], ..., int_array[n], where n is the number of items 
      currently in the stack.
   4. An item in the stack is defined by an integer, called the value.
   5. A stack consists of the following.
      a. n = last index in the stack that is being used.
           =  number of items currently in the stack.
      b. n_allocated = the number of elements allocated to int_array (minus 1).
      c. int_array:  The items are stored in int_array[1], ..., int_array[n].
   6. Written 12/12/11.  Based on neighbor_list from c:\sewell\research\stable\reductions\reduce\bigraph.h.
*/

/*************************************************************************************************/

class stack_int {
public:
   stack_int()    {  n = 0; n_allocated = 0; int_array = NULL;}
   explicit stack_int(const int nn) {  assert(nn > 0);
                                       n = 0; 
                                       n_allocated = nn;
                                       int_array = new int[n_allocated + 1];
                                       if(int_array == NULL) {
                                          fprintf(stderr, "Out of space for int_array in stack_int\n");
                                          exit(1);
                                       }
                                    };
   void  allocate(const int nn) {   assert(nn > 0);
                                    n = 0;
                                    n_allocated = nn;
                                    int_array = new int[n_allocated + 1];
                                    if(int_array == NULL) {
                                       fprintf(stderr, "Out of space for int_array in stack_int\n");
                                       exit(1);
                                    }
                                };
   ~stack_int()   { delete [] int_array;}
   int& operator[] (int i) const {assert((1 <= i) && (i <= n)); return int_array[i];}
   void  push(const int value, int max_allocated = 0)  
               {  
                  if(n >= n_allocated) {
                     if(max_allocated <= 0) max_allocated = INT_MAX;
                     n_allocated = max(2, min(2 * n_allocated, max_allocated));
                     if(int_array == NULL) {
                        int_array = new int[n_allocated + 1];
                        if(int_array == NULL) {
                           fprintf(stderr, "Out of space for int_array in stack_int\n");
                           exit(1);
                        }
                     } else {
                        int      *temp_array;
                        temp_array = new int[n_allocated + 1];
                        if(temp_array == NULL) {
                           fprintf(stderr, "Out of space for temp_array in stack_int\n");
                           exit(1);
                        }
                        memcpy(temp_array, int_array, (n + 1) * sizeof(int));
                        delete [] int_array;
                        int_array = temp_array;
                     }
                  }
                  int_array[++n] = value;
               };
   bool  empty()  const {return(n == 0);}
   int   pop()    {  assert(n > 0); return(int_array[n--]);}
   int   top()    {  assert(n > 0); return(int_array[n]);}
   void  reset()  {  n = 0;}
   int   size()   const {return n;}
   void  print()  {  for(int i = 1; i <= n; i++) printf("%4d ", int_array[i]); }
private:
   int      n;             // The number of elements in the vector.
   int      n_allocated;   // The number of elements allocated to the vector.
   int      *int_array;
};

#endif
