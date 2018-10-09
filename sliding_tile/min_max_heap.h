#ifndef _min_max_heap_
#define _min_max_heap_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include "heap_record.h"

/*
   The following class and functions implement a min-max heap, which can be used for implementing a 
   distributed best bound first search.
   1. Min-max heaps.  See Symmetric Min-Max Heap: A Simpler Data Structure for Double-Ended Priority Queue
      by Arvind and Rangan; Information Processing Letters, 69, 1999 for an explanation of min-max heaps.  
      This heap supports the following operations in O(log n) time:
      a. Insert an item into the heap.
      b. Delete the minimum item from the heap.
      c. Delete the maximum item from the heap.
   2. The heap is represented by a binary tree where the root node is empty.  If x is a node in the tree, 
      let Tx be the subtree rooted at x.  It maintains the following invariant for every node x in the tree.
      a. The node with the maximum key in Tx (excluding x itself) is in the right child of x.
      b. The node with the minimum key in Tx (excluding x itself) is in the left child of x.  (Note: The paper did not say "excluding x itself" here, but there example indicates this must be the case.)
   3. The items in the heap are stored in heap[1], ..., heap[n], where n is the number of items 
      currently in the heap minus 1.
   4. Parent, child, and sibling are defined in the usual way.
   5. Let z be a node such that z > 3 and g be its grandparent.  
      Left(z) is the left child of g and right(z) is the right child of g.
   6. Given a node k, the indices of its relatives can be computed as follows:
      a. parent(k) = floor(k / 2)         k > 1
      b. left_child(k) = 2 * k            2*k > n => no left child
      c. right_child(k) = 2 * k + 1       2*k + 1 > n => no right child
      d. grandparent(k) = floor(k / 4)    k > 3
      e. left_sibling(k) = k - 1          if k > 1 is odd
      f. right_sibling(k) = k + 1         if k > 1 is even and k < n
      g. left(k) = 2 * floor(k / 4)       if k > 3
      h. right(k) = 2 * floor(k / 4) + 1  if k > 3
   7. The authors prove that a heap is a min-max heap iff each node x satisfies:
      a. Q1: left(x) is undefined or key(x) >= key(left(x)).
      b. Q2: right(x) is undefined or key(x) <= key(right(x)).
      c. key(left sibling) <= key(right sibling).
      These are the conditions that are actually checked and enforced during siftup and siftdown operations.
   8. An item in the heap is defined by
      a. key = a number that is used to sort the heap.
      b. Any other data that you wish to store with the item.
   9. A heap consists of the following.
      a. last = last index in the heap that is being used.
              = 1 plus number of items currently in the heap.
      b. max_size = maximum number of items in the heap.
      c. heap:  The items are stored in heap[2], ..., heap[last].  heap[1] is a dummy item.
  10. Written 6/12/07.  Based on the heaps in c:\sewell\research\matlab\min_max_heap.
*/

/*************************************************************************************************/


class min_max_heap {
public:
   min_max_heap()    {  last = 1; max_size = 0; heap = NULL;}
   min_max_heap(const min_max_heap&); // copy constructor
   ~min_max_heap()   { delete [] heap;}
   void  initialize(const int maximum_size)  {  assert(maximum_size > 0);
                                                heap = new heap_record[maximum_size + 2];
                                                assert(heap != NULL);
                                                max_size = maximum_size;
                                                last = 1;
                                                heap[1].key = -1;
                                                heap[1].state_index = -1;
                                             };
   heap_record& operator[] (int i) const {assert((2 <= i) && (i <= last)); return heap[i];}
   int   n_of_items()   const {return last - 1;}
   int   last_index()   const {return last;}
   void  null()         {last = 1; max_size = 0; heap = NULL;}
   void  clear()         {last = 1;}
   bool  empty()        const {return(last == 1);}
   bool  is_full()      const {return(last == max_size - 1);}
   bool  is_not_full()  const {return(last < max_size - 1);}
   bool  is_null()      const {return(heap == NULL);}
   int   maximum_size() const {return max_size;}
   void  swap(const int i, const int j)   {  heap_record temp;
                                             assert((2 <= i) && (i <= last));
                                             assert((2 <= j) && (j <= last));
                                             temp = heap[i];
                                             heap[i] = heap[j];
                                             heap[j] = temp;
                                          };
   void  print();
   void  insert(heap_record item);
   void  siftup(int k);
   heap_record get_min();
   heap_record delete_min();
   void  siftdown_min(int k);
   heap_record get_max();
   heap_record delete_max();
   void  siftdown_max(int k);
   heap_record replace_min(heap_record item);
   heap_record replace_max(heap_record item);
   void  check_heap(int k, double *min_key, double *max_key);
private:
   int            last;             // = last index in the heap that is being used.
                                    // = 1 plus number of items currently in the heap.
   int            max_size;         // = max number of items in the heap.
   heap_record    *heap;            // The items are stored in heap[2], ..., heap[last].
                                    // heap[1] is a dummy item.
};

#endif

