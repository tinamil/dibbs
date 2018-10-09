/*
   1. This file was copied on 7/20/17 from c:\sewell\research\15puzzle\15puzzle_code2\min_max_heap.cpp.
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <math.h>
#include <string.h>
#include "min_max_heap.h"


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

//_________________________________________________________________________________________________

min_max_heap::min_max_heap(const min_max_heap& mm_heap){
   last = mm_heap.last;
   max_size = mm_heap.max_size;

	heap = new heap_record[max_size + 2];
   assert(heap != NULL);
	memcpy(heap, mm_heap.heap, (last + 1) * sizeof(heap_record));
}

//_________________________________________________________________________________________________

void min_max_heap::insert(heap_record item)
// INSERT inserts an item into the heap.
{
   if(last >= max_size) {
      fprintf(stderr, "Out of space for heap\n");
      exit(1);
   }

   last++;
   heap[last] = item;
   this->siftup(last);
}

//_________________________________________________________________________________________________

void min_max_heap::siftup(int k)
/*
   SIFT_UP performs a siftup operation on the item in position k of the heap.
   1. It performs the following operations.
      a. If key(k) < key(left_sibling(k)), then swap k with its left sibling.
      b. It then repeatedly swaps the item in position k 
         i. with left(k) if key(k) < key(left(k))
        ii. with right(k) if key(k) > key(right(k))
         until either it is in position 2 or 3 or until no more swaps are needed.
   2. Written 6/14/07.
*/
{
   int      k1, stop, left_right;

   if(k < 2) {
      fprintf(stderr, "k < 2 in siftup\n");
      exit(1);
   }

   if( ((k % 2) == 1) && (heap[k].key < heap[k-1].key)) {
      // Swap k with its left sibling.
      k1 = k - 1;
      this->swap(k, k1);
      k = k1;
   }

   if(k > 3) {
      left_right = 2 * (k / 4);              // Don't need the floor function due to integral division.
      stop = 0;
   } else {
      stop = 1;
   }
   while(stop == 0) {
      if(heap[k].key < heap[left_right].key) {
         this->swap(k, left_right);
         k = left_right;
         left_right = 2 * (k / 4);
      } else {
         left_right++;
         if(heap[k].key > heap[left_right].key) {
            this->swap(k, left_right);
            k = left_right;
            left_right = 2 * (k / 4);
         } else {
            stop = 1;
         }
      }
      if(k < 4) {
         stop = 1;
      }
   }
}

//_________________________________________________________________________________________________

heap_record min_max_heap::get_min()
/*
   1. MIN_ITEM returns the item with minimum key from the heap (without deleting it).
   2. A copy of the item with the minimum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 3/4/08.
*/
{
   heap_record item;

   if(last <= 1) {                     // If there are no items in the heap, return item with -1 in the key.
      item.key = -1;
      return(item);
   }

   // If the heap is not empty, then the item with the minimum key is in position 2.
   // Return this item.

   item = heap[2];
   return(item);
}

//_________________________________________________________________________________________________

heap_record min_max_heap::delete_min()
/*
   1. DELETE_MIN deletes the item with minimum key from the heap.   
   2. A copy of the item with the minimum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 6/15/07.
*/
{
   heap_record item;

   if(last <= 1) {                     // If there are no items in the heap, return item with -1 in the key.
      item.key = -1;
      return(item);
   }

   // If the heap is not empty, then the item with the minimum key is in position 2.
   // Return this item.  Put the last item into position 2 and perform a siftdown operation.

   item = heap[2];
   heap[2] = heap[last];
   last--;
   siftdown_min(2);
   return(item);
}

//_________________________________________________________________________________________________

void min_max_heap::siftdown_min(int k)
/*
   1. This function performs a siftdown operation on the item in position k of the heap.
   2. It repeatedly performs the following operations.
      a. If key(k) > key(right_sibling(k)), then swap k with its right sibling.
      b. If key(k) is greater than either its left child or its left nephew, then it swaps the item
         in position k with either its left child or its left nephew (whichever one has the smaller key).
      until no more swaps are needed.
   3. k must be a left child, which means k must be even.
   4. Written 6/15/07.
*/
{
   int      k1, left_nephew;
   double   key1, left_nephew_key;

   if( (k < 2) || ((k % 2) == 1)) {
      fprintf(stderr, "k < 2 or k is odd in siftdown_min\n");
      exit(1);
   }

   while(k < last) {
      
      // If the key of the item in position k is greater than its right sibling, then swap it with its right sibling.
      
      k1 = k + 1;   
      if(heap[k].key > heap[k1].key) {
         this->swap(k, k1);
         continue;
      }

      // If the key of the item in position k is greater than either its left child or its left nephew,
      // then swap it with the one with the smaller key.
      
      k1 = 2 * k;                                     // k1 = left child of k.
      if(k1 > last) {
         break;
      }
      key1 = heap[k1].key;
      
      left_nephew = 2 * k + 2;                       // Note: This formula for the left nephew assumes that k is a left child.
      if(left_nephew <= last) {
         left_nephew_key = heap[left_nephew].key;
         if(key1 > left_nephew_key) {
            k1 = left_nephew;
            key1 = left_nephew_key;
         }
      }

      if(heap[k].key > key1) {
         this->swap(k, k1);
         k = k1;
         continue;
      } else {
         break;
      }
   }
}

//_________________________________________________________________________________________________

heap_record min_max_heap::get_max()
/*
   1. MAX_ITEM returns the item with maximum key from the heap (without deleting it).   
   2. A copy of the item with the maximum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 3/4/08.
*/
{
   heap_record item;

   if(last <= 1) {                     // If there are no items in the heap, return item with -1 in the key.
      item.key = -1;
      return(item);
   } else {
      if(last == 2) {                  // If there is precisely one item in the heap, then it is in
         item = heap[2];               // position 2.
         return(item);
      }
   }

   // If there is more than one item in the heap, then the item with the maximum key is in position 3.
   // Return this item.

   item = heap[3];
   return(item);
}

//_________________________________________________________________________________________________

heap_record min_max_heap::delete_max()
/*
   1. DELETE_MAX deletes the item with maximum key from the heap.   
   2. A copy of the item with the maximum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 6/15/07.
*/
{
   heap_record item;

   if(last <= 1) {                     // If there are no items in the heap, return item with -1 in the key.
      item.key = -1;
      return(item);
   } else {
      if(last == 2) {                  // If there is precisely one item in the heap, then it is in
         item = heap[2];               // position 2. Return this item and decrement last (so that
         last--;                       // the item is deleted from the heap and the heap is empty).
         return(item);
      }
   }

   // If there is more than one item in the heap, then the item with the maximum key is in position 3.
   // Return this item.  Put the last item into position 3 and perform a siftdown operation.

   item = heap[3];
   heap[3] = heap[last];
   last--;
   siftdown_max(3);
   return(item);
}

//_________________________________________________________________________________________________

void min_max_heap::siftdown_max(int k)
/*
   1. This function performs a siftdown operation on the item in position k of heaps(m).
   2. It repeatedly performs the following operations.
      a. If key(k) < key(left_sibling(k)), then swap k with its left sibling.
      b. If key(k) is less than either its right child or its right nephew, then it swaps the item
         in position k with either its right child or its right nephew (whichever one has the larger key).
      c. If k does not have a right child, then if key(k) is less than the key of its left child or any of its nephews,
         then k is swapped with the one with the largest key.  If k becomes a left child, then the process is halted.
      until no more swaps are needed.
   3. k must be a right child, which means k must be odd.
   4. Written 6/15/07.
*/
{
   int      i, k1, right_child;
   double   key1, right_child_key;

   if( (k < 3) || ((k % 2) == 2)) {
      fprintf(stderr, "k < 2 or k is even in siftdown_max\n");
      exit(1);
   }

   while(k <= last) {
      
      // If the key of the item in position k is less than its left sibling, then swap it with its left sibling.
            
      k1 = k - 1;   
      if(heap[k].key < heap[k1].key) {
         this->swap(k, k1);
         continue;
      }

      // Check if k has a right child.  If it does not, then if key(k) is less than the key of its left child or any of its nephews,
      // then k is swapped with the one with the largest key.  If k becomes a left child, then the process is halted.
      
      if(2 * k + 1 > last) {
         key1 = INT_MIN;
         i = 2 * k - 2;
         while(i <= last) {
            if(key1 < heap[i].key) {
               k1 = i;
               key1 = heap[i].key;
            }
            i++;
         }
         if(heap[k].key < key1) {
            this->swap(k, k1);
            k = k1;
            
            // If k is now a left child, then break out of the loop.
            
            if((k % 2) == 0) {
               break;
            } else {
               continue;
            }
         } else {
            break;
         }
      }
      
      // If the key of the item in position k is less than either its rightt child or its right nephew,
      // then swap it with the one with the larger key.
      
      k1 = 2 * k - 1;                                 // k1 = right nephew of k. Note: This formula for the right nephew assumes that k is a right child.
      key1 = heap[k1].key;

      right_child = 2 * k + 1;
      right_child_key = heap[right_child].key;
      if(key1 < right_child_key) {
         k1 = right_child;
         key1 = right_child_key;
      }

      if(heap[k].key < key1) {
         this->swap(k, k1);
         k = k1;
         continue;
      } else {
         break;
      }
   }
}

//_________________________________________________________________________________________________

heap_record min_max_heap::replace_min(heap_record item)
/*
   1. REPLACE_MIN replaces the item with minimum key from the heap with the new item supplied in the input parameters.
      If the heap is empty, then the new item is simply inserted.
   2. A copy of the item with the minimum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 6/15/07.
*/
{
   heap_record min_item;

   if(last <= 1) {                     // If there are no items in the heap, then insert the new item
      min_item.key = -1;               // and return an item with -1 in the key.
      insert(item);
      return(min_item);
   }

   // If the heap is not empty, then the item with the minimum key is in position 2.
   // Return this item.  Put the new item into position 2 and perform a siftdown operation.

   min_item = heap[2];
   heap[2] = item;
   siftdown_min(2);
   return(min_item);
}

//_________________________________________________________________________________________________

heap_record min_max_heap::replace_max(heap_record item)
/*
   1. REPLACE_MAX replaces the item with maximum key from the heap with the new item supplied in the input parameters.
      If the heap is empty, then the new item is simply inserted.
   2. A copy of the item with the maximum key is returned.
   3. If the heap is empty, the key of the item that is returned is set equal to -1.
   4. Written 6/15/07.
*/
{
   heap_record max_item;

   if(last <= 1) {                     // If there are no items in the heap, then insert the new item
      max_item.key = -1;               // and return an item with -1 in the key.
      insert(item);
      return(max_item);
   } else {
      if(last == 2) {                  // If there is precisely one item in the heap, then the item with
         max_item = heap[2];           // the maximum key is in position 2.  Return this item.
         heap[2] = item;               // Put the new item in positon 2.
         return(max_item);
      }
   }

   // If there is more than one item in the heap, then the item with the maximum key is in position 3.
   // Return this item.  Put the new item into position 3 and perform a siftdown operation.

   max_item = heap[3];
   heap[3] = item;
   siftdown_max(3);
   return(max_item);
}

//_________________________________________________________________________________________________

void min_max_heap::check_heap(int k, double *min_key, double *max_key)
/*
   CHECK_HEAP checks the heap to make sure that it satisfies the min-max heap criterion.
   1. It recursively finds the minimum and maximum key value in the subtree rooted at k.
   2. It checks that key(left_child(k)) <= minimum key in subtree rooted at k (excluding k itself).
   3. It checks that key(right_child(k)) >= maximum key in subtree rooted at k (excluding k itself).
   4. It returns the minimum and maximum key values in the subtree rooted at k.
   5. Written 6/15/07.
*/
{
   int      child;
   double   key, min_right_key, max_right_key, min_left_key, max_left_key;
   double   right_child_key, left_child_key;

   if(k > last) {
      fprintf(stderr, "k > last\n");
      exit(1);
   }

   child = 2 * k;
   if(child > last) {
      *min_key = heap[k].key;
      *max_key = heap[k].key;
      return;
   } else {
      check_heap(child, &min_left_key, &max_left_key);
      left_child_key = heap[child].key;
   }

   child++;
   if(child > last) {
      min_right_key = INT_MAX;
      max_right_key = INT_MIN;
      right_child_key =INT_MAX;
   } else {
      check_heap(child, &min_right_key, &max_right_key);
      right_child_key = heap[child].key;
   }

   if( (left_child_key > min_left_key) | (left_child_key > min_right_key) ) {
      fprintf(stderr, "Error in check_heap\n");
      exit(1);
   }
   if( (right_child_key < max_left_key) | (right_child_key < max_right_key) ) {
      fprintf(stderr, "Error in check_heap\n");
      exit(1);
   }

   key = heap[k].key;
   if(min_left_key <= min_right_key) {
      *min_key = min_left_key;
   } else {
      *min_key = min_right_key;
   }
   if(key < *min_key) *min_key = key;
   if(max_left_key >= max_right_key) {
      *max_key = max_left_key;
   } else {
      *max_key = max_right_key;
   }
   if(key > *max_key) *max_key = key;
}

//_________________________________________________________________________________________________

void min_max_heap::print()
{
   char     s[80], w[65];
   int      cnt, d, i, field_width, n_levels, start, stop, width;

   for(i = 1; i <= last; i++) {
      printf("%2d ", heap[i].key);
   }
   printf("\n");

   n_levels = (int) ceil(log((double) (last + 1)) / log(2.0));
   field_width = 4;
   width = field_width * (1 << (n_levels - 2));
   cnt = 0;
   for(d = 1; d <= n_levels - 1; d++) {
      _itoa_s(field_width / 2, w, 10);
      strcpy_s(s, "%");
      strcat_s(s, w);
      strcat_s(s, "c");
      printf(s, ' ');
      stop = 1 << (d - 1);
      for(i = 1; i  <= stop; i++) {
         cnt = cnt++;
         _itoa_s(width, w, 10);
         strcpy_s(s, "%");
         strcat_s(s, w);
         strcat_s(s, "d%");
         strcat_s(s, w);
         strcat_s(s, "c");     
         printf(s, heap[cnt].key, ' ');
      }
      printf("\n");
      width = width / 2;                     // Don't need the floor function due to integral division.
   }
   start = 1 << (d - 1); 
   for(i = start; i <= last; i++) {
      printf(" %3d", heap[i].key);
   }
   printf("\n");

}