#ifndef _vec_
#define _vec_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>

using namespace std;

/*
   The following template, class, and functions implement an array to hold items of type T.
   1. This data structure supports the following operations in constant time*:
      a. Push an item onto the top of the vec.  (*This requires linear time if additional space must be allocated.)
      b. Pop the item from the top of the vec.
      c. Remove an item at a given index by replacing it with the last item in vec.
   2. The vec is stored in an array of type T. 
   3. The items are stored in v[1], ..., v[n], where n is the number of items 
      currently in the vec.
   4. An item in the vec is defined by an object of type T, called the value.
   5. A vec consists of the following.
      a. n = last index in the stack that is being used.
           =  number of items currently in the vec.
      b. n_allocated = the number of elements allocated to int_array (minus 1).
      c. v:  The items are stored in v[1], ..., v[n].
   6. Written 5/17/19.  Based on c:\sewell\research\15puzzle\15puzzle_code2\stack_int.h.
*/

/*************************************************************************************************/

template <class T>
class vec {
public:
   vec()    {  n = 0; n_allocated = 0; v = NULL;}
   explicit vec(const __int64 nn)   {  assert(nn > 0);
                                       n = 0; 
                                       n_allocated = nn;
                                       v = new T[n_allocated + 1];   if (v == NULL) { fprintf(stderr, "Out of space for v in vec.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
                                    };
   int   allocate(const __int64 nn) {   assert(nn > 0);
                                       n = 0;
                                       n_allocated = nn;
                                       v = new T[n_allocated + 1];   if (v == NULL) { return(-1); }
                                       return(0);
                                    };
   ~vec()   { delete [] v;}
   T&    operator[] (__int64 i) const {assert((1 <= i) && (i <= n)); return v[i];}
   __int64  push(const T value)  
               {  
                  if(n >= n_allocated) {
                     n_allocated = 2 * n_allocated;
                     if (n_allocated < 2) n_allocated = 2;
                     if(v == NULL) {
                        v = new T[n_allocated + 1];   if (v == NULL) { return(-1); }
                     } else {
                        T     *temp_array;
                        temp_array = new T[n_allocated + 1];   if (temp_array == NULL) { return(-1); }
                        memcpy(temp_array, v, (n + 1) * sizeof(T));
                        delete [] v;
                        v = temp_array;
                     }
                  }
                  v[++n] = value;
                  return(n);
               };
   void     clear()     { n = 0; }
   bool     empty()     const {return(n == 0);}
   __int64  n_alloc()   const { return n_allocated; }
   T        pop()       {  assert(n > 0); return(v[n--]);}
   T        top()       {  assert(n > 0); return(v[n]);}
   void     release_memory() { delete[] v; n = 0; n_allocated = 0; }
   __int64  remove(const __int64 i)    // To use remove, the calling function must know the index of the element to be deleted.  This class does not keep track of that information.
   {                                   // Hence, after this function moves the last element to position i, the calling function must update the new index for the element that was moved.
                                       // Warning: This function has not been debugged.  Need to add variables in bistate to store the index of the element.
      assert((1 <= i) && (i <= n));
      if (i == n) {
         n--;
         return(-1);          // In this case, no element needs to be moved.  Return -1 so the calling function knows that no element was moved, hence no index needs to be updated.
      } else {
         v[i] = v[n--];       // Move the last element to position i.  Decrement n.  Note: The element that was in the last position is still in memory.  If this class is used to store large objects, we might want to explicitly delete this from memory.
                              // Note: If this function is used, then the order in which the elements were added to vec will not be preserved.
         return(i);           // Return the index so that the calling function knows that it needs to update the index for the value now stored in v[i].
      }
   }
   __int64  remove(const __int64 i, const T value)    // To use remove, the calling function must know the index of the element to be deleted.  This class does not keep track of that information.
                                                      // Hence, after this function moves the last element to position i, the calling function must update the new index for the element that was moved.
                                                      // Warning: This function has not been debugged.  Need to add variables in bistate to store the index of the element.
   {  // This overloaded version of remove includes an assertion to check that the value stored in v[i] is what we think it is.
      assert((1 <= i) && (i <= n));
      assert(v[i] == value);
      if (i == n) {
         n--;
         return(-1);          // In this case, no element needs to be moved.  Return -1 so the calling function knows that no element was moved, hence no index needs to be updated.
      } else {
         v[i] = v[n--];       // Move the last element to position i.  Decrement n.  Note: The element that was in the last position is still in memory.  If this class is used to store large objects, we might want to explicitly delete this from memory.
                              // Note: If this function is used, then the order in which the elements were added to vec will not be preserved.
         return(i);           // Return the index so that the calling function knows that it needs to update the index for the value now stored in v[i].
      }
   }
   void     reset()  {  n = 0;}
   __int64  size()   const {return n;}
   void     print() { for (int i = 1; i <= n; i++) cout << v[i] << " "; cout << endl; }
   void     print_stats()  { printf("n = %12I64d  n_allocated = %12I64d\n", n, n_allocated); }
private:
   __int64  n;             // The number of elements in the vector.
   __int64  n_allocated;   // The number of elements allocated to the vector.
   T        *v;            // The items are stored in v[1], ..., v[n].
};

#endif
