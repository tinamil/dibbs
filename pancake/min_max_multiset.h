#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>

using namespace std;

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
   3. Written 1/9/19.
*/

/*************************************************************************************************/

class min_max_multiset {
public:
   min_max_multiset()  { max_size_value = 0; min_value = INT_MAX; max_value = -1; n_elements = NULL; total_n_elements = 0; }
   ~min_max_multiset() {  if(n_elements != NULL) delete [] n_elements;}
   void  initialize(const int maximum_value) 
               {  assert(maximum_value > 0);
                  max_size_value = maximum_value;
                  total_n_elements = 0;
                  n_elements = new __int64[max_size_value + 1];
                  if(n_elements == NULL) {
                     fprintf(stderr, "Out of space for multiset\n");
                     exit(1);
                  }
                  for(int v = 0; v <= max_size_value; v++) n_elements[v] = 0;
						 min_value = INT_MAX;
						 max_value = -1;
               }
   __int64& operator[] (int v) const {assert((0 <= v) && (v <= max_size_value)); return n_elements[v];}
   __int64  n_of_elements(int v)    const {return (__int64) n_elements[v];}
   bool  empty()                    const {return(total_n_elements == 0);}
   void  clear()                    {min_value = INT_MAX; max_value = -1; for(int i = 0; i <= max_size_value; i++) n_elements[i] = 0; total_n_elements = 0;}
   bool  is_null()                  const {return(n_elements == NULL);}
   __int64  cardinality()           const {return(total_n_elements);}
   void  print();
   void  print_stats();
   void  insert(int v);
   void  delete_element(int v);
   int   get_min();
   int   delete_min();
   void  update_min(int v);
   int   get_max();
   int   delete_max();
   void  update_max(int v);
   void  check_multiset();
private:
   int            max_size_value;      // = max value permitted in the multiset.
   int            min_value;           // = min value currently in the multiset.
   int            max_value;           // = max value currently in the multiset.
   __int64        *n_elements;         // n_elements[v] = the number of elements in the multiset with value v.
   __int64        total_n_elements;    // = total number of elements currently in the multiset.
};
