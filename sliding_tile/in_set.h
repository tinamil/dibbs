#ifndef _in_set_
#define _in_set_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>

using namespace std;

/*
   Modified 12/27/11 to permit use of index 0 in int_array.
*/

/*************************************************************************************************/

class in_set {
public:
   in_set()    {  counter = 0; n = 0; int_array = NULL;}
   explicit in_set(const int nn)    {  assert(nn > 0);
                                       counter = 0; 
                                       n = nn;
                                       try {
                                          int_array = new int[n + 1];
                                       }
                                       catch(bad_alloc x) {
                                          cerr << "bad_alloc caught in in_set" << endl;
                                          exit(1);
                                       }
                                       catch(...) {cerr << "Unknown exception caught in in_set" << endl;}
                                       for(int i = 0; i <= n; i++) int_array[i] = 0;
                                    };
   void  initialize(const int nn)   {  assert(nn > 0);
                                       counter = 0;
                                       n = nn;
                                       if(int_array != NULL) delete [] int_array;
                                       try {
                                          int_array = new int[n + 1];
                                       }
                                       catch(bad_alloc x) {
                                          cerr << "bad_alloc caught in in_set" << endl;
                                          exit(1);
                                       }
                                       catch(...) {cerr << "Unknown exception caught in in_set" << endl;}
                                       for(int i = 0; i <= n; i++) int_array[i] = 0;
                                    };
   ~in_set()      { delete [] int_array;}
   bool  operator[] (const int i) const {assert((0 <= i) && (i <= n)); return(int_array[i] == counter);}
   void  set(const int i)     {  assert((0 <= i) && (i <= n)); int_array[i] = counter;}
   void  set0(const int i)    {  assert((0 <= i) && (i <= n)); int_array[i] = 0;}
   void  increment()    {  counter++;
                           if(counter == INT_MAX) {
                              counter = 1;
                              for(int i = 0; i <= n; i++) int_array[i] = 0;
                           }
                        };
   int   size()   const {return n;}
   void  print()     {    for(int i = 0; i <= n; i++) printf("%4d ", int_array[i]); printf("\n");}
   void  print_in_set()   {    for(int i = 0; i <= n; i++) if(int_array[i] == counter) printf("%4d ", i); printf("\n");}
;
private:
   int      counter;
   int      n;             // The number of elements in the vector (not counting position 0)
   int      *int_array;
};

//_________________________________________________________________________________________________

class in_set_uchar {
public:
   in_set_uchar() {  counter = 0; n = 0; uchar_array = NULL;}
   explicit in_set_uchar(const __int64 nn)   {  assert(nn > 0);
                                                counter = 0; 
                                                n = nn;
                                                try {
                                                   uchar_array = new unsigned char[n + 1];
                                                }
                                                catch(bad_alloc x) {
                                                   cerr << "bad_alloc caught in in_set_uchar" << endl;
                                                   exit(1);
                                                }
                                                catch(...) {cerr << "Unknown exception caught in in_set_uchar" << endl;}
                                                for(int i = 0; i <= n; i++) uchar_array[i] = 0;
                                             };
   void  initialize(const __int64 nn)  {  assert(nn > 0);
                                          counter = 0;
                                          n = nn;
                                          if(uchar_array != NULL) delete [] uchar_array;
                                          try {
                                             uchar_array = new unsigned char[n + 1];
                                          }
                                          catch(bad_alloc x) {
                                             cerr << "bad_alloc caught in in_set_uchar" << endl;
                                             exit(1);
                                          }
                                          catch(...) {cerr << "Unknown exception caught in in_set_uchar" << endl;}
                                          for(int i = 0; i <= n; i++) uchar_array[i] = 0;
                                       };
   ~in_set_uchar()   { delete [] uchar_array;}
   bool  operator[] (const __int64 i) const {assert((0 <= i) && (i <= n)); return(uchar_array[i] == counter);}
   void  set(const __int64 i)    {  assert((0 <= i) && (i <= n)); uchar_array[i] = counter;}
   void  set0(const __int64 i)   {  assert((0 <= i) && (i <= n)); uchar_array[i] = 0;}
   void  increment()    {  counter++;
                           if(counter == UCHAR_MAX) {
                              counter = 1;
                              for(int i = 0; i <= n; i++) uchar_array[i] = 0;
                           }
                        };
   __int64  size()   const {return n;}
   void  print()     {    for(__int64 i = 0; i <= n; i++) printf("%4d ", uchar_array[i]); printf("\n");}
   void  print_in_set()   {    for(__int64 i = 0; i <= n; i++) if(uchar_array[i] == counter) printf("%4d ", i); printf("\n");}
;
private:
   unsigned char  counter;
   __int64        n;             // The number of elements in the vector (not counting position 0)
   unsigned char  *uchar_array;
};

#endif
