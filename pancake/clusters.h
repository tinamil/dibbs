#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <vector>
#include <set>
#include <algorithm>
#include <tuple>
#include "bistates.h"
#include "vec.h"

using namespace std;

/*
   The following class and functions implement a 3D matrix for storing the open nodes in a bidirectional search algorithm.
	 It keeps track of how many times each value is in the multiset and the minimum and maximum value in the multiset.
	 It is designed to be used for keeping track of the minimum value of g or f among the open nodes in a bidirectional search algorithm.
   1. The following assumptions must be satisfied in order to use this data structure.
      a. Each value of g, h1, and h2 must be a nonnegative integer (unsigned char).
      b. The maximum search depth can be bounded prior to the search.
      c. The maximum value of h1,h2 can be bounded prior to the search.
   2. The stacks consists of the following.
      a. forward_clusters[g1][h1][h2] = {v in O1: g1(v) = g1, h1(v) = h1, h2(v) = h2}, where O1 is the set of open nodes in the forward direction.
      b. reverse_clusters[g2][h1][h2] = {v in O2: g2(v) = g2, h1(v) = h1, h2(v) = h2}, where O2 is the set of open nodes in the reverse direction.
      c. max_size_g = max depth of the search tree.
      d. max_size_h = max h1,h2 value.
   3. Written 5/11/19.
*/

/*************************************************************************************************/

class Min_values {
public:
   Min_values() { g1_min = g2_min = f1_min = f2_min = f1_bar_min = f2_bar_min = f1_hat_min = f2_hat_min = UCHAR_MAX; p1_min = p2_min = DBL_MAX; }
   unsigned char  g1_min, g2_min, f1_min, f2_min, f1_bar_min, f2_bar_min, f1_hat_min, f2_hat_min;
   double         p1_min, p2_min;
};


/*************************************************************************************************/

class Cluster_indices {
public:
   Cluster_indices() { g = 0; h1 = 0; h2 = 0; }
   Cluster_indices(const unsigned char gg, const unsigned char hh1, const unsigned char hh2) { g = gg; h1 = hh1; h2 = hh2; }
   ~Cluster_indices() {}
   bool operator<(const Cluster_indices& c) const
   {
      if (g < c.g) return(true);
      if (g > c.g) return(false);
      if (h1 < c.h1) return(true);
      if (h1 > c.h1) return(false);
      if (h2 < c.h2) return(true);
      return(false);
   }
   unsigned char  g, h1, h2;
};

/*************************************************************************************************/

class Clusters {
public:
   Clusters() { initialized = false; eps = 0; max_size_g = 0; max_size_h = 0; n_open_forward = 0; n_open_reverse = 0; }
   ~Clusters() {} 
   //~Clusters() {  for (int g = 0; g <= max_size_g; g++) {
   //                  for (int h1 = 0; h1 <= max_size_h; h1++) {
   //                     for (int h2 = 0; h2 <= max_size_h; h2++) {
   //                        forward_clusters[g][h1][h2].release_memory();   // Is this necessary?
   //                        reverse_clusters[g][h1][h2].release_memory();   // Is this necessary?
   //                     }
   //                     delete[] forward_clusters[g][h1];
   //                     delete[] reverse_clusters[g][h1];
   //                  }
   //                  delete[] forward_clusters[g];
   //                  delete[] reverse_clusters[g];
   //               }
   //            }
   void  initialize(const unsigned char epsilon, const unsigned char maximum_size_g, const unsigned char maximum_size_h)
   {
      unsigned char  g, h1, h2;
      eps = epsilon;
      assert(maximum_size_g > 0);   assert(maximum_size_h > 0);
      max_size_g = maximum_size_g;   max_size_h = maximum_size_h;
      n_open_forward = 0;   n_open_reverse = 0;
      try { forward_clusters.resize(max_size_g + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); exit(1); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');}
      try { reverse_clusters.resize(max_size_g + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
      for (g = 0; g <= max_size_g; g++) {
         try { forward_clusters[g].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); exit(1); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');}
         try { reverse_clusters[g].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
         for (h1 = 0; h1 <= max_size_h; h1++) {
            try { forward_clusters[g][h1].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
            try { reverse_clusters[g][h1].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
         }
      }
      try { LB1_h1_h2.resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); exit(1); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }
      try { LB2_h1_h2.resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); exit(1); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }
      for (h1 = 0; h1 <= max_size_h; h1++) {
         try { LB1_h1_h2[h1].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
         try { LB2_h1_h2[h1].resize(max_size_h + 1); }   catch (bad_alloc x) { fprintf(stderr, "Out of space in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
         for (h2 = 0; h2 <= max_size_h; h2++) {
            LB1_h1_h2[h1][h2] = UCHAR_MAX;
            LB2_h1_h2[h1][h2] = UCHAR_MAX;
         }
      }
      initialized = true;
   }
   //__int64& operator[] (int v) const { assert((0 <= v) && (v <= max_size_value)); return n_elements[v]; }
   // The following overload of the () operator is not functioning correctly.  It appears that it may be returning a copy of the cluster instead of a pointer to the original cluster.
   //vec<State_index> operator() (const int direction, const unsigned char g, const unsigned char h1, const unsigned char h2) const 
   //{  assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   //   if (direction == 1) {
   //      return(forward_clusters[g][h1][h2]);
   //   } else {
   //      return(reverse_clusters[g][h1][h2]);
   //   }
   //}
   __int64  n_forward()    const { return(n_open_forward); }
   __int64  n_reverse()    const { return(n_open_reverse); }
   bool  empty()           const { return(n_open_forward + n_open_reverse == 0); }
   bool  empty(const int direction, const unsigned char g, const unsigned char h1, const unsigned char h2) const 
   {  assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
      if(direction == 1) return(forward_clusters[g][h1][h2].empty()); else return(reverse_clusters[g][h1][h2].empty());
   }
   void  clear() {   unsigned char  g, h1, h2;
                     for (g = 0; g <= max_size_g; g++) {
                        for (h1 = 0; h1 <= max_size_h; h1++) {
                           for (h2 = 0; h2 <= max_size_h; h2++) {
                              forward_clusters[g][h1][h2].clear();
                              forward_clusters[g][h1][h2].clear();
                           }
                        }
                     }
                     nonempty_forward_clusters.clear();
                     nonempty_reverse_clusters.clear();
                     ready_forward_clusters.clear();
                     ready_reverse_clusters.clear();
                     n_open_forward = 0;
                     n_open_reverse = 0;
                     for (h1 = 0; h1 <= max_size_h; h1++) {
                        for (h2 = 0; h2 <= max_size_h; h2++) {
                           LB1_h1_h2[h1][h2] = UCHAR_MAX;
                           LB2_h1_h2[h1][h2] = UCHAR_MAX;
                        }
                     }
                  }
   __int64     insert(int direction, unsigned char g, unsigned char h1, unsigned char h2, State_index state_ind, bistates_array *states);
   void        delete_element(int direction, unsigned char g, unsigned char h1, unsigned char h2, State_index state_ind, __int64 index_in_cluster, bistates_array *states);
   State_index Clusters::pop(int direction, unsigned char g, unsigned char h1, unsigned char h2, bistates_array *states);
   int         insert_cluster_indices(int direction, unsigned char g, unsigned char h1, unsigned char h2);
   void        delete_cluster_indices(int direction, unsigned char g, unsigned char h1, unsigned char h2);
   tuple<int, Cluster_indices, Min_values> choose_cluster(int priority_rule);
   void        compute_LBs();
   void        compute_LB(const int direction, const Cluster_indices indices);
   Min_values  compute_min_values(int priority_rule);
   double      compute_priority(int direction, unsigned char g, unsigned char h1, unsigned char h2, int rule);
   unsigned char  LB1(const unsigned char h1, const unsigned char h2)   const { assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));   return(LB1_h1_h2[h1][h2]); }
   unsigned char  LB2(const unsigned char h1, const unsigned char h2)   const { assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));   return(LB2_h1_h2[h1][h2]); }
   bool        is_initialized() const { return(initialized); }
   void        set_eps(const unsigned char epsilon) { eps = epsilon; }
   //int   get_min();
   //int   delete_min();
   //void  update_min(int v);
   //int   get_max();
   //int   delete_max();
   //void  update_max(int v);
   void     check_clusters();
   void     print_min_g();
   void     print_min_g_n_open();
   void     print_cluster(const int direction, const unsigned char g, const unsigned char h1, const unsigned char h2);
   void     print_nonempty_clusters();
   void     print_LBs();
   //void   print_stats();
private:
   bool                 initialized;         // = true if clusters has already been intialized.
   unsigned char        eps;                 // = length of the longest edge in the graph.
   unsigned char        max_size_g;          // = max depth of the search tree.
   unsigned char        max_size_h;          // = max h1,h2 value.
   __int64              n_open_forward;      // number of open nodes in the forward direction.
   __int64              n_open_reverse;      // number of open nodes in the reverse direction.
   vector<vector<vector<vec<State_index>>>>  forward_clusters; // forward_clusters[g1][h1][h2] = {v in O1: g1(v) = g1, h1(v) = h1, h2(v) = h2}, where O1 is the set of open nodes in the forward direction.
   vector<vector<vector<vec<State_index>>>>  reverse_clusters; // reverse_clusters[g2][h1][h2] = {v in O2: g2(v) = g2, h1(v) = h1, h2(v) = h2}, where O2 is the set of open nodes in the reverse direction.
   set<Cluster_indices> nonempty_forward_clusters;   // = set of indices of nonempty clusters in the forward direction.
   set<Cluster_indices> nonempty_reverse_clusters;   // = set of indices of nonempty clusters in the forward direction.
   vector<Cluster_indices> ready_forward_clusters;   // = set of indices of clusters whose p1 = p1_min.
   vector<Cluster_indices> ready_reverse_clusters;   // = set of indices of clusters whose p2 = p2_min.
   vector<vector<unsigned char>> LB1_h1_h2;  // LB1_h1_h2[h1][h2] = lower bound to complete a path from a node v to node t, where h1(v) = h1 and h2(v) = h2.
                                             //                   = UCHAR_MAX if it has not been computed yet.
   vector<vector<unsigned char>> LB2_h1_h2;  // LB2_h1_h2[h1][h2] = lower bound to complete a reverse path from a node v to node s, where h1(v) = h1 and h2(v) = h2.
                                             //                   = UCHAR_MAX if it has not been computed yet.
};
