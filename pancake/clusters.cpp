#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <math.h>
#include <string.h>
#include <vector>
#include "clusters.h"

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

//_________________________________________________________________________________________________

__int64 Clusters::insert(int direction, unsigned char g, unsigned char h1, unsigned char h2, State_index state_ind, bistates_array *states)
/*
   1. This function inserts an element with value state_ind into the appropriate cluster.
   2. Input Variables
      a. Let v be the node that is to be added to clusters.
      b. direction = 1 = forward direction = insert v in the forward clusters.
                   = 2 = reverse direction = insert v in the reverse clusters.
      c. g = g1(v) if direction = 1, o.w., g = g2(v).
      d. h1 = h1(v).
      e. h2 = h2(v).
      f. state_ind = the index of the state (in states).
   3. Output Variables
      a. index = the index at which the element was inserted into forward_clusters[g][h1][h2] or reverse_clusters[g][h1][h2].
            -1 is returned if an error occurs, such as running out of memory.
   4. Written 5/17/19.
*/
{
   int               status;
   __int64           index;
   pair<set<Cluster_indices>::iterator, bool>   iterator_status;

   assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   if (direction == 1) {
      index = forward_clusters[g][h1][h2].push(state_ind);
      //forward_clusters[g][h1][h2].print();
      if (index >= 0) {
         n_open_forward++;
         (*states)[state_ind].cluster_index1 = index;
         if (forward_clusters[g][h1][h2].size() == 1) {
            status = insert_cluster_indices(1, g, h1, h2);
            if (status == -1) return(-1);
         }
      }
   } else {
      index = reverse_clusters[g][h1][h2].push(state_ind);
      if (index >= 0) {
         n_open_reverse++;
         (*states)[state_ind].cluster_index2 = index;
         if (reverse_clusters[g][h1][h2].size() == 1) {
            status = insert_cluster_indices(2, g, h1, h2);
            if (status == -1) return(-1);
         }
      }
   }

   return(index);
}

//_________________________________________________________________________________________________

// Overload the insert function to accept a pointer to the state instead of g, h1, h2.

//_________________________________________________________________________________________________

void Clusters::delete_element(int direction, unsigned char g, unsigned char h1, unsigned char h2, State_index state_ind, __int64 index_in_cluster, bistates_array *states)
/*
   1. This function deletes an element with value state_ind from the appropriate cluster.
   2. Input Variables
      a. Let v be the node that is to be added to clusters.
      b. direction = 1 = forward direction = delete v from the forward clusters.
                   = 2 = reverse direction = delete v from the reverse clusters.
      c. g = g1(v) if direction = 1, o.w., g = g2(v).
      d. h1 = h1(v).
      e. h2 = h2(v).
      f. state_ind = the index of the state (in states).
      g. index_in_cluster = the index of the element in forward_clusters[g][h1][h2] or reverse_clusters[g][h1][h2].
   3. Output Variables
   4. Written 5/22/19.
   5. Warning: This code has not been debugged.  Need to add variables in bistate to store the index of the element.
*/
{
   unsigned int      n_erased;
   __int64           index;

   assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   if (direction == 1) {
      index = forward_clusters[g][h1][h2].remove(index_in_cluster, state_ind);
      (*states)[state_ind].cluster_index1 = -1;
      if (index >= 0) {
         // The last item in forward_clusters[g][h1][h2] was moved to position index.  Need to update this index in the bistate.
         (*states)[forward_clusters[g][h1][h2][index]].cluster_index1 = index;
      }
      n_open_forward--;
      if (forward_clusters[g][h1][h2].size() == 0) {
         delete_cluster_indices(1, g, h1, h2);        // Delete the cluster from nonempty_forward_clusters.
         compute_LBs();                               // Compute the LBs because they might have changed due to deleting a cluster.
      }
   } else {
      index = reverse_clusters[g][h1][h2].remove(index_in_cluster, state_ind);
      (*states)[state_ind].cluster_index2 = -1;
      if (index >= 0) {
         // The last item in forward_clusters[g][h1][h2] was moved to position index.  Need to update this index in the bistate.
         (*states)[reverse_clusters[g][h1][h2][index]].cluster_index2 = index;
      }
      n_open_reverse--;
      if (reverse_clusters[g][h1][h2].size() == 0) {
         delete_cluster_indices(2, g, h1, h2);        // Delete the cluster from nonempty_reverse_clusters.
         compute_LBs();                               // Compute the LBs because they might have changed due to deleting a cluster.
      }
   }
}

//_________________________________________________________________________________________________

State_index Clusters::pop(int direction, unsigned char g, unsigned char h1, unsigned char h2, bistates_array *states)
/*
   1. This function pops the top element from the forward_clusters[g][h1][h2] or reverse_clusters[g][h1][h2].
   2. Input Variables
      a. Let v be the node that is to be popped from clusters.
      b. direction = 1 = forward direction = pop v from the forward clusters.
                   = 2 = reverse direction = pop v from the reverse clusters.
      c. g = g1(v) if direction = 1, o.w., g = g2(v).
      d. h1 = h1(v).
      e. h2 = h2(v).
   3. Output Variables
      a. state_index = the index of the state (in states).
   4. Written 5/22/19.
*/
{
   unsigned int      n_erased;
   State_index       index;

   assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   if (direction == 1) {
      index = forward_clusters[g][h1][h2].pop();
      (*states)[index].cluster_index1 = -1;
      n_open_forward--;
      if (forward_clusters[g][h1][h2].size() == 0) {
         delete_cluster_indices(1, g, h1, h2);        // Delete the cluster from nonempty_forward_clusters.
         compute_LBs();                               // Compute the LBs because they might have changed due to deleting a cluster.
      }
   } else {
      index = reverse_clusters[g][h1][h2].pop();
      (*states)[index].cluster_index2 = -1;
      n_open_reverse--;
      if (reverse_clusters[g][h1][h2].size() == 0) {
         delete_cluster_indices(2, g, h1, h2);        // Delete the cluster from nonempty_reverse_clusters.
         compute_LBs();                               // Compute the LBs because they might have changed due to deleting a cluster.
      }
   }
   return(index);
}

//_________________________________________________________________________________________________

int Clusters::insert_cluster_indices(int direction, unsigned char g, unsigned char h1, unsigned char h2)
/*
   1. This function inserts a triple of cluster indices into nonempty_forward_clusters or nonempty_reverse_clusters.
   2. Input Variables
      a. direction = 1 = forward direction = insert the triple of indices into nonempty_forward_clusters.
                   = 2 = reverse direction = insert the triple of indices into nonempty_reverse_clusters.
      b. g = g1(v) if direction = 1, o.w., g = g2(v).
      c. h1 = h1(v).
      d. h2 = h2(v).
   e. state_ind = the index of the state (in states).
   3. Output Variables
      a. -1 is returned if an error occurs, such as running out of memory.
   4. Written 5/22/19.
*/
{
   pair<set<Cluster_indices>::iterator, bool>   iterator_status;

   assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   if (direction == 1) {
      try { iterator_status = nonempty_forward_clusters.insert(Cluster_indices(g, h1, h2)); }   catch (bad_alloc x) { return(-1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
      if (!iterator_status.second) { fprintf(stderr, "Attempting to insert indices that already are in the set.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
   } else {
      try { iterator_status = nonempty_reverse_clusters.insert(Cluster_indices(g, h1, h2)); }   catch (bad_alloc x) { return(-1); }   catch (...) { fprintf(stderr, "Unknown exception caught in clusters.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
      if (!iterator_status.second) { fprintf(stderr, "Attempting to insert indices that already are in the set.  Press ENTER to continue\n"); cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); exit(1); }
   }
   return(0);
}

//_________________________________________________________________________________________________

void Clusters::delete_cluster_indices(int direction, unsigned char g, unsigned char h1, unsigned char h2)
/*
   1. This function deletes a triple of cluster indices from nonempty_forward_clusters or nonempty_reverse_clusters.
   2. Input Variables
      a. direction = 1 = forward direction = delete the triple of indices from nonempty_forward_clusters.
                   = 2 = reverse direction = delete the triple of indices from nonempty_reverse_clusters.
      b. g = g1(v) if direction = 1, o.w., g = g2(v).
      c. h1 = h1(v).
      d. h2 = h2(v).
   3. Output Variables
   4. Written 5/22/19.
*/
{
   int   n_erased;

   assert((0 <= g) && (g <= max_size_g));   assert((0 <= h1) && (h1 <= max_size_h));   assert((0 <= h2) && (h2 <= max_size_h));
   if (direction == 1) {
      n_erased = nonempty_forward_clusters.erase(Cluster_indices(g, h1, h2));
      assert(n_erased == 1);
   } else {
      n_erased = nonempty_reverse_clusters.erase(Cluster_indices(g, h1, h2));
      assert(n_erased == 1);
   }
}

//_________________________________________________________________________________________________

tuple<int, Cluster_indices, Min_values> Clusters::choose_cluster(int priority_rule)
/*
   1. This function chooses the next cluster to be expanded.
   2. Input Variables
      a. priority_rule = which rule to use to compute the priority.
   3. Output Variables
      a. direction = the direction in which to expand the chosen cluster.
      b. indices = indices of the chosen cluster.
   4. Written 5/29/19.
*/
{
   unsigned char     g, g_min;
   int               direction;
   Cluster_indices   indices;
   Min_values        min_vals;
   vector<Cluster_indices>::iterator   pnt;

   // Compute the front-to-front LBs for the nonempty forward clusters.

   compute_LBs();

   // Compute the minimum values: g1_min, g2_min, f1_min, f2_min, f1_bar_min, f2_bar_min, f1_hat_min, f2_hat_min, p1_min, p2_min.
   // Store the indices of the clusters whose p1 = p1_min (p2 = p2_min) in ready_forward_clusters (ready_reverse_clusters).

   min_vals = compute_min_values(priority_rule);

   // Choose the cluster.

   if (min_vals.p1_min <= min_vals.p2_min) {    // Best First Direction (BFD).

      // Choose the cluster with the largest g1 from among the ready clusters in the forward direction.

      direction = 1;
      g_min = UCHAR_MAX;
      for (pnt = ready_forward_clusters.begin(); pnt != ready_forward_clusters.end(); pnt++) {
         if (pnt->g < g_min) {
            g_min = pnt->g;
            indices = *pnt;
         }
      }
   } else {

      // Choose the cluster with the largest g2 from among the ready clusters in the reverse direction.

      direction = 2;
      g_min = UCHAR_MAX;
      for (pnt = ready_reverse_clusters.begin(); pnt != ready_reverse_clusters.end(); pnt++) {
         if (pnt->g < g_min) {
            g_min = pnt->g;
            indices = *pnt;
         }
      }
   }

   return(make_tuple(direction, indices, min_vals));
}

//_________________________________________________________________________________________________

void Clusters::compute_LBs()
/*
   1. This function computes LB1_h1_h2 and LB2_h1_h2 for all nonempty clusters.
   2. Input Variables
   3. Output Variables
   4. Written 5/22/19.
*/
{
   unsigned char  g1, g2, h1_f, h1_r, h2_f, h2_r;
   int            min_LB;
   set<Cluster_indices>::iterator   pnt1, pnt2;

   // Compute the front-to-front LB for the nonempty forward clusters.

   for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
      g1 = pnt1->g;   h1_f = pnt1->h1;   h2_f = pnt1->h2;
      min_LB = UCHAR_MAX;
      for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
         g2 = pnt2->g;   h1_r = pnt2->h1;   h2_r = pnt2->h2;
         min_LB = min(min_LB, max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps) + g2);
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB1_h1_h2[h1_f][h2_f] = (unsigned char) min_LB;
   }

   // Compute the front-to-front LB for the nonempty reverse clusters.

   for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
      g2 = pnt2->g;   h1_r = pnt2->h1;   h2_r = pnt2->h2;
      min_LB = UCHAR_MAX;
      for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
         g1 = pnt1->g;   h1_f = pnt1->h1;   h2_f = pnt1->h2;
         min_LB = min(min_LB, g1 + max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps));
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB2_h1_h2[h1_r][h2_r] = (unsigned char)min_LB;
   }
}

//_________________________________________________________________________________________________

void Clusters::compute_LB(const int direction, const Cluster_indices indices)
/*
   1. This function computes LB1_h1_h2 or LB2_h1_h2 for the cluster specified by indices.
   2. Input Variables
      a. direction = 1 = forward direction = compute LB1_h1_h2 for the forward cluster specified by indices.
                   = 2 = reverse direction = compute LB2_h1_h2 for the reverse cluster specified by indices.
      b. indices = the indice (g, h1, h2) of the cluster.
   3. Output Variables
   4. Written 6/7/19.
*/
{
   unsigned char  g1, g2, h1_f, h1_r, h2_f, h2_r;
   int            min_LB;
   set<Cluster_indices>::iterator   pnt1, pnt2;

   if (direction == 1) {

      // Compute the front-to-front LB for the forward cluster.

      g1 = indices.g;   h1_f = indices.h1;   h2_f = indices.h2;
      min_LB = UCHAR_MAX;
      for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
         g2 = pnt2->g;   h1_r = pnt2->h1;   h2_r = pnt2->h2;
         min_LB = min(min_LB, max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps) + g2);
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB1_h1_h2[h1_f][h2_f] = (unsigned char)min_LB;
   } else {
    
      // Compute the front-to-front LB for the reverse cluster.

      g2 = indices.g;   h1_r = indices.h1;   h2_r = indices.h2;
      min_LB = UCHAR_MAX;
      for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
         g1 = pnt1->g;   h1_f = pnt1->h1;   h2_f = pnt1->h2;
         min_LB = min(min_LB, g1 + max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps));
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB2_h1_h2[h1_r][h2_r] = (unsigned char)min_LB;
   }
}

//_________________________________________________________________________________________________

Min_values Clusters::compute_min_values(int priority_rule)
/*
   1. This function 
      a. Computes the minimum values of g1_min, g2_min, f1_min, f2_min, f1_bar_min, f2_bar_min, f1_hat_min, f2_hat_min, p1_min, p2_min among the open nodes.
      b. It stores the indices of the clusters whose p1 = p1_min (p2 = p2_min) in ready_forward_clusters (ready_reverse_clusters).
   2. This function assumes that LB1_h1_h2 and LB2_h1_h2 have already been computed.
   2. Input Variables
      a. priority_rule = which rule to use to compute the priority.
   3. Output Variables
      a. min_vals contains g1_min, g2_min, f1_min, f2_min, f1_bar_min, f2_bar_min, f1_hat_min, f2_hat_min, p1_min, p2_min.
   4. Written 5/27/19.
*/
{
   unsigned char  g1, g2, h1, h2;
   double         p1, p2;
   set<Cluster_indices>::iterator   pnt1, pnt2;
   Min_values     min_vals;

   // Compute the minimum values of g1, f1, f1_bar, f1_hat, and p1.

   for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
      g1 = pnt1->g;   h1 = pnt1->h1;   h2 = pnt1->h2;
      if (g1 < min_vals.g1_min) min_vals.g1_min = g1;
      if (g1 + h1 < min_vals.f1_min) min_vals.f1_min = g1 + h1;
      if (2 * g1 + h1 - h2 < min_vals.f1_bar_min) min_vals.f1_bar_min = 2 * g1 + h1 - h2;
      if (g1 + LB1_h1_h2[h1][h2] < min_vals.f1_hat_min) min_vals.f1_hat_min = g1 + LB1_h1_h2[h1][h2];
      p1 = compute_priority(1, g1, h1, h2, priority_rule);
      if (p1 < min_vals.p1_min) {
         min_vals.p1_min = p1;
         ready_forward_clusters.clear();
         ready_forward_clusters.push_back(*pnt1);
      } else {
         if (p1 == min_vals.p1_min) ready_forward_clusters.push_back(*pnt1);
      }
   }

   // Compute the minimum values of g1, f1, f1_bar, f1_hat, and p1.

   for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
      g2 = pnt2->g;   h1 = pnt2->h1;   h2 = pnt2->h2;
      if (g2 < min_vals.g2_min) min_vals.g2_min = g2;
      if (g2 + h2 < min_vals.f2_min) min_vals.f2_min = g2 + h2;
      if (2 * g2 + h2 - h1 < min_vals.f2_bar_min) min_vals.f2_bar_min = 2 * g2 + h2 - h1;
      if (LB2_h1_h2[h1][h2] + g2 < min_vals.f2_hat_min) min_vals.f2_hat_min = LB2_h1_h2[h1][h2] + g2;
      p2 = compute_priority(2, g2, h1, h2, priority_rule);
      if (p2 < min_vals.p2_min) {
         min_vals.p2_min = p2;
         ready_reverse_clusters.clear();
         ready_reverse_clusters.push_back(*pnt2);
      } else {
         if (p2 == min_vals.p2_min) ready_reverse_clusters.push_back(*pnt2);
      }
   }

   return(min_vals);
}

//_________________________________________________________________________________________________

double Clusters::compute_priority(int direction, unsigned char g, unsigned char h1, unsigned char h2, int rule)
/*
   1. This function computes the priority of a node.
   2. Input Variables
      a. direction = 1 to compute the priority in the forward direction
                   = 2 to compute the priority in the reverse direction.
      b. g  = number of moves that have been made so far in the given direction.
      c. h1 = lower bound on the number of moves needed to reach the goal postion.
      d. h2 = lower bound on the number of moves needed to reach the source postion.
      e. rule = which rule to use to compute the priority.
   3. Output Variables
      a. priority = the priority of this node.
   4. Written 5/27/19.
*/
{
   double         priority;

   switch (rule) {
      case 1:  // priority = f_d = g_d + h_d
         if (direction == 1) {
            priority = g + h1;
         } else {
            priority = g + h2;
         }
         break;
      case 2:  // priority = f_bar = f_d + g_d - h_d' = 2*g_d + h_d - h_d'.
         if (direction == 1) {
            priority = 2 * g + h1 - h2;
         } else {
            priority = 2 * g + h2 - h1;
         }
         break;
      case 3:  // priority = f_d - (g_d /(max_size_g + 1) Break ties in f_d in favor of states with larger g_d.
         if (direction == 1) {
            priority = g + h1 - (double)g / (double)(max_size_g + 1);
         } else {
            priority = g + h2 - (double)g / (double)(max_size_g + 1);
         }
         break;
      case 4:  // priority = f_bar_d - (g_d /(max_size_g + 1) Break ties in fbar_d in favor of states with larger g_d.
         if (direction == 1) {
            priority = 2 * g + h1 - h2 - (double)g / (double)(max_size_g + 1);
         } else {
            priority = 2 * g + h2 - h1 - (double)g / (double)(max_size_g + 1);
         }
         break;
      case 5:  // priority = max(2*g_d, f_d) MM priority function.
         if (direction == 1) {
            priority = max(2 * g, g + h1);
         } else {
            priority = max(2 * g, g + h2);
         }
         break;
      case 6:  // priority = max(2*g_d + 1, f_d) MMe priority function.
         if (direction == 1) {
            priority = max(2 * g + 1, g + h1);
         } else {
            priority = max(2 * g + 1, g + h2);
         }
         break;
      case 7:  // priority = max(2*g_d, f_d) + (g_d /(max_size_g + 1) MM priority function.  Break ties in favor of states with smaller g_d.
         if (direction == 1) {
            priority = max(2 * g, g + h1) + (double)g / (double)(max_size_g + 1);
         } else {
            priority = max(2 * g, g + h2) + (double)g / (double)(max_size_g + 1);
         }
         break;
      case 8:  // priority = max(2*g_d + 1, f_d) + (g_d /(max_size_g + 1) MMe priority function.  Break ties in favor of states with smaller g_d.
         if (direction == 1) {
            priority = max(2 * g + 1, g + h1) + (double)g / (double)(max_size_g + 1);
         } else {
            priority = max(2 * g + 1, g + h2) + (double)g / (double)(max_size_g + 1);
         }
         break;
      case 9:  // priority = max(2*g_d + 1, f_d) - (g_d /(max_size_g + 1) MMe priority function.  Break ties in favor of states with larger g_d.
         if (direction == 1) {
            priority = max(2 * g + 1, g + h1) - (double)g / (double)(max_size_g + 1);
         } else {
            priority = max(2 * g + 1, g + h2) - (double)g / (double)(max_size_g + 1);
         }
         break;
      case 10:  // priority = max(2*g_d + 1, f_d) + (f_d /(max_size_g + 1) MMe priority function.  Break ties in favor of states with smaller f_d.
         if (direction == 1) {
            priority = max(2 * g + 1, g + h1) + (double)(g + h1) / (double)(max_size_g + 1);
         } else {
            priority = max(2 * g + 1, g + h2) + (double)(g + h2) / (double)(max_size_g + 1);
         }
         break;
      case 11:  // priority = max(2*g_d + 1, f_d) + (h_d /(max_size_g + 1) MMe priority function.  Break ties in favor of states with smaller h_d.
         if (direction == 1) {
            priority = max(2 * g + 1, g + h1) + (double)(h1) / (double)(max_size_g + 1);
         } else {
            priority = max(2 * g + 1, g + h2) + (double)(h2) / (double)(max_size_g + 1);
         }
         break;
      default:
         fprintf(stderr, "Unknown priority measure\n");
         exit(1);
         break;
   }

   return(priority);
}

//_________________________________________________________________________________________________

void Clusters::check_clusters()
/*
   1. check_clusters performs some elementary checks on the multiset.
   2. Written 5/20/19.
*/
{
   unsigned char  g, g1, g2, h1, h1_f, h1_r, h2, h2_f, h2_r, max1_h1, max1_h2, max2_h1, max2_h2;
   int            min_LB;
   __int64        n_open_f, n_open_r;
   vector<vector<unsigned char>>    LB1, LB2, min_g1, min_g2;
   set<Cluster_indices>::iterator   pnt, pnt1, pnt2;

   min_g1.resize(max_size_h + 1);
   min_g2.resize(max_size_h + 1);
   for (h1 = 0; h1 <= max_size_h; h1++) {
      min_g1[h1].resize(max_size_h + 1);
      min_g2[h1].resize(max_size_h + 1);
      for (h2 = 0; h2 <= max_size_h; h2++) {
         min_g1[h1][h2] = max_size_g;
         min_g2[h1][h2] = max_size_g;
      }
   }
   max1_h1 = 0;   max1_h2 = 0;   max2_h1 = 0;   max2_h2 = 0;

   // Check if n_open_forward and n_open_reverse are correct.
   // Check if every nonempty forward (reverse) cluster is in nonempty_forward_cluster (nonempty_reverse_cluster).

   n_open_f = 0;
   n_open_r = 0;
   for (g = 0; g <= max_size_g; g++) {
      for (h1 = 0; h1 <= max_size_h; h1++) {
         for (h2 = 0; h2 <= max_size_h; h2++) {
            n_open_f += forward_clusters[g][h1][h2].size();
            n_open_r += reverse_clusters[g][h1][h2].size();
            if (!forward_clusters[g][h1][h2].empty()) {
               if (g < min_g1[h1][h2]) min_g1[h1][h2] = g;
               if (h1 > max1_h1) max1_h1 = h1;
               if (h2 > max1_h2) max1_h2 = h2;
               pnt = nonempty_forward_clusters.find(Cluster_indices(g, h1, h2));
               if (pnt == nonempty_forward_clusters.end()) {
                  fprintf(stderr, "Error in check_clusters: cluster_indices not found in nonempty_forward_clusters.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                  exit(1);
               }
            }
            if (!reverse_clusters[g][h1][h2].empty()) {
               if (g < min_g2[h1][h2]) min_g2[h1][h2] = g;
               if (h1 > max2_h1) max2_h1 = h1;
               if (h2 > max2_h2) max2_h2 = h2;
               pnt = nonempty_reverse_clusters.find(Cluster_indices(g, h1, h2));
               if (pnt == nonempty_reverse_clusters.end()) {
                  fprintf(stderr, "Error in check_clusters: cluster_indices not found in nonempty_reverse_clusters.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                  exit(1);
               }
            }
         }
      }
   }
   if (n_open_f != n_open_forward) {
      fprintf(stderr, "Error in check_clusters: n_open_f != n_open_forward.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }
   if (n_open_r != n_open_reverse) {
      fprintf(stderr, "Error in check_clusters: n_open_r != n_open_revers.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      exit(1);
   }

   // Check if every cluster in nonempty_forward_clusters actually is nonempty.

   for (pnt = nonempty_forward_clusters.begin(); pnt != nonempty_forward_clusters.end(); pnt++) {
      if(forward_clusters[pnt->g][pnt->h1][pnt->h2].empty()) {
         fprintf(stderr, "Error in check_clusters: nonempty_forward_clusters contains an empty cluster.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
   }
   for (pnt = nonempty_reverse_clusters.begin(); pnt != nonempty_reverse_clusters.end(); pnt++) {
      if (reverse_clusters[pnt->g][pnt->h1][pnt->h2].empty()) {
         fprintf(stderr, "Error in check_clusters: nonempty_reverse_clusters contains an empty cluster.  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         exit(1);
      }
   }

   // Initialize LB1 and LB2 to check the LBs.

   LB1.resize(max1_h1 + 1);
   for (h1 = 0; h1 <= max1_h1; h1++) {
      LB1[h1].resize(max1_h2 + 1);
      for (h2 = 0; h2 <= max1_h2; h2++) LB1[h1][h2] = UCHAR_MAX;
   }
   LB2.resize(max2_h1 + 1);
   for (h1 = 0; h1 <= max2_h1; h1++) {
      LB2[h1].resize(max2_h2 + 1);
      for (h2 = 0; h2 <= max2_h2; h2++) LB2[h1][h2] = UCHAR_MAX;
   }

   // Compute the front-to-front LB for the nonempty forward clusters.

   for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
      g1 = pnt1->g;   h1_f = pnt1->h1;   h2_f = pnt1->h2;
      min_LB = UCHAR_MAX;
      for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
         g2 = pnt2->g;   h1_r = pnt2->h1;   h2_r = pnt2->h2;
         min_LB = min(min_LB, max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps) + g2);
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB1[h1_f][h2_f] = (unsigned char)min_LB;
   }

   // Compute the front-to-front LB for the nonempty reverse clusters.

   for (pnt2 = nonempty_reverse_clusters.begin(); pnt2 != nonempty_reverse_clusters.end(); pnt2++) {
      g2 = pnt2->g;   h1_r = pnt2->h1;   h2_r = pnt2->h2;
      min_LB = UCHAR_MAX;
      for (pnt1 = nonempty_forward_clusters.begin(); pnt1 != nonempty_forward_clusters.end(); pnt1++) {
         g1 = pnt1->g;   h1_f = pnt1->h1;   h2_f = pnt1->h2;
         min_LB = min(min_LB, g1 + max(max(abs(h1_f - h1_r), abs(h2_f - h2_r)), (int)eps));
      }
      assert((0 <= min_LB) && (min_LB <= UCHAR_MAX));
      LB2[h1_r][h2_r] = (unsigned char)min_LB;
   }

   // Check the front-to-front LB for the nonempty forward clusters.

   for (h1 = 0; h1 <= max1_h1; h1++) {
      for (h2 = 0; h2 <= max1_h2; h2++) {
         if (LB1[h1][h2] != LB1_h1_h2[h1][h2]) {
            fprintf(stderr, "Error in check_clusters: LB1[h1][h2] != LB1_h1_h2[h1][h2].  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
      }
   }
   // Check the front-to-front LB for the nonempty reverse clusters.

   for (h1 = 0; h1 <= max2_h1; h1++) {
      for (h2 = 0; h2 <= max2_h2; h2++) {
         if (LB2[h1][h2] != LB2_h1_h2[h1][h2]) {
            fprintf(stderr, "Error in check_clusters: LB2[h1][h2] != LB2_h1_h2[h1][h2].  Press ENTER to continue\n");  cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            exit(1);
         }
      }
   }
}

//_________________________________________________________________________________________________

void Clusters::print_min_g()
/*
   1. Print the minimum value of g for each combination of h1,h2.
*/
{
   unsigned char  g, h1, h2, i, j, max1_h1, max1_h2, max2_h1, max2_h2, **min_g1, **min_g2;

   // Compute the largest h1,h2 values in both directions.
   // Compute the smallest g-value for each combination of h1,h2.

   min_g1 = new unsigned char*[max_size_h + 1];
   min_g2 = new unsigned char*[max_size_h + 1];
   for (h1 = 0; h1 <= max_size_h; h1++) {
      min_g1[h1] = new unsigned char[max_size_h + 1];
      min_g2[h1] = new unsigned char[max_size_h + 1];
      for (h2 = 0; h2 <= max_size_h; h2++) {
         min_g1[h1][h2] = max_size_g;
         min_g2[h1][h2] = max_size_g;
      }
   }
   max1_h1 = 0;
   max1_h2 = 0;
   max2_h1 = 0;
   max2_h2 = 0;
   for (g = 0; g <= max_size_g; g++) {
      for (h1 = 0; h1 <= max_size_h; h1++) {
         for (h2 = 0; h2 <= max_size_h; h2++) {
            if (!forward_clusters[g][h1][h2].empty()) {
               if (g < min_g1[h1][h2]) min_g1[h1][h2] = g;
               if (h1 > max1_h1) max1_h1 = h1;
               if (h2 > max1_h2) max1_h2 = h2;
            }
            if (!reverse_clusters[g][h1][h2].empty()) {
               if (g < min_g2[h1][h2]) min_g2[h1][h2] = g;
               if (h1 > max2_h1) max2_h1 = h1;
               if (h2 > max2_h2) max2_h2 = h2;
            }
         }
      }
   }

   // Print the minimum value of g1 for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max1_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max1_h2; h2++) {
         if (min_g1[h1][h2] < max_size_g) {
            printf(" %2d", min_g1[h1][h2]);
         } else {
            printf("  *");
         }
      }
      printf("\n");
   }
   printf("\n");

   // Print the minimum value of g2 for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max2_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max2_h2; h2++) {
         if (min_g2[h1][h2] < max_size_g) {
            printf(" %2d", min_g2[h1][h2]);
         } else {
            printf("  *");
         }
      }
      printf("\n");
   }
   printf("\n");

   for (h1 = 0; h1 <= max_size_h; h1++) {
      delete[] min_g1[h1];
      delete[] min_g2[h1];
   }
   delete[] min_g1;
   delete[] min_g2;
}

//_________________________________________________________________________________________________

void Clusters::print_min_g_n_open()
/*
   1. Print the minimum value of g and the number of open nodes for each combination of h1,h2.
*/
{
   unsigned char  g, h1, h2, i, j, max1_h1, max1_h2, max2_h1, max2_h2, **min_g1, **min_g2;

   // Compute the largest h1,h2 values in both directions.
   // Compute the smallest g-value for each combination of h1,h2.

   min_g1 = new unsigned char*[max_size_h + 1];
   min_g2 = new unsigned char*[max_size_h + 1];
   for (h1 = 0; h1 <= max_size_h; h1++) {
      min_g1[h1] = new unsigned char[max_size_h + 1];
      min_g2[h1] = new unsigned char[max_size_h + 1];
      for (h2 = 0; h2 <= max_size_h; h2++) {
         min_g1[h1][h2] = max_size_g;
         min_g2[h1][h2] = max_size_g;
      }
   }
   max1_h1 = 0;
   max1_h2 = 0;
   max2_h1 = 0;
   max2_h2 = 0;
   for (g = 0; g <= max_size_g; g++) {
      for (h1 = 0; h1 <= max_size_h; h1++) {
         for (h2 = 0; h2 <= max_size_h; h2++) {
            if (!forward_clusters[g][h1][h2].empty()) {
               if (g < min_g1[h1][h2]) min_g1[h1][h2] = g;
               if (h1 > max1_h1) max1_h1 = h1;
               if (h2 > max1_h2) max1_h2 = h2;
            }
            if (!reverse_clusters[g][h1][h2].empty()) {
               if (g < min_g2[h1][h2]) min_g2[h1][h2] = g;
               if (h1 > max2_h1) max2_h1 = h1;
               if (h2 > max2_h2) max2_h2 = h2;
            }
         }
      }
   }

   // Print the minimum value of g1 and the number of open nodes for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf("           %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max1_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max1_h2; h2++) {
         g = min_g1[h1][h2];
         if (g < max_size_g) {
            printf(" (%2d, %6I64d)", g, forward_clusters[g][h1][h2].size());
         } else {
            printf("            *");
         }
      }
      printf("\n");
   }
   printf("\n");

   // Print the minimum value of g2 and the number of open nodes for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max2_h2; h2++) printf("           %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max2_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max2_h2; h2++) {
         g = min_g2[h1][h2];
         if (g < max_size_g) {
            printf(" (%2d, %6I64d)", g, reverse_clusters[g][h1][h2].size());
         } else {
            printf("            *");
         }
      }
      printf("\n");
   }
   printf("\n");

   for (h1 = 0; h1 <= max_size_h; h1++) {
      delete[] min_g1[h1];
      delete[] min_g2[h1];
   }
   delete[] min_g1;
   delete[] min_g2;
}

//_________________________________________________________________________________________________

void Clusters::print_cluster(const int direction, const unsigned char g, const unsigned char h1, const unsigned char h2)
{
   __int64        i;
   State_index    *pnt;

   if (direction == 1) {
      
      // Print the nonempty forward clusters.

      printf("Forward ");
      printf("(%2d, %2d, %2d) ", g, h1, h2);
      for (i = 1; i <= forward_clusters[g][h1][h2].size(); i++) {
         printf(" %10d\n", forward_clusters[g][h1][h2][i]);
      }
   } else {
    
      // Print the nonempty reverse clusters.

      printf("Reverse ");
      printf("(%2d, %2d, %2d) ", g, h1, h2);
      for (i = 1; i <= reverse_clusters[g][h1][h2].size(); i++) {
         printf(" %10d\n", reverse_clusters[g][h1][h2][i]);
      }
   }
}

//_________________________________________________________________________________________________

void Clusters::print_nonempty_clusters()
{
   unsigned char  g, h1, h2;
   set<Cluster_indices>::iterator   pnt;

   // Print the nonempty forward clusters.

   printf("Forward \n");
   printf(" g1 h1 h2 f1_hat f1 Cardinality \n");
   for (pnt = nonempty_forward_clusters.begin(); pnt != nonempty_forward_clusters.end(); pnt++) {
      g = pnt->g;   h1 = pnt->h1;   h2 = pnt->h2;
      printf(" %2d %2d %2d     %2d %2d %10I64d\n",  g, h1, h2, g + LB1_h1_h2[h1][h2], g + h1, forward_clusters[g][h1][h2].size());
   }

   // Print the nonempty reverse clusters.

   printf("Reverse \n");
   printf(" g2 h1 h2 f2_hat f2 Cardinality \n");
   for (pnt = nonempty_reverse_clusters.begin(); pnt != nonempty_reverse_clusters.end(); pnt++) {
      g = pnt->g;   h1 = pnt->h1;   h2 = pnt->h2;
      printf(" %2d %2d %2d     %2d %2d %10I64d\n", g, h1, h2, g + LB2_h1_h2[h1][h2], g + h2, reverse_clusters[g][h1][h2].size());
   }

}

//_________________________________________________________________________________________________

void Clusters::print_LBs()
/*
   1. Print LB1_h1_h2 and LB2_h1_h2 for each combination of h1,h2.
*/
{
   unsigned char  g, h1, h2, i, j, max1_h1, max1_h2, max2_h1, max2_h2, **min_g1, **min_g2;

   // Compute the largest h1,h2 values in both directions.
   // Compute the smallest g-value for each combination of h1,h2.

   min_g1 = new unsigned char*[max_size_h + 1];
   min_g2 = new unsigned char*[max_size_h + 1];
   for (h1 = 0; h1 <= max_size_h; h1++) {
      min_g1[h1] = new unsigned char[max_size_h + 1];
      min_g2[h1] = new unsigned char[max_size_h + 1];
      for (h2 = 0; h2 <= max_size_h; h2++) {
         min_g1[h1][h2] = max_size_g;
         min_g2[h1][h2] = max_size_g;
      }
   }
   max1_h1 = 0;
   max1_h2 = 0;
   max2_h1 = 0;
   max2_h2 = 0;
   for (g = 0; g <= max_size_g; g++) {
      for (h1 = 0; h1 <= max_size_h; h1++) {
         for (h2 = 0; h2 <= max_size_h; h2++) {
            if (!forward_clusters[g][h1][h2].empty()) {
               if (g < min_g1[h1][h2]) min_g1[h1][h2] = g;
               if (h1 > max1_h1) max1_h1 = h1;
               if (h2 > max1_h2) max1_h2 = h2;
            }
            if (!reverse_clusters[g][h1][h2].empty()) {
               if (g < min_g2[h1][h2]) min_g2[h1][h2] = g;
               if (h1 > max2_h1) max2_h1 = h1;
               if (h2 > max2_h2) max2_h2 = h2;
            }
         }
      }
   }

   // Print LB1_h1_h2 for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max1_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max1_h2; h2++) {
         if (LB1_h1_h2[h1][h2] < UCHAR_MAX) {
            printf(" %2d", LB1_h1_h2[h1][h2]);
         } else {
            printf("  *");
         }
      }
      printf("\n");
   }
   printf("\n");

   // Print LB2_h1_h2 for each combination of h1,h2.

   printf("   ");  for (h2 = 0; h2 <= max1_h2; h2++) printf(" %2d", h2); printf("\n");
   for (h1 = 0; h1 <= max2_h1; h1++) {
      printf("%2d:", h1);
      for (h2 = 0; h2 <= max2_h2; h2++) {
         if (LB2_h1_h2[h1][h2] < UCHAR_MAX) {
            printf(" %2d", LB2_h1_h2[h1][h2]);
         } else {
            printf("  *");
         }
      }
      printf("\n");
   }
   printf("\n");

   for (h1 = 0; h1 <= max_size_h; h1++) {
      delete[] min_g1[h1];
      delete[] min_g2[h1];
   }
   delete[] min_g1;
   delete[] min_g2;
}
