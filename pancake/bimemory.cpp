#include "main.h"

/*****************************************************************************/

/*
   1. The following functions manage the list of states for the bidirectional search for the pancake problem.
   2. Conceptually, a subproblem consists of a sequence of the pancakes.  
      To reduce memory requirements, do not store the moves that were made for each subproblem.  
      Instead, use pointers to reconstruct the moves.  A subproblem consists of:
      a. g1 = objective function value = number of flips that have been made so far in the forward direction.
      b. h1 = lower bound on the number of moves needed to reach the goal postion.
      c. open1 = 2 if this subproblem has not yet been generated in the forward direction
               = 1 if this subproblem is open in the forward direction
               = 0 if this subproblem closed in the forward direction
      d. parent1 = index (in states) of the parent subproblem in the forward direction.
      e. g2 = objective function value = number of flips that have been made so far in the reverse direction.
      f. h2 = lower bound on the number of flips needed to reach the source postion.
      g. open2 = 2 if this subproblem has not yet been generated in the reverse direction
               = 1 if this subproblem is open in the reverse direction
               = 0 if this subproblem closed in the reverse direction
      h. parent2 = index (in states) of the parent subproblem in the reverse direction.
      i. hash_value = the index in the hash table to begin searching for the state.
      j. seq[i] = the number of the pancake that is position i (i.e., order of the pancakes).
         seq[0] = the number of pancakes.
   3. Created 7/21/17 by modifying c:\sewell\research\15puzzle\15puzzle_code2\bimemory.cpp.
*/

//_________________________________________________________________________________________________

int search_bimemory(bistate *new_bistate, bistates_array *bistates, searchparameters *parameters, Hash_table *hash_table, int hash_value)
/*
   1. This routine attempts to find state in bidirectional memory.
   2. Input Variables
      a. new_bistate is the new state.
      b. bistates is the list of states in memory.
      c. hash_table is the hash table used to find a bistate in memory.
      d. hash_value = the hash value for new_state.  This is the index in the hash table to begin searching for new_bistate.
   4. Output Variables
      a. -1 is returned if the new state is not found in memory,
          index (in hash_table) is returned if the new state is found in memory.
   3. Written 2/17/15.
*/
{
   int      hash_index, status;

   status = hash_table->find_bistate(new_bistate->seq, hash_value, &hash_index, bistates);
   if(status == -1) 
      return(-1);
   else
      return(hash_index);
}

//_________________________________________________________________________________________________

pair<int, int> find_or_insert(bistate *new_state, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, Hash_table *hash_table, int hash_value, Clusters *clusters, Cluster_indices indices)
/*
   1. This routine attempts to find state in bidirectional memory.  
   2. If the subproblem is found, then
      a. If the new subproblem dominates the old one, then 
         i. The old one is replaced by the new one.
        ii. The index where it is stored in the list of states is inserted in the appropriate heap (unless the heap is full).
            Note: If this subproblem was already open in the forward direction, then this subproblem will be in the heap twice.
      b. If the old subproblem dominates the new one, then the new one is discarded.
   3. If the subproblem is not found, then it is added to memory (by calling add_to_bimemory)
      a. It is added to the list of states (unless states is full).
      b. The index where it is stored in the list of states is inserted into the hash table (unless the hash table is full).
      c. The index where it is stored in the list of states is inserted in the appropriate heap (unless the heap is full).
         The key in the heap will be set equal to best.
   4. Input Variables
      a. new_bistate is the new state.
      b. best = best measure for this subproblem in the current direction.
      c. depth = depth of subproblem in search tree.
      d. direction = 1 if currently searching in the forward direction
                   = 2 if currently searching in the reverse direction.
      e. bistates is the list of states in memory.
      f. hash_table is the hash table used to find a bistate in memory.
      g. hash_value = the hash value for new_state.  This is the index in the hash table to begin searching for new_bistate.
   5. Output Variables
      a. status =  1 if this subproblem was found
                = -1 if this subproblem was not found

      b. state_index = index (in states) of the state
                     = -1 unable to add to memory (states, hash table, or heap).

   6. Written 7/24/17.
   7. Modified 1/15/19 to add nodes to open_g1_values, open_g2_values, open_f1_values, open_f2_values.
*/
{
   unsigned char  f1, f2, g1, g2, h1, h2;
   int            hash_index, status, state_index;
   __int64        index_in_clusters;
   bistate        *old_state;
   heap_record    record;

   status = hash_table->find_bistate(new_state->seq, hash_value, &hash_index, bistates);
   if(status == 1) {

      // The subproblem has been found.  Determine if the new subproblem dominates the old subproblem.

      state_index = (*hash_table)[hash_index].state_index;
      old_state = &(*bistates)[state_index];
      if(direction == 1) {
         if(new_state->g1 < old_state->g1) {
            g1 = old_state->g1; h1 = old_state->h1; h2 = old_state->h2; f1 = g1 + h1;
            assert(old_state->open1 != 0);                     // Do not use this assert if you are using an inconsistent heuristic.
            assert(new_state->h1 == h1);  assert(new_state->h2 == old_state->h2);
            if (old_state->open1 == 1) {
               open_g1_values[f1].delete_element(g1);             // Delete an element with value old_state->g1 from the multiset of open g1 values.
               open_f1_values.delete_element(f1);                 // Delete an element with value old_state->g1 + old_state->h1 from the multiset of open f1 values.
               open_g1_h1_h2_values[h1][h2].delete_element(g1);   // Delete an element with value old_state->g1 from the multiset of open g1 values for the given values of h1 and h2.
                                                                  // The number of elements in the multiset will not equal the number of items in the heaps because the item is not being deleted from the heaps.
               if (clusters != NULL) {
                  clusters->delete_element(1, g1, h1, h2, state_index, old_state->cluster_index1, bistates);
               }
            }
            old_state->g1 = new_state->g1;
            old_state->open1 = 1;
            old_state->parent1 = new_state->parent1;
            record.key = best;
            record.state_index = state_index;
            index_in_clusters = insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, clusters, indices);     // Note: If this subproblem was already open in the forward direction, then this subproblem will be in the heaps twice.
            if (index_in_clusters < 0) state_index = -1;
            open_g1_values[new_state->g1 + new_state->h1].insert(new_state->g1);       // Insert an element with value old_state->g1 into the multiset of open g1 values.
            open_f1_values.insert(new_state->g1 + new_state->h1);                      // Insert an element with value old_state->g1 + old_state->h1 into the multiset of open f1 values.
            open_g1_h1_h2_values[new_state->h1][new_state->h2].insert(new_state->g1);  // Insert an element with value old_state->g1 into the multiset of open g1 values for the given values of h1 and h2.
         }
      } else {
         if (new_state->g2 < old_state->g2) {
            g2 = old_state->g2; h1 = old_state->h1; h2 = old_state->h2; f2 = g2 + h2;
            assert(old_state->open2 != 0);                     //  Do not use this assert if you are using an inconsistent heuristic. 
            assert(new_state->h1 == old_state->h1);  assert(new_state->h2 == h2);
            if (old_state->open2 == 1) {
               open_g2_values[f2].delete_element(g2);             // Delete an element with value old_state->g1 from the multiset of open g1 values.
               open_f2_values.delete_element(f2);                 // Delete an element with value old_state->g2 + old_state->h2 from the multiset of open f2 values.
               open_g2_h1_h2_values[h1][h2].delete_element(g2);   // Delete an element with value old_state->g2 from the multiset of open g2 values for the given values of h1 and h2.
                                                                  // The number of elements in the multiset will not equal the number of items in the heaps because the item is not being deleted from the heaps.
               if (clusters != NULL) {
                  clusters->delete_element(1, g2, h1, h2, state_index, old_state->cluster_index1, bistates);
               }
            }
            old_state->g2 = new_state->g2;
            old_state->open2 = 1;
            old_state->parent2 = new_state->parent2;
            record.key = best;
            record.state_index = state_index;
            index_in_clusters = insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, clusters, indices);     // Note: If this subproblem was already open in the reverse direction, then this subproblem will be in the heaps twice.
            if (index_in_clusters < 0) state_index = -1;
            open_g2_values[new_state->g2 + new_state->h2].insert(new_state->g2);       // Insert an element with value old_state->g1 into the multiset of open g1 values.
            open_f2_values.insert(new_state->g2 + new_state->h2);                      // Insert an element with value old_state->g2 + old_state->h2 into the multiset of open f2 values.
            open_g2_h1_h2_values[new_state->h1][new_state->h2].insert(new_state->g2);  // Insert an element with value old_state->g2 into the multiset of open g2 values for the given values of h1 and h2.
         }
      }
      return(make_pair(status,state_index));
   }  else {

      // The subproblem was not found, so insert it into memory.

      state_index = add_to_bimemory(new_state, best, depth, direction, bistates, parameters, info, hash_table, hash_index, clusters, indices);
      return(make_pair(status,state_index));
   }
}

//_________________________________________________________________________________________________

int add_to_bimemory(bistate *bistate, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, Hash_table *hash_table, int hash_index, Clusters *clusters, Cluster_indices indices)
/*
   1. This routine attempts to add a new state to memory.
      a. If there is sufficient memory available in states and the appropriate heap, then the
         state is added to memory.
      b. If there is insufficient memory in either states or the appropriate heap, then it will
         check if this state should replace an existing state in memory.
   2. -1 is returned if the new state is not added to memory,
      o.w. the index (in states) is returned.
   3. Written 5/31/07.
   4. Modified 1/15/19 to add nodes to open_g1_values, open_g2_values, open_f1_values, open_f2_values.
*/
{
   int         index;
   __int64     index_in_clusters;
   //clock_t  start_time;
   hash_record hrecord;
   heap_record record;

   //start_time = clock();
   
   switch(parameters->search_strategy) {		
		case 1:  // dfs
			break;
		case 2:  // breadth fs
			break;
		case 3:  // best fs
         if(direction == 1) {
            // Do not replace states when states is full.
            if (bistates->is_not_full() && forward_bfs_heap.is_not_full()) {
               index = bistates->add_bistate(bistate);
               hash_table->insert_at_index(index, hash_index);
               record.key = best;
               record.state_index = index;
               insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, NULL, { 0,0,0 });
               open_f1_values.insert(bistate->g1 + bistate->h1);
               open_g1_h1_h2_values[bistate->h1][bistate->h2].insert(bistate->g1);
            } else {
               info->optimal = 0;
               index = -1;
            }
         } else {
            // Do not replace states when states is full.
            if (bistates->is_not_full() && reverse_bfs_heap.is_not_full()) {
               index = bistates->add_bistate(bistate);
               hash_table->insert_at_index(index, hash_index); 
               record.key = best;
               record.state_index = index;
               insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, NULL, { 0,0,0 });
               open_g2_values[bistate->g2 + bistate->h2].insert(bistate->g2);
               open_f2_values.insert(bistate->g2 + bistate->h2);
               open_g2_h1_h2_values[bistate->h1][bistate->h2].insert(bistate->g2);
            } else {
               info->optimal = 0;
               index = -1;
            }
         }
			break;
      case 4:  // best first search using clusters
         if (direction == 1) {
            if (bistates->is_not_full()) {
               index = bistates->add_bistate(bistate);
               hash_table->insert_at_index(index, hash_index);
               record.key = best;
               record.state_index = index;
               index_in_clusters = insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, clusters, indices);
               if (index_in_clusters < 0) { info->optimal = 0; index = -1; }
               open_g1_values[bistate->g1 + bistate->h1].insert(bistate->g1);
               open_f1_values.insert(bistate->g1 + bistate->h1);
               open_g1_h1_h2_values[bistate->h1][bistate->h2].insert(bistate->g1);
            } else {
               info->optimal = 0;
               index = -1;
            }
         } else {
            if (bistates->is_not_full()) {
               index = bistates->add_bistate(bistate);
               hash_table->insert_at_index(index, hash_index);
               record.key = best;
               record.state_index = index;
               index_in_clusters = insert_bidirectional_unexplored(bistates, record, depth, direction, parameters, clusters, indices);
               if (index_in_clusters < 0) { info->optimal = 0; index = -1; }
               open_g2_values[bistate->g2 + bistate->h2].insert(bistate->g2);
               open_f2_values.insert(bistate->g2 + bistate->h2);
               open_g2_h1_h2_values[bistate->h1][bistate->h2].insert(bistate->g2);
            } else {
               info->optimal = 0;
               index = -1;
            }
         }
         break;
      default:
         fprintf(stderr, "This method has not been implemented yet\n\n");
         exit(1);
         break;

   }

   //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;

   return(index);
}

//_________________________________________________________________________________________________

int get_bistate(bistates_array *states, int direction, searchparameters *parameters, searchinfo *info, Clusters *clusters, Cluster_indices indices)
/*
   1. This function tries to obtain an un-explored state from the unexplored
      stl container, depending on which algorithm is selected.
   2. Input arguments
   3. Output
	   Index to of the next unexplored state in the state list.
	   o.w. -1 if there are no more unexplored states.
   4. Written 5/31/07.
   5. Modified 1/15/19 to delete nodes from open_g1_values, open_g2_values, open_f1_values, open_f2_values.
   */
{
	static int depth = 0, max_depth;
	int index;
   //clock_t  start_time;
   heap_record item;
   
   //start_time = clock();
   switch(direction) {
      case 1:  // Forward Direction
	      switch(parameters->search_strategy) {
		      case 1:  // dfs
			      break;
		      case 2:  // breadth fs
			      break;
		      case 3:  // best fs
               if(forward_bfs_heap.empty()) return -1;
			      item = forward_bfs_heap.delete_min();
			      index = item.state_index;
               if((*states)[index].open1 == 1) {
                  open_g1_values[(*states)[index].g1 + (*states)[index].h1].delete_element((*states)[index].g1);
                  open_f1_values.delete_element((*states)[index].g1 + (*states)[index].h1);
                  open_g1_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g1);
               }
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			      return(index);
			      break;
            case 4:  // best first search using clusters
               if ((*clusters).empty(1, indices.g, indices.h1, indices.h2)) return(-1);
               index = clusters->pop(1, indices.g, indices.h1, indices.h2, states);
               if ((*states)[index].open1 == 1) {
                  open_g1_values[(*states)[index].g1 + (*states)[index].h1].delete_element((*states)[index].g1);
                  open_f1_values.delete_element((*states)[index].g1 + (*states)[index].h1);
                  open_g1_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g1);
               }
               return(index);
               break;
            default:
               fprintf(stderr, "This method has not been implemented yet\n\n");
               exit(1);
               break;
         }
         break;
      case 2:  // Reverse Direction
         switch (parameters->search_strategy) {
            case 1:  // dfs
               break;
            case 2:  // breadth fs
               break;
            case 3:  // best fs
               if (reverse_bfs_heap.empty()) return -1;
               item = reverse_bfs_heap.delete_min();
               index = item.state_index;
               if((*states)[index].open2 == 1) {
                  open_g2_values[(*states)[index].g2 + (*states)[index].h2].delete_element((*states)[index].g2);
                  open_f2_values.delete_element((*states)[index].g2 + (*states)[index].h2);
                  open_g2_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g2);
               }
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
               return(index);
               break;
            case 4:  // best first search using clusters
               if ((*clusters).empty(2, indices.g, indices.h1, indices.h2)) return(-1);
               index = clusters->pop(2, indices.g, indices.h1, indices.h2, states);
               if ((*states)[index].open2 == 1) {
                  open_g2_values[(*states)[index].g2 + (*states)[index].h2].delete_element((*states)[index].g2);
                  open_f2_values.delete_element((*states)[index].g2 + (*states)[index].h2);
                  open_g2_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g2);
               }
               return(index);
               break;
            default:
               fprintf(stderr, "This method has not been implemented yet\n\n");
               exit(1);
               break;
         }
         break;
   }

   //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;

	return 0;
}

//_________________________________________________________________________________________________

State_index get_bistate_from_cluster(Clusters *clusters, Cluster_indices indices, bistates_array *states, int direction)
/*
   1. This function gets an unexplored state from the cluster specified by indices.
   2. Input arguments
      a. clusters = stores the clusters of open nodes.
      b. indices = the indices of the cluster from which to get the node.
      c. states = store the states.
   3. Output
      index = index (in states) of the next unexplored state in the state list.
      o.w. -1 if there are no more unexplored states.
   4. Created 5/31/19 by modifying get_bistate from c:\sewell\research\pancake\pancake_code\meet_in_middle.cpp.
      a. Deleted the various cases that were in get_bistate.  This function is designed only to get a node from a cluster.
      b. Deleted search_parameters and search_info from the input parameters.
*/
{
   State_index index;

   if (direction == 1) {
      if ((*clusters).empty(1, indices.g, indices.h1, indices.h2)) return(-1);
      index = clusters->pop(1, indices.g, indices.h1, indices.h2, states);
      if ((*states)[index].open1 == 1) {
         open_g1_values[(*states)[index].g1 + (*states)[index].h1].delete_element((*states)[index].g1);
         open_f1_values.delete_element((*states)[index].g1 + (*states)[index].h1);
         open_g1_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g1);
      }
      return(index);
   } else {
      if ((*clusters).empty(1, indices.g, indices.h1, indices.h2)) return(-1);
      index = clusters->pop(2, indices.g, indices.h1, indices.h2, states);
      if ((*states)[index].open2 == 1) {
         open_g2_values[(*states)[index].g2 + (*states)[index].h2].delete_element((*states)[index].g2);
         open_f2_values.delete_element((*states)[index].g2 + (*states)[index].h2);
         open_g2_h1_h2_values[(*states)[index].h1][(*states)[index].h2].delete_element((*states)[index].g2);
      }
      return(index);
   }
}

//_________________________________________________________________________________________________

__int64 insert_bidirectional_unexplored(bistates_array *states, heap_record rec, int depth, int direction, searchparameters *parameters, Clusters *clusters, Cluster_indices indices)
{
   __int64     index_in_clusters = 0;

   switch(direction) {
      case 1:  // Forward Direction
	      switch(parameters->search_strategy){
            case 1:  // dfs
			      //forward_stack_dfs.push(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 2:  // breadth fs
			      //forward_queue_bfs.push(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 3:  // best fs
			      forward_bfs_heap.insert(rec);
			      break;
		      case 4:  // best first search using clusters
               index_in_clusters = clusters->insert(1, indices.g, indices.h1, indices.h2, rec.state_index, states);
			      break;
         }
         break;
      case 2:  // Reverse Direction
	      switch(parameters->search_strategy){
            case 1:  // dfs
			      //reverse_stack_dfs.push(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 2:  // breadth fs
			      //reverse_queue_bfs.push(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 3:  // best fs
			      reverse_bfs_heap.insert(rec);
			      break;
            case 4:  // best first search using clusters
               index_in_clusters = clusters->insert(2, indices.g, indices.h1, indices.h2, rec.state_index, states);
               break;
         }
         break;
   }
   return(index_in_clusters);
}
