#include "main.h"

/*****************************************************************************/

/*
   1. The following functions manage the list of states for the 15-puzzle.
   2. Conceptually, a subproblem consists of a configuration of the tiles plus the moves that were made to reach this configuration.  
      To reduce memory requirements, do not store the moves that were made for each subproblem.  
      Instead, use pointers to reconstruct the moves.  A subproblem consists of:
      a. z = objective function value = number of moves that have been made so far.
      b. LB = lower bound on the number of moves needed to reach the goal postion.
      c. empty_location = location of the empty tile.
      d. prev_location = location of the empty tile in the parent of this subproblem.
      e. parent = index of the parent subproblem.
      f. tile_in_location[i] = the tile that is in location i.
   3. Created 12/2/11 by modifying c:\sewell\research\facility\facility_cbfns\memory.cpp.
      a. I created a class to store the states.
*/

/*
   Memory Dominance
   1. I added a hash table to implement memory based dominance.
   2. state should contain a field that indicates whether it is open or closed.
   3. Suppose subproblem A has just been generated and it dominates subproblem B which was generated previously.
   4. get_state should skip closed states.
      a. If B is closed, then B is not in the heaps (by which I mean that the heaps do not contain an item that points to B).
      b. If B is open, then B is in the heaps.  There is no easy way to remove B from the heaps.  Close B.  get_state should skip B
         when it is encountered in the heaps.
   5. Replacing subproblems.
      a. For some types of problems, it may be possible for A to replace B in the list of states.  This would reduce memory requirements.
      b. Let AB be subproblem A stored in the old location of B.  AB should be set to open.
      c. We cannot remove B from the heaps nor modify its best measure.  AB needs to be added to the heaps.  AB may be added to a
         different heap than B, resulting in two items in the heaps pointing to the same location in states.  It is possible that
         B will be selected from the heaps before A (this shouldn't happen if A and B are in the same heap because A should have
         a better best measure).  Thus, the CBFS strategy may not be strictly followed.  This should not cause incorrect results, though.
   6. 10/6/12.
*/
//_________________________________________________________________________________________________

int search_memory(state *new_state, states_array *states, Hash_table *hash_table, int hash_value, int *hash_index)
/*
   1. This routine attempts to find state in memory.
   2. If the state is found in memory (call it mem_state), then the following dominance test is applied.
      a. If new_state.z >= mem_state.z, then mem_state dominates new_state.
      b. O.w., new_state dominates mem_state.
   3. Input Variables
      a. new_state is the new state.
      b. states is the list of states in memory.
      c. hash_table is the hash table used to find a state in memory.
      d. hash_value = the hash value for new_state.  This is the index in the hash table to begin searching for new_state.
   4. Output Variables
      a. -1 is returned if the new state is dominated by another state in memory,
          0 is returned if the new state is not found in memory,
          1 is returned if the new state dominates another state in memory.
      b. If new_state dominates a state in memory, then the index (in hash_table) of the dominated state is returned in hash_index.
   3. Written 11/20/12.
*/
{
   int      status;

   status = hash_table->find(new_state->tile_in_location, hash_value, hash_index, states);
   if(status == -1) return(0);

   if(new_state->z >= (*states)[(*hash_table)[*hash_index].state_index].z)
      return(-1);
   else
      return(1);
}
//_________________________________________________________________________________________________

int add_to_memory(state *state, double best, int depth, states_array *states, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_index, int dominance)
/*
   1. This routine attempts to add a new state to memory.
      a. If there is sufficient memory available in states and the appropriate heap, then the
         state is added to memory.
      b. If there is insufficient memory in either states or the appropriate heap, then it will
         check if this state should replace an existing state in memory.
   2. -1 is returned if the new state is not added to memory,
      o.w. the index (in states) is returned.
   3. Written 5/31/07.
*/
{
   unsigned char  LB;
   int         index, replace;
   //clock_t  start_time;
   hash_record hrecord;
   heap_record record;
   best_index  best_index;

   //start_time = clock();
   LB = state->z + state->LB;

   switch(algorithm) {		
		case 1:  // dfs
         // Do not replace states in DFS when states is full.
         // A recursive DFS would eliminate states after they have been processed, thus it should not run out of memory (except for the hash table).
         // But under the methods used here, it is not easy to know when all the children of a subproblem have been processed,
         // so we cannot delete the subproblem from states.
         if(states->is_not_full()) {
            index = states->add_state(state);
            record.key = best;
            record.state_index = index;
            insert_unexplored(record, depth, LB, cbfs_stacks);
         } else {
            info->optimal = 0;
            index = -1;
         }
			break;
		case 2:  // breadth fs
         // Do not replace states in breadth first search when states is full.
			if(states->is_not_full()) {
            index = states->add_state(state);
            record.key = best;
            record.state_index = index;
            insert_unexplored(record, depth, LB, cbfs_stacks);
         } else {
            info->optimal = 0;
            index = -1;
         }
			break;
		case 3:  // best fs
			if(states->is_not_full() && bfs_heap.is_not_full()) {
            index = states->add_state(state);
            record.key = best;
            record.state_index = index;
            insert_unexplored(record, depth, LB, cbfs_stacks);
         } else {
            // Either states is full or the bfs_heap is full.  In either case, replace the minimum element in bfs_heap
            // and replace the corresponding subproblem in states.

            info->optimal = 0;
            record = bfs_heap.get_max(); 
            if(record.key == -1) {
               // There is no state available at this depth to replace.  This implies that states is full.
               // With the current data structures, there is no way to store another state at this depth.
               // It is possible that there exist states at other levels which could lead to an improved solution.

               //info->optimal = -1;
               index = -1;
               //fprintf (stderr, "No state is available to be replaced\n\n");
               //exit(1);
            } else {
               if(best < record.key) {
                  index = record.state_index;
                  states->replace_state(index, state);
                  record.key = best;
                  record.state_index = index;
                  bfs_heap.replace_max(record);
               } else {
                  index = -1;
               }
            }
         }
			break;
		case 4:  //cbfs using min-max heaps
         /* replace = -1 if state needs to replace an existing state (due to states or heaps full), 
                         but either the correct heap is empty or state is not better than the worse state in the heap.
                    =  0 if there is room in states and the correct heap, and state does not dominate an existing state.
                    =  1 if state dominates an existing state and there is room in the correct heap.
                    =  2 if state replaces a state in states and the correct heap (including the case where state dominates existing state but the heap is full). */

         replace = 0;
         if(dominance == 1) {
            index = (*hash_table)[hash_index].state_index;
            replace = 1;
         } else {
            if(states->is_full()) replace = 2;
         }
         if(cbfs_heaps[depth].is_full() || (replace >= 2)) {
            replace = 2;
            info->optimal = 0;
            record = cbfs_heaps[depth].get_max(); 
            if(record.key == -1) {
               // There is no state available at this depth to replace.
               // With the current data structures, there is no way to store another state at this depth.
               // It is possible that there exist states at other levels which could lead to an improved solution.

               replace = -1;
            } else {
               if(best < record.key) {
                  index = record.state_index;
               } else {
                  replace = -1;
               }
            }
         }

         switch(replace) {
		      case -1:    // Unable to add the state or replace an exsiting one.
               index = -1; 
               break;
            case  0:    // Add the state to states, heaps, and hash table.
               index = states->add_state(state);
               record.key = best;
               record.state_index = index;
               insert_unexplored(record, depth, LB, cbfs_stacks);
               hash_table->insert_at_index(index, hash_index);
               break;
            case  1:    // Replace in states and add to heaps and hash table.
               states->replace_state(index, state);
               record.key = best;
               record.state_index = index;
               insert_unexplored(record, depth, LB, cbfs_stacks);
               if(dominance == 1) {
                  hash_table->replace_at_index(index, hash_index);
               } else {
                  hash_table->insert_at_index(index, hash_index);
               }
               break;
            case  2:    // Replace in states and heaps and add to hash table.
               states->replace_state(index, state);
               record.key = best;
               record.state_index = index;
               cbfs_heaps[depth].replace_max(record);
               if(dominance == 1) {
                  hash_table->replace_at_index(index, hash_index);
               } else {
                  hash_table->insert_at_index(index, hash_index);
               }
               break;
            default: fprintf(stderr,"Illegal value of replace in add_to_memory\n"); exit(1); break;
         }
			break;
		case 5:  //cbfs using min-max stacks
         if(states->is_not_full()) {            // The min-max stacks expand as needed, so they will not be full, unless we have run out of memory.
            index = states->add_state(state);
            record.key = best;
            record.state_index = index;
            insert_unexplored(record, depth, LB, cbfs_stacks);
            //cbfs_stacks->check_stacks();
         } else {
            // states is full.  The min-max stacks expand as needed.  Replace the minimum element in dbfs_stacks[depth]
            // and replace the corresponding subproblem in states.

            info->optimal = 0;
            best_index = cbfs_stacks->get_max(depth); 
            if(best_index.state_index == -1) {
               // There is no state available at this depth to replace.  This implies that states is full.
               // With the current data structures, there is no way to store another state at this depth.
               // It is possible that there exist states at other levels which could lead to an improved solution.

               //info->optimal = -1;
               index = -1;
               //fprintf (stderr, "No state is available to be replaced\n\n");
               //exit(1);
            } else {
               if(best < best_index.best) {
                  index = best_index.state_index;
                  states->replace_state(index, state);
                  cbfs_stacks->replace_max(depth, (int) best, index);
                  //cbfs_stacks->check_stacks();
               } else {
                  index = -1;
               }
            }
         }
			break;
		case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
         if(states->is_not_full() && cbfs_heaps[LB].is_not_full()) {
            index = states->add_state(state);
            record.key = best;
            record.state_index = index;
            insert_unexplored(record, depth, LB, cbfs_stacks);
         } else {
            // Either states is full or the dbfs_heaps[LB] is full.  In either case, replace the minimum element in dbfs_heaps[LB]
            // and replace the corresponding subproblem in states.

            info->optimal = 0;
            record = cbfs_heaps[LB].get_max(); 
            if(record.key == -1) {
               // There is no state available with LB to replace.  This implies that states is full.
               // With the current data structures, there is no way to store another state with this LB.
               // It is possible that there exist states at other levels which could lead to an improved solution.

               //info->optimal = -1;
               index = -1;
               //fprintf (stderr, "No state is available to be replaced\n\n");
               //exit(1);
            } else {
               if(best < record.key) {
                  index = record.state_index;
                  states->replace_state(index, state);
                  record.key = best;
                  record.state_index = index;
                  cbfs_heaps[LB].replace_max(record);
                  //cbfs_heaps[state.depth].check_heap(1, &min_key, &max_key);
               } else {
                  index = -1;
               }
            }
         }
			break;

   }

   //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;

   return(index);
}

//_________________________________________________________________________________________________

int get_state(searchinfo *info, min_max_stacks *cbfs_stacks)
/*
   1. This function tries to obtain an un-explored state from the unexplored
      stl container, depending on which algorithm is selected.
   2. Input arguments
   3. Output
	   Index to of the next unexplored state in the state list.
	   o.w. -1 if there are no more unexplored states.
   4. Written 5/31/07.
*/
{
	static int depth = 0, LB = 0, max_depth;
	int index;
   //clock_t  start_time;
   heap_record item;
   //best_index  best_index;
   
   //start_time = clock();

	switch(algorithm){
		
		case 1:  // dfs
			if(stack_dfs.empty()) return -1;
			index = (stack_dfs.top()).state_index;
			stack_dfs.pop();
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
         return(index);
			break;
		case 2:  // breadth fs
			if(queue_bfs.empty()) return -1;
			index = (queue_bfs.front()).state_index;
			queue_bfs.pop();
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
         return(index);
			break;
		case 3:  // best fs
         if(bfs_heap.empty()) return -1;
			item = bfs_heap.delete_min();
			index = item.state_index;
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			return(index);
			break;
		case 4:  //cbfs using min-max heaps
			
         max_depth = UB;
			for(int i= 0; i <= max_depth; i++){
				depth = (depth + 1) % (max_depth + 1);
				if(!cbfs_heaps[depth].empty()) {
					item = cbfs_heaps[depth].delete_min();
					index = item.state_index;
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
					return(index);
				}
			}
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			return -1; //went through all level and everything is empty
			break;
		case 5:  //cbfs using min-max stacks
			/*
         max_depth = UB;
			for(int i= 0; i <= max_depth; i++){
				depth = (depth + 1) % (max_depth + 1);
            //info->cnt++;
            if(!cbfs_stacks->empty(depth)) {
               best_index = cbfs_stacks->delete_min(depth);
					index = best_index.state_index;
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
					return(index);
				}
			}
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			return -1; //went through all level and everything is empty
         */
         index = cbfs_stacks->get_state(UB);
         return(index);
			break;
		case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
			
         max_depth = UB;
			for(int i= 0; i <= max_depth; i++){
				LB = (LB + 1) % (max_depth + 1);
				if(!cbfs_heaps[LB].empty()) {
					item = cbfs_heaps[LB].delete_min();
					index = item.state_index;
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
					return(index);
				}
			}
         //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			return -1; //went through all level and everything is empty
			break;

   }

   //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;

	return 0;
}

//_________________________________________________________________________________________________

void insert_unexplored(heap_record rec, int depth, unsigned char LB, min_max_stacks *cbfs_stacks)
{
	switch(algorithm){
      case 1:  // dfs
			stack_dfs.push(rec);
			break;
		case 2:  // breadth fs
			queue_bfs.push(rec);
			break;
		case 3:  // best fs
			bfs_heap.insert(rec);
			break;
		case 4:  //cbfs using min-max heaps
         cbfs_heaps[depth].insert(rec);
			break;
		case 5:  //cbfs using min-max stacks
         cbfs_stacks->insert(depth, (int) rec.key, rec.state_index);
         //cbfs_stacks->check_stacks();
			break;
		case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
         cbfs_heaps[LB].insert(rec);
			break;
   }
}
