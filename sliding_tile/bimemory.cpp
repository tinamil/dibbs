#include "main.h"

/*****************************************************************************/

/*
   1. The following functions manage the list of states for the bidirectional search for the 15-puzzle.
   2. Conceptually, a subproblem consists of a configuration of the tiles plus the moves that were made to reach this configuration.  
      To reduce memory requirements, do not store the moves that were made for each subproblem.  
      Instead, use pointers to reconstruct the moves.  A subproblem consists of:
      a. g1 = number of moves that have been made so far in the forward direction.
      b. h1 = lower bound on the number of moves needed to reach the goal postion.
      c. open1 = 2 if this subproblem has not yet been generated in the forward direction
               = 1 if this subproblem is open in the forward direction
               = 0 if this subproblem closed in the forward direction
      d. empty_location = location of the empty tile.
      e. prev_location1 = location of the empty tile in the parent of this subproblem in the forward direction.
      f. parent1 = index of the parent subproblem in the forward direction.
      g. g2 = number of moves that have been made so far in the reverse direction.
      h. h2 = lower bound on the number of moves needed to reach the source postion.
      i. open2 = 2 if this subproblem has not yet been generated in the reverse direction
               = 1 if this subproblem is open in the reverse direction
               = 0 if this subproblem closed in the reverse direction
      j. prev_location2 = location of the empty tile in the parent of this subproblem in the reverse direction
      k. parent2 = index of the parent subproblem in the reverse direction.
      l. tile_in_location[i] = the tile that is in location i.
   3. Created 2/17/17 by modifying c:\sewell\research\15puzzle\15puzzle_code\memory.cpp.
      a. Eliminated memory dominance.
      b. Eliminated the ability to replace subproblems.
*/

//_________________________________________________________________________________________________

int search_bimemory(bistate *new_bistate, bistates_array *bistates, Hash_table *hash_table, int hash_value)
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

   status = hash_table->find_bistate(new_bistate->tile_in_location, hash_value, &hash_index, bistates);
   if(status == -1) 
      return(-1);
   else
      return(hash_index);
}

//_________________________________________________________________________________________________

pair<int, int> find_or_insert(bistate *new_state, double best, int depth, int direction, bistates_array *bistates, searchparameters *parameters, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_value)
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
   6.  Created 6/23/18 by modifying c:\sewell\research\pancake\pancake_code\bimemory.cpp.
*/
{
   int      hash_index, status, state_index;
   bistate  *old_state;
   heap_record record;

   status = hash_table->find_bistate(new_state->tile_in_location, hash_value, &hash_index, bistates);
   if (status == 1) {

      // The subproblem has been found.  Determine if the new subproblem dominates the old subproblem.

      state_index = (*hash_table)[hash_index].state_index;
      old_state = &(*bistates)[state_index];
      if (direction == 1) {
         if (new_state->g1 < old_state->g1) {
            assert(old_state->open1 != 0); assert(new_state->h1 == old_state->h1);  assert(new_state->h2 == old_state->h2);
            old_state->g1 = new_state->g1;
            old_state->open1 = 1;
            old_state->parent1 = new_state->parent1;
            record.key = best;
            record.state_index = state_index;
            insert_bidirectional_unexplored(record, depth, old_state->h1, direction, cbfs_stacks);     // Note: If this subproblem was already open in the forward direction, then this subproblem will be in the heaps twice.
         }
      } else {
         if (new_state->g2 < old_state->g2) {
            assert(old_state->open2 != 0); assert(new_state->h1 == old_state->h1);  assert(new_state->h2 == old_state->h2);
            old_state->g2 = new_state->g2;
            old_state->open2 = 1;
            old_state->parent2 = new_state->parent2;
            record.key = best;
            record.state_index = state_index;
            insert_bidirectional_unexplored(record, depth, old_state->h2, direction, cbfs_stacks);     // Note: If this subproblem was already open in the forward direction, then this subproblem will be in the heaps twice.
         }
      }
      return(make_pair(status, state_index));
   } else {

      // The subproblem was not found, so insert it into memory.

      state_index = add_to_bimemory(new_state, best, depth, direction, bistates, info, cbfs_stacks, hash_table, hash_index);
      return(make_pair(status, state_index));
   }
}

//_________________________________________________________________________________________________

pair<int, int> find_or_insert2(bistate *new_state, int direction, bistates_array *bistates, Hash_table *hash_table, int hash_value)
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
      b. direction = 1 if currently searching in the forward direction
                   = 2 if currently searching in the reverse direction.
      c. bistates is the list of states in memory.
      d. hash_table is the hash table used to find a bistate in memory.
      e. hash_value = the hash value for new_state.  This is the index in the hash table to begin searching for new_bistate.
   5. Output Variables
      a. status =  1 if this subproblem was found
                = -1 if this subproblem was not found

      b. state_index = index (in states) of the state
                     = -1 unable to add to memory (states, hash table, or heap).
   6.  Created 7/14/18 by modifying find_or_insert from this file.
      a. This version was designed for ID_DIBBS.
      b. It does not add the state to a heap, so best, info, parameters, and cbfs_stacks are not needed.
*/
{
   int      hash_index, index, status, state_index;
   bistate  *old_state;

   status = hash_table->find_bistate(new_state->tile_in_location, hash_value, &hash_index, bistates);
   if (status == 1) {

      // The subproblem has been found.  Determine if the new subproblem dominates the old subproblem.

      state_index = (*hash_table)[hash_index].state_index;
      old_state = &(*bistates)[state_index];
      if (direction == 1) {
         if (new_state->g1 < old_state->g1) {
            assert(old_state->open1 != 0); assert(new_state->h1 == old_state->h1);  assert(new_state->h2 == old_state->h2);
            old_state->g1 = new_state->g1;
            old_state->open1 = 1;
            old_state->parent1 = new_state->parent1;
          }
      } else {
         if (new_state->g2 < old_state->g2) {
            assert(old_state->open2 != 0); assert(new_state->h1 == old_state->h1);  assert(new_state->h2 == old_state->h2);
            old_state->g2 = new_state->g2;
            old_state->open2 = 1;
            old_state->parent2 = new_state->parent2;
         }
      }
      return(make_pair(status, state_index));
   } else {

      // The subproblem was not found, so insert it into memory.

      if (bistates->is_not_full()) {
         state_index = bistates->add_bistate(new_state);
         hash_table->insert_at_index(state_index, hash_index);
      } else {
         state_index = -1;
      }
      return(make_pair(status, state_index));
   }
}

//_________________________________________________________________________________________________

int add_to_bimemory(bistate *bistate, double best, int depth, int direction, bistates_array *bistates, searchinfo *info, min_max_stacks *cbfs_stacks, Hash_table *hash_table, int hash_index)
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
   int         index;
   //clock_t  start_time;
   hash_record hrecord;
   heap_record record;
   best_index  best_index;

   //start_time = clock();
   if(direction == 1) {
      LB = bistate->g1 + bistate->h1;
   } else {
      LB = bistate->g2 + bistate->h2;
   }

   switch(algorithm) {		
		case 1:  // dfs
			break;
		case 2:  // breadth fs
			break;
		case 3:  // best fs
         // Do not replace states when states is full.
         if(bistates->is_not_full() && (((direction == 1) && forward_bfs_heap.is_not_full()) || ((direction == 2) && reverse_bfs_heap.is_not_full())) ) {
            index = bistates->add_bistate(bistate);
            hash_table->insert_at_index(index, hash_index);
            record.key = best;
            record.state_index = index;
            insert_bidirectional_unexplored(record, depth, LB, direction, cbfs_stacks);
         } else {
            info->optimal = 0;
            index = -1;
         }
			break;
		case 4:  //cbfs using min-max heaps
			break;
		case 5:  //cbfs using min-max stacks
			break;
		case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
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

int get_bistate(int direction, searchinfo *info, min_max_stacks *cbfs_stacks)
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
   switch(direction) {
      case 1:  // Forward Direction
	      switch(algorithm) {
		      case 1:  // dfs
			      break;
		      case 2:  // breadth fs
			      break;
		      case 3:  // best fs
               if(forward_bfs_heap.empty()) return -1;
			      item = forward_bfs_heap.delete_min();
			      index = item.state_index;
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
			      return(index);
			      break;
		      case 4:  //cbfs using min-max heaps
			      break;
		      case 5:  //cbfs using min-max stacks
			      break;
		      case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
			      break;
            default:
               fprintf(stderr, "This method has not been implemented yet\n\n");
               exit(1);
               break;
         }
      case 2:  // Reverse Direction
         switch (algorithm) {
            case 1:  // dfs
               break;
            case 2:  // breadth fs
               break;
            case 3:  // best fs
               if (reverse_bfs_heap.empty()) return -1;
               item = reverse_bfs_heap.delete_min();
               index = item.state_index;
               //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;
               return(index);
               break;
            case 4:  //cbfs using min-max heaps
               break;
            case 5:  //cbfs using min-max stacks
               break;
            case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
               break;
            default:
               fprintf(stderr, "This method has not been implemented yet\n\n");
               exit(1);
               break;
         }
   }

   //info->states_cpu += (double) (clock() - start_time) / CLOCKS_PER_SEC;

	return 0;
}

//_________________________________________________________________________________________________

void insert_bidirectional_unexplored(heap_record rec, int depth, unsigned char LB, int direction, min_max_stacks *cbfs_stacks)
{
   switch(direction) {
      case 1:  // Forward Direction
	      switch(algorithm){
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
		      case 4:  //cbfs using min-max heaps
               //forward_cbfs_heaps[depth].insert(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 5:  //cbfs using min-max stacks
               //forward_cbfs_stacks->insert(depth, (int) rec.key, rec.state_index);
               //cbfs_stacks->check_stacks();
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
               //forward_cbfs_heaps[LB].insert(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
         }
         break;
      case 2:  // Reverse Direction
	      switch(algorithm){
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
		      case 4:  //cbfs using min-max heaps
               //reverse_cbfs_heaps[depth].insert(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 5:  //cbfs using min-max stacks
               //reverse_cbfs_stacks->insert(depth, (int) rec.key, rec.state_index);
               //cbfs_stacks->check_stacks();
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
		      case 6:  //CBFS: Cylce through LB instead of depth.  Use min-max heaps.
               //reverse_cbfs_heaps[LB].insert(rec);
               fprintf(stderr, "This method has not been implemented yet\n\n");
			      break;
         }
         break;
   }
}
