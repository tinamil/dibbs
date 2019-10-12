/*
   1. This project was created on 7/17/17.
      a. I copied the following files from c:\sewell\research\15puzzle\15puzzle_code2:
         heap_record.h, main.h, io.cpp, main.cpp, memory.cpp.
         These files (together with several files that were not needed for this project) implemented a CBFS for the sliding tile puzzle.
   2. This project implements various branch and bound algorithms for the Pancake problem.
*/

#include "Pancake.h"

/*************************************************************************************************/

uint8_t Pancake::gap_lb(Direction dir) const
/*
   1. This function computes the GAP LB for a sequence of pancakes.
   2. Input Variables
      a. source = the order of the pancakes.
      b. direction = = 1(2) for forward (reverse) search.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
      a. LB = GAP LB for the sequence of pancakes is returned.
   5. Created 7/17/17 by modifying c:\sewell\research\pancake\matlab\gap_lb.m.
   6. Modified 8/5/17 to compute the GAP-X LB.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
*/
{
  unsigned char  LB;
  int            i;

  LB = 0;
  if (GAPX == -1) return(LB);

  if (dir == Direction::forward) {
    for (i = 2; i <= NUM_PANCAKES; i++) {
      if ((source[i] <= GAPX) || (source[i - 1] <= GAPX)) continue;
      if (abs(source[i] - source[i - 1]) > 1) LB = LB + 1;
    }
    if ((abs(NUM_PANCAKES + 1 - source[NUM_PANCAKES]) > 1) && (source[NUM_PANCAKES] > GAPX)) LB = LB + 1;
  }
  else if (DUAL_SOURCE() != nullptr) {
    for (i = 2; i <= NUM_PANCAKES; i++) {
      if ((source[i] <= GAPX) || (source[i - 1] <= GAPX)) continue;
      if (abs(DUAL_SOURCE()[source[i]] - DUAL_SOURCE()[source[i - 1]]) > 1) LB = LB + 1;
    }
    if ((abs(NUM_PANCAKES + 1 - DUAL_SOURCE()[source[NUM_PANCAKES]]) > 1) && (source[NUM_PANCAKES] > GAPX)) LB = LB + 1;
  }

  return(LB);
}

//_________________________________________________________________________________________________

uint8_t Pancake::update_gap_lb(Direction dir, int i, uint8_t LB) const
/*
   1. This function updates the GAP lower bound when a flip is made at position i.
   2. Input Variables
      a. source = the order of the pancakes.
      b. direction = = 1(2) for forward (reverse) search.
      c. i = position where the flip is to be made.  I.e., the new sequence is obtained by reversing the sequence
             of pancakes in positions 1 through i.
      d. LB = the GAP LB for source before the flip.
   3. Global Variables
      a. n = number of pancakes.
      b. inv_source = inverse of source.
      c. source[i] = the number of the pancake that is position i (i.e., order of the pancakes).
   4. Output Variables
      a. LB = GAP LB for the sequence after the flip has been made is returned.
   5. Created 7/17/17 by modifying c:\sewell\research\pancake\matlab\update_gap_lb.m.
   6. Modified 8/7/17 to compute the GAP-X LB.  See "Bidirectional Search That Is Guaranteed to Meet in the Middle."
   */
{
  int            inv_p1, inv_pi, inv_pi1, p1, pi, pi1;

  if (GAPX == -1) return(0);

  assert((1 <= i) && (i <= NUM_PANCAKES));

  if (dir == Direction::forward) {
    p1 = source[1];
    pi = source[i];
    if (i < NUM_PANCAKES)
      pi1 = source[i + 1];
    else
      pi1 = NUM_PANCAKES + 1;

    if ((pi <= GAPX) || (pi1 <= GAPX) || (abs(pi1 - pi) <= 1)) LB = LB + 1;
    if ((p1 <= GAPX) || (pi1 <= GAPX) || (abs(pi1 - p1) <= 1)) LB = LB - 1;
  }
  else if (DUAL_SOURCE() != nullptr) {
    p1 = source[1];
    pi = source[i];
    inv_p1 = DUAL_SOURCE()[p1];
    inv_pi = DUAL_SOURCE()[pi];
    if (i < NUM_PANCAKES) {
      pi1 = source[i + 1];
      inv_pi1 = DUAL_SOURCE()[source[i + 1]];
    }
    else {
      pi1 = NUM_PANCAKES + 1;
      inv_pi1 = NUM_PANCAKES + 1;
    }
    if ((pi <= GAPX) || (pi1 <= GAPX) || (abs(inv_pi1 - inv_pi) <= 1)) LB = LB + 1;
    if ((p1 <= GAPX) || (pi1 <= GAPX) || (abs(inv_pi1 - inv_p1) <= 1)) LB = LB - 1;
  }

  return(LB);
}

/*************************************************************************************************/

//_________________________________________________________________________________________________

int Pancake::check_inputs() const
/*
   1. This routine performs some simple checks on seq.
      If an error is found, 0 is returned, otherwise 1 is returned.
   2. Written 7/18/17.
*/
{
  int* used = new int[NUM_PANCAKES + 1];
  for (int i = 0; i <= NUM_PANCAKES; i++) used[i] = 0;

  // Check that all the indices in seq are legitimate and that there are no duplicates.

  for (int i = 1; i <= NUM_PANCAKES; i++) {
    if ((source[i] < 1) || (source[i] > NUM_PANCAKES)) {
      fprintf(stderr, "illegal number in seq\n");
      delete[] used;
      return(0);
    }
    if (used[source[i]]) {
      fprintf(stderr, "seq contains the same number twice\n");
      delete[] used;
      return(0);
    }
    used[source[i]] = 1;
  }

  delete[] used;

  return(1);
}

