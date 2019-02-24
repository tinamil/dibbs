#include "hash_table.h"

// The following classes and functions implement a min-max heap, which can be used for implementing a
// distributed best bound first search.
// See Fundamentals of Data Structure by Horowitz and Sahni and Algorithms in
// C++ by Sedgwick for explanations of hashing.
/*
   1. HASH_SIZE = the length of the hash table.  It must be a prime number.
      Currently using linear probing, but if quadratic probing is used, then
      it must be a prime of the form 4j+3.  1019=4*254+3   4000039=4*1000009+3
   2. hash_record does not store a key.  Instead it stores state_index, which is the index (in states) of the corresponding state.
   3. The state_index of every entry in the hash table is initialized to -1.
      If -1 is a valid state_index, then some other value must be used.
   4. n = # of entries stored in the hash table.
   5. This implements a very simple hash table where the hash record contains only a state_index.
   6. Created 10/3/12 by modifying c:\sewell\research\statmatch\boss_cpp\hash_table.cpp, which was based on the hash tables in:
      a. c:\sewell\research\cdc\phase2\search06\search06.cpp (which was written as classes).
      b. c:\sewell\research\schedule\rtardy\bbr\memoroy.cpp.
*/

/*************************************************************************************************/

//_________________________________________________________________________________________________

int Hash_table::hash_configuration (unsigned char *tile_in_location)
/*
   1. This routine computes the hash value of a configuration.
   2. It uses a simple modular hash function.
   3. tile_in_location[i] = the tile that is in location i.
      The elements of tile_in_location are stored beginning in tile_in_location[0].
   4. hash_value[t][i] = random hash value associated with tile t assigned to location i: U[0,HASH_SIZE).
   5. The hash value is returned.
   6. Written 11/20/12.
*/
{
  int      i, index, t;

  index = 0;
  for (i = 0; i < N_LOCATIONS; i++)
  {
    t = tile_in_location[i];
    assert ( (0 <= t) && (t < N_LOCATIONS) );
    index = (index + hash_value[t][i]) % HASH_SIZE;
  }
  return (index);
}

//_________________________________________________________________________________________________

int Hash_table::update_hash_value (unsigned char *tile_in_location, unsigned char empty_location, unsigned char new_location, int index)
/*
   1. This routine updates the hash value of a configuration when the empty tile is moved from empty_location to new_location.
   2. tile_in_location[i] = the tile that is in location i.
      The elements of tile_in_location are stored beginning in tile_in_location[0].
   3. empty_location = location of the empty tile.
   4. new_location = new location of the empty tile.
   5. index = the hash value corresponding to the configuration represented by tile_in_location prior to the move.
   4. hash_value[t][i] = random hash value associated with tile t assigned to location i: U[0,HASH_SIZE).
   5. The hash value is returned.
   6. Written 11/20/12.
*/
{
  int      tile;

  assert ( (0 <= empty_location) && (empty_location < N_LOCATIONS) );
  assert ( (0 <= new_location) && (new_location < N_LOCATIONS) );
  assert (tile_in_location[empty_location] == 0);
  tile = tile_in_location[new_location];
  assert ( (0 <= tile) && (tile < N_LOCATIONS) );

  //index = (index - hash_value[0][empty_location]) % HASH_SIZE;    // Do not use the % operator because it may return a negative result.
  //index = (index - hash_value[tile][new_location]) % HASH_SIZE;
  if (index > hash_value[0][empty_location])
    index = index - hash_value[0][empty_location];
  else
    index = index - hash_value[0][empty_location] + HASH_SIZE;
  if (index > hash_value[tile][new_location])
    index = index - hash_value[tile][new_location];
  else
    index = index - hash_value[tile][new_location] + HASH_SIZE;
  index = (index + hash_value[0][new_location]) % HASH_SIZE;
  index = (index + hash_value[tile][empty_location]) % HASH_SIZE;

  return (index);
}

//_________________________________________________________________________________________________

int Hash_table::find (unsigned char *tile_in_location, int hash_value, int *hash_index, states_array *states)
/*
   1. This routine searches the hash table for a subproblem whose tile_in_location is the same as the input arguement.
   2. It uses linear probing.
   3. If the subproblem is found, the index where it is stored in the table is returned,
      o.w., -1 is returned.
   4. hash_value = the index in the hash table to begin searching for the subprolem.
   5. Created 10/4/12 by modifying find_key in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
   6. Modified 11/21/12 to return a status and the hash index.
      a. Status
         1 = the subproblem was found,
        -1 = the subproblem was not found.
      b. The index (in the hash table) is returned in hash_index.
         hash_index = the index where the subproblem is, if it was found.
                    = the index where the subproblem should be added to the hash table, if it was not found.
*/
{
  int      index, state_index;

  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_value) && (hash_value < HASH_SIZE) );
  index = hash_value;

  while ( (state_index = table[index].state_index) != -1)
  {
    if (memcmp ( (*states) [state_index].tile_in_location, tile_in_location, N_LOCATIONS) == 0)
    {
      *hash_index = index;
      return (1);
    }
    else
    {
      index = (index + 1) % HASH_SIZE;
    }
  }
  *hash_index = index;
  return (-1);
}

//_________________________________________________________________________________________________

int Hash_table::insert (int state_index, unsigned char *tile_in_location, int hash_index, states_array *states)
/*
   1. This routine uses the linear probing method to insert an entry into the hash table.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. -1 is returned if the hash table is full,
      -2 is returned if the key is already in the hash table,
      o.w. 1 is returned.
   4. Created 10/5/12 by modifying insert in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
*/
{
  int      index;

  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_index) && (hash_index < HASH_SIZE) );
  index = hash_index;

  while ( (state_index = table[index].state_index) != -1)
  {
    if (memcmp ( (*states) [state_index].tile_in_location, tile_in_location, N_LOCATIONS) == 0)
    {
      fprintf (stderr, "Already in the hash table\n");
      return (-2);
    }
    else
    {
      index = (index + 1) % HASH_SIZE;
    }
  }

  table[index].state_index = state_index;
  n++;

  return (1);
}

//_________________________________________________________________________________________________

int Hash_table::insert_at_index (int state_index, int hash_index)
/*
   1. This routine inserts an entry into the hash table at a particular index.
      It is designed to be used in a situation where the correct index in the hash table has already
      been found (by the find function).  This avoids a second search through the hash table to
      find the correct location.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. -1 is returned if the hash table is full,
      -2 is returned if the hash table already has an entry in the given index,
      o.w. 1 is returned.
   4. Written 10/5/12.
*/
{
  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_index) && (hash_index < HASH_SIZE) );
  if (table[hash_index].state_index != -1)
  {
    fprintf (stderr, "The hash table already contains an entry in this location\n");
    return (-2);
  }

  table[hash_index].state_index = state_index;
  n++;

  return (1);
}

//_________________________________________________________________________________________________

int Hash_table::replace_at_index (int state_index, int hash_index)
/*
   1. This routine replaces an entry into the hash table at a particular index.
      It is designed to be used in a situation where the correct index in the hash table has already
      been found (by the find function).  This avoids a second search through the hash table to
      find the correct location.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. The program is aborted if the hash table is full,
      -2 is returned if there is no entry in the hash table in the given index,
      o.w. 1 is returned.
   4. Written 12/14/12.
*/
{
  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_index) && (hash_index < HASH_SIZE) );
  if (table[hash_index].state_index == -1)
  {
    fprintf (stderr, "The hash table does not contain an entry in this location\n");
    return (-2);
  }

  table[hash_index].state_index = state_index;

  return (1);
}

//_________________________________________________________________________________________________

void Hash_table::print_hash_values()
{
  printf ("\n");
  for (int t = 0; t < N_LOCATIONS; t++)
  {
    for (int i = 0; i < N_LOCATIONS; i++)
    {
      printf ("%9d ", hash_value[t][i]);
    }
    printf ("\n");
  }
  printf ("\n");
}

//_________________________________________________________________________________________________

int Hash_table::find_bistate (unsigned char *tile_in_location, int hash_value, int *hash_index, bistates_array *bistates)
/*
   1. This routine searches the hash table for a subproblem whose tile_in_location is the same as the input arguement.
   2. It uses linear probing.
   3. If the subproblem is found, the index where it is stored in the table is returned,
      o.w., -1 is returned.
   4. hash_value = the index in the hash table to begin searching for the subprolem.
   5. Created 10/4/12 by modifying find_key in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
   6. Modified 11/21/12 to return a status and the hash index.
      a. Status
         1 = the subproblem was found,
        -1 = the subproblem was not found.
      b. The index (in the hash table) is returned in hash_index.
         hash_index = the index where the subproblem is, if it was found.
                    = the index where the subproblem should be added to the hash table, if it was not found.
   7. Created 2/17/15 by modifying find.
*/
{
  int      index, state_index;

  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_value) && (hash_value < HASH_SIZE) );
  index = hash_value;

  while ( (state_index = table[index].state_index) != -1)
  {
    if (memcmp ( (*bistates) [state_index].tile_in_location, tile_in_location, N_LOCATIONS) == 0)
    {
      *hash_index = index;
      return (1);
    }
    else
    {
      index = (index + 1) % HASH_SIZE;
    }
  }
  *hash_index = index;
  return (-1);
}

//_________________________________________________________________________________________________

int Hash_table::insert_bistate (int state_index, unsigned char *tile_in_location, int hash_index, bistates_array *bistates)
/*
   1. This routine uses the linear probing method to insert an entry into the hash table.
   2. hash_index = the index in the hash table to begin attempting to insert the key.
   3. -1 is returned if the hash table is full,
      -2 is returned if the key is already in the hash table,
      o.w. 1 is returned.
   4. Created 10/5/12 by modifying insert in c:\sewell\research\statmatch\boss_cpp\hash_table.cpp.
   5. Created 2/17/15 by modifying insert.
*/
{
  int      index;

  if (n >= HASH_SIZE)
  {
    fprintf (stderr, "Hash table is full\n");
    exit (1);
  }

  assert ( (0 <= hash_index) && (hash_index < HASH_SIZE) );
  index = hash_index;

  while ( (state_index = table[index].state_index) != -1)
  {
    if (memcmp ( (*bistates) [state_index].tile_in_location, tile_in_location, N_LOCATIONS) == 0)
    {
      fprintf (stderr, "Already in the hash table\n");
      return (-2);
    }
    else
    {
      index = (index + 1) % HASH_SIZE;
    }
  }

  table[index].state_index = state_index;
  n++;

  return (1);
}

