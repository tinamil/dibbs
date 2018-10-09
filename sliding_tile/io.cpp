#include "main.h"

//_________________________________________________________________________________________________
/*
void read_data(char *f)
{
   FILE     *in;
   int      i, j;

   if (fopen_s(&in, f, "r") != 0) {
      fprintf(stderr,"Unable to open %s for input\n",f);
      exit(1);
   }

   fscanf_s(in,"%d", &n_sites);

   distances = new int*[n_sites + 1];
   flow = new int*[n_sites + 1];
   for(i = 1; i <= n_sites; i++) {
      distances[i] = new int[n_sites + 1];
      flow[i] = new int[n_sites + 1];
   }

   for(i = 1; i <= n_sites; i++) {
      for(j = 1; j <= n_sites; j++) {
         fscanf_s(in, "%d", &distances[i][j]);
      }
   }
   for(i = 1; i <= n_sites; i++) {
      for(j = 1; j <= n_sites; j++) {
         fscanf_s(in, "%d", &flow[i][j]);
      }
   }

   fclose(in);
}
*/
//_________________________________________________________________________________________________

void prn_data(unsigned char *tile_in_location)
{
   int      i;
 
   printf("\n");
   printf("%3d %3d %3d\n", n_tiles, n_rows, n_cols);
   for(i = 0; i <= n_tiles; i++) printf(" %2d", tile_in_location[i]);
   printf("\n");
   prn_configuration(tile_in_location);
}

//_________________________________________________________________________________________________

void prnvec(int n, int *vec)
{
   int      i;

   for (i = 1; i <= n; i++) {
      printf("%6d%s", vec[i], (i % 30) == 0 ? "\n":" ");
   }
   if ( (i % 30) != 0 )  printf("\n");
}

//_________________________________________________________________________________________________

void prn_double_vec(int n, double *vec)
{
   int      i;

   for (i = 1; i <= n; i++) {
      printf("%10.6f%s", vec[i], (i % 30) == 0 ? "\n":" ");
   }
   if ( (i % 30) != 0 )  printf("\n");
}

//_________________________________________________________________________________________________

void prnmatrix( int **matrix, int m, int n)
{
   int      i;

   for (i = 1; i <= m; i++) prnvec(n, matrix[i]);
}

//_________________________________________________________________________________________________

void prn_configuration(unsigned char *tile_in_location)
{
   int      cnt, i, j;

   cnt = 0;
   for(i = 1; i <= n_rows; i++) {
      for(j = 1; j <= n_cols; j++) {
         printf(" %2d", tile_in_location[cnt]);
         cnt++;
      }
      printf("\n");
   }
}

//_________________________________________________________________________________________________

void prn_distances()
{
   int      i, j;

   for(i = 0; i <= n_tiles; i++) {
      for(j = 0; j <= n_tiles; j++) {
         printf(" %2d", distances[i][j]);
      }
      printf("\n");
   }
}

//_________________________________________________________________________________________________

void prn_moves()
{
   int      i, j;

   for(i = 0; i <= n_tiles; i++) {
      for(j = 1; j <= moves[i][0]; j++) {
         printf(" %2d", moves[i][j]);
      }
      printf("\n");
   }
}

//_________________________________________________________________________________________________

void prn_dfs_subproblem(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB, unsigned char z)
{
   printf("z = %3d  LB = %3d  z + LB = %3d  empty_location = %2d  prev_location = %2d\n", z, LB, z + LB, empty_location, prev_location);  
   prn_configuration(tile_in_location);
}

//_________________________________________________________________________________________________

void prn_dfs_subproblem2(unsigned char *tile_in_location, unsigned char empty_location, unsigned char prev_location, unsigned char LB_to_goal, unsigned char LB_to_source, unsigned char z)
{
   printf("z = %3d  LB_to_goal = %3d  LB_to_source = %3d  z + LB_to_source = %3d  empty_location = %2d  prev_location = %2d\n", z, LB_to_goal, LB_to_source, z + LB_to_source, empty_location, prev_location);  
   prn_configuration(tile_in_location);
}

//_________________________________________________________________________________________________

void prn_forward_dfs_subproblem(unsigned char *tile_in_location, unsigned char bound1, unsigned char g1, unsigned char h1, unsigned char h2, unsigned char empty_location, unsigned char prev_location, int prn_config)
{
   printf("bound1 = %3d  g1 = %3d  h1 = %3d  h2 = %3d  f1 = %3d  f1_bar = %3d  empty_location = %2d  prev_location = %2d\n", bound1, g1, h1, h2, g1+h1, 2*g1+h1-h2, empty_location, prev_location);
   if(prn_config > 0) prn_configuration(tile_in_location);
}

//_________________________________________________________________________________________________

void prn_reverse_dfs_subproblem(unsigned char *tile_in_location, unsigned char bound2, unsigned char g2, unsigned char h1, unsigned char h2, unsigned char empty_location, unsigned char prev_location, int prn_config)
{
   printf("bound2 = %3d  g2 = %3d  h1 = %3d  h2 = %3d  f2 = %3d  f2_bar = %3d  empty_location = %2d  prev_location = %2d\n", bound2, g2, h1, h2, g2 + h2, 2 * g2 + h2 - h1, empty_location, prev_location);
   if (prn_config > 0) prn_configuration(tile_in_location);
}

//_________________________________________________________________________________________________

void prn_solution(unsigned char *tile_in_location, unsigned char *solution, unsigned char z, DPDB *DPDB)
{
   unsigned char  empty_location, *goal, h1, h2, new_location, *source, tile;  
   int      i;

   // Store the source (initial) configuration in source.

   source = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) source[i] = tile_in_location[i];

   // Load the goal configuration into goal.

   goal = new unsigned char[n_tiles + 1];
   for(i = 0; i <= n_tiles; i++) goal[i] = i;

   // Compute the Manhattan lower bounds.

   if (dpdb_lb > 0) {
      h1 = DPDB->compute_lb(tile_in_location);
   } else {
      h1 = compute_Manhattan_LB(tile_in_location);
   }
   h2 = compute_Manhattan_LB2(source, tile_in_location);

   printf("%3d %3d %3d    %3d %3d %3d %3d %3d\n", 0, h1, h1, z, h2, z + h2, 0 - h2, z - h1);
   prn_configuration(tile_in_location);   printf("\n");

   for(i = 1; i <= z; i++) {
      empty_location = solution[i-1];
      new_location = solution[i];
      tile = tile_in_location[new_location];
      tile_in_location[empty_location] = tile;
      tile_in_location[new_location] = 0;
      if (dpdb_lb > 0) {
         h1 = DPDB->compute_lb(tile_in_location);
      }
      else {
         h1 = compute_Manhattan_LB2(tile_in_location, goal);
      }
      h2 = compute_Manhattan_LB2(source, tile_in_location);
      //h1 = compute_Manhattan_LB2(tile_in_location, goal);
      printf("%3d %3d %3d    %3d %3d %3d %3d %3d %3d %3d\n", i, h1, i + h1, z - i, h2, z - i + h2, i - h2, z - i - h1, i + h1 + i - h2, z - i + h2 + z - i - h1);
      prn_configuration(tile_in_location); printf("\n");
   }

   delete [] goal;
   delete [] source;
}

//_________________________________________________________________________________________________

void prn_heap_info()
{
   int      i;

   for(i = 0; i <= UB; i++) {
      if(cbfs_heaps[i].empty()) {
         printf("%4d \n", i);
      } else {
         printf("%4d %6d %8.2f %8.2f\n", i, cbfs_heaps[i].n_of_items(), cbfs_heaps[i].get_min().key, cbfs_heaps[i].get_max().key);
      }
   }
}
