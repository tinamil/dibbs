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

void prn_data(unsigned char *seq, int n_seq)
{
   int      i;
 
   printf("\n");
   printf("%3d\n", n_seq);
   for(i = 1; i <= n_seq; i++) printf(" %2d", seq[i]);
   printf("\n");
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

void prn_sequence(unsigned char *seq, int n_seq)
{
   int      i;

   for(i = 1; i <= n_seq; i++) printf(" %2d", seq[i]);
   printf("\n");
}

//_________________________________________________________________________________________________

void prn_sequence2(unsigned char *seq, int n_seq)
{
   int      i;

   printf("{%2d, ", n_seq);
	for(i = 1; i <= n_seq; i++) {
		printf(" %2d", seq[i]);
		if(i < n_seq) printf(",");
	}
	printf("}\n");
}

//_________________________________________________________________________________________________

void prn_dfs_subproblem(unsigned char bound, unsigned char g1, unsigned char h1, unsigned char *seq)
{
   printf("bound = %2d  g1 = %2d  h1 = %2d  f1 = %2d: ", bound, g1, h1, g1 + h1);  
   prn_sequence(seq, n);
}

//_________________________________________________________________________________________________

void prn_dfs_subproblem2(unsigned char g1, unsigned char h1, unsigned char *seq)
{
   printf("            g1 = %2d  h1 = %2d  f1 = %2d: ", g1, h1, g1 + h1);
   prn_sequence(seq, n);
}

//_________________________________________________________________________________________________

void prn_a_star_subproblem(bistate *state, int direction, unsigned char UB, searchinfo *info)
{
   unsigned char  g1, h1, g2, h2;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if(direction == 1) {
      printf("Forward  UB  f1 f1b  g1  h1  g2  h2  e1  e2    n_exp_f    n_gen_f\n");
      if(state->open2 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", UB, g1 + h1, 2*g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("        %3d %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d: ", UB, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      prn_sequence(state->seq, n);
   } else {
      printf("Reverse  UB  f2 f2b  g1  h1  g2  h2  e1  e2    n_exp_r    n_gen_r\n");
      if (state->open1 != 2)
         printf("        %3d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", UB, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("        %3d %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d: ", UB, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      prn_sequence(state->seq, n);
   }
}

//_________________________________________________________________________________________________

void prn_a_star_subproblem2(bistate *state, int direction, int status, searchinfo *info)
{
   unsigned char  g1, h1, g2, h2;

   g1 = state->g1;
   h1 = state->h1;
   g2 = state->g2;
   h2 = state->h2;

   if (direction == 1) {
      if (state->open2 != 2)
         printf("         %2d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", status, g1 + h1, 2 * g1 + h1 - h2, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_forward, info->n_generated_forward);
      else
         printf("         %2d %3d %3d %3d %3d   * %3d %3d   * %10I64d %10I64d: ", status, g1 + h1, 2 * g1 + h1 - h2, g1, h1, h2, g1 - h2, info->n_explored_forward, info->n_generated_forward);
      prn_sequence(state->seq, n);
   }
   else {
      if (state->open1 != 2)
         printf("         %2d %3d %3d %3d %3d %3d %3d %3d %3d %10I64d %10I64d: ", status, g2 + h2, 2 * g2 + h2 - h1, g1, h1, g2, h2, g1 - h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      else
         printf("         %2d %3d %3d   * %3d %3d %3d   * %3d %10I64d %10I64d: ", status, g2 + h2, 2 * g2 + h2 - h1, h1, g2, h2, g2 - h1, info->n_explored_reverse, info->n_generated_reverse);
      prn_sequence(state->seq, n);
   }
}

//_________________________________________________________________________________________________

void prn_open_g_h1_h2_values(int direction)
{
   int            h1, h2, max_h1, max_h2;

   max_h1 = 0;
   max_h2 = 0;
   if (direction == 1) {
      for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
         for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
            if (!open_g1_h1_h2_values[h1][h2].empty()) {
               if (h1 > max_h1) max_h1 = h1;
               if (h2 > max_h2) max_h2 = h2;
            }
         }
      }
      printf("   ");  for (h2 = 0; h2 <= max_h2; h2++) printf(" %2d", h2); printf("\n");
      for (h1 = 0; h1 <= max_h1; h1++) {
         printf("%2d:", h1);
         for (h2 = 0; h2 <= max_h2; h2++ ) {
            if (open_g1_h1_h2_values[h1][h2].empty()) {
               printf("  *");
            }
            else {
               printf(" %2d", open_g1_h1_h2_values[h1][h2].get_min());
            }
         }
         printf("\n");
      }
   } else {
      for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
         for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
            if (!open_g2_h1_h2_values[h1][h2].empty()) {
               if (h1 > max_h1) max_h1 = h1;
               if (h2 > max_h2) max_h2 = h2;
            }
         }
      }
      printf("   ");  for (h2 = 0; h2 <= max_h2; h2++) printf(" %2d", h2); printf("\n");
      for (h1 = 0; h1 <= max_h1; h1++) {
         printf("%2d:", h1);
         for (h2 = 0; h2 <= max_h2; h2++ ) {
            if (open_g2_h1_h2_values[h1][h2].empty()) {
               printf("  *");
            }
            else {
               printf(" %2d", open_g2_h1_h2_values[h1][h2].get_min());
            }
         }
         printf("\n");
      }

   }
}

//_________________________________________________________________________________________________

void prn_open_g_h1_h2_values2(int direction)
{
   int            g, h1, h2, max_h1, max_h2;

   max_h1 = 0;
   max_h2 = 0;
   if (direction == 1) {
      for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
         for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
            if (!open_g1_h1_h2_values[h1][h2].empty()) {
               if (h1 > max_h1) max_h1 = h1;
               if (h2 > max_h2) max_h2 = h2;
            }
         }
      }
      printf("   ");  for (h2 = 0; h2 <= max_h2; h2++) printf("           %2d", h2); printf("\n");
      for (h1 = 0; h1 <= max_h1; h1++) {
         printf("%2d:", h1);
         for (h2 = 0; h2 <= max_h2; h2++) {
            if (open_g1_h1_h2_values[h1][h2].empty()) {
               printf("            *");
            } else {
               g = open_g1_h1_h2_values[h1][h2].get_min();
               printf(" (%2d, %6I64d)", g, open_g1_h1_h2_values[h1][h2].n_of_elements(g));
            }
         }
         printf("\n");
      }
   } else {
      for (h1 = 0; h1 <= MAX_DEPTH; h1++) {
         for (h2 = 0; h2 <= MAX_DEPTH; h2++) {
            if (!open_g2_h1_h2_values[h1][h2].empty()) {
               if (h1 > max_h1) max_h1 = h1;
               if (h2 > max_h2) max_h2 = h2;
            }
         }
      }
      printf("   ");  for (h2 = 0; h2 <= max_h2; h2++) printf("           %2d", h2); printf("\n");
      for (h1 = 0; h1 <= max_h1; h1++) {
         printf("%2d:", h1);
         for (h2 = 0; h2 <= max_h2; h2++) {
            if (open_g2_h1_h2_values[h1][h2].empty()) {
               printf("            *");
            } else {
               g = open_g2_h1_h2_values[h1][h2].get_min();
               printf(" (%2d, %6I64d)", g, open_g2_h1_h2_values[h1][h2].n_of_elements(g));
            }
         }
         printf("\n");
      }
   }
}

//_________________________________________________________________________________________________

//void prn_heap_info()
//{
//   int      i;
//
//   for(i = 0; i <= UB; i++) {
//      if(cbfs_heaps[i].empty()) {
//         printf("%4d \n", i);
//      } else {
//         printf("%4d %6d %8.2f %8.2f\n", i, cbfs_heaps[i].n_of_items(), cbfs_heaps[i].get_min().key, cbfs_heaps[i].get_max().key);
//      }
//   }
//}
