/* 
   1. This file was copied on 10/5/12 from c:\sewell\research\facility\facility_cbfns\random.cpp.
   2. Modified the functions to explicitly cast to integers in order to eliminate compiler warnings.
*/

//_________________________________________________________________________________________________

double ggubfs(double *dseed)
{
   int      div;
   double   product;

      product = 16807.0 * *dseed;
      div = (int) (product / 2147483647.0);
      *dseed = product - (div * 2147483647.0);
      return( *dseed / 2147483648.0 );
}

//_________________________________________________________________________________________________

int randomi(int n, double *dseed)
{
   int      truncate;

   truncate = (int) (n * ggubfs(dseed)) + 1;
   return(truncate);
}

//_________________________________________________________________________________________________

int random_int_0n(int n, double *dseed)
{
   int      truncate;

   truncate = (int) ((n+1) * ggubfs(dseed));
   return(truncate);
}

//_________________________________________________________________________________________________

void random_permutation(int n_s, int *s, double *dseed)
/*
   1. This routine generates a random permutation of the elements in s.
   2. The permutation is returned in s.
   3. Based on Chapter 2 of Data Structures and Algorithm Analysis in C 
      (Dr. Dobb's Algorithms Collection).
   4. Written 10/20/03.
   5. Modified 1/21/04 to permute a vector of int's.
   6. Created 8/3/10 by modifying c:\sewell\research\nms\tsp\c\random.c.
      a. Made the correction that is in c:\sewell\research\nms\tsp\c\random2.c.
         It was computing index correctly.
*/
{
   int      i, index;
   int      temp;

   for(i = 1; i < n_s; i++) {
      temp = s[i];
      index = i + random_int_0n(n_s - i, dseed);
      s[i] = s[index];
      s[index] = temp;
   }
}
