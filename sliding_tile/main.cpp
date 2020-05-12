#include "sliding_tile.h"
#include "astar.h"
#include "dibbs.h"
#include "dvcbs.h"
#include "GBFHS.h"
#include "id-d.h"
#include "nbs.h"
#include "ida.h"
#include "dibbs-2phase.h"
#include <iostream>
#include <cassert>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>


//#define A_STAR
//#define REVERSE_ASTAR
//#define IDA
//#define IDD
//#define DIBBS
//#define GBFHS
//#define NBS
//#define DVCBS
#define DIBBS_NBS


void define_problems15(int i, unsigned char* tile_in_location)
{
  int            j;
  unsigned char  problems[101][16] =
  {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 14, 13, 15, 7, 11, 12, 9, 5, 6, 0, 2, 1, 4, 8, 10, 3 },
    { 13, 5, 4, 10, 9, 12, 8, 14, 2, 3, 7, 1, 0, 15, 11, 6 },
    { 14, 7, 8, 2, 13, 11, 10, 4, 9, 12, 5, 0, 3, 6, 1, 15 },
    { 5, 12, 10, 7, 15, 11, 14, 0, 8, 2, 1, 13, 3, 4, 9, 6 },
    { 4, 7, 14, 13, 10, 3, 9, 12, 11, 5, 6, 15, 1, 2, 8, 0 },
    { 14, 7, 1, 9, 12, 3, 6, 15, 8, 11, 2, 5, 10, 0, 4, 13 },
    { 2, 11, 15, 5, 13, 4, 6, 7, 12, 8, 10, 1, 9, 3, 14, 0 },
    { 12, 11, 15, 3, 8, 0, 4, 2, 6, 13, 9, 5, 14, 1, 10, 7 },
    { 3, 14, 9, 11, 5, 4, 8, 2, 13, 12, 6, 7, 10, 1, 15, 0 },
    { 13, 11, 8, 9, 0, 15, 7, 10, 4, 3, 6, 14, 5, 12, 2, 1 },
    { 5, 9, 13, 14, 6, 3, 7, 12, 10, 8, 4, 0, 15, 2, 11, 1 },
    { 14, 1, 9, 6, 4, 8, 12, 5, 7, 2, 3, 0, 10, 11, 13, 15 },
    { 3, 6, 5, 2, 10, 0, 15, 14, 1, 4, 13, 12, 9, 8, 11, 7 },
    { 7, 6, 8, 1, 11, 5, 14, 10, 3, 4, 9, 13, 15, 2, 0, 12 },
    { 13, 11, 4, 12, 1, 8, 9, 15, 6, 5, 14, 2, 7, 3, 10, 0 },
    { 1, 3, 2, 5, 10, 9, 15, 6, 8, 14, 13, 11, 12, 4, 7, 0 },
    { 15, 14, 0, 4, 11, 1, 6, 13, 7, 5, 8, 9, 3, 2, 10, 12 },
    { 6, 0, 14, 12, 1, 15, 9, 10, 11, 4, 7, 2, 8, 3, 5, 13 },
    { 7, 11, 8, 3, 14, 0, 6, 15, 1, 4, 13, 9, 5, 12, 2, 10 },
    { 6, 12, 11, 3, 13, 7, 9, 15, 2, 14, 8, 10, 4, 1, 5, 0 },
    { 12, 8, 14, 6, 11, 4, 7, 0, 5, 1, 10, 15, 3, 13, 9, 2 },
    { 14, 3, 9, 1, 15, 8, 4, 5, 11, 7, 10, 13, 0, 2, 12, 6 },
    { 10, 9, 3, 11, 0, 13, 2, 14, 5, 6, 4, 7, 8, 15, 1, 12 },
    { 7, 3, 14, 13, 4, 1, 10, 8, 5, 12, 9, 11, 2, 15, 6, 0 },
    { 11, 4, 2, 7, 1, 0, 10, 15, 6, 9, 14, 8, 3, 13, 5, 12 },
    { 5, 7, 3, 12, 15, 13, 14, 8, 0, 10, 9, 6, 1, 4, 2, 11 },
    { 14, 1, 8, 15, 2, 6, 0, 3, 9, 12, 10, 13, 4, 7, 5, 11 },
    { 13, 14, 6, 12, 4, 5, 1, 0, 9, 3, 10, 2, 15, 11, 8, 7 },
    { 9, 8, 0, 2, 15, 1, 4, 14, 3, 10, 7, 5, 11, 13, 6, 12 },
    { 12, 15, 2, 6, 1, 14, 4, 8, 5, 3, 7, 0, 10, 13, 9, 11 },
    { 12, 8, 15, 13, 1, 0, 5, 4, 6, 3, 2, 11, 9, 7, 14, 10 },
    { 14, 10, 9, 4, 13, 6, 5, 8, 2, 12, 7, 0, 1, 3, 11, 15 },
    { 14, 3, 5, 15, 11, 6, 13, 9, 0, 10, 2, 12, 4, 1, 7, 8 },
    { 6, 11, 7, 8, 13, 2, 5, 4, 1, 10, 3, 9, 14, 0, 12, 15 },
    { 1, 6, 12, 14, 3, 2, 15, 8, 4, 5, 13, 9, 0, 7, 11, 10 },
    { 12, 6, 0, 4, 7, 3, 15, 1, 13, 9, 8, 11, 2, 14, 5, 10 },
    { 8, 1, 7, 12, 11, 0, 10, 5, 9, 15, 6, 13, 14, 2, 3, 4 },
    { 7, 15, 8, 2, 13, 6, 3, 12, 11, 0, 4, 10, 9, 5, 1, 14 },
    { 9, 0, 4, 10, 1, 14, 15, 3, 12, 6, 5, 7, 11, 13, 8, 2 },
    { 11, 5, 1, 14, 4, 12, 10, 0, 2, 7, 13, 3, 9, 15, 6, 8 },
    { 8, 13, 10, 9, 11, 3, 15, 6, 0, 1, 2, 14, 12, 5, 4, 7 },
    { 4, 5, 7, 2, 9, 14, 12, 13, 0, 3, 6, 11, 8, 1, 15, 10 },
    { 11, 15, 14, 13, 1, 9, 10, 4, 3, 6, 2, 12, 7, 5, 8, 0 },
    { 12, 9, 0, 6, 8, 3, 5, 14, 2, 4, 11, 7, 10, 1, 15, 13 },
    { 3, 14, 9, 7, 12, 15, 0, 4, 1, 8, 5, 6, 11, 10, 2, 13 },
    { 8, 4, 6, 1, 14, 12, 2, 15, 13, 10, 9, 5, 3, 7, 0, 11 },
    { 6, 10, 1, 14, 15, 8, 3, 5, 13, 0, 2, 7, 4, 9, 11, 12 },
    { 8, 11, 4, 6, 7, 3, 10, 9, 2, 12, 15, 13, 0, 1, 5, 14 },
    { 10, 0, 2, 4, 5, 1, 6, 12, 11, 13, 9, 7, 15, 3, 14, 8 },
    { 12, 5, 13, 11, 2, 10, 0, 9, 7, 8, 4, 3, 14, 6, 15, 1 },
    { 10, 2, 8, 4, 15, 0, 1, 14, 11, 13, 3, 6, 9, 7, 5, 12 },
    { 10, 8, 0, 12, 3, 7, 6, 2, 1, 14, 4, 11, 15, 13, 9, 5 },
    { 14, 9, 12, 13, 15, 4, 8, 10, 0, 2, 1, 7, 3, 11, 5, 6 },
    { 12, 11, 0, 8, 10, 2, 13, 15, 5, 4, 7, 3, 6, 9, 14, 1 },
    { 13, 8, 14, 3, 9, 1, 0, 7, 15, 5, 4, 10, 12, 2, 6, 11 },
    { 3, 15, 2, 5, 11, 6, 4, 7, 12, 9, 1, 0, 13, 14, 10, 8 },
    { 5, 11, 6, 9, 4, 13, 12, 0, 8, 2, 15, 10, 1, 7, 3, 14 },
    { 5, 0, 15, 8, 4, 6, 1, 14, 10, 11, 3, 9, 7, 12, 2, 13 },
    { 15, 14, 6, 7, 10, 1, 0, 11, 12, 8, 4, 9, 2, 5, 13, 3 },
    { 11, 14, 13, 1, 2, 3, 12, 4, 15, 7, 9, 5, 10, 6, 8, 0 },
    { 6, 13, 3, 2, 11, 9, 5, 10, 1, 7, 12, 14, 8, 4, 0, 15 },
    { 4, 6, 12, 0, 14, 2, 9, 13, 11, 8, 3, 15, 7, 10, 1, 5 },
    { 8, 10, 9, 11, 14, 1, 7, 15, 13, 4, 0, 12, 6, 2, 5, 3 },
    { 5, 2, 14, 0, 7, 8, 6, 3, 11, 12, 13, 15, 4, 10, 9, 1 },
    { 7, 8, 3, 2, 10, 12, 4, 6, 11, 13, 5, 15, 0, 1, 9, 14 },
    { 11, 6, 14, 12, 3, 5, 1, 15, 8, 0, 10, 13, 9, 7, 4, 2 },
    { 7, 1, 2, 4, 8, 3, 6, 11, 10, 15, 0, 5, 14, 12, 13, 9 },
    { 7, 3, 1, 13, 12, 10, 5, 2, 8, 0, 6, 11, 14, 15, 4, 9 },
    { 6, 0, 5, 15, 1, 14, 4, 9, 2, 13, 8, 10, 11, 12, 7, 3 },
    { 15, 1, 3, 12, 4, 0, 6, 5, 2, 8, 14, 9, 13, 10, 7, 11 },
    { 5, 7, 0, 11, 12, 1, 9, 10, 15, 6, 2, 3, 8, 4, 13, 14 },
    { 12, 15, 11, 10, 4, 5, 14, 0, 13, 7, 1, 2, 9, 8, 3, 6 },
    { 6, 14, 10, 5, 15, 8, 7, 1, 3, 4, 2, 0, 12, 9, 11, 13 },
    { 14, 13, 4, 11, 15, 8, 6, 9, 0, 7, 3, 1, 2, 10, 12, 5 },
    { 14, 4, 0, 10, 6, 5, 1, 3, 9, 2, 13, 15, 12, 7, 8, 11 },
    { 15, 10, 8, 3, 0, 6, 9, 5, 1, 14, 13, 11, 7, 2, 12, 4 },
    { 0, 13, 2, 4, 12, 14, 6, 9, 15, 1, 10, 3, 11, 5, 8, 7 },
    { 3, 14, 13, 6, 4, 15, 8, 9, 5, 12, 10, 0, 2, 7, 1, 11 },
    { 0, 1, 9, 7, 11, 13, 5, 3, 14, 12, 4, 2, 8, 6, 10, 15 },
    { 11, 0, 15, 8, 13, 12, 3, 5, 10, 1, 4, 6, 14, 9, 7, 2 },
    { 13, 0, 9, 12, 11, 6, 3, 5, 15, 8, 1, 10, 4, 14, 2, 7 },
    { 14, 10, 2, 1, 13, 9, 8, 11, 7, 3, 6, 12, 15, 5, 4, 0 },
    { 12, 3, 9, 1, 4, 5, 10, 2, 6, 11, 15, 0, 14, 7, 13, 8 },
    { 15, 8, 10, 7, 0, 12, 14, 1, 5, 9, 6, 3, 13, 11, 4, 2 },
    { 4, 7, 13, 10, 1, 2, 9, 6, 12, 8, 14, 5, 3, 0, 11, 15 },
    { 6, 0, 5, 10, 11, 12, 9, 2, 1, 7, 4, 3, 14, 8, 13, 15 },
    { 9, 5, 11, 10, 13, 0, 2, 1, 8, 6, 14, 12, 4, 7, 3, 15 },
    { 15, 2, 12, 11, 14, 13, 9, 5, 1, 3, 8, 7, 0, 10, 6, 4 },
    { 11, 1, 7, 4, 10, 13, 3, 8, 9, 14, 0, 15, 6, 5, 2, 12 },
    { 5, 4, 7, 1, 11, 12, 14, 15, 10, 13, 8, 6, 2, 0, 9, 3 },
    { 9, 7, 5, 2, 14, 15, 12, 10, 11, 3, 6, 1, 8, 13, 0, 4 },
    { 3, 2, 7, 9, 0, 15, 12, 4, 6, 11, 5, 14, 8, 13, 10, 1 },
    { 13, 9, 14, 6, 12, 8, 1, 2, 3, 4, 0, 7, 5, 10, 11, 15 },
    { 5, 7, 11, 8, 0, 14, 9, 13, 10, 12, 3, 15, 6, 1, 4, 2 },
    { 4, 3, 6, 13, 7, 15, 9, 0, 10, 5, 8, 11, 2, 12, 1, 14 },
    { 1, 7, 15, 14, 2, 6, 4, 9, 12, 11, 13, 3, 0, 8, 5, 10 },
    { 9, 14, 5, 7, 8, 15, 1, 2, 10, 4, 13, 6, 12, 0, 11, 3 },
    { 0, 11, 3, 12, 5, 2, 1, 9, 8, 10, 14, 15, 7, 4, 13, 6 },
    { 7, 15, 4, 0, 10, 9, 2, 5, 12, 11, 13, 6, 1, 3, 14, 8 },
    { 11, 4, 0, 8, 6, 10, 5, 13, 12, 7, 14, 3, 1, 2, 9, 15 }
  };

  assert((1 <= i) && (i <= 100));
  for (j = 0; j <= 15; j++) tile_in_location[j] = problems[i][j];
}

//_________________________________________________________________________________________________

void define_problems24(int i, unsigned char* tile_in_location)
{
  int            j;
  unsigned char  problems[51][25] =
  {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 14, 5, 9, 2, 18, 8, 23, 19, 12, 17, 15, 0, 10, 20, 4, 6, 11, 21, 1, 7, 24, 3, 16, 22, 13 },
    { 16, 5, 1, 12, 6, 24, 17, 9, 2, 22, 4, 10, 13, 18, 19, 20, 0, 23, 7, 21, 15, 11, 8, 3, 14 },
    { 6, 0, 24, 14, 8, 5, 21, 19, 9, 17, 16, 20, 10, 13, 2, 15, 11, 22, 1, 3, 7, 23, 4, 18, 12 },
    { 18, 14, 0, 9, 8, 3, 7, 19, 2, 15, 5, 12, 1, 13, 24, 23, 4, 21, 10, 20, 16, 22, 11, 6, 17 },
    { 17, 1, 20, 9, 16, 2, 22, 19, 14, 5, 15, 21, 0, 3, 24, 23, 18, 13, 12, 7, 10, 8, 6, 4, 11 },
    { 2, 0, 10, 19, 1, 4, 16, 3, 15, 20, 22, 9, 6, 18, 5, 13, 12, 21, 8, 17, 23, 11, 24, 7, 14 },
    { 21, 22, 15, 9, 24, 12, 16, 23, 2, 8, 5, 18, 17, 7, 10, 14, 13, 4, 0, 6, 20, 11, 3, 1, 19 },
    { 7, 13, 11, 22, 12, 20, 1, 18, 21, 5, 0, 8, 14, 24, 19, 9, 4, 17, 16, 10, 23, 15, 3, 2, 6 },
    { 3, 2, 17, 0, 14, 18, 22, 19, 15, 20, 9, 7, 10, 21, 16, 6, 24, 23, 8, 5, 1, 4, 11, 12, 13 },
    { 23, 14, 0, 24, 17, 9, 20, 21, 2, 18, 10, 13, 22, 1, 3, 11, 4, 16, 6, 5, 7, 12, 8, 15, 19 },
    { 15, 11, 8, 18, 14, 3, 19, 16, 20, 5, 24, 2, 17, 4, 22, 10, 1, 13, 9, 21, 23, 7, 6, 12, 0 },
    { 12, 23, 9, 18, 24, 22, 4, 0, 16, 13, 20, 3, 15, 6, 17, 8, 7, 11, 19, 1, 10, 2, 14, 5, 21 },
    { 21, 24, 8, 1, 19, 22, 12, 9, 7, 18, 4, 0, 23, 14, 10, 6, 3, 11, 16, 5, 15, 2, 20, 13, 17 },
    { 24, 1, 17, 10, 15, 14, 3, 13, 8, 0, 22, 16, 20, 7, 21, 4, 12, 9, 2, 11, 5, 23, 6, 18, 19 },
    { 24, 10, 15, 9, 16, 6, 3, 22, 17, 13, 19, 23, 21, 11, 18, 0, 1, 2, 7, 8, 20, 5, 12, 4, 14 },
    { 18, 24, 17, 11, 12, 10, 19, 15, 6, 1, 5, 21, 22, 9, 7, 3, 2, 16, 14, 4, 20, 23, 0, 8, 13 },
    { 23, 16, 13, 24, 5, 18, 22, 11, 17, 0, 6, 9, 20, 7, 3, 2, 10, 14, 12, 21, 1, 19, 15, 8, 4 },
    { 0, 12, 24, 10, 13, 5, 2, 4, 19, 21, 23, 18, 8, 17, 9, 22, 16, 11, 6, 15, 7, 3, 14, 1, 20 },
    { 16, 13, 6, 23, 9, 8, 3, 5, 24, 15, 22, 12, 21, 17, 1, 19, 10, 7, 11, 4, 18, 2, 14, 20, 0 },
    { 4, 5, 1, 23, 21, 13, 2, 10, 18, 17, 15, 7, 0, 9, 3, 14, 11, 12, 19, 8, 6, 20, 24, 22, 16 },
    { 24, 8, 14, 5, 16, 4, 13, 6, 22, 19, 1, 10, 9, 12, 3, 0, 18, 21, 20, 23, 15, 17, 11, 7, 2 },
    { 7, 6, 3, 22, 15, 19, 21, 2, 13, 0, 8, 10, 9, 4, 18, 16, 11, 24, 5, 12, 17, 1, 23, 14, 20 },
    { 24, 11, 18, 7, 3, 17, 5, 1, 23, 15, 21, 8, 2, 4, 19, 14, 0, 16, 22, 6, 9, 13, 20, 12, 10 },
    { 14, 24, 18, 12, 22, 15, 5, 1, 23, 11, 6, 19, 10, 13, 7, 0, 3, 9, 4, 17, 2, 21, 16, 20, 8 },
    { 3, 17, 9, 8, 24, 1, 11, 12, 14, 0, 5, 4, 22, 13, 16, 21, 15, 6, 7, 10, 20, 23, 2, 18, 19 },
    { 22, 21, 15, 3, 14, 13, 9, 19, 24, 23, 16, 0, 7, 10, 18, 4, 11, 20, 8, 2, 1, 6, 5, 17, 12 },
    { 9, 19, 8, 20, 2, 3, 14, 1, 24, 6, 13, 18, 7, 10, 17, 5, 22, 12, 21, 16, 15, 0, 23, 11, 4 },
    { 17, 15, 7, 12, 8, 3, 4, 9, 21, 5, 16, 6, 19, 20, 1, 22, 24, 18, 11, 14, 23, 10, 2, 13, 0 },
    { 10, 3, 6, 13, 1, 2, 20, 14, 18, 11, 15, 7, 5, 12, 9, 24, 17, 22, 4, 8, 21, 23, 19, 16, 0 },
    { 8, 19, 7, 16, 12, 2, 13, 22, 14, 9, 11, 5, 6, 3, 18, 24, 0, 15, 10, 23, 1, 20, 4, 17, 21 },
    { 19, 20, 12, 21, 7, 0, 16, 10, 5, 9, 14, 23, 3, 11, 4, 2, 6, 1, 8, 15, 17, 13, 22, 24, 18 },
    { 1, 12, 18, 13, 17, 15, 3, 7, 20, 0, 19, 24, 6, 5, 21, 11, 2, 8, 9, 16, 22, 10, 4, 23, 14 },
    { 11, 22, 6, 21, 8, 13, 20, 23, 0, 2, 15, 7, 12, 18, 16, 3, 1, 17, 5, 4, 9, 14, 24, 10, 19 },
    { 5, 18, 3, 21, 22, 17, 13, 24, 0, 7, 15, 14, 11, 2, 9, 10, 1, 8, 6, 16, 19, 4, 20, 23, 12 },
    { 2, 10, 24, 11, 22, 19, 0, 3, 8, 17, 15, 16, 6, 4, 23, 20, 18, 7, 9, 14, 13, 5, 12, 1, 21 },
    { 2, 10, 1, 7, 16, 9, 0, 6, 12, 11, 3, 18, 22, 4, 13, 24, 20, 15, 8, 14, 21, 23, 17, 19, 5 },
    { 23, 22, 5, 3, 9, 6, 18, 15, 10, 2, 21, 13, 19, 12, 20, 7, 0, 1, 16, 24, 17, 4, 14, 8, 11 },
    { 10, 3, 24, 12, 0, 7, 8, 11, 14, 21, 22, 23, 2, 1, 9, 17, 18, 6, 20, 4, 13, 15, 5, 19, 16 },
    { 16, 24, 3, 14, 5, 18, 7, 6, 4, 2, 0, 15, 8, 10, 20, 13, 19, 9, 21, 11, 17, 12, 22, 23, 1 },
    { 2, 17, 4, 13, 7, 12, 10, 3, 0, 16, 21, 24, 8, 5, 18, 20, 15, 19, 14, 9, 22, 11, 6, 1, 23 },
    { 13, 19, 9, 10, 14, 15, 23, 21, 24, 16, 12, 11, 0, 5, 22, 20, 4, 18, 3, 1, 6, 2, 7, 17, 8 },
    { 16, 6, 20, 18, 23, 19, 7, 11, 13, 17, 12, 9, 1, 24, 3, 22, 2, 21, 10, 4, 8, 15, 14, 5, 0 },
    { 7, 4, 19, 12, 16, 20, 15, 23, 8, 10, 1, 18, 2, 17, 14, 24, 9, 5, 0, 21, 6, 3, 11, 13, 22 },
    { 8, 12, 18, 3, 2, 11, 10, 22, 24, 17, 1, 13, 23, 4, 20, 16, 6, 15, 9, 21, 19, 5, 14, 0, 7 },
    { 9, 7, 16, 18, 12, 1, 23, 8, 22, 0, 6, 19, 4, 13, 2, 24, 11, 15, 21, 17, 20, 3, 10, 14, 5 },
    { 1, 16, 10, 14, 17, 13, 0, 3, 5, 7, 4, 15, 19, 2, 21, 9, 23, 8, 12, 6, 11, 24, 22, 20, 18 },
    { 21, 11, 10, 4, 16, 6, 13, 24, 7, 14, 1, 20, 9, 17, 0, 15, 2, 5, 8, 22, 3, 12, 18, 19, 23 },
    { 2, 22, 21, 0, 23, 8, 14, 20, 12, 7, 16, 11, 3, 5, 1, 15, 4, 9, 24, 10, 13, 6, 19, 17, 18 },
    { 2, 21, 3, 7, 0, 8, 5, 14, 18, 6, 12, 11, 23, 20, 10, 15, 17, 4, 9, 16, 13, 19, 24, 22, 1 },
    { 23, 1, 12, 6, 16, 2, 20, 10, 21, 18, 14, 13, 17, 19, 22, 0, 15, 24, 3, 7, 4, 8, 5, 9, 11 }
  };

  assert((1 <= i) && (i <= 50));
  for (j = 0; j <= 24; j++) tile_in_location[j] = problems[i][j];
}


void benchmarks(std::ostream& stream)
{
  uint8_t z_optimal[101];
  int n_problems;

  switch (NUM_TILES) {
  case 16:
    z_optimal[0] = 0; z_optimal[1] = 57; z_optimal[2] = 55; z_optimal[3] = 59; z_optimal[4] = 56;
    z_optimal[5] = 56; z_optimal[6] = 52; z_optimal[7] = 52; z_optimal[8] = 50; z_optimal[9] = 46;
    z_optimal[10] = 59; z_optimal[11] = 57; z_optimal[12] = 45; z_optimal[13] = 46; z_optimal[14] = 59;
    z_optimal[15] = 62; z_optimal[16] = 42; z_optimal[17] = 66; z_optimal[18] = 55; z_optimal[19] = 46;
    z_optimal[20] = 52; z_optimal[21] = 54; z_optimal[22] = 59; z_optimal[23] = 49; z_optimal[24] = 54;
    z_optimal[25] = 52; z_optimal[26] = 58; z_optimal[27] = 53; z_optimal[28] = 52; z_optimal[29] = 54;
    z_optimal[30] = 47; z_optimal[31] = 50; z_optimal[32] = 59; z_optimal[33] = 60; z_optimal[34] = 52;
    z_optimal[35] = 55; z_optimal[36] = 52; z_optimal[37] = 58; z_optimal[38] = 53; z_optimal[39] = 49;
    z_optimal[40] = 54; z_optimal[41] = 54; z_optimal[42] = 42; z_optimal[43] = 64; z_optimal[44] = 50;
    z_optimal[45] = 51; z_optimal[46] = 49; z_optimal[47] = 47; z_optimal[48] = 49; z_optimal[49] = 59;
    z_optimal[50] = 53; z_optimal[51] = 56; z_optimal[52] = 56; z_optimal[53] = 64; z_optimal[54] = 56;
    z_optimal[55] = 41; z_optimal[56] = 55; z_optimal[57] = 50; z_optimal[58] = 51; z_optimal[59] = 57;
    z_optimal[60] = 66; z_optimal[61] = 45; z_optimal[62] = 57; z_optimal[63] = 56; z_optimal[64] = 51;
    z_optimal[65] = 47; z_optimal[66] = 61; z_optimal[67] = 50; z_optimal[68] = 51; z_optimal[69] = 53;
    z_optimal[70] = 52; z_optimal[71] = 44; z_optimal[72] = 56; z_optimal[73] = 49; z_optimal[74] = 56;
    z_optimal[75] = 48; z_optimal[76] = 57; z_optimal[77] = 54; z_optimal[78] = 53; z_optimal[79] = 42;
    z_optimal[80] = 57; z_optimal[81] = 53; z_optimal[82] = 62; z_optimal[83] = 49; z_optimal[84] = 55;
    z_optimal[85] = 44; z_optimal[86] = 45; z_optimal[87] = 52; z_optimal[88] = 65; z_optimal[89] = 54;
    z_optimal[90] = 50; z_optimal[91] = 57; z_optimal[92] = 57; z_optimal[93] = 46; z_optimal[94] = 53;
    z_optimal[95] = 50; z_optimal[96] = 49; z_optimal[97] = 44; z_optimal[98] = 54; z_optimal[99] = 57;
    z_optimal[100] = 54;
    n_problems = 100;
    break;
  case 25:
    z_optimal[0] = 0; z_optimal[1] = 95; z_optimal[2] = 96; z_optimal[3] = 97; z_optimal[4] = 98;
    z_optimal[5] = 100; z_optimal[6] = 101; z_optimal[7] = 104; z_optimal[8] = 108; z_optimal[9] = 113;
    z_optimal[10] = 114; z_optimal[11] = 106; z_optimal[12] = 109; z_optimal[13] = 101; z_optimal[14] = 111;
    z_optimal[15] = 103; z_optimal[16] = 96; z_optimal[17] = 109; z_optimal[18] = 110; z_optimal[19] = 106;
    z_optimal[20] = 92; z_optimal[21] = 103; z_optimal[22] = 95; z_optimal[23] = 104; z_optimal[24] = 107;
    z_optimal[25] = 81; z_optimal[26] = 105; z_optimal[27] = 99; z_optimal[28] = 98; z_optimal[29] = 88;
    z_optimal[30] = 92; z_optimal[31] = 99; z_optimal[32] = 97; z_optimal[33] = 106; z_optimal[34] = 102;
    z_optimal[35] = 98; z_optimal[36] = 90; z_optimal[37] = 100; z_optimal[38] = 96; z_optimal[39] = 104;
    z_optimal[40] = 82; z_optimal[41] = 106; z_optimal[42] = 108; z_optimal[43] = 104; z_optimal[44] = 93;
    z_optimal[45] = 101; z_optimal[46] = 100; z_optimal[47] = 92; z_optimal[48] = 107; z_optimal[49] = 100;
    z_optimal[50] = 113;
    n_problems = 50;
    break;
  default: std::cerr << "Illegal value of N_LOCATIONS in benchmarks\n"; exit(1); break;
  }

  stream << "Sliding Tile = " << NUM_TILES << " with Manhattan Distance\n";
  stream << "Algorithms: ";
#ifdef A_STAR
  stream << "A* ";
#endif
#ifdef REVERSE_ASTAR
  stream << "RA* ";
#endif
#ifdef IDD
  stream << "IDD ";
#endif
#ifdef DIBBS
  stream << "DIBBS ";
#endif
#ifdef GBFHS
  stream << "GBFHS ";
#endif
#ifdef NBS
  stream << "NBS ";
#endif
#ifdef DVCBS
  stream << "DVCBS ";
#endif
#ifdef DIBBS_NBS
  stream << "DIBBS_NBS ";
#endif
  stream << "\n";

  std::stringstream expansion_stream;
  std::stringstream time_stream;
  std::stringstream memory_stream;
  typedef std::chrono::nanoseconds precision;
  uint8_t tile_in_location[NUM_TILES];

  {
#ifdef A_STAR
    std::cout << "\nA*: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Astar::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef REVERSE_ASTAR
    std::cout << "\nRA: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Astar::search(goal_state, starting_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }


  {
#ifdef IDA
    std::cout << "\nIDA*: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = IDAstar::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef IDD
    std::cout << "\nIDD: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = ID_D::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef DIBBS
    std::cout << "\nDIBBS: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Dibbs::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef GBFHS
    std::cout << "\nGBFHS: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Gbfhs::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }


  {
#ifdef NBS
    std::cout << "\nNBS: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Nbs::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef DVCBS
    std::cout << "\nDVCBS: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Dvcbs::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef DIBBS_NBS
    std::cout << "\nDIBBS_NBS: ";
    for (int i = 1; i <= n_problems; i++) {
      std::cout << i << " ";
      switch (NUM_TILES) {
      case 16: define_problems15(i, tile_in_location); break;
      case 25: define_problems24(i, tile_in_location); break;
      default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
      }

      SlidingTile::initialize(tile_in_location);
      SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
      SlidingTile starting_state(tile_in_location, Direction::forward);

      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = DibbsNbs::search(starting_state, goal_state);
      auto end = std::chrono::system_clock::now();
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (!std::isinf(cstar) && z_optimal[i] != cstar) { std::cout << "ERROR Cstar mismatch: " << std::to_string(cstar) << " instead of " << std::to_string(z_optimal[i]); return; }
    }

    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }
}

std::string return_formatted_time(std::string format)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), format.c_str());
  return ss.str();
}

void run_test() {
  std::ofstream file;
  std::string dir = R"(C:\Users\John\Dropbox\UIUC\Research\SlidingTileData\)";
  std::string name = "output" + std::to_string(NUM_TILES) + "_MD_" + return_formatted_time("%y%b%d-%H%M%S");
#ifdef A_STAR
  name += "_A";
#endif
#ifdef REVERSE_ASTAR
  name += "_RA";
#endif
#ifdef IDA_STAR
  name += "_IDA";
#endif
#ifdef IDD
  name += "_IDD";
#endif
#ifdef DIBBS
  name += "_DIBBS";
#endif
#ifdef GBFHS
  name += "_GBFHS";
#endif
#ifdef NBS
  name += "_NBS";
#endif
#ifdef DVCBS
  name += "_DVCBS";
#endif
#ifdef DIBBS_NBS
  name += "_DBSNBS";
#endif
  name += ".txt";
  file.open(dir + name, std::ios::app);

  if (!file)
  {
    std::cout << "Error in creating file!!!" << std::endl;
    return;
  }

  benchmarks(file);
}

#pragma region PerfectSolution

std::vector<int8_t> last, prev, head;
std::vector<int8_t> dist, Q, matching;
std::vector<bool> used, vis;

inline void init(int8_t _n1, int8_t _n2) {
  last.resize(_n1);
  std::fill(last.begin(), last.end(), -1);
  dist.resize(_n1);
  Q.resize(_n1);
  matching.resize(_n2);
  used.resize(_n1);
  vis.resize(_n1);
}

inline void addEdge(int8_t u, int8_t v) {
  head.push_back(v);
  prev.push_back(last[u]);
  last[u] = head.size() - 1;
}

inline void bfs() {
  std::fill(dist.begin(), dist.end(), -1);
  int sizeQ = 0;
  for (int u = 0; u < dist.size(); ++u) {
    if (!used[u]) {
      Q[sizeQ++] = u;
      dist[u] = 0;
    }
  }
  for (int i = 0; i < sizeQ; i++) {
    int u1 = Q[i];
    for (int e = last[u1]; e >= 0; e = prev[e]) {
      int u2 = matching[head[e]];
      if (u2 >= 0 && dist[u2] < 0) {
        dist[u2] = dist[u1] + 1;
        Q[sizeQ++] = u2;
      }
    }
  }
}

inline bool dfs(int8_t u1) {
  vis[u1] = true;
  for (int e = last[u1]; e >= 0; e = prev[e]) {
    int v = head[e];
    int u2 = matching[v];
    if (u2 < 0 || !vis[u2] && dist[u2] == dist[u1] + 1 && dfs(u2)) {
      matching[v] = u1;
      used[u1] = true;
      return true;
    }
  }
  return false;
}

inline int maxMatching() {
  std::fill(used.begin(), used.end(), false);
  std::fill(matching.begin(), matching.end(), -1);
  for (int res = 0;;) {
    bfs();
    std::fill(vis.begin(), vis.end(), false);
    int f = 0;
    for (int u = 0; u < vis.size(); ++u)
      if (!used[u] && dfs(u))
        ++f;
    if (!f)
      return res;
    res += f;
  }
}

void GeneratePerfectCounts() {
  std::cout << "Sliding Tile perfect counts: ";
  uint8_t tile_in_location[NUM_TILES];
  for (int i = 1; i <= 100; i++) {
    switch (NUM_TILES) {
    case 16: define_problems15(i, tile_in_location); break;
    case 25: define_problems24(i, tile_in_location); break;
    default: fprintf(stderr, "Illegal value of N_LOCATIONS in benchmarks\n"); exit(1); break;
    }

    SlidingTile::initialize(tile_in_location);
    SlidingTile goal_state = SlidingTile::GetSolvedPuzzle(Direction::backward);
    SlidingTile starting_state(tile_in_location, Direction::forward);

    uint32_t cstar;
    {
      std::unordered_set<SlidingTile, SlidingTileHash> closed_b, closed_f;
      {
        Astar backwardInstance;
        auto [cstar_b, expansions_b, memory_b] = backwardInstance.run_search(goal_state, starting_state);
        cstar = cstar_b;
        closed_b = backwardInstance.closed;
      }
      {
        Astar forwardInstance;
        auto [cstar_f, expansions_f, memory_f] = forwardInstance.run_search(starting_state, goal_state);
        closed_f = forwardInstance.closed;
        if (cstar_f != cstar) std::cout << "ERROR";
      }


      int bsize = closed_b.size();
      init(closed_f.size(), closed_b.size());
      int findex = 0;
      for (const auto& f : closed_f) {
        int bindex = 0;
        for (const auto& b : closed_b) {
          if (f.g + b.g + 1 <= cstar && f.f_bar + b.f_bar <= 2 * cstar && f.f + b.delta <= cstar && b.f + f.delta <= cstar) {
            addEdge(findex, bindex);
          }
          bindex += 1;
        }
        findex += 1;
      }
    }
    std::cout << maxMatching() << " ";
  }
}

#pragma endregion


int main(int ac, char** av)
{
  //unsigned char  tile_in_location[16] = {1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};   // Simple test problem.
  //unsigned char  tile_in_location[16] = {4, 2, 6, 1, 5, 0, 7, 3, 8, 9, 10, 11, 12, 13, 14, 15};   // Simple test problem.
  //unsigned char  tile_in_location[16] = { 14, 13, 15, 7, 11, 12, 9, 5, 6, 0, 2, 1, 4, 8, 10, 3 };   // Problem 1 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {13,  5,  4, 10,  9, 12,  8, 14,  2,  3,  7,  1,  0, 15, 11,  6};   // Problem 2 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = { 2, 11, 15, 5,  13,  4,  6,  7, 12,  8, 10,  1,  9,  3, 14,  0};   // Problem 7 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = { 5,  9, 13, 14,  6,  3,  7, 12, 10,  8,  4,  0, 15,  2, 11,  1};   // Problem 11 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {14,  1,  9,  6,  4,  8, 12,  5,  7,  2,  3,  0, 10, 11, 13, 15};   // Problem 12 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = { 3,  6,  5,  2, 10,  0, 15, 14,  1,  4, 13, 12,  9,  8, 11,  7 };  // Problem 13 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {15, 14,  0,  4, 11,  1,  6, 13,  7,  5,  8,  9,  3,  2, 10, 12};   // Problem 17 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {12,  8, 15, 13,  1,  0,  5,  4,  6,  3,  2, 11,  9,  7, 14, 10};   // Problem 31 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {14, 10,  2,  1, 13,  9,  8, 11,  7,  3,  6, 12, 15,  5,  4,	0};   // Problem 82 from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {14,  7,  8,  2, 13, 11, 10,  4,  9, 12,  5,  0,  3,  6,  1, 15};   // Problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {15,  2, 12, 11, 14, 13,  9,  5,  1,  3,  8,  7,  0, 10,  6,  4};   // Hardest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {};   // Problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {14, 1, 9, 6, 4, 8, 12, 5, 7, 2, 3, 0, 10, 11, 13, 15};   // Easiest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[16] = {15, 2, 12, 11, 14, 13, 9, 5, 1, 3, 8, 7, 0, 10, 6, 4};   // Hardest problem from Depth-First Iterative-Deepening: An Optimal Admissible Tree Search
  //unsigned char  tile_in_location[25] = {14,  5,  9,  2, 18,  8, 23, 19, 12, 17, 15,  0, 10, 20,  4,  6, 11, 21,  1,  7, 24,  3, 16, 22, 13};   // First problem from Disjoint Pattern Database Heuristics
  //unsigned char  tile_in_location[25] = {16,  5,  1, 12,  6, 24, 17,  9,  2, 22,  4, 10, 13, 18, 19, 20,  0, 23,  7, 21, 15, 11,  8,  3, 14};   // Second problem from Disjoint Pattern Database Heuristics
  //unsigned char  tile_in_location[25] = { 3, 17,  9,  8, 24,  1, 11, 12, 14,  0,  5,  4, 22, 13, 16, 21, 15,  6,  7, 10, 20, 23,  2, 18, 19};   // Problem 25 from Disjoint Pattern Database Heuristics
  //unsigned char  tile_in_location[25] = { 1, 12, 18, 13, 17, 15,  3,  7, 20,  0, 19, 24,  6,  5, 21, 11,  2,  8,  9, 16, 22, 10,  4, 23, 14};   // Problem 32 from Disjoint Pattern Database Heuristics
  //unsigned char  tile_in_location[25] = {10,  3, 24, 12,  0,  7,  8, 11, 14, 21, 22, 23,  2,  1,  9, 17, 18,  6, 20,  4, 13, 15,  5, 19, 16};   // Easiest problem from Disjoint Pattern Database Heuristics
  //unsigned char  tile_in_location[25] = {23,  1, 12,  6, 16,  2, 20, 10, 21, 18, 14, 13, 17, 19, 22,  0, 15, 24,  3,  7,  4,  8,  5,  9, 11};   // Hardest problem from Disjoint Pattern Database Heuristics
  //GeneratePerfectCounts();
  run_test();
}

