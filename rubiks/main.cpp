#include <iostream>
#include <ctime>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"

using namespace std;

int main()
{
  const uint8_t start_state[] =
  {
    2,  2,  8,  1, 17, 1,  9,  0, 15,  0,  7,  1, 18,  1, 14,  0,  3,  0, 13,  1,  1,  0, 10,  0,
    12,  0,  6,  1,  5,  1,  4,  1, 11,  0,  0,  2, 16,  1, 19,  1
  };
  std::clock_t c_start = std::clock();
  //search::a_star (start_state);
  //search::dibbs (start_state);
  Rubiks::generate_all_dbs();

  std::clock_t c_end = std::clock();

  auto time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
  std::cout << "CPU time used: " << time_elapsed_ms << " s" << std::endl;
  return 0;
}
