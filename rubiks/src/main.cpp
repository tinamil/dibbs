#include <iostream>
#include "a_star.h"

using namespace std;

int main()
{
  const uint8_t start_state[] =
  {
    2,  2,  8,  1, 17, 1,  9,  0, 15,  0,  7,  1, 18,  1, 14,  0,  3,  0, 13,  1,  1,  0, 10,  0,
    12,  0,  6,  1,  5,  1,  4,  1, 11,  0,  0,  2, 16,  1, 19,  1
  };

  search::a_star (start_state);
  return 0;
}
