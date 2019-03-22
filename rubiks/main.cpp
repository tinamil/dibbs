#include <iostream>
#include <ctime>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"
#include "rubiks_loader.h"

using namespace std;
using namespace Rubiks;

void search_cubes()
{
  vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  PDB type = PDB::a1997;

  for (size_t i = 0; i < cubes.size(); ++i)
  {
    clock_t c_start = clock();
    search::a_star(cubes[i], type);
    clock_t c_end = clock();
    auto time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "IDA* CPU time used: " << time_elapsed_ms << " s" << endl;
    c_start = clock();
    search::dibbs(cubes[i], type);
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    cout << "DIBBS CPU time used: " << time_elapsed_ms << " s" << endl;
  }
}

int main()
{
  search_cubes();
  //system("PAUSE");
  return 0;
}
