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
  PDB type = PDB::a12;
  vector<uint64_t> count_results;
  vector<long> time_results;

  for (size_t i = 0; i < cubes.size(); ++i)
  {
    clock_t c_start = clock();
    count_results.push_back(search::a_star(cubes[i], type));
    clock_t c_end = clock();
    auto time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "IDA* CPU time used: " << time_elapsed_ms << " s" << endl;
    c_start = clock();
    //search::id_dibbs(cubes[i], type);
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    //cout << "DIBBS CPU time used: " << time_elapsed_ms << " s" << endl;
  }

  for (size_t i = 0; i < count_results.size(); ++i) {
    std::cout << i << " " << count_results[i] << " nodes expanded; " << time_results[i] << " s to solve\n";
  }
}

int main()
{
  std::cout << "Size of " << sizeof(Node) << std::endl;
  search_cubes();
  return 0;
}
