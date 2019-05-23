#include <iostream>
#include <ctime>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"
#include "rubiks_loader.h"
#include "gbfhs.h"
#include "id_gbfhs.h"
#include "id_dibbs.h"

using namespace std;
using namespace Rubiks;

//#define ASTAR 
//#define DIBBS 
#define GBFHS

void search_cubes()
{
  vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  PDB type = PDB::a12;
  vector<uint64_t> count_results;
  vector<int64_t> time_results;

  clock_t c_start, c_end;
  int64_t time_elapsed_ms;
  for (size_t i = 0; i < cubes.size(); ++i)
  {
    #ifdef ASTAR
    c_start = clock();
    count_results.push_back(search::a_star(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "IDA* CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
    #ifdef DIBBS
    c_start = clock();
    count_results.push_back(search::id_dibbs(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "DIBBS CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
    #ifdef GBFHS
    c_start = clock();
    count_results.push_back(search::id_gbfhs(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "GBFHS CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
  }

  for (size_t i = 0; i < count_results.size(); ++i) {
    std::cout << i << " " << count_results[i] << " nodes expanded; " << time_results[i] << " s to solve\n";
  }
}

int main()
{
  std::cout << "Size of Node=" << sizeof(Node) << std::endl;
  search_cubes();
  return 0;
}
