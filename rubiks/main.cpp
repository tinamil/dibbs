#include <iostream>
#include <ctime>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"
#include "rubiks_loader.h"
#include "gbfhs.h"
#include "id_gbfhs.h"
#include "id_dibbs.h"
#include "utility.h"

using namespace std;
using namespace Rubiks;

#define ASTAR 
#define DIBBS 
//#define GBFHS

void search_cubes()
{
  vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  PDB type = PDB::a888;
  vector<uint64_t> count_results;
  vector<int64_t> time_results;

  clock_t c_start, c_end;
  int64_t time_elapsed_ms;
  for (size_t i = 0; i < cubes.size(); ++i)
  {
    //Trigger PDB generation before beginning search to remove from timing
    Rubiks::pattern_lookup(Rubiks::__goal, type);
    Rubiks::pattern_lookup(__goal, cubes[i], type);

    #ifdef ASTAR
    c_start = clock();
    count_results.push_back(search::ida_star(cubes[i], type));
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
    Rubiks::pattern_lookup(nullptr, cubes[i], Rubiks::PDB::clear_state);
  }

  for (size_t i = 0; i < count_results.size(); ++i) {
    std::cout << i << " " << count_results[i] << " nodes expanded; " << time_results[i] << " s to solve\n";
  }
}

int main()
{
  std::cout << "Size of Node=" << sizeof(Node) << std::endl;
  uint8_t* test_state = RubiksLoader::scramble("B2 L' F' U2 R2 D  R2 B  U2 R B' L  R  F F' R' L' B R' U2 B' R2 D' R2 U2 F L B2");
  for (int i = 0; i < 40; ++i) {
    if (Rubiks::__goal[i] != test_state[i]) {
      throw std::exception("Failed to rotate and counter-rotate properly.");
    }
  }
  delete[] test_state;
  search_cubes();
  return 0;
}
