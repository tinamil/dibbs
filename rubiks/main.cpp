#include <iostream>
#include <ctime>
#include <iomanip>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"
#include "rubiks_loader.h"
#include "gbfhs.h"
#include "id_gbfhs.h"
#include "id_dibbs.h"
#include "utility.h"
#include "multithreaded_id_dibbs.h"
#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include "DiskHash.hpp"
#include "PEMM.h"
using namespace std;
using namespace Rubiks;

//#define ASTAR 
#define DIBBS 
//#define DISK_DIBBS
//#define COMPRESSED_DIBBS 
//#define GBFHS
//#define PEMM

void search_cubes()
{
  vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  PDB type = PDB::a888;
  vector<uint64_t> count_results;
  vector<double> time_results;

  for (size_t i = 0; i < cubes.size(); ++i)
  {
    if (i != 10) continue;
    //Trigger PDB generation before beginning search to remove from timing
    Rubiks::pattern_lookup(Rubiks::__goal, type);
    Rubiks::pattern_lookup(__goal, cubes[i], type);
    PROCESS_MEMORY_COUNTERS memCounter;
    BOOL res = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
    if (res) {
      cout << memCounter.PeakPagefileUsage / 1024.0 / 1024 / 1024 << "GB\n";
    }
    break;
#ifdef ASTAR
    {
      auto [count, time_elapsed] = search::multithreaded_ida_star(cubes[i], type, false);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "IDA* CPU time used: " << time_elapsed << " s" << endl;
    }
#endif
#ifdef DIBBS 
    {
      auto [count, time_elapsed] = search::multithreaded_id_dibbs(cubes[i], type);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "DIBBS CPU time used: " << time_elapsed << " s" << endl;
      PROCESS_MEMORY_COUNTERS memCounter;
      BOOL res = GetProcessMemoryInfo(GetCurrentProcess(), &memCounter, sizeof(memCounter));
      if (res) {
        cout << memCounter.PeakPagefileUsage / 1024.0 / 1024 / 1024 << "GB\n";
      }
    }
#endif
#ifdef COMPRESSED_DIBBS 
    {
      auto [count, time_elapsed, memPeak] = search::multithreaded_compressed_id_dibbs(cubes[i], type);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "COMPRESED DIBBS CPU time used: " << time_elapsed << " s" << endl;
      cout << (memPeak / 1024.0 / 1024 / 1024) << "GB\n";
    }
#endif
#ifdef DISK_DIBBS
    {
      auto [count, time_elapsed] = search::multithreaded_disk_dibbs(cubes[i], type);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "IDD-DISK CPU time used: " << time_elapsed << " s" << endl;
    }
#endif
#ifdef PEMM
    {
      auto [count, time_elapsed, total_size] = Nathan::pemm(cubes[i], type);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "PEMM CPU time used: " << time_elapsed << " s" << endl;
      cout << (total_size / (double)(1ui64 << 30)) << "GB\n";
    }
#endif
#ifdef GBFHS
    {
      auto [count, time_elapsed] = search::id_gbfhs(cubes[i], type);
      count_results.push_back(count);
      time_results.push_back(time_elapsed);
      cout << "ID-GBFHS CPU time used: " << time_elapsed << " s" << endl;
    }
#endif
    Rubiks::pattern_lookup(nullptr, cubes[i], Rubiks::PDB::clear_state);
  }

  for (size_t i = 0; i < count_results.size(); ++i) {
    std::cout << i << " " << count_results[i] << " nodes expanded; " << time_results[i] << " s to solve\n";
  }

  for each (auto state in cubes)
  {
    delete[] state;
  }
}

int main()
{
  _setmaxstdio(8192);
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
