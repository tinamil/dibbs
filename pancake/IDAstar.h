#pragma once

#include "Pancake.h"

class IDAstar
{

private:
  IDAstar() {}
  std::pair<double, size_t> run_search(Pancake start, Pancake goal);

public:
  static inline std::pair<double, size_t> search(Pancake start, Pancake goal) {
    IDAstar instance;
    return instance.run_search(start, goal);
  }
  //std::pair<uint64_t, double> multithreaded_ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
};
