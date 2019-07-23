#pragma once
#include "rubiks.h"

namespace search
{
  std::pair<uint64_t, double> ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
  std::pair<uint64_t, double> multithreaded_ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
}
