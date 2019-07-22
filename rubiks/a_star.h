#pragma once
#include "rubiks.h"

namespace search
{
  uint64_t ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
  uint64_t multithreaded_ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, const bool reverse);
}
