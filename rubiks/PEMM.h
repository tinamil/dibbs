#pragma once

#include "rubiks.h"

namespace Nathan
{
  std::tuple<uint64_t, double, size_t> pemm(const uint8_t* state, const Rubiks::PDB pdb_type);
}