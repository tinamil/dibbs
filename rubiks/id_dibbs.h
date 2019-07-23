#pragma once
#include "rubiks.h"

namespace search
{
  std::pair<uint64_t, double> id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
}
