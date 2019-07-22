#pragma once
#include "rubiks.h"

namespace search
{
  size_t multithreaded_id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
}
