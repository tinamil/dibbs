#pragma once
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <set>
#include <queue>
#include "node.h"
#include "rubiks.h"

namespace search
{
  size_t dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  size_t id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
}
