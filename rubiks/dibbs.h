#pragma once
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <set>
#include <queue>
#include <functional>
#include "node.h"
#include "rubiks.h"

namespace search
{
  std::pair<uint64_t, double> dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
}
