#pragma once
#include <stdint.h>
#include <stack>
#include <queue>
#include "node.h"
#include "rubiks.h"

namespace search
{
  void a_star(const uint8_t state[], const Rubiks::PDB pdb_type);
}
