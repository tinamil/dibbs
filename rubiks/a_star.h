#pragma once
#include <stdint.h>
#include <stack>
#include <queue>
#include "node.h"
#include "rubiks.h"
#include "thread_safe_stack.hpp"

namespace search
{
  uint64_t a_star(const uint8_t* state, const Rubiks::PDB pdb_type);
}
