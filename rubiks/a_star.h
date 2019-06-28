#pragma once
#include <stdint.h>
#include <stack>
#include <queue>
#include <atomic>
#include "node.h"
#include "rubiks.h"
#include "thread_safe_stack.hpp"

namespace search
{
  uint64_t ida_star(const uint8_t* start_state, const Rubiks::PDB pdb_type, bool reverse);
  uint64_t multithreaded_ida_star(const uint8_t* state, const Rubiks::PDB pdb_type);
}
