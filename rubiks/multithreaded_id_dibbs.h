#pragma once
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <set>
#include <queue>
#include <functional>
#include <algorithm>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include "node.h"
#include "rubiks.h"
#include "thread_safe_stack.hpp"

namespace search
{
  size_t multithreaded_id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
}
