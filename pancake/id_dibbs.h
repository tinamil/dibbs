#pragma once
#include <unordered_set>
#include <concurrent_unordered_set.h>
#include <limits>
#include <shared_mutex>
#include <atomic>

namespace search
{
  std::pair<uint64_t, double> multithreaded_id_dibbs(const uint8_t* start_state, const unsigned int n, const unsigned int gap);
}
