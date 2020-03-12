#pragma once
#include <unordered_set>
#include <concurrent_unordered_set.h>
#include <limits>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include "rubiks.h"
#include "node.h"
#include "thread_safe_stack.hpp"
#include "DiskHash.hpp"

#include "tsl/hopscotch_set.h"

namespace search
{

  typedef std::stack<Node> stack;
  typedef thread_safe_stack<Node> tstack;
  typedef concurrency::concurrent_unordered_set<Node, NodeHash, NodeEqual> concurrent_set;
  typedef tsl::hopscotch_set<uint64_t> concurrent_int_set;
  typedef DiskHash<uint64_t> disk_set;

  template <class a, class b>
  void move_nodes(a& origin, b& destination) {
    std::vector<Node> list;
    while (!origin.empty()) {
      list.push_back(origin.top());
      origin.pop();
    }
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      destination.push((list[i]));
    }
    return;
  }

  std::pair<uint64_t, double> multithreaded_id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  std::tuple<uint64_t, double, size_t> multithreaded_compressed_id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  std::pair<uint64_t, double> multithreaded_disk_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  std::pair<uint8_t, uint64_t> solve_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type);


  bool reached_depth_limit(unsigned int iteration, const Rubiks::PDB pdb_type);
}
