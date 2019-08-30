#pragma once
#include <unordered_set>
#include <concurrent_unordered_set.h>
#include <limits>
#include <shared_mutex>
#include <atomic>
#include "rubiks.h"
#include "node.h"
#include "thread_safe_stack.hpp"
#include "DiskHash.hpp"

namespace search
{

#ifndef HISTORY

  typedef std::stack<Node> stack;
  typedef thread_safe_stack<Node> tstack;
  typedef concurrency::concurrent_unordered_set<Node, NodeHash, NodeEqual> concurrent_set;
  typedef DiskHash<Node, NodeHash, NodeEqual> disk_set;

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

#else

  typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node> > > stack;
  typedef thread_safe_stack<std::shared_ptr<Node> > tstack;
  typedef concurrency::concurrent_unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;

  template <class a, class b>
  void move_nodes(a& origin, b& destination) {
    std::vector<std::shared_ptr<Node>> list;
    while (!origin.empty()) {
      list.push_back(std::move(origin.top()));
      origin.pop();
    }
    for (int i = (int)list.size() - 1; i >= 0; --i) {
      destination.push(std::move(list[i]));
    }
    return;
  }

#endif

  std::pair<uint64_t, double> multithreaded_id_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  std::pair<uint64_t, double> multithreaded_disk_dibbs(const uint8_t* state, const Rubiks::PDB pdb_type);
  std::pair<uint8_t, uint64_t> solve_disk_dibbs(const uint8_t* start_state, const Rubiks::PDB pdb_type, unsigned int iteration, uint8_t upper_bound);
}
