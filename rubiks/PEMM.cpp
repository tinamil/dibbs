//
//  MM.cpp
//  hog2 glut
//
//  Created by Nathan Sturtevant on 8/31/15.
//  Copyright (c) 2015 University of Denver. All rights reserved.
//

#include "PEMM.h"
#include <string>
#include <unordered_set>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <mutex>
#include <atomic>
#include "node.h"
#include <cstdint>
#include<filesystem>

#include "tsl/hopscotch_set.h"

namespace Nathan {
  const uint64_t bucket_bits = 5;
  const int bucket_mask = ((1 << bucket_bits) - 1);//0x1F;


  const char* prefix1;
  const char* prefix2;

  bool finished;

  int bestSolution;
  int currentC;
  int minGForward;
  int minGBackward;
  int minFForward;
  int minFBackward;
  uint64_t expanded;
  std::vector<uint64_t> gDistForward;
  std::vector<uint64_t> gDistBackward;
  std::mutex printLock;
  std::mutex countLock;
  std::mutex openLock;

  Rubiks::PDB pdb_type;
  Node startState, goalState;
  //unsigned __int128 i;
  //typedef RubiksState diskState;
  typedef uint64_t diskState;
#define MY_SET

  typedef tsl::hopscotch_set<diskState> bucketSet;


  void GetBucketAndData(const Node& s, int& bucket, diskState& data);
  void GetState(uint8_t* s, int bucket, diskState data);
  int GetBucket(const Node& s);

  const uint8_t* cube;

  enum  tSearchDirection {
    kForward,
    kBackward,
  };


  // This tells us the open list buckets
  struct openData {
    tSearchDirection dir;  // at most 2 bits
    uint8_t priority;      // at most 6 bits
    uint8_t gcost;         // at most 4 bits
    uint16_t bucket;        // at most (3) bits
  };

  static bool operator==(const openData& a, const openData& b)
  {
    return (a.dir == b.dir && a.priority == b.priority && a.gcost == b.gcost && a.bucket == b.bucket);
  }

  static std::ostream& operator<<(std::ostream& out, const openData& d)
  {
    out << "[" << ((d.dir == kForward) ? "forward" : "backward") << ", p:" << +d.priority << ", g:" << +d.gcost;
    out << ", b:" << +d.bucket << "]";
    return out;
  }

  struct openDataHash
  {
    std::size_t operator()(const openData& x) const
    {
      return (x.dir) | (x.priority << 2) | (x.gcost << 8) | (x.bucket << 20);
    }
  };

  struct openList {
    openList() :f(0), writtenStates(0), minf(0xFF) {}
    FILE* f;
    uint64_t writtenStates;
    uint8_t minf;
  };

  static std::ostream& operator<<(std::ostream& out, const openList& d)
  {
    out << "[" << d.writtenStates << ", " << +d.minf << "]";
    return out;
  }

  struct closedData {
    tSearchDirection dir;
    uint8_t depth;
    uint8_t bucket;
  };

  struct closedList {
    closedList() :f(0) {}
    FILE* f;
  };

  static bool operator==(const closedData& a, const closedData& b)
  {
    return (a.dir == b.dir && a.depth == b.depth && a.bucket == b.bucket);
  }

  struct closedDataHash
  {
    std::size_t operator()(const closedData& x) const
    {
      return (x.dir) | (x.bucket << 2) | (x.bucket << 7);
    }
  };

  std::unordered_map<closedData, closedList, closedDataHash> closed;
  std::unordered_map<openData, openList, openDataHash> open;

  std::string GetClosedName(closedData d)
  {
    std::string s;
    if (d.dir == kForward)
    {
      if ((d.bucket % 16) < 8)
      {
        s += prefix1;
      }
      else {
        s += prefix2;
      }
      s += "forward-";
    }
    else {
      if ((d.bucket % 16) >= 8)
      {
        s += prefix1;
      }
      else {
        s += prefix2;
      }
      s += "backward-";
    }
    s += std::to_string(d.bucket);
    s += "-";
    s += std::to_string(d.depth);
    s += ".closed";
    return s;
  }

  std::string GetOpenName(const openData& d)
  {
    /* Previously we split across disks according to forward/backward, but
     * this is unbalanced. Now we split according to buckets. With an even
     * number of buckets the files should now be even split across two disks.
     */
    std::string s;
    if ((d.bucket % 16) < 8)
    {
      s += prefix1;
    }
    else {
      s += prefix2;
    }
    if (d.dir == kForward)
    {
      s += "forward-";
    }
    else {
      s += "backward-";
    }
    s += std::to_string(d.priority);
    s += "-";
    s += std::to_string(d.gcost);
    s += "-";
    s += std::to_string(d.bucket);
    s += ".open";
    return s;
  }

  openData GetBestFile()
  {
    minGForward = 100;
    minGBackward = 100;
    minFForward = 100;
    minFBackward = 100;
    // actually do priority here
    openData best = (open.begin())->first;
    //return (open.begin())->first;
    for (const auto& s : open)
    {
      //std::cout << "--: " << s.first << " minf: " << +s.second.minf << "\n";
      if (s.first.dir == kForward && s.first.gcost < minGForward)
        minGForward = s.first.gcost;
      else if (s.first.dir == kBackward && s.first.gcost < minGBackward)
        minGBackward = s.first.gcost;

      if (s.first.dir == kForward && s.second.minf < minFForward)
        minFForward = s.second.minf;
      else if (s.first.dir == kBackward && s.second.minf < minFBackward)
        minFBackward = s.second.minf;

      if (s.first.priority < best.priority)
      {
        best = s.first;
      }
      else if (s.first.priority == best.priority)
      {
        if (s.first.gcost < best.gcost)
          best = s.first;
        else if (s.first.gcost == best.gcost)
        {
          if (s.first.dir == best.dir)
          {
            //					if (s.first.fcost < best.fcost)
            //						best = s.first;
            //					else if (s.first.fcost == best.fcost)
            //					{
            if (s.first.bucket < best.bucket)
              best = s.first;
            //					}
          }
          else if (s.first.dir == kForward)
            best = s.first;
        }
      }
    }
    //printf("Min f forward: %d backward: %d\n", minFForward, minFBackward);
    return best;
  }

  void GetOpenData(const Node& from, tSearchDirection dir, int cost,
    openData& d, diskState& data, uint8_t& fcost)
  {
    int bucket;
    GetBucketAndData(from, bucket, data);
    d.dir = dir;
    d.gcost = cost;
    fcost = from.combined;
    d.bucket = bucket;
    d.priority = std::max(fcost, uint8_t(d.gcost * 2));
  }

  void AddStatesToQueue(const openData& d, diskState* data, size_t count, uint8_t minFcost)
  {
    openLock.lock();
    auto iter = open.find(d);
    if (iter == open.end())
    {
      open[d].f = fopen(GetOpenName(d).c_str(), "w+b");
      if (open[d].f == 0)
      {
        printf("Error opening %s; Aborting!\n", GetOpenName(d).c_str());
        perror("Reason: ");
        exit(0);
      }
      iter = open.find(d);
      iter->second.minf = 100; // new file, start with high f-min
    }
    if (iter->second.f == 0)
    {
      printf("Error - file is null!\n");
      exit(0);
    }
    //iter->second.minf = std::min(iter->second.minf, d.fcost);
    size_t written = fwrite(data, sizeof(diskState), count, open[d].f);
    open[d].writtenStates += written;
    open[d].minf = std::min(open[d].minf, minFcost);
    openLock.unlock();
  }

  void AddStateToQueue(openData& d, diskState data, uint8_t fcost)
  {
    AddStatesToQueue(d, &data, 1, fcost);
  }

  void AddStateToQueue(const Node& start, tSearchDirection dir, int cost)
  {
    openData d;
    diskState rank;
    uint8_t fcost;
    GetOpenData(start, dir, cost, d, rank, fcost);
    AddStateToQueue(d, rank, fcost);
  }


  bool CanTerminateSearch()
  {
    int val;
    if (bestSolution <= (val = std::max(currentC, std::max(minFForward, std::max(minFBackward, minGBackward + minGForward)))))
    {
      printf("Done!\n");
      printf("Min fforward %d; minfbackward: %d; g: %d+%d; currentC: %d; solution: %d\n",
        minFForward, minFBackward, minGBackward, minGForward, currentC, bestSolution);
      printf("%llu nodes expanded\n", expanded);
      printf("Forward Distribution:\n");
      for (int x = 0; x < gDistForward.size(); x++)
        if (gDistForward[x] != 0)
          printf("%d\t%llu\n", x, gDistForward[x]);
      printf("Backward Distribution:\n");
      for (int x = 0; x < gDistBackward.size(); x++)
        if (gDistBackward[x] != 0)
          printf("%d\t%llu\n", x, gDistBackward[x]);
      finished = true;
      if (val == currentC)
        printf("-Triggered by current priority\n");
      if (val == minFForward)
        printf("-Triggered by f in the forward direction\n");
      if (val == minFBackward)
        printf("-Triggered by f in the backward direction\n");
      if (val == minGBackward + minGForward)
        printf("-Triggered by gforward+gbackward\n");
      return true;
    }
    return false;
  }

  void FindSolutionThread(const openData& d, const openList& l, int g, const bucketSet& states)
  {
    const size_t bufferSize = 128;
    diskState buffer[bufferSize];
    if (l.f == 0)
    {
      std::cout << "Error opening " << d << "\n";
      exit(0);
    }
    rewind(l.f);
    size_t numRead;
    do {
      numRead = fread(buffer, sizeof(diskState), bufferSize, l.f);
      for (int x = 0; x < numRead; x++)
      {
        if (states.find(buffer[x]) != states.end())
        {
          printLock.lock();
          printf("\nFound solution cost %d+%d=%d\n", d.gcost, g, d.gcost + g);
          bestSolution = std::min(d.gcost + g, bestSolution);
          printf("Current best solution: %d\n", bestSolution);
          printLock.unlock();

          if (CanTerminateSearch())
            return;
        }
      }
    } while (numRead == bufferSize);
  }

  void CheckSolution(std::unordered_map<openData, openList, openDataHash> currentOpen, openData d,
    const bucketSet& states)
  {
    //std::vector<std::thread*> threads;
    for (const auto& s : currentOpen)
    {
      // Opposite direction, same bucket AND could be a solution (g+g >= C)
      // TODO: only need to check if we find a better solution (g+g < U)
      if (s.first.dir != d.dir && s.first.bucket == d.bucket &&
        d.gcost + s.first.gcost >= currentC && d.gcost + s.first.gcost < bestSolution)// && d.hcost2 == s.first.hcost)
      {
        //			std::thread *t = new std::thread(FindSolutionThread, s.first, s.second, states);
        //			threads.push_back(t);
        FindSolutionThread(s.first, s.second, d.gcost, states);
      }
    }
    //	while (threads.size() > 0)
    //	{
    //		threads.back()->join();
    //		delete threads.back();
    //		threads.pop_back();
    //	}
  }

  void ReadBucket(bucketSet& states, openData d)
  {
    const size_t bufferSize = 128;
    diskState buffer[bufferSize];
    rewind(open[d].f);

    size_t numRead;
    do {
      numRead = fread(buffer, sizeof(diskState), bufferSize, open[d].f);
      for (int x = 0; x < numRead; x++)
        states.insert(buffer[x]);
    } while (numRead == bufferSize);
    fclose(open[d].f);
    remove(GetOpenName(d).c_str());
    open[d].f = 0;
  }

  void RemoveDuplicates(bucketSet& states, openData d)
  {
    for (int depth = d.gcost - 2; depth < d.gcost; depth++)
    {
      closedData c;
      c.bucket = d.bucket;
      c.depth = depth;
      c.dir = d.dir;

      closedList& cd = closed[c];
      if (cd.f == 0)
        continue;
      rewind(cd.f);

      const size_t bufferSize = 1024;
      diskState buffer[bufferSize];
      rewind(cd.f);
      size_t numRead;
      do {
        numRead = fread(buffer, sizeof(diskState), bufferSize, cd.f);
        for (int x = 0; x < numRead; x++)
        {
          auto i = states.find(buffer[x]);
          if (i != states.end())
            states.erase(i);
        }
      } while (numRead == bufferSize);
    }
  }

  void WriteToClosed(bucketSet& states, openData d)
  {
    closedData c;
    c.bucket = d.bucket;
    c.depth = d.gcost;
    c.dir = d.dir;

    closedList& cd = closed[c];
    if (cd.f == 0)
    {
      cd.f = fopen(GetClosedName(c).c_str(), "w+b");
    }
    for (const auto& i : states)
    {
      fwrite(&i, sizeof(diskState), 1, cd.f);
    }
  }

  void ParallelExpandBucket(openData d, bucketSet& states, int myThread, int totalThreads, bool reverse)
  {
    const int cacheSize = 1024;
    // stores the min f for this bucket, the files to write
    std::unordered_map<openData, std::pair<uint8_t, std::vector<diskState>>, openDataHash> cache;
    uint8_t tmp[40];
    uint64_t localExpanded = 0;
    int count = 0;
    for (const auto& values : states)
    {
      count++;
      if (myThread != count % totalThreads)
        continue;
      diskState v = values;
      localExpanded++;
      GetState(tmp, d.bucket, v);
      for (int face = 0; face < 6; ++face)
      {
        for (int rotation = 0; rotation < 3; ++rotation)
        {
          Node new_node;
          memcpy(new_node.state, tmp, 40);
          Rubiks::rotate(new_node.state, face, rotation);
          if (reverse == false) {
            new_node.heuristic = Rubiks::pattern_lookup(new_node.state, pdb_type);
          }
          else {
            new_node.heuristic = Rubiks::pattern_lookup(new_node.state, startState.state, pdb_type);
          }
          new_node.depth = d.gcost + 1;
          new_node.face = face;
          new_node.combined = d.gcost + 1 + new_node.heuristic;


          openData newData;
          diskState newRank;
          uint8_t fcost;
          GetOpenData(new_node, d.dir, d.gcost + 1, newData, newRank, fcost);

          auto& i = cache[newData];
          std::vector<diskState>& c = i.second;
          if (i.first == 0)
            i.first = fcost;
          else
            i.first = std::min(i.first, fcost);
          c.push_back(newRank);
          if (c.size() > cacheSize)
          {
            AddStatesToQueue(newData, &c[0], c.size(), i.first);
            c.clear();
          }
        }
      }
    }
    for (auto& i : cache)
    {
      if (i.second.second.size() > 0)
        AddStatesToQueue(i.first, &(i.second.second[0]), i.second.second.size(), i.second.first);
    }

    countLock.lock();
    expanded += localExpanded;
    if (d.dir == kForward)
    {
      gDistForward[d.gcost] += localExpanded;
    }
    else {
      gDistBackward[d.gcost] += localExpanded;
    }

    countLock.unlock();
  }

  void ReadAndDDBucket(bucketSet& states, const openData& d)
  {
    ReadBucket(states, d);
    RemoveDuplicates(states, d); // delayed duplicate detection
    WriteToClosed(states, d); // this could run in parallel! (it is if we pre-load)
  }

#define DO_PRELOAD

  void ExpandNextFile(bool reset = false)
  {
    static bool preLoaded = false;
    static bucketSet nextStates(0);
    static bucketSet states(0);
    static openData next;

    if (reset) {
      preLoaded = false;
      nextStates.clear();
      states.clear();
      next.bucket = 0;
      next.dir = kForward;
      next.gcost = 0;
      next.priority = 0;
      return;
    }

    // 1. Get next expansion target
    openData d = GetBestFile();
    currentC = d.priority;

    if (CanTerminateSearch())
      return;

    states.clear();
    if (preLoaded == false)
    {
      ReadAndDDBucket(states, d);
    }
    else {
      if (d == next)
      {
        states.swap(nextStates);
        preLoaded = false;
      }
      else {
        std::cout << "ERROR: pre-loading changed buckets!\n";
        ReadAndDDBucket(states, d);
      }
    }

    //printLock.lock();
    //std::cout << "Next: " << d << " (" << states.size() << " entries " << open.find(d)->second.writtenStates << ") ";
    //printLock.unlock();
    open.erase(open.find(d));

    // Read in opposite buckets to check for solutions in parallel to expanding this bucket
    openLock.lock();
    std::thread t(CheckSolution, open, d, std::ref(states));
    openLock.unlock();

    std::thread* pre = 0;

#ifdef DO_PRELOAD
    // 2. Pre-read next bucket if it is the same as us
    next = GetBestFile();
    if (next.dir == d.dir && next.gcost == d.gcost && next.bucket != d.bucket)
    {
      pre = new std::thread(ReadAndDDBucket, std::ref(nextStates), std::ref(next));
      preLoaded = true;
    }
#endif

    // 3. expand all states in current bucket & write out successors
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread*> threads;
    for (int x = 0; x < numThreads; x++)
      threads.push_back(new std::thread(ParallelExpandBucket, d, std::ref(states), x, numThreads, d.dir == kBackward));
    for (int x = 0; x < threads.size(); x++)
    {
      threads[x]->join();
      delete threads[x];
    }

    // Close thread that is doing DSD
    t.join();

    // Close thread that is reading previous bucket
    if (pre != 0)
    {
      pre->join();
      delete pre;
      pre = 0;
    }
    //printLock.lock();
    //std::cout << "\n";
    //printLock.unlock();
  }

  int GetBucket(const Node& s)
  {
    const uint64_t corner_hash = Rubiks::get_corner_index(s.state);
    return bucket_mask & corner_hash;
  }

  void GetBucketAndData(const Node& s, int& bucket, diskState& data)
  {
    const uint64_t edge_hash = Rubiks::get_edge_index12(s.state);
    const uint64_t corner_hash = Rubiks::get_corner_index(s.state);
    const size_t hash = bucket_mask & corner_hash;
    bucket = bucket_mask & corner_hash;
    data = (corner_hash >> bucket_bits) * 1961990553600ui64 + edge_hash;
  }

  void GetBucketAndData(const Node& s, int& bucket, Node& data)
  {
    bucket = GetBucket(s);
    data = s;
  }

  void GetState(uint8_t* s, int bucket, diskState data)
  {
    Rubiks::restore_index12(data % 1961990553600ui64, s);
    Rubiks::restore_corner(bucket | ((data / 1961990553600ui64) << bucket_bits), s);
  }

  void GetState(Node& s, int bucket, Node data)
  {
    s = data;
  }

  std::tuple<uint64_t, double, size_t> pemm(const uint8_t* start_state, const Rubiks::PDB pdb)
  {
    if (Rubiks::is_solved(start_state))
    {
      std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
      return std::make_tuple(0, 0, 0);
    }
    finished = false;
    pdb_type = pdb;
    startState = Node(start_state, Rubiks::__goal, pdb_type);
    goalState = Node(Rubiks::__goal, start_state, pdb_type);
    prefix1 = "forward/";
    prefix2 = "backward/";

    for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
      std::filesystem::remove(entry);
    }
    for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
      std::filesystem::remove(entry);
    }

    bestSolution = std::numeric_limits<int>::max();
    gDistBackward.clear();
    gDistForward.clear();
    gDistBackward.resize(12);
    gDistForward.resize(12);
    expanded = 0;

    currentC = 0;
    minGForward = 0;
    minGBackward = 0;
    minFForward = 0;
    minFBackward = 0;

    ExpandNextFile(true);

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout << std::setprecision(2);

    auto c_start = clock();

    std::printf("---MM*---\n");
    AddStateToQueue(startState, kForward, 0);
    AddStateToQueue(goalState, kBackward, 0);
    while (!open.empty() && !finished)
    {
      ExpandNextFile();
    }
    auto c_end = clock();
    auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;

    for (const auto& fs : open) {
      if (fs.second.f != 0)
        fclose(fs.second.f);
    }
    for (const auto& fs : closed) {
      if (fs.second.f != 0) {
        fclose(fs.second.f);
      }
    }

    size_t total_size = 0;
    for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
      total_size += std::filesystem::file_size(entry);
    }
    for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
      total_size += std::filesystem::file_size(entry);
    }

    for (const auto& entry : std::filesystem::directory_iterator(prefix1)) {
      std::filesystem::remove(entry);
    }
    for (const auto& entry : std::filesystem::directory_iterator(prefix2)) {
      std::filesystem::remove(entry);
    }
    closed.clear();
    open.clear();
    return std::make_tuple(expanded, time_elapsed, total_size);
  }
}