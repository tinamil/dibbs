#include "road.h"
#include "node.h"
#include "astar.h"
//#include "dibbs.h"
//#include "dvcbs.h"
#include "gbfhs.h"
//#include "idd.h"
//#include "cbbs.h"
//#include "ffgbs.h"
#include <iostream>
#include <iomanip> 
#include <chrono>
#include <random>
#include <functional>

constexpr size_t NUM_PROBLEMS = 1000;
constexpr Type MAP_TYPE = Type::NY;

constexpr uint32_t SEED = 0;
//32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
std::mt19937 rng_engine(SEED);
std::vector<std::string> algorithm_names;
std::vector<std::function<std::tuple<double, size_t, size_t>(Node, Node)>> algorithms;
void choose_algorithms()
{
  #ifdef ASTAR
  algorithm_names.push_back("AStar");
  algorithms.push_back(Astar::search);
  #endif

  #ifdef DIBBS
  algorithm_names.push_back("DIBBS");
  algorithms.push_back(Dibbs::search);
  #endif

  #ifdef DVCBS
  algorithm_names.push_back("DVCBS");
  algorithms.push_back(Dvcbs::search);
  #endif

  #ifdef GBFHS
  algorithm_names.push_back("GBFHS");
  algorithms.push_back(Gbfhs::search);
  #endif

  #ifdef IDD
  algorithm_names.push_back("IDD");
  algorithms.push_back(Idd::search);
  #endif

  #ifdef CBBS
  algorithm_names.push_back("CBBS");
  algorithms.push_back(Cbbs::search);
  #endif

  #ifdef FFGBS
  algorithm_names.push_back("FFGBS");
  algorithms.push_back(Ffgbs::search);
  #endif

}

struct Result
{
  std::string_view algorithm;
  double UB;
  size_t expansions;
  size_t memory;
  size_t microseconds;
};

Result test_problem(std::function<std::tuple<double, size_t, size_t>(Node, Node)> algorithm, Node start, Node goal)
{
  auto start_time = std::chrono::system_clock::now();
  auto [UB, expansions, memory] = algorithm(start, goal);
  auto end_time = std::chrono::system_clock::now();
  return Result{
    .UB = UB,
    .expansions = expansions,
    .memory = memory,
    .microseconds = static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count())
  };
}

std::vector<Result> random_problem(const std::uniform_int_distribution<uint32_t>& rng, bool run = true)
{
  uint32_t start_index = 0, goal_index = 0;
  while(start_index == goal_index) {
    start_index = rng(rng_engine);
    goal_index = rng(rng_engine);
  }
  if(!run) return {};
  Node::goal_node_index = goal_index;
  Node::start_node_index = start_index;

  uint32_t dist = Road::haversine_distance(Node::start_node_index, Node::goal_node_index);

  Node start{
    .vertex_index = Node::start_node_index,
    .g = 0,
    .h = dist,
    .f = dist,
    .h2 = 0,
    .f_bar = dist,
    .delta = 0,
    .dir = Direction::forward,
    .threshold = dist == 0,
  };

  Node goal{
    .vertex_index = Node::goal_node_index,
    .g = 0,
    .h = dist,
    .f = dist,
    .h2 = 0,
    .f_bar = dist,
    .delta = 0,
    .dir = Direction::backward,
    .threshold = dist == 0,
  };

  uint32_t prev_UB = UINT32_MAX;
  std::vector<Result> results;
  for(int i = 0; i < algorithms.size(); ++i) {
    results.push_back(test_problem(algorithms[i], start, goal));
    results.back().algorithm = algorithm_names[i];
    if(prev_UB == UINT32_MAX) prev_UB = static_cast<uint32_t>(results.back().UB);
    if(prev_UB != results.back().UB) {
      std::cout << "ERROR: " << algorithm_names[i] << " wrong answer! First UB = " << prev_UB << " This UB = " << results.back().UB << std::endl;
    }
  }
  return results;
}

std::string return_formatted_time(std::string format)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  tm tm_struct;
  localtime_s(&tm_struct, &in_time_t);
  ss << std::put_time(&tm_struct, format.c_str());
  return ss.str();
}

void output(std::unordered_map<std::string_view, std::vector<Result>>& alg_results)
{
  if(alg_results.size() == 0) return;
  std::ofstream file;
  std::string dir = R"(C:\Users\John\Dropbox\UIUC\Research\RoadData\)";
  std::string name = std::string("output_") + type_str[MAP_TYPE] + "_";

  for(auto& pair : alg_results) {
    name += std::string(pair.first) + "_";
  }
  name += return_formatted_time("%y%b%d") + ".txt";

  do file.open(dir + name, std::ios::out);
  while(!file);

  for(auto& pair : alg_results) {
    file << pair.first << "\n";
    for(auto& result : pair.second) {
      file << std::to_string(result.expansions) << " ";
    }
    file << "\n";
    for(auto& result : pair.second) {
      file << std::to_string(result.microseconds) << " ";
    }
    file << "\n\n";
  }
  file.close();
}

int main()
{
  choose_algorithms();
  uint32_t num_vertices = Road::LoadGraph(MAP_TYPE);

  std::uniform_int_distribution<uint32_t> uniform_dist(0, num_vertices);

  std::unordered_map<std::string_view, std::vector<Result>> alg_results;
  for(int i = 0; i < NUM_PROBLEMS; ++i) {
    std::cout << i << std::endl;
    
    auto results = random_problem(uniform_dist, i >= 475);

    for(auto& r : results) {
      alg_results[r.algorithm].push_back(r);
      std::cout << r.algorithm << " " << std::fixed << std::setprecision(1) << std::to_string(r.microseconds / 1000000.) << " seconds; " << r.expansions << " expansions" << std::endl;
    }

    output(alg_results);
  }

  return 0;
}

