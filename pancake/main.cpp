#include "Pancake.h"
#include "Astar.h"
#include "id-d.h"
#include "IDAstar.h"
#include "dibbs.h"
#include "GBFHS.h"
#include "nbs.h"
#include "problems.h"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>


//#define A_STAR
//#define REVERSE_ASTAR
//#define IDA_STAR
//#define IDD
//#define DIBBS
//#define GBFHS
#define NBS


void generate_random_instance(double& seed, uint8_t problem[]) {
  problem[0] = NUM_PANCAKES;
  for (int i = 1; i <= NUM_PANCAKES; i++) problem[i] = i;
  random_permutation2(NUM_PANCAKES, problem, seed);
}



void output_data(std::ostream& stream) {

  stream << "Pancakes = " << NUM_PANCAKES << " gap-x = " << GAPX << '\n';
  stream << "Algorithms: ";
#ifdef A_STAR
  stream << "A* ";
#endif
#ifdef REVERSE_ASTAR
  stream << "RA* ";
#endif
#ifdef IDA_STAR
  stream << "IDA ";
#endif
#ifdef IDD
  stream << "IDD ";
#endif
#ifdef DIBBS
  stream << "DIBBS ";
#endif
#ifdef GBFHS
  stream << "GBFHS ";
#endif
#ifdef NBS
  stream << "NBS ";
#endif
  stream << "\n";

  typedef std::chrono::nanoseconds precision;

  std::stringstream expansion_stream;
  std::stringstream time_stream;
  uint8_t problem[NUM_PANCAKES + 1];
  {
#ifdef A_STAR

    std::cout << "A*\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = Astar::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef REVERSE_ASTAR
    std::cout << "RA*\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::backward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = Astar::search(goal, node);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef IDA_STAR
    std::cout << "IDA\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = IDAstar::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      expansion_stream << std::to_string(expansions) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef IDD
    //ID-D
    std::cout << "ID-D\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = ID_D::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }
  {
#ifdef DIBBS
    //DIBBS
    std::cout << "DIBBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = Dibbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
    }

  {
#ifdef GBFHS
    std::cout << "GBFHS\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = Gbfhs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef NBS
    std::cout << "NBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= 100; ++i) {
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions] = Nbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
#endif
  }
  }

std::string return_formatted_time(std::string format)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), format.c_str());
  return ss.str();
}

int main()
{
  std::ofstream file;
  std::string name = "output" + std::to_string(NUM_PANCAKES) + "_" + std::to_string(GAPX) + "_" + return_formatted_time("%y%b%d-%H%M%S");
#ifdef A_STAR
  name += "_A";
#endif
#ifdef REVERSE_ASTAR
  name += "_RA";
#endif
#ifdef IDA_STAR
  name += "_IDA";
#endif
#ifdef IDD
  name += "_IDD";
#endif
#ifdef DIBBS
  name += "_DIBBS";
#endif
#ifdef GBFHS
  name += "_GBFHS";
#endif
#ifdef NBS
  name += "_NBS";
#endif
  name += ".txt";
  file.open(name, std::ios::app);

  if (!file)
  {
    std::cout << "Error in creating file!!!" << std::endl;
    return 0;
}

  output_data(file);
  }