#include "Pancake.h"
#include "Astar.h"
#include "id-d.h"
#include "IDAstar.h"
#include "dibbs.h"
#include "GBFHS.h"
#include "Nbs.h"
#include "dvcbs.h"
#include "problems.h"
#include "dibbs-array.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include "asymmetric.h"

#include "PerfectSolution.h"


//#define IDA_STAR

//#define A_STAR
#define REVERSE_ASTAR
//#define IDD
#define DIBBS
//#define GBFHS
//#define NBS
//#define DVCBS
//#define DIBBS_NBS
#define ASSYMETRIC

//constexpr int NUM_PROBLEMS = 100;


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
#ifdef DVCBS
  stream << "DVCBS ";
#endif
#ifdef DIBBS_NBS
  stream << "DIBBS_NBS ";
#endif
#ifdef ASSYMETRIC
  stream << "ASSYMETRIC ";
#endif
  stream << "\n";

  typedef std::chrono::nanoseconds precision;

  std::stringstream expansion_stream;
  std::stringstream memory_stream;
  std::stringstream time_stream;
  uint8_t problem[NUM_PANCAKES + 1];
  double answers[NUM_PROBLEMS + 1];
  for (int i = 0; i <= NUM_PROBLEMS; ++i) {
    answers[i] = -1;
  }
  {
#ifdef HISTORY
    std::cout << "Problem:\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      for (int i = 0; i <= NUM_PANCAKES; ++i) {
        std::cout << " " << std::to_string(problem[i]) << " ";
      }
      std::cout << std::endl;
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }
  {
#ifdef A_STAR
    std::cout << "A*\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);

      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Astar::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) { std::cout << "ERROR Cstar mismatch"; return; }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef REVERSE_ASTAR
    std::cout << "\nRA*\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::backward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Astar::search(goal, node);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef IDA_STAR
    std::cout << "\nIDA\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = IDAstar::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      expansion_stream << std::to_string(expansions) << " ";

      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef IDD
    //ID-D
    std::cout << "\nID-D\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //if (i != 19) continue;
      /*std::cout << "Problem:";
      for (int j = 0; j < 21; ++j) {
        std::cout << std::to_string(problem[j]) << " ";
      }
      std::cout << "\n";*/
      //define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = ID_D::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }
  {
#ifdef DIBBS
    //DIBBS
    std::cout << "\nDIBBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Dibbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef GBFHS
    std::cout << "\nGBFHS\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Gbfhs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef NBS
    std::cout << "\nNBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Nbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }

  {
#ifdef DVCBS
    std::cout << "\nDVCBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = Dvcbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch"; return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }
  {
#ifdef DIBBS_NBS
    //DIBBS_NBS
    std::cout << "\nDIBBS_NBS\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //if (i != 19) continue;
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = DibbsNbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
#endif
  }
  {
#ifdef ASSYMETRIC
    //DIBBS_NBS
    std::cout << "ASSYMETRIC\n";
    double seed = 3.1567;
    for (int i = 1; i <= NUM_PROBLEMS; ++i) {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //if (i != 19) continue;
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = AssymetricSearch::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        expansion_stream << "NAN ";
      }
      else {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if (answers[i] < 0 && !std::isinf(cstar)) {
        answers[i] = cstar;
      }
      else if (!std::isinf(cstar) && answers[i] != cstar) {
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
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

void run_random_test() {
  std::ofstream file;
  std::string dir = R"(C:\Users\John\Dropbox\UIUC\Research\PancakeData\)";
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
#ifdef DVCBS
  name += "_DVCBS";
#endif
#ifdef DIBBS_NBS
  name += "_DIBNBS";
#endif
#ifdef ASSYMETRIC
  name += "_ASSYM";
#endif
  name += ".txt";
  file.open(dir + name, std::ios::app);

  if (!file)
  {
    std::cout << "Error in creating file!!!" << std::endl;
    return;
  }

  output_data(file);
}

int main()
{
  GeneratePerfectCounts();
  //run_random_test();
  return 0;

  uint8_t problem[] = { 14,  8,  11,  4,  2,  10,  3,  14,  7,  12,  13,  1,  6,  5,  9 };
  Pancake::Initialize_Dual(problem);

  Pancake idd_f(problem, Direction::forward);
  Pancake idd_b = Pancake::GetSortedStack(Direction::backward);
  std::cout << "s " << " g = " << (int)idd_f.g << " h = " << (int)idd_f.h << " h2 = " << (int)idd_f.h2 << " fbar = " << (int)idd_f.f_bar << "\n";
  std::cout << "t " << " g = " << (int)idd_b.g << " h = " << (int)idd_b.h << " h2 = " << (int)idd_b.h2 << " fbar = " << (int)idd_b.f_bar << "\n";
  std::vector<int> moves = { 7, 14, 11, 12, 9, 7, 10, 5, 8, 11 };
  for (int i = 0; i < moves.size(); ++i) {
    idd_f = idd_f.apply_action(moves[i]);
    std::cout << "After move " << moves[i] << " g = " << (int)idd_f.g << " h = " << (int)idd_f.h << " h2 = " << (int)idd_f.h2 << " fbar = " << (int)idd_f.f_bar << "\n";

    //std::cout << "Node: ";
    //for (int i = 0; i < NUM_PANCAKES + 1; ++i) {
    //  std::cout << std::to_string(idd_f.source[i]) << " ";
    //}
    //std::cout << "\n";
  }
  std::vector<int> moves2 = { 10, 13, 3};
  for (int i = moves2.size() - 1; i >= 0; --i) {
    idd_b = idd_b.apply_action(moves2[i]);
    std::cout << "After backward move " << moves2[i] << " g = " << (int)idd_b.g << " h = " << (int)idd_b.h << " h2 = " << (int)idd_b.h2 << " fbar = " << (int)idd_b.f_bar << "\n";
  }
  assert(idd_f == idd_b);

  std::cout << "Middle pancake:\n";
  for (int i = 0; i < NUM_PANCAKES; ++i) {
    std::cout << std::to_string(idd_f.source[i]) << " ";
  }
  std::cout << "\n";
  for (int i = 0; i < NUM_PANCAKES; ++i) {
    std::cout << std::to_string(idd_b.source[i]) << " ";
  }
  std::cout << "\n";

  std::cout << "\nMeeting point: ";
  for (int i = 0; i < NUM_PANCAKES + 1; ++i) {
    std::cout << std::to_string(idd_f.source[i]) << " ";
  } 
  
}