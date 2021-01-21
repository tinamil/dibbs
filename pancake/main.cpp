
//#define IDA_STAR
#define A_STAR
//#define REVERSE_ASTAR
//#define IDD
#define DIBBS
#define GBFHS
//#define NBS
#define DVCBS
#include "dibbs-2phase.hpp"
//#include "2phase-lookahead.h"
#include "ftf-dibbs.h"
//#include "dibbs-ftf-hybrid.h"

#include "Pancake.h"
#ifdef A_STAR
#include "Astar.h"
#endif
#ifdef IDD
#include "id-d.h"
#endif
//#include "IDAstar.h"
#ifdef DIBBS
#include "dibbs.h"
#endif
#ifdef GBFHS
#include "GBFHS.h"
#endif
#ifdef NBS
#include "Nbs.h"
#endif
#ifdef DVCBS
#include "dvcbs.h"
#endif
#include "problems.h"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "PerfectSolution.h"

#include "hash_table.h"


#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include <type_traits>
#include <chrono>
#include <ctime>    

//Windows includes
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <ShellScalingAPI.h>
#include <thread>
#include "mycuda.h"
//constexpr int NUM_PROBLEMS = 100;

float small_rand()
{
  return -1 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 2);
}

void generate_random_instance(double& seed, uint8_t problem[])
{
  problem[0] = NUM_PANCAKES;
  for(int i = 1; i <= NUM_PANCAKES; i++) problem[i] = i;
  random_permutation2(NUM_PANCAKES, problem, seed);
}

void output_data(std::ostream& stream)
{
  std::cout << "Pancakes = " << NUM_PANCAKES << " gap-x = " << GAPX << '\n';
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
  stream << DIBBS_NBS << " ";
  #endif
  #ifdef TWO_PHASE_LOOKAHEAD
  stream << TWO_PHASE_LOOKAHEAD << " ";
  #endif
  #ifdef FTF_PANCAKE
  stream << FTF_PANCAKE << " ";
  #endif
  #ifdef FTF_PANCAKE_HYBRID
  stream << FTF_PANCAKE_HYBRID << " ";
  #endif
  stream << "\n";

  typedef std::chrono::nanoseconds precision;

  std::stringstream expansion_stream;
  std::stringstream memory_stream;
  std::stringstream time_stream;
  std::stringstream expansions_after_cstar_stream;
  std::stringstream expansions_after_ub_stream;
  uint8_t problem[NUM_PANCAKES + 1];
  double* answers = new double[NUM_PROBLEMS + 1];
  for(int i = 0; i <= NUM_PROBLEMS; ++i)
  {
    answers[i] = -1;
  }
  {
    #ifdef HISTORY
    std::cout << "Problem:\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      for(int i = 0; i <= NUM_PANCAKES; ++i)
      {
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
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar) { std::cout << "ERROR Cstar mismatch"; return; }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef REVERSE_ASTAR
    std::cout << "\nRA*\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef IDA_STAR
    std::cout << "\nIDA\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    stream << memory_stream.rdbuf() << std::endl;
    stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef IDD
        //ID-D
    std::cout << "\nID-D\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //if(i != 2348) continue;
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef DIBBS
        //DIBBS
    std::cout << "\nDIBBS\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << "ERROR Cstar mismatch: " << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef GBFHS
    std::cout << "\nGBFHS\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef NBS
    std::cout << "\nNBS\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }

  {
    #ifdef DVCBS
    std::cout << "\nDVCBS\n";
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
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
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";
      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch"; return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef DIBBS_NBS
    std::cout << '\n' << DIBBS_NBS << '\n';
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      //if (i != 19) continue;
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory, expansions_after_cstar, expansions_after_UB] = DibbsNbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
        expansions_after_cstar_stream << std::to_string(expansions_after_cstar) << " ";
        expansions_after_ub_stream << std::to_string(expansions_after_UB) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef TWO_PHASE_LOOKAHEAD
    std::cout << '\n' << TWO_PHASE_LOOKAHEAD << '\n';
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory, expansions_after_cstar, expansions_after_UB] = TWO_PHASE::TwoPhase::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
        expansions_after_cstar_stream << std::to_string(expansions_after_cstar) << " ";
        expansions_after_ub_stream << std::to_string(expansions_after_UB) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef FTF_PANCAKE
    std::cout << '\n' << FTF_PANCAKE << '\n';
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = FTF_Dibbs::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch ";
        std::cout << expansions << "\n";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef FTF_PANCAKE_HYBRID
    std::cout << '\n' << FTF_PANCAKE_HYBRID << '\n';
    double seed = 3.1567;
    for(int i = 1; i <= NUM_PROBLEMS; ++i)
    {
      std::cout << i << " ";
      generate_random_instance(seed, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto start = std::chrono::system_clock::now();
      auto [cstar, expansions, memory] = dibbs_ftf_hybrid::search(node, goal);
      auto end = std::chrono::system_clock::now();
      //stream << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if(std::isinf(cstar))
      {
        expansion_stream << "NAN ";
      }
      else
      {
        expansion_stream << std::to_string(expansions) << " ";
      }
      memory_stream << std::to_string(memory) << " ";

      time_stream << std::to_string(std::chrono::duration_cast<precision>(end - start).count()) << " ";

      if(answers[i] < 0 && !std::isinf(cstar))
      {
        answers[i] = cstar;
      }
      else if(!std::isinf(cstar) && answers[i] != cstar)
      {
        std::cout << std::to_string(i) << ": " << std::to_string(answers[i]) << " previous C* is different than the new C* of " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    //stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    //stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  delete[] answers;
}

std::string return_formatted_time(std::string format)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), format.c_str());
  return ss.str();
}

void run_random_test()
{
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
  name += std::string("_") + DIBBS_NBS;
  #endif
  #ifdef TWO_PHASE_LOOKAHEAD
  name += std::string("_") + TWO_PHASE_LOOKAHEAD;
  #endif
  #ifdef FTF_PANCAKE
  name += std::string("_") + FTF_PANCAKE;
  #endif
  #ifdef FTF_PANCAKE_HYBRID
  name += std::string("_") + FTF_PANCAKE_HYBRID;
  #endif
  name += ".txt";
  file.open(dir + name, std::ios::app);

  if(!file)
  {
    std::cout << "Error in creating file!!!" << std::endl;
    return;
  }

  output_data(file);
}

int main()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for(device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "Device " << device << " has compute capability " << deviceProp.major << "." << deviceProp.minor << "\n";
  }
  hash_table::initialize_hash_values();
  mycuda::initialize();
  run_random_test();

  //std::cout << "\nDone\n";
  //while(true);
  return EXIT_SUCCESS;
}
