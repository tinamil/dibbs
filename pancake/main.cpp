
//#define IDA_STAR
//#define A_STAR
//#define REVERSE_ASTAR
//#define IDD
#define DIBBS
//#define GBFHS
//#define NBS
//#define DVCBS
//#define ASSYMETRIC

#include "ftf-dibbs.h"
#include <StackArray.h>
#include "Transform.h"
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
#include "dibbs-2phase.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#ifdef ASSYMETRIC
#include "asymmetric.h"
#endif

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

#include "Renderer.h"
#include "Timer.hpp"
#include "Archetype.h"
#include "ECSManager.h"
#include "Transform.h"
#include "static_helpers.h"
#include "Input.h"
#include "ScreenElementProcessor.hpp"
#include "Screen.h"
#include "Serialization.h"
#include "UISystem.h"

#include "sprite.h"
#include "FontComponents.h"
#include "Model.hpp"
#include <VulkanLineRenderer.h>

//constexpr int NUM_PROBLEMS = 100;

//TODO: Setup cpu to gpu line buffer for line rendering

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


void SolveAStar(uint8_t* problem)
{
  glm::vec3 nodeScale = { .03f, .03f, .03f };

  Pancake fnode(problem, Direction::forward);


  Pancake::Initialize_Dual(problem);
  Pancake bnode(problem, Direction::backward);
  Pancake goal = Pancake::GetSortedStack(Direction::backward);

  std::vector<const Pancake*> expansions;
  uint32_t cstar;

  Astar backwardInstance;
  auto [cstar_b, expansions_b, memory_b] = backwardInstance.run_search(goal, bnode, &expansions);
  cstar = cstar_b;

  auto s = globalECS.create_entity(0, Mesh{ 0, 0 }, dWorldPosition{ { 0, 0, 0 } }, Scale{ nodeScale });
  auto t = globalECS.create_entity(0, Mesh{ 0, 0 }, dWorldPosition{ { 0, 0, cstar } }, Scale{ nodeScale });

  assert(*expansions[0] == goal);

  std::queue<Pancake> queue;
  queue.push(fnode);
  std::unordered_set<Pancake, PancakeHash> closed;

  while(queue.empty() == false && closed.size() < 1000)
  {
    Pancake parent = queue.front();
    queue.pop();
    for(int j = 2; j <= NUM_PANCAKES; ++j)
    {
      Pancake node = parent.apply_action(j);
      if(!closed.contains(node))
      {
        queue.push(node);
        closed.insert(node);
        auto next_entity = globalECS.create_entity(0, Mesh{ 0, 0 },
          dWorldPosition{ { small_rand(), small_rand(), small_rand() + node.g } },
          Scale{ nodeScale }, RepulsiveForce{}, Velocity{}, Acceleration{}, Drag{ 0.1 });
        //globalECS.create_entity(1, SpringForce{ s, next_entity, 1.f });
        //globalECS.create_entity(1, Line{ s, next_entity });
      }
    }
  }
  //goal = Pancake::GetSortedStack(Direction::forward);
  //{
  //  Astar forwardInstance;
  //  auto [cstar_f, expansions_f, memory_f] = forwardInstance.run_search(fnode, goal);
  //  closed_f = forwardInstance.closed;

  //  if (cstar_f != cstar) std::cout << "ERROR";
  //}


  //std::vector<Pancake> pancake_expansions;
  //auto [cstar_dibbs, expansions, memory] = Dibbs::search(fnode, goal, &pancake_expansions);

  //tsl::hopscotch_map<int, float> offsets;
  //for (const auto& x : pancake_expansions) {
  //  auto b = closed_b.find(x);
  //  auto f = closed_f.find(x);
  //  if (b == closed_b.end() && f == closed_f.end()) {
  //    std::cout << "ERROR in finding both";
  //  }
  //  if (x.dir == Direction::forward) {
  //    if (offsets.count(x.g) == 0) offsets[x.g] = 0;
  //    globalECS.create_entity(0, Mesh{ 1, 0 }, dWorldPosition{ { 0., offsets[x.g], x.g } }, Scale{ { .01f, .01f, .01f }, RepulsiveForce{1.f} });
  //  }
  //}


 /* for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
    Pancake new_action = fnode.apply_action(i);
    auto next_entity = globalECS.create_entity(0, Mesh{ 0, 0 },
      dWorldPosition{ { small_rand(), small_rand(), small_rand() } },
      Scale{ nodeScale }, RepulsiveForce{}, Velocity{}, Acceleration{}, Drag{ 0.1 });
    globalECS.create_entity(1, SpringForce{ s, next_entity, 1.f });
    globalECS.create_entity(1, Line{ s, next_entity });

    for (int k = 2, l = NUM_PANCAKES; k <= l; ++k) {
      Pancake new_action2 = new_action.apply_action(k);
      auto next_entity2 = globalECS.create_entity(1, Mesh{ 0, 0 },
        dWorldPosition{ { small_rand(), small_rand(), small_rand() } },
        Scale{ nodeScale }, RepulsiveForce{}, Velocity{}, Acceleration{}, Drag{ 0.1 });
      globalECS.create_entity(2, SpringForce{ next_entity, next_entity2, 1.f });
      globalECS.create_entity(2, Line{ next_entity, next_entity2 });
    }
  }*/

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
  #ifdef ASSYMETRIC
  stream << "ASSYMETRIC ";
  #endif
  stream << "\n";

  typedef std::chrono::nanoseconds precision;

  std::stringstream expansion_stream;
  std::stringstream memory_stream;
  std::stringstream time_stream;
  std::stringstream expansions_after_cstar_stream;
  std::stringstream expansions_after_ub_stream;
  uint8_t problem[NUM_PANCAKES + 1];
  double answers[NUM_PROBLEMS + 1];
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
    stream << memory_stream.rdbuf() << std::endl;
    stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    stream << expansions_after_ub_stream.rdbuf() << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    //stream << time_stream.rdbuf() << std::endl;
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
    //stream << time_stream.rdbuf() << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch"; return;
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
        std::cout << "ERROR Cstar mismatch";
        return;
      }
    }
    stream << expansion_stream.rdbuf() << std::endl;
    //stream << time_stream.rdbuf() << std::endl;
    //stream << memory_stream.rdbuf() << std::endl;
    stream << expansions_after_cstar_stream.rdbuf() << std::endl;
    stream << expansions_after_ub_stream.rdbuf() << std::endl;
    #endif
  }
  {
    #ifdef ASSYMETRIC
        //DIBBS_NBS
    std::cout << "ASSYMETRIC\n";
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
      auto [cstar, expansions, memory] = AssymetricSearch::search(node, goal);
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
        std::cout << std::to_string(i) << " " << std::to_string(answers[i]) << " " << std::to_string(cstar) << std::endl;
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
  #ifdef ASSYMETRIC
  name += "_ASSYM";
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

class InputHelper
{
public:
  static void clear()
  {
    Input::clear();
  }
};

class Engine
{
public:
  std::unique_ptr<Serialization> serializer;
  std::unique_ptr<vks::Renderer> renderer;
  std::unique_ptr<Camera> camera;

  inline static bool ProcessWin32()
  {
    static MSG msg;
    while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) == TRUE)
    {
      if(msg.message == WM_QUIT)
      {
        return false;
      }
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    return true;
  }

  void MainLoop()
  {
    while(true)
    {
      Timer::Run();
      globalECS.ClearDeleteQueue();
      InputHelper::clear();
      if(!ProcessWin32()) break;
      serializer->Run();
      ScreenElementProcessor::Run();

      UISystem::Run();

      for(auto x : globalECS.Query<Acceleration>())
      {
        x.get<Acceleration>()->value = glm::dvec3(0);
      }

      for(auto x : globalECS.Query<RepulsiveForce, dWorldPosition, Acceleration>())
      {
        auto x_pos = x.get<dWorldPosition>();
        auto x_accel = x.get<Acceleration>();
        for(auto y : globalECS.Query<RepulsiveForce, dWorldPosition, Acceleration>())
        {
          if(x == y) continue;
          auto y_accel = y.get<Acceleration>();
          auto y_pos = y.get<dWorldPosition>();
          auto q = x_pos->value - y_pos->value + glm::dvec3(0, 0.001, 0);
          auto inv_dist = glm::normalize(q) / std::max(0.01, q.x * q.x + q.y * q.y + q.z * q.z);
          if(x_accel != nullptr)
            x_accel->value += inv_dist;
          if(y_accel != nullptr)
            y_accel->value -= inv_dist;
        }
      }

      for(auto x : globalECS.Query<SpringForce>())
      {
        auto e1 = globalECS.get<dWorldPosition>(x.get<SpringForce>()->obj1);
        auto e2 = globalECS.get<dWorldPosition>(x.get<SpringForce>()->obj2);
        auto e1a = globalECS.get<Acceleration>(x.get<SpringForce>()->obj1);
        auto e2a = globalECS.get<Acceleration>(x.get<SpringForce>()->obj2);
        auto q = e1->value - e2->value + glm::dvec3(0, 0.001, 0);;
        auto dist = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        if(e1a != nullptr)
          e1a->value -= glm::normalize(q) * dist;
        if(e2a != nullptr)
          e2a->value += glm::normalize(q) * dist;
      }

      for(auto x : globalECS.Query<Drag, Velocity>())
      {
        x.get<Velocity>()->value *= glm::clamp(1 - x.get<Drag>()->strength, 0., 1.);
      }

      for(auto x : globalECS.Query<dWorldPosition, Velocity>())
      {
        x.get<dWorldPosition>()->value += x.get<Velocity>()->value * Timer::deltaTime;
        if(std::isnan(x.get<dWorldPosition>()->value.x))
        {
          x.get<dWorldPosition>()->value = glm::vec3(0);
        }
      }

      for(auto x : globalECS.Query<Acceleration, Velocity>())
      {
        x.get<Velocity>()->value += x.get<Acceleration>()->value * Timer::deltaTime;
        if(std::isnan(x.get<Velocity>()->value.x))
        {
          x.get<Velocity>()->value = glm::vec3(0);
        }
      }

      for(auto x : globalECS.Query<Line>())
      {
        auto line = x.get<Line>();
        line->positions.clear();
        line->positions.push_back({ globalECS.get<dWorldPosition>(line->a)->value, glm::dvec4(1) });
        line->positions.push_back({ globalECS.get<dWorldPosition>(line->b)->value, glm::dvec4(1) });
      }

      camera->Run();
      renderer->Run();
    }
  }

  void Initialize(HINSTANCE hInstance)
  {
    Timer::Init();
    serializer = std::make_unique<Serialization>();
    renderer = std::make_unique<vks::Renderer>("Pancake", hInstance);
    camera = std::make_unique<Camera>();
    camera->Initialize();
    UISystem::Init();
  }
};

int APIENTRY WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE, _In_ LPSTR, _In_ int)
{
  hash_table::initialize_hash_values();
  Window window("pancake", hInstance, true);
  std::thread t(run_random_test);
  //std::thread t(GeneratePerfectCounts);
  std::thread t2([]() {
    using namespace std::chrono_literals;
    while(true)
    {
      Engine::ProcessWin32();
      std::this_thread::sleep_for(10ms);
    }
  });

  t.join();
  std::cout << "\nDone\n";
  while(true);
  return EXIT_SUCCESS;

  //try {
  //  Engine e;
  //  e.Initialize(hInstance);
  //  //Mesh mesh{ 0, 0 };
  //  //dWorldPosition dpos{ { 0., 0., 0. } };
  //  //Orientation o{ glm::rotate(glm::identity<glm::quat>(), glm::radians(90.f), glm::vec3(1, 0, 0)) };
  //  //globalECS.create_entity(0, mesh, dpos, o);

  //  //run_random_test();
  //  //GeneratePerfectCounts();
  //  double seed = 3.1567;
  //  uint8_t problem[NUM_PANCAKES + 1];
  //  generate_random_instance(seed, problem);
  //  SolveAStar(problem);

  //  e.MainLoop();
  //}
  //catch (const std::exception& e) {
  //  vks::tools::exitFatal(std::string("Fatal exception thrown: ") + e.what(), 0);
  //  return EXIT_FAILURE;
  //}
  //return EXIT_SUCCESS;
}


