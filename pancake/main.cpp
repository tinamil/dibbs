#include "Pancake.h"
#include "Astar.h"
#include "id-d.h"
#include "IDAstar.h"
#include "dibbs.h"
#include "problems.h"
#include <iostream>
#include <string>

//#define A_STAR
//#define REVERSE_ASTAR
//#define IDA_STAR
//#define IDD
#define DIBBS

int main()
{
  std::cout << "Pancakes = " << NUM_PANCAKES << " gap-x = " << GAPX << '\n';
  {
#ifdef A_STAR
    std::cout << "A*: ";
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto [cstar, expansions] = Astar::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        std::cout << "NAN ";
      }
      else {
        std::cout << std::to_string(expansions) << " ";
      }
    }
    std::cout << '\n';
#endif
  }
  {
#ifdef IDA_STAR
    std::cout << "IDA*: ";
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto [cstar, expansions] = IDAstar::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      std::cout << std::to_string(expansions) << " ";
    }
    std::cout << '\n';
#endif
  }
  {
#ifdef REVERSE_ASTAR
    std::cout << "RA*: ";
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::backward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = Astar::search(goal, node);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        std::cout << "NAN ";
      }
      else {
        std::cout << std::to_string(expansions) << " ";
      }
    }
    std::cout << '\n';
#endif
  }

  {
#ifdef IDD
    //ID-D
    std::cout << "ID-D: ";
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = ID_D::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        std::cout << "NAN ";
      }
      else {
        std::cout << std::to_string(expansions) << " ";
      }
    }
#endif
  }
  {
#ifdef DIBBS
    //DIBBS
    std::cout << "DIBBS: ";
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = Dibbs::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      if (std::isinf(cstar)) {
        std::cout << "NAN ";
      }
      else {
        std::cout << std::to_string(expansions) << " ";
      }
    }
#endif
  }
  std::cout << "\nDone.\n";
  int x;
  std::cin >> x;
}