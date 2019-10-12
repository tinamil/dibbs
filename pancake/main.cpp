#include "Pancake.h"
#include "Astar.h"
#include "id-d.h"
#include "problems.h"
#include <iostream>
#include <string>

//#define A_STAR
//#define REVERSE_ASTAR
#define IDD

int main()
{
  {
#ifdef A_STAR
    //Forward A*
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto [cstar, expansions] = Astar::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      std::cout << std::to_string(expansions) << " ";
    }
    std::cout << '\n';
#endif
  }
  {
#ifdef REVERSE_ASTAR
    //Backwards A*
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::backward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = Astar::search(goal, node);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      std::cout << std::to_string(expansions) << " ";
    }
    std::cout << '\n';
#endif
  }

  {
#ifdef IDD
    //ID-D
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      //easy_problem(NUM_PANCAKES, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = ID_D::search(node, goal);
      //std::cout << std::to_string((int)cstar) << " , " << std::to_string(expansions) << "\n";
      std::cout << std::to_string(expansions) << " ";
    }
    std::cout << '\n';
#endif
  }
  int x;
  std::cin >> x;
}