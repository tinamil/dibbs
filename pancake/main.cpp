#include "Pancake.h"
#include "Astar.h"
#include "problems.h"
#include <iostream>
#include <string>


int main()
{
  {
    //Forward A*
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake node(problem, Direction::forward);
      Pancake goal = Pancake::GetSortedStack(Direction::forward);
      auto [cstar, expansions] = Astar::search(node, goal);
      std::cout << i << ": depth = " << std::to_string((int)cstar) << "; expansions = " << std::to_string(expansions) << "\n";
    }
  }
  {
    //Backwards A*
    uint8_t problem[NUM_PANCAKES + 1];
    for (int i = 1; i <= 100; ++i) {
      define_problems(NUM_PANCAKES, GAPX, i, problem);
      Pancake::Initialize_Dual(problem);
      Pancake node(problem, Direction::backward);
      Pancake goal = Pancake::GetSortedStack(Direction::backward);
      auto [cstar, expansions] = Astar::search(goal, node);
      std::cout << i << ": depth = " << std::to_string((int)cstar) << "; expansions = " << std::to_string(expansions) << "\n";
    }
  }
  std::cout << "Done\n";
  int x;
  std::cin >> x;
}