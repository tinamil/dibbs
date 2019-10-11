#include "Pancake.h"
#include "Astar.h"
#include "problems.h"
#include <iostream>
#include <string>

int main()
{
  Pancake::initialize_goal(NUM_PANCAKES);
  uint8_t problem[NUM_PANCAKES + 1];
  for (int i = 1; i <= 100; ++i) {
    define_problems(NUM_PANCAKES, GAPX, i, problem);
    Pancake node(problem, Direction::forward);
    auto [cstar, expansions] = Astar::search(node);
    std::cout << i << ": depth = " << std::to_string((int)cstar) << "; expansions = " << std::to_string(expansions) << "\n";
  }   
  std::cout << "Done\n";
  int x;
  std::cin >> x;
}