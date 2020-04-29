#include <cstdint>
#include <stack>
#include <unordered_set>
#include <ctime>
#include "IDAstar.h"

typedef std::stack<Pancake, std::vector<Pancake>> stack;
std::pair<double, size_t> IDAstar::run_search(Pancake start, Pancake goal)
{
  auto c_start = clock();
  stack state_stack;

  if (start == goal)
  {
    std::cout << "Given a solved pancake problem.  Nothing to solve." << std::endl;
    return std::make_pair(0, 0);
  }

  uint8_t id_depth = start.f;

  state_stack.push(start);
  size_t c_star;
  uint64_t count = 0;
  bool done = false;
  while (done == false)
  {
    if (state_stack.empty())
    {
      id_depth += 1;
      state_stack.push(start);
      //std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
    }

    auto next_val = state_stack.top();
    state_stack.pop();

    ++count;
    //if (count % 1000000 == 0)
    //{
      //std::cout << count << std::endl;
    //}

    for (int i = 2, j = NUM_PANCAKES; i <= j; ++i) {
      Pancake new_action = next_val.apply_action(i);
      
      if (new_action.f <= id_depth) {

        if (new_action == goal) {
          c_star = new_action.g;
          //std::cout << "Solution: " << UB << '\n';
          //std::cout << "Actions: ";
          //for (int i = 0; i < next_val.actions.size(); ++i) {
           // std::cout << std::to_string(next_val.actions[i]) << " ";
          //}
          //std::cout << std::endl;
          assert(new_action.h == 0);
          done = true;
          break;
        }
        state_stack.push(new_action);
      }
    }
  }
  auto c_end = clock();
  auto time_elapsed = (c_end - c_start) / CLOCKS_PER_SEC;
  return std::make_pair((double)c_star, count);
}
