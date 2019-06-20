#include <iostream>
#include <ctime>
#include <iomanip>
#include "a_star.h"
#include "dibbs.h"
#include "rubiks.h"
#include "rubiks_loader.h"
#include "gbfhs.h"
#include "id_gbfhs.h"
#include "id_dibbs.h"
#include "utility.h"

using namespace std;
using namespace Rubiks;

#define ASTAR 
#define DIBBS 
//#define GBFHS

void test() {
  typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;

  auto start_state = RubiksLoader::scramble("L' R' B  L' D' L  U2 B2 D' F2 D  B2 R' F' R  U2 B2 D2 L' B' D  R2 D2 R  D' R2 F' R  U  B' U2 B2 D' R' D2 F  U  D' F' U2 L2 U  F2 R' B' U' L  B' R' U' L' U2 B2 D  B' R  B  D  B2 R' U2 D' R' F  L' F' D' B  L2 R' B  R' B' D  R  U' R2 B' D2 F' R  F  L' U  L2 B2 R2 F' U' F D  B' L' F2 B  L2 U2 D' L2 B2");
  auto pdb_type = PDB::a888;

  auto l0 = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  auto l1 = std::make_shared<Node>(l0, start_state, 1, Face::down, Rotation::half, false, pdb_type);
  auto l2 = std::make_shared<Node>(l1, start_state, 2, Face::front, Rotation::half, false, pdb_type);
  auto l3 = std::make_shared<Node>(l2, start_state, 3, Face::down, Rotation::half, false, pdb_type);
  auto l4 = std::make_shared<Node>(l3, start_state, 4, Face::right, Rotation::half, false, pdb_type);
  auto l5 = std::make_shared<Node>(l4, start_state, 5, Face::up, Rotation::clockwise, false, pdb_type);
  auto l6 = std::make_shared<Node>(l5, start_state, 6, Face::left, Rotation::clockwise, false, pdb_type);
  auto l7 = std::make_shared<Node>(l6, start_state, 7, Face::back, Rotation::half, false, pdb_type);
  auto l8 = std::make_shared<Node>(l7, start_state, 8, Face::up, Rotation::clockwise, false, pdb_type);
  auto l9 = std::make_shared<Node>(l8, start_state, 9, Face::front, Rotation::counterclockwise, false, pdb_type);
  auto l10 = std::make_shared<Node>(l9, start_state, 10, Face::down, Rotation::counterclockwise, false, pdb_type);
  auto l11 = std::make_shared<Node>(l10, start_state, 11, Face::back, Rotation::clockwise, false, pdb_type);
  auto l12 = std::make_shared<Node>(l11, start_state, 12, Face::right, Rotation::clockwise, false, pdb_type);
  auto l13 = std::make_shared<Node>(l12, start_state, 13, Face::up, Rotation::clockwise, false, pdb_type);
  auto l14 = std::make_shared<Node>(l13, start_state, 14, Face::front, Rotation::clockwise, false, pdb_type);
  auto l15 = std::make_shared<Node>(l14, start_state, 15, Face::up, Rotation::half, false, pdb_type);
  auto l16 = std::make_shared<Node>(l15, start_state, 16, Face::left, Rotation::clockwise, false, pdb_type);
  auto l17 = std::make_shared<Node>(l16, start_state, 17, Face::down, Rotation::half, false, pdb_type);
  auto l18 = std::make_shared<Node>(l17, start_state, 18, Face::back, Rotation::counterclockwise, false, pdb_type);


  auto r0 = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);
  auto r1 = std::make_shared<Node>(r0, start_state, 1, Face::back, Rotation::clockwise, true, pdb_type);
  auto r2 = std::make_shared<Node>(r1, start_state, 2, Face::down, Rotation::half, true, pdb_type);
  auto r3 = std::make_shared<Node>(r2, start_state, 3, Face::left, Rotation::counterclockwise, true, pdb_type);
  auto r4 = std::make_shared<Node>(r3, start_state, 4, Face::up, Rotation::half, true, pdb_type);
  auto r5 = std::make_shared<Node>(r4, start_state, 5, Face::front, Rotation::counterclockwise, true, pdb_type);
  auto r6 = std::make_shared<Node>(r5, start_state, 6, Face::up, Rotation::counterclockwise, true, pdb_type);
  auto r7 = std::make_shared<Node>(r6, start_state, 7, Face::right, Rotation::counterclockwise, true, pdb_type);
  auto r8 = std::make_shared<Node>(r7, start_state, 8, Face::back, Rotation::counterclockwise, true, pdb_type);
  auto r9 = std::make_shared<Node>(r8, start_state, 9, Face::down, Rotation::clockwise, true, pdb_type);


  cout << l18->print_solution() << "\n\n";

  auto middle = std::make_shared<Node>(r9, start_state, 10, Face::front, Rotation::clockwise, true, pdb_type);
  hash_set x;
  x.insert(middle);

  auto search = x.find(l8);
  if (search != x.end())
  {
    cout << "Matched\n";
    cout << "Reverse cost =" << std::to_string((*search)->depth) << "\n";
    cout << "My cost: " << std::to_string(l8->depth) << "\n";
    l8->set_reverse(middle);
  }

  cout << l8->print_solution() << "\n\n";


  cout << "Solved l18= " << std::to_string(memcmp(Rubiks::__goal, l18->state, 40) == 0) << '\n';

  delete[] start_state;
  return;
}


typedef std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>> node_stack;
void generate_statistics(const uint8_t* state, const Rubiks::PDB pdb_type) {

  node_stack state_stack;

  std::shared_ptr<Node> original_node = std::make_shared<Node>();
  memcpy(original_node->state, state, 40);
  uint8_t id_depth = Rubiks::pattern_lookup(original_node->state, pdb_type);
  original_node->heuristic = id_depth;
  original_node->combined = id_depth;
  state_stack.push(original_node);

  size_t stats_array[20 * 20];
  for (int i = 0; i < 400; ++i) {
    stats_array[i] = 0;
  }

  bool done = false;
  while (done == false)
  {
    if (state_stack.empty())
    {
      id_depth += 1;
      state_stack.push(original_node);
      std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << std::endl;
    }

    const size_t stats_round = id_depth * 20ui64;

    std::shared_ptr<Node> next_node = state_stack.top();
    state_stack.pop();

    for (int face = 0; face < 6; ++face)
    {
      if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
      {
        continue;
      }

      for (int rotation = 0; rotation < 3; ++rotation)
      {
        std::shared_ptr<Node> new_node = std::make_shared<Node>(next_node, state, next_node->depth + 1, face, rotation, false, pdb_type);

        if (new_node->combined < next_node->combined) {
          std::cout << "Consistency error: " << unsigned(new_node->combined) << " < " << unsigned(next_node->combined) << " " << std::endl;
        }

        if (new_node->combined > id_depth)
        {
          continue;
        }

        stats_array[stats_round + new_node->depth - new_node->reverse_heuristic] += 1;

        if (Rubiks::is_solved(new_node->state))
        {
          //std::cout << "Solved IDA*: " << unsigned int(id_depth) << " Count = " << unsigned long long(count) << std::endl;
          //std::cout << "Solution: " << new_node->print_solution() << std::endl;
          done = true;
        }
        state_stack.push(new_node);
      }
    }
  }

  size_t field_width = 10;
  cout << "\n" << setw(3) << " ";
  for (int i = 0; i < 20; ++i) {
    cout << setw(field_width) << std::right << i;
  }
  cout << '\n';
  for (int diff = 0; diff < 20; ++diff) {
    cout << setw(2) << std::right << diff << " ";
    for (int depth = 0; depth < 20; ++depth) {
      cout << setw(field_width) << std::right << stats_array[depth * 20 + diff];
    }
    cout << "\n";
  }
}

void search_cubes()
{
  vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  PDB type = PDB::a888;
  vector<uint64_t> count_results;
  vector<int64_t> time_results;

  clock_t c_start, c_end;
  int64_t time_elapsed_ms;
  for (size_t i = 0; i < cubes.size(); ++i)
  {
    //Trigger PDB generation before beginning search to remove from timing
    Rubiks::pattern_lookup(Rubiks::__goal, type);
    Rubiks::pattern_lookup(__goal, cubes[i], type);

    #ifdef ASTAR
    c_start = clock();
    count_results.push_back(search::ida_star(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "IDA* CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
    #ifdef DIBBS
    c_start = clock();
    count_results.push_back(search::id_dibbs(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "DIBBS CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
    #ifdef GBFHS
    c_start = clock();
    count_results.push_back(search::id_gbfhs(cubes[i], type));
    c_end = clock();
    time_elapsed_ms = (c_end - c_start) / CLOCKS_PER_SEC;
    time_results.push_back(time_elapsed_ms);
    cout << "GBFHS CPU time used: " << time_elapsed_ms << " s" << endl;
    #endif
    Rubiks::pattern_lookup(nullptr, cubes[i], Rubiks::PDB::clear_state);
  }

  for (size_t i = 0; i < count_results.size(); ++i) {
    std::cout << i << " " << count_results[i] << " nodes expanded; " << time_results[i] << " s to solve\n";
  }
}

int main()
{
  std::cout << "Size of Node=" << sizeof(Node) << std::endl;
  uint8_t* test_state = RubiksLoader::scramble("B2 L' F' U2 R2 D  R2 B  U2 R B' L  R  F F' R' L' B R' U2 B' R2 D' R2 U2 F L B2");
  for (int i = 0; i < 40; ++i) {
    if (Rubiks::__goal[i] != test_state[i]) {
      throw std::exception("Failed to rotate and counter-rotate properly.");
    }
  }
  delete[] test_state;
  search_cubes();
  /*vector<uint8_t*> cubes = RubiksLoader::load_cubes("korf1997.txt");
  for each (auto cube in cubes) {
    generate_statistics(cube, Rubiks::PDB::a888);
    Rubiks::pattern_lookup(nullptr, cube, Rubiks::PDB::clear_state);
  }*/

  return 0;
}
