#include "id_gbfhs.h"

typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;
typedef std::vector<std::shared_ptr<Node> > vector;
typedef std::set<std::shared_ptr<Node>, NodeCompare> tree_set;

void id_gbfhs_split(const uint8_t f_lim, uint8_t& g_lim_a, uint8_t& g_lim_b) {
  g_lim_b = f_lim / 2;
  //g_lim_b = 0;
  g_lim_a = f_lim - g_lim_b;
}

bool id_gbfhs_is_expandable(const std::shared_ptr<Node> node, const uint8_t f_lim, const uint8_t g_lim) {
  return (node->combined <= f_lim && node->depth < g_lim);
}

std::shared_ptr<Node> make_id_gbfhs_node(const hash_set & other_set,
  std::shared_ptr<Node> prev_node,
  const uint8_t * start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::shared_ptr<Node> & best_node,
  uint8_t & upper_bound)
{
  auto new_node = std::make_shared<Node>(prev_node, start_state, prev_node->depth + 1, face, rotation, reverse, type);

  uint8_t reverse_cost = 0;
  auto search = other_set.find(new_node);
  if (search != other_set.end())
  {
    reverse_cost = (*search)->depth;
    if (new_node->depth + reverse_cost < upper_bound)
    {
      upper_bound = new_node->depth + reverse_cost;
      best_node = new_node;
      best_node->set_reverse(*search);
      std::cout << "New upper bound: " << unsigned int(upper_bound) << std::endl;
    }
  }

  return new_node;
}


void id_gbfhs_expand_level(const uint8_t g_lim,
  const uint8_t f_lim,
  hash_set & my_open,
  hash_set & other_open,
  size_t & count,
  const Rubiks::PDB type,
  uint8_t & upper_bound,
  std::shared_ptr<Node> & best_node,
  const uint8_t * start_state,
  const bool is_reverse) {

  vector expandable;
  for each (auto var in my_open) {
    if (id_gbfhs_is_expandable(var, f_lim, g_lim)) {
      expandable.push_back(var);
    }
  }

  while (!expandable.empty()) {
    std::shared_ptr<Node> next_node = expandable.back();
    expandable.pop_back();
    my_open.erase(next_node);

    count += 1;
    if (count % 1000000 == 0) {
      std::cout << count << "\n";
    }

    for (int face = 0; face < 6; ++face)
    {
      if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        auto new_node = make_id_gbfhs_node(other_open, next_node, start_state, face, rotation, is_reverse, type, best_node, upper_bound);
        if (upper_bound <= f_lim) {
          return;
        }

        auto find = my_open.find(new_node);
        if (find != my_open.end() && (*find)->depth <= new_node->depth) {
          continue;
        }
        else if (find != my_open.end()) {
          my_open.erase(find);
        }
        my_open.insert(new_node);
        if (id_gbfhs_is_expandable(new_node, f_lim, g_lim)) {
          expandable.push_back(new_node);
        }
      }
    }
  }

}

size_t search::id_gbfhs(const uint8_t * start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "ID-GBFHS" << std::endl;
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  hash_set front_open, back_open;

  auto start = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  auto goal = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);

  front_open.insert(start);
  back_open.insert(goal);

  bool explore_forward = true;
  std::shared_ptr<Node> best_node = nullptr;
  uint8_t best_cost = 255;

  size_t count = 0;

  uint8_t f_lim = std::max(start->heuristic, goal->heuristic);
  uint8_t g_lim_f, g_lim_b;

  while (best_cost > f_lim && !front_open.empty() && !back_open.empty())
  {
    std::cout << "Expanding f_lim = " << std::to_string(f_lim) << std::endl;
    id_gbfhs_split(f_lim, g_lim_f, g_lim_b);
    id_gbfhs_expand_level(g_lim_f, f_lim, front_open, back_open, count, pdb_type, best_cost, best_node, start_state, false);
    id_gbfhs_expand_level(g_lim_b, f_lim, back_open, front_open, count, pdb_type, best_cost, best_node, start_state, true);
    f_lim += 1;
  }

  std::cout << "Solved GBFHS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}