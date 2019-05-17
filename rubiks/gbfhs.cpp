#include "gbfhs.h"

typedef std::unordered_set<std::shared_ptr<Node>, NodeHash, NodeEqual> hash_set;
typedef std::vector<std::shared_ptr<Node> > vector;

void split(const uint8_t f_lim, uint8_t& g_lim_a, uint8_t& g_lim_b) {
  g_lim_b = f_lim / 2;
  g_lim_a = f_lim - g_lim_b;
}

bool is_expandable(const std::shared_ptr<Node> node, const uint8_t f_lim, const uint8_t g_lim) {
  return (node->combined <= f_lim && node->depth < g_lim);
}

std::shared_ptr<Node> make_gbfhs_node(const hash_set& other_set,
  std::shared_ptr<Node> prev_node,
  const uint8_t* start_state,
  const int face,
  const int rotation,
  const bool reverse,
  const Rubiks::PDB type,
  std::shared_ptr<Node>& best_node,
  uint8_t& upper_bound)
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

void expand_node(std::shared_ptr<Node> next_node,
  hash_set& my_set,
  const hash_set& other_set,
  uint8_t& upper_bound,
  std::shared_ptr<Node>& best_node,
  const bool reverse,
  const Rubiks::PDB type,
  const uint8_t* start_state,
  size_t& count,
  vector& my_list) {

  count += 1;
  if (count % 10000 == 0) {
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
      auto new_node = make_gbfhs_node(other_set, next_node, start_state, face, rotation, reverse, type, best_node, upper_bound);
      auto find = my_set.find(new_node);

      if (find != my_set.end() && (*find)->depth <= new_node->depth) {
        continue;
      }
      else if (find != my_set.end()) {
        my_set.erase(find);
        for (int i = 0; i < my_list.size(); ++i) {
          if (memcmp(my_list[i]->state, new_node->state, 40) == 0)
          {
            my_list.erase(my_list.begin() + i);
            break;
          }
        }
      }

      my_list.push_back(new_node);
      my_set.insert(new_node);
    }
  }
}

void expand_level(const uint8_t g_lim_f,
  const uint8_t g_lim_b,
  const uint8_t f_lim,
  hash_set & front_open,
  hash_set & back_open,
  vector & front_list,
  vector & back_list,
  size_t & count,
  const Rubiks::PDB type,
  uint8_t & upper_bound,
  std::shared_ptr<Node> & best_node,
  const uint8_t * start_state) {

  for (int node_index = 0; node_index < front_list.size(); ++node_index)
  {
    auto var = front_list[node_index];
    if (is_expandable(var, f_lim, g_lim_f)) {
      expand_node(var, front_open, back_open, upper_bound, best_node, false, type, start_state, count, front_list);
      front_open.erase(var);
      for (int i = 0; i < front_list.size(); ++i) {
        if (memcmp(front_list[i]->state, var->state, 40) == 0) {
          front_list.erase(front_list.begin() + i);
          break;
        }
      }
    }
  }
  for (int node_index = 0; node_index < back_list.size(); ++node_index)
  {
    auto var = back_list[node_index];
    if (is_expandable(var, f_lim, g_lim_b)) {
      expand_node(var, back_open, front_open, upper_bound, best_node, true, type, start_state, count, back_list);
      back_open.erase(var);
      for (int i = 0; i < back_list.size(); ++i) {
        if (memcmp(back_list[i]->state, var->state, 40) == 0) {
          back_list.erase(back_list.begin() + i);
          break;
        }
      }
    }
  }
}

size_t search::gbfhs(const uint8_t * start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "GBFHS" << std::endl;
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

  vector front_list, back_list;
  front_list.push_back(start);
  back_list.push_back(goal);

  bool explore_forward = true;
  std::shared_ptr<Node> best_node = nullptr;
  uint8_t best_cost = 255;

  size_t count = 0;

  uint8_t f_lim = std::max(start->heuristic, goal->heuristic);
  uint8_t g_lim_f, g_lim_b;

  while (best_cost > f_lim && !front_open.empty() && !back_open.empty())
  {
    std::cout << "Expanding f_lim = " << std::to_string(f_lim) << std::endl;
    split(f_lim, g_lim_f, g_lim_b);
    expand_level(g_lim_f, g_lim_b, f_lim, front_open, back_open, front_list, back_list, count, pdb_type, best_cost, best_node, start_state);
    f_lim += 1;
  }

  std::cout << "Solved GBFHS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}