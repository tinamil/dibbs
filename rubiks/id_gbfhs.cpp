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

bool is_perimiter(const std::shared_ptr<Node> node, const uint8_t f_lim, const uint8_t g_lim) {
  return (node->combined <= f_lim && node->depth == g_lim);
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
  hash_set & frontier,
  size_t & count,
  const Rubiks::PDB type,
  uint8_t & upper_bound,
  std::shared_ptr<Node> & best_node,
  const uint8_t * start_state,
  const bool is_reverse,
  const bool is_building_frontier,
  bool& is_finished,
  std::stack<std::shared_ptr<Node>, vector> & expandable,
  const size_t node_limit) {

  if (is_building_frontier) {
    frontier.clear();
  }

  int min_delta = 255;
  while (!expandable.empty()) {
    std::shared_ptr<Node> next_node = expandable.top();
    expandable.pop();

    count += 1;
    if (count % 1000000 == 0) {
      std::cout << count << " " << frontier.size() << "\n";
    }

    for (int face = 0; face < 6; ++face)
    {
      if (next_node->depth > 0 && Rubiks::skip_rotations(next_node->get_face(), face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        std::shared_ptr<Node> new_node;
        if (is_building_frontier) {
          new_node = std::make_shared<Node>(next_node, start_state, next_node->depth + 1, face, rotation, is_reverse, type);
          auto find = frontier.find(new_node);
          if (find != frontier.end() && (*find)->depth <= new_node->depth) {
            continue;
          }
          else if (find != frontier.end()) {
            frontier.erase(find);
          }
          if (is_perimiter(new_node, f_lim, g_lim)) {
            frontier.insert(new_node);
          }
        }
        else {
          new_node = make_id_gbfhs_node(frontier, next_node, start_state, face, rotation, is_reverse, type, best_node, upper_bound);
          if (upper_bound <= f_lim) {
            return;
          }
        }

        if (id_gbfhs_is_expandable(new_node, f_lim, g_lim)) {
          expandable.push(new_node);
        }
        else if (is_perimiter(new_node, f_lim, g_lim)) {
          int delta = new_node->depth - new_node->reverse_heuristic;
          if (delta < min_delta) {
            min_delta = delta;
          }
        }
      }
    }

    if (is_building_frontier && frontier.size() >= node_limit) {
      return;
    }
  }
  std::cout << "Min-delta = " << min_delta << std::endl;
  is_finished = true;
}

//TODO: Reverse directions to put the smaller size into the frontier and iterative deepening the larger size
//TODO: Use DIBBS, 2G
size_t search::id_gbfhs(const uint8_t * start_state, const Rubiks::PDB pdb_type)
{
  std::cout << "ID-GBFHS" << std::endl;

  const uint64_t max_nodes = 200000000Ui64; //200 million 
  const uint8_t epsilon = 1;

  if (Rubiks::is_solved(start_state))
  {
    std::cout << "Given a solved cube.  Nothing to solve." << std::endl;
    return 0;
  }

  hash_set frontier;

  const auto start = std::make_shared<Node>(start_state, Rubiks::__goal, pdb_type);
  const auto goal = std::make_shared<Node>(Rubiks::__goal, start_state, pdb_type);

  bool explore_forward = true;
  std::shared_ptr<Node> best_node = nullptr;
  uint8_t best_cost = 255;

  size_t count = 0;

  uint8_t f_lim = std::max(start->heuristic, goal->heuristic);
  uint8_t g_lim_f, g_lim_b;
  std::stack<std::shared_ptr<Node>, vector> forward_expandable_stack, backward_expandable_stack;
  while (best_cost > f_lim)
  {
    while (!forward_expandable_stack.empty()) {
      forward_expandable_stack.pop();
    }
    forward_expandable_stack.push(start);

    std::cout << "Expanding f_lim = " << std::to_string(f_lim) << " frontier = " << std::to_string(frontier.size()) << " count = " << std::to_string(count) << std::endl;
    bool is_finished = false;
    size_t start_node = 0;
    id_gbfhs_split(f_lim, g_lim_f, g_lim_b);
    while (!is_finished && best_cost > f_lim) {
      while (!backward_expandable_stack.empty()) {
        backward_expandable_stack.pop();
      }
      backward_expandable_stack.push(goal);
      const bool is_reverse = true;
      const bool is_building_frontier = true;
      id_gbfhs_expand_level(g_lim_f, f_lim, frontier, count, pdb_type, best_cost, best_node, start_state, !is_reverse, is_building_frontier, is_finished, forward_expandable_stack, max_nodes);
      std::cout << "Finished expanding forward cycle, " << std::to_string(frontier.size()) << "\n";
      bool throwaway_finished;
      id_gbfhs_expand_level(g_lim_b, f_lim, frontier, count, pdb_type, best_cost, best_node, start_state, is_reverse, !is_building_frontier, throwaway_finished, backward_expandable_stack, 0);
      std::cout << "Finished expanding backward cycle\n";
    }
    f_lim += 1;
  }

  std::cout << "Solved GBFHS: " << " Count = " << count << std::endl;
  std::cout << "Solution: " << best_node->print_solution() << std::endl;
  return count;
}