#include "rubiks.h"


void Rubiks::rotate(uint8_t* new_state, const uint8_t face, const uint8_t rotation)
{
  const uint8_t rotation_index = 6 * rotation + face;
  if (rotation == 2)
  {
    for (int i = 0; i < 20; i++)
    {
      new_state[i * 2] = __turn_position_lookup[new_state[i * 2]][rotation_index];
    }
  }
  else
  {
    for (int i = 0; i < 20; i++)
    {
      uint8_t index = i * 2;
      if (__turn_lookup[new_state[index]][face])
      {
        if (__corner_booleans[index])
        {
          new_state[index + 1] = __corner_rotation[face][new_state[index + 1]];
        }
        else if (face == 2 || face == 5) // Face left and right
        {
          new_state[index + 1] = 1 - new_state[index + 1];
        }
        new_state[index] = __turn_position_lookup[new_state[index]][rotation_index];
      }
    }
  }
}


uint32_t Rubiks::get_corner_index(const uint8_t* state)
{
  const static int size = 8;
  uint8_t corners[size];

  for (int i = 0; i < size; ++i)
  {
    corners[i] = __cube_translations[state[__corner_pos_indices[i]]];
  }

  uint32_t corner_index = (uint32_t)mr::get_rank(size, corners);
  corner_index *= 2187;

  for (int i = 0; i < size - 1; ++i)
  {
    corner_index += state[__corner_rot_indices[i]] * base3[i];
  }
  return corner_index;
}

uint64_t FactorialUpperK(const int n, const int k)
{
  static const uint64_t result[13][13] =
  {
    { 1 }, // n = 0
  { 1, 1 }, // n = 1
  { 2, 2, 1 }, // n = 2
  { 6, 6, 3, 1 }, // n = 3
  { 24, 24, 12, 4, 1 }, // n = 4
  { 120, 120, 60, 20, 5, 1 }, // n = 5
  { 720, 720, 360, 120, 30, 6, 1 }, // n = 6
  { 5040, 5040, 2520, 840, 210, 42, 7, 1 }, // n = 7
  { 40320, 40320, 20160, 6720, 1680, 336, 56, 8, 1 }, // n = 8
  { 362880, 362880, 181440, 60480, 15120, 3024, 504, 72, 9, 1 }, // n = 9
  { 3628800, 3628800, 1814400, 604800, 151200, 30240, 5040, 720, 90, 10, 1 }, // n = 10
  { 39916800, 39916800, 19958400, 6652800, 1663200, 332640, 55440, 7920, 990, 110, 11, 1 }, // n = 11
  { 479001600, 479001600, 239500800, 79833600, 19958400, 3991680, 665280, 95040, 11880, 1320, 132, 12, 1 } // n = 12
  };
  return result[n][k];
}


uint64_t Rubiks::get_edge_index(const uint8_t* state, const bool is_a, const Rubiks::PDB type)
{
  switch (type)
  {
    case PDB::a1997:
      if (is_a)
        return get_edge_index6a(state);
      else
        return get_edge_index6b(state);
    case PDB::a888:
      if (is_a)
        return get_edge_index8a(state);
      else
        return get_edge_index8b(state);
    case PDB::a12:
      return get_new_edge_pos_index(state);
    case PDB::zero:
      throw std::runtime_error("Tried to get edge index when using zero heuristic.");
  }
  throw std::runtime_error("Failed to find edge_index type");
}

uint64_t Rubiks::get_edge_index(const uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices)
{
  uint8_t edge_pos[12];
  for (int i = 0; i < 12; ++i)
  {
    edge_pos[i] = __cube_translations[state[edge_pos_indices_12[i]]];
  }
  uint8_t puzzle[12] = { UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX };
  uint8_t dual[12];
  uint8_t newdual[12];
  for (int x = 0; x < 12; x++)
    dual[edge_pos[x]] = x;
  for (int x = 0; x < size; x++)
  {
    newdual[x] = dual[edges[x]];
    puzzle[dual[edges[x]]] = x;
  }
  uint64_t edge_index = mr::k_rank(puzzle, newdual, size, 12);

  edge_index *= 1i64 << size;

  for (int i = 0; i < size; ++i)
  {
    edge_index += state[edge_rot_indices[i]] * 1i64 << (size - i - 1);
  }
  return edge_index;
}

uint64_t Rubiks::get_new_edge_pos_index(const uint8_t* state)
{
  uint8_t edge_pos[12];
  for (int i = 0; i < 12; ++i)
  {
    edge_pos[i] = __cube_translations[state[edge_pos_indices_12[i]]];
  }

  return mr::get_rank(12, edge_pos);
}

uint64_t Rubiks::get_new_edge_rot_index(const uint8_t* state) {
  uint8_t edge_rots[12];
  for (int i = 0; i < 12; ++i)
  {
    edge_rots[__cube_translations[state[edge_pos_indices_12[i]]]] = state[edge_rot_indices_12[i]];
  }

  uint64_t edge_index(0);
  for (int i = 0; i < 11; ++i)
  {
    edge_index += uint64_t(edge_rots[i]) * base2[i];
  }
  edge_index *= 2187;
  for (int i = 0; i < 8; ++i)
  {
    edge_rots[__cube_translations[state[__corner_pos_indices[i]]]] = state[__corner_rot_indices[i]];
  }
  for (int i = 0; i < 7; ++i)
  {
    edge_index += uint64_t(edge_rots[i]) * base3[i];
  }
  return edge_index;
}


bool Rubiks::is_solved(const uint8_t* cube)
{
  for (int i = 0; i < 40; ++i)
  {
    if (__goal[i] != cube[i])
    {
      return false;
    }
  }
  return true;
}


uint8_t Rubiks::pattern_lookup(const uint8_t* state, const uint8_t* start_state, const PDB type)
{
  if (type == PDB::zero)
  {
    return 0;
  }

  // static std::unordered_map<const uint8_t*, PDBVectors*, StateHash, StateHash> initialized_pdbs;
  static std::vector<const uint8_t*> initialized_pdbs_locs;
  static std::vector<std::shared_ptr<PDBVectors>> initialized_pdbs;
  std::shared_ptr<PDBVectors>  vectors = nullptr;
  for (int i = 0; i < initialized_pdbs_locs.size(); ++i) {
    if (memcmp(initialized_pdbs_locs[i], start_state, 40) == 0) {
      vectors = initialized_pdbs[i];
      break;
    }
  }

  uint64_t rot_index = get_new_edge_rot_index(state);
  uint64_t pos_index = get_new_edge_pos_index(state);
  uint64_t corner_index = get_corner_index(state);
  if (vectors == nullptr)
  {
    vectors = std::make_shared<PDBVectors>();
    initialized_pdbs_locs.push_back(start_state);
    initialized_pdbs.push_back(vectors);

    std::string name;
    for (int i = 0; i < 40; ++i)
    {
      name += std::to_string(start_state[i]);
    }
    std::cout << "Loading PDBs from disk for " << name << '\n';
    std::string corner_name = "corner_db_" + name + ".npy";
    if (!utility::test_file(corner_name))
    {
      generate_pattern_database_multithreaded(corner_name, start_state, corner_max_depth, corner_max_count, get_corner_index);
    }
    std::vector<uint64_t> shape{ 1 };
    npy::LoadArrayFromNumpy<uint8_t>(corner_name, shape, vectors->corner_db);

    shape.clear();
    shape.push_back(1);
    std::string edge_name_a;
    if (type == PDB::a1997)
    {
      edge_name_a = "edge_db_6a_" + name + ".npy";

      if (!utility::test_file(edge_name_a))
      {
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_6_max_depth, edge_6_max_count, get_edge_index6a);
      }
    }
    else if (type == PDB::a888)
    {
      edge_name_a = "edge_db_8a_" + name + ".npy";

      if (!utility::test_file(edge_name_a))
      {
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_8_max_depth, edge_8_max_count, get_edge_index8a);
      }
    }
    else if (type == PDB::a12) {
      edge_name_a = "edge_db_12_pos_" + name + ".npy";
      if (!utility::test_file(edge_name_a))
      {
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_12_pos_max_depth, edge_12_pos_max_count, get_new_edge_pos_index);
      }
    }

    npy::LoadArrayFromNumpy<uint8_t>(edge_name_a, shape, vectors->edge_a);


    shape.clear();
    shape.push_back(1);
    std::string edge_name_b;
    if (type == PDB::a1997)
    {
      edge_name_b = "edge_db_6b_" + name + ".npy";

      if (!utility::test_file(edge_name_b))
      {
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_6_max_depth, edge_6_max_count, get_edge_index6b);
      }
    }
    else if (type == PDB::a888)
    {
      edge_name_b = "edge_db_8b_" + name + ".npy";

      if (!utility::test_file(edge_name_b))
      {
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_8_max_depth, edge_8_max_count, get_edge_index8b);
      }
    }
    else if (type == PDB::a12)
    {
      edge_name_b = "edge_db_12_rot_" + name + ".npy";

      if (!utility::test_file(edge_name_b))
      {
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_20_rot_max_depth, edge_20_rot_max_count, get_new_edge_rot_index);
      }
    }
    npy::LoadArrayFromNumpy<uint8_t>(edge_name_b, shape, vectors->edge_b);
  }
  uint8_t best = vectors->corner_db[corner_index];
  if (type == PDB::a12) {
    uint8_t a = vectors->edge_a[pos_index];
    uint8_t b = vectors->edge_b[rot_index];
    if (a > best) best = a;
    if (b > best) best = b;
  }
  else {
    best = std::max(best, vectors->edge_a[get_edge_index(state, true, type)]);
    best = std::max(best, vectors->edge_b[get_edge_index(state, false, type)]);
  }
  return best;
}

void Rubiks::generate_pattern_database(
  const std::string filename,
  const uint8_t* state,
  const uint8_t max_depth,
  const size_t max_count,
  const std::function<size_t(const uint8_t* state)> lookup_func
)
{
  std::cout << "Generating edges db\n";
  std::stack<RubiksIndex> stack;
  RubiksIndex ri(state, 0, 0);
  stack.push(ri);

  //uint64_t all_edges = max_count//npr(12, size) * uint64_t(pow(2, size));
  std::cout << "Edges: " << max_count << "\n";
  std::vector<uint8_t> pattern_lookup(max_count);
  std::fill(pattern_lookup.begin(), pattern_lookup.end(), max_depth);

  pattern_lookup[lookup_func(state)] = 0;

  uint8_t id_depth = 1;
  uint64_t count = 1;

  while (count < max_count && id_depth < max_depth)
  {
    if (stack.empty())
    {
      id_depth += 1;
      stack.push(ri);
      std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << "\n";
    }
    auto prev_ri = stack.top();
    stack.pop();
    auto prev_index = lookup_func(prev_ri.state);
    auto prev_db_val = pattern_lookup[prev_index];
    for (int face = 0; face < 6; ++face)
    {
      if (prev_ri.depth > 0 && Rubiks::skip_rotations(prev_ri.last_face, face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        uint8_t new_state_depth = prev_ri.depth + 1;
        RubiksIndex next_ri(prev_ri.state, new_state_depth, face);
        rotate(next_ri.state, face, rotation);
        uint64_t new_state_index = lookup_func(next_ri.state);
        if (new_state_depth < id_depth)
        {
          stack.push(next_ri);
        }
        else if (pattern_lookup[new_state_index] == max_depth)
        {
          pattern_lookup[new_state_index] = prev_db_val + 1;
          ++count;
          if (count % 10000000 == 0)
          {
            std::cout << count << "\n";
          }
        }
      }
    }
  }
  for (int i = 0; i < max_count; ++i) {
    if (pattern_lookup[i] > max_depth) {
      std::cout << "ERROR in PDB Generation, index " << i << " has depth = " << unsigned int(pattern_lookup[i]) << std::endl;
    }
  }
  const uint64_t shape[] = { max_count };
  npy::SaveArrayAsNumpy<uint8_t>(filename, false, 1, shape, pattern_lookup);
}

void Rubiks::pdb_expand_nodes(
  moodycamel::ConcurrentQueue<PDB_Value>& results_queue,
  moodycamel::BlockingConcurrentQueue<RubiksIndex>& input_queue,
  std::vector<uint8_t>& pattern_lookup,
  const std::function<size_t(const uint8_t* state)> lookup_func,
  const uint8_t id_depth,
  std::atomic_bool& finished
)
{
  std::stack<RubiksIndex> stack;
  moodycamel::ProducerToken ptok(results_queue);
  bool done = false;
  while (!done) {
    if (stack.empty()) {
      RubiksIndex queue_ri;
      if (input_queue.try_dequeue(queue_ri) == false) {
        done = true;
        break;
      }
      else {
        stack.push(queue_ri);
      }
    }
    auto prev_ri = stack.top();
    stack.pop();
    auto prev_index = lookup_func(prev_ri.state);
    auto prev_db_val = pattern_lookup[prev_index];
    for (int face = 0; face < 6; ++face)
    {
      if (prev_ri.depth > 0 && Rubiks::skip_rotations(prev_ri.last_face, face))
      {
        continue;
      }
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        uint8_t new_state_depth = prev_ri.depth + 1;
        RubiksIndex next_ri(prev_ri.state, new_state_depth, face);
        rotate(next_ri.state, face, rotation);
        uint64_t new_state_index = lookup_func(next_ri.state);
        if (new_state_depth < id_depth)
        {
          stack.push(next_ri);
        }
        else
        {
          results_queue.enqueue(PDB_Value(new_state_index, prev_db_val + 1));
        }
      }
    }
  }

  finished = true;
}

void Rubiks::generate_pattern_database_multithreaded(
  const std::string filename,
  const uint8_t* state,
  const uint8_t max_depth,
  const size_t max_count,
  const std::function<size_t(const uint8_t* state)> lookup_func
)
{
  constexpr size_t thread_count = 4;
  constexpr size_t bulk_size = 50;

  std::cout << "Generating PDB\n";
  std::cout << "Count: " << max_count << "\n";
  std::vector<uint8_t> pattern_lookup(max_count);
  std::fill(pattern_lookup.begin(), pattern_lookup.end(), max_depth);

  pattern_lookup[lookup_func(state)] = 0;

  uint8_t id_depth = 2;
  size_t count = 1;
  moodycamel::BlockingConcurrentQueue<RubiksIndex> input_queue;
  moodycamel::ConcurrentQueue<PDB_Value> results_queue;
  moodycamel::ConsumerToken ctok(results_queue);
  PDB_Value results_array[200];
  using namespace std::chrono_literals;

  for (int face = 0; face < 6; ++face)
  {
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      uint8_t new_state_depth = 1;
      input_queue.enqueue(RubiksIndex(state, new_state_depth, face, rotation));
    }
  }

  while (count < max_count && id_depth < max_depth)
  {
    std::thread threads[thread_count];
    std::atomic_bool finished_flags[thread_count];
    for (unsigned int i = 0; i < thread_count; ++i) {
      finished_flags[i] = false;
      threads[i] = std::thread(Rubiks::pdb_expand_nodes, std::ref(results_queue), std::ref(input_queue), std::ref(pattern_lookup), lookup_func, id_depth, std::ref(finished_flags[i]));
    }

    auto is_done = [&finished_flags, thread_count]() {
      for (unsigned int i = 0; i < thread_count; ++i) {
        if (finished_flags[i] == false) return false;
      }
      return true;
    };

    while (!is_done()) {
      size_t num_results = results_queue.try_dequeue_bulk(ctok, results_array, 200);
      for (size_t i = 0; i < num_results; ++i) {
        auto [index, value] = results_array[i];
        if (pattern_lookup[index] > value) {
          pattern_lookup[index] = value;
          ++count;
          if (count % 10000000 == 0)
          {
            std::cout << count << "\n";
          }
        }
      }
      if (num_results == 0) {
        std::this_thread::sleep_for(1ms);
      }
    }

    for (unsigned int i = 0; i < thread_count; ++i) {
      threads[i].join();
    }

    id_depth += 1;
    std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << "\n";

    for (int face = 0; face < 6; ++face)
    {
      for (int rotation = 0; rotation < 3; ++rotation)
      {
        uint8_t new_state_depth = 1;
        input_queue.enqueue(RubiksIndex(state, new_state_depth, face, rotation));
      }
    }
  }
  for (int i = 0; i < max_count; ++i) {
    if (pattern_lookup[i] > max_depth) {
      std::cout << "ERROR in PDB Generation, index " << i << " has depth = " << unsigned int(pattern_lookup[i]) << std::endl;
    }
  }
  const uint64_t shape[] = { max_count };
  npy::SaveArrayAsNumpy<uint8_t>(filename, false, 1, shape, pattern_lookup);
}
