#include "rubiks.h"

void Rubiks::rotate(uint8_t* __restrict new_state, const uint8_t face, const uint8_t rotation)
{
  static constexpr unsigned int __turn_position_lookup[18][20] =
  {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 15, 12, 18, 13, 19, 16, 14 },
  { 0, 1, 2, 3, 4, 7, 11, 19, 8, 9, 6, 18, 12, 13, 14, 15, 16, 5, 10, 17 },
  { 5, 1, 2, 10, 4, 17, 6, 7, 3, 9, 15, 11, 0, 13, 14, 8, 16, 12, 18, 19 },
  { 2, 4, 7, 1, 6, 0, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 },
  { 12, 8, 0, 3, 4, 5, 6, 7, 13, 1, 10, 11, 14, 9, 2, 15, 16, 17, 18, 19 },
  { 0, 1, 14, 3, 9, 5, 6, 2, 8, 16, 10, 4, 12, 13, 19, 15, 11, 17, 18, 7 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 13, 18, 12, 15, 17 },
  { 0, 1, 2, 3, 4, 17, 10, 5, 8, 9, 18, 6, 12, 13, 14, 15, 16, 19, 11, 7 },
  { 12, 1, 2, 8, 4, 0, 6, 7, 15, 9, 3, 11, 17, 13, 14, 10, 16, 5, 18, 19 },
  { 5, 3, 0, 6, 1, 7, 4, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 },
  { 2, 9, 14, 3, 4, 5, 6, 7, 1, 13, 10, 11, 0, 8, 12, 15, 16, 17, 18, 19 },
  { 0, 1, 7, 3, 11, 5, 6, 19, 8, 4, 10, 16, 12, 13, 2, 15, 9, 17, 18, 14 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 18, 17, 16, 15, 14, 13, 12 },
  { 0, 1, 2, 3, 4, 19, 18, 17, 8, 9, 11, 10, 12, 13, 14, 15, 16, 7, 6, 5 },
  { 17, 1, 2, 15, 4, 12, 6, 7, 10, 9, 8, 11, 5, 13, 14, 3, 16, 0, 18, 19 },
  { 7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 },
  { 14, 13, 12, 3, 4, 5, 6, 7, 9, 8, 10, 11, 2, 1, 0, 15, 16, 17, 18, 19 },
  { 0, 1, 19, 3, 16, 5, 6, 14, 8, 11, 10, 9, 12, 13, 7, 15, 4, 17, 18, 2 }
  };

  static constexpr uint8_t __corner_rotation[][3] =
  {
    { 0, 2, 1 },
  { 1, 0, 2 },
  { 2, 1, 0 },
  { 0, 2, 1 },
  { 1, 0, 2 },
  { 2, 1, 0 }
  };

  static constexpr bool __turn_lookup[6][20] =
  {
    { false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true },
  { false, false, false, false, false, true, true, true, false, false, true, true, false, false, false, false, false, true, true, true },
  { true, false, false, true, false, true, false, false, true, false, true, false, true, false, false, true, false, true, false, false },
  { true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false },
  { true, true, true, false, false, false, false, false, true, true, false, false, true, true, true, false, false, false, false, false },
  { false, false, true, false, true, false, false, true, false, true, false, true, false, false, true, false, true, false, false, true },
  };

  static constexpr bool __corner_booleans[] = {
    true, false, true, false,
    false, true, false, true,
    false, false, false, false,
    true, false, true, false,
    false, true, false, true
  };

  const uint8_t rotation_index = 6 * rotation + face;
  const unsigned int* __restrict turn_pos = __turn_position_lookup[rotation_index];
  if (rotation == 2)
  {
    for (int i = 0; i < 20; i++)
    {
      new_state[i] = turn_pos[new_state[i]];
    }
  }
  else
  {
    const bool* __restrict do_turn = __turn_lookup[face];
    const uint8_t* __restrict corner_rotation = __corner_rotation[face];
    for (int i = 20; i < 40; i++) {
      if (do_turn[new_state[i - 20]])
      {
        if (__corner_booleans[i - 20])
        {
          new_state[i] = corner_rotation[new_state[i]];
        }
        else if (face == 2 || face == 5) // Face left and right
        {
          new_state[i] = 1 - new_state[i];
        }
      }
    }
    for (int i = 0; i < 20; ++i)
    {
      new_state[i] = turn_pos[new_state[i]];
    }
  }
}


uint32_t Rubiks::get_corner_index(const uint8_t* state)
{
  constexpr int size = 8;
  uint8_t puzzle[size];
  uint8_t dual[size];

  for (uint8_t i = 0; i < 8; ++i) {
    dual[i] = __cube_translations[state[__corner_pos_indices[i]]];
    puzzle[dual[i]] = i;
  }

  uint32_t corner_index = (uint32_t)mr::k_rank(puzzle, dual, 8, 8);

  uint32_t rot_index = 0;
  for (int i = 0; i < size - 1; ++i)
  {
    rot_index += state[__corner_rot_indices[i]] * base3[i];
  }
  return corner_index * 2187 + rot_index;
}

void Rubiks::restore_corner(const size_t hash_index, uint8_t* state) {
  uint8_t puzzle[12];
  uint8_t dual[12];
  const size_t pos_index = hash_index / 2187ui64;
  mr::unrank(pos_index, puzzle, dual, 8, 8);
  for (int i = 0; i < 20; ++i) {
    state[i] = 20ui8;
  }
  for (int i = 20; i < 40; ++i) {
    state[i] = 3ui8;
  }

  for (int i = 0; i < 8; ++i) {
    state[__corner_pos_indices[i]] = __corner_pos_indices[dual[i]];
  }

  size_t rot_index = hash_index % 2187ui64;
  for (int i = 6; i >= 0; --i)
  {
    state[__corner_rot_indices[i]] = rot_index % 3ui8;
    rot_index /= 3;
  }
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
      if (is_a)
        return get_new_edge_pos_index(state);
      else
        return get_new_edge_rot_index(state);
    case PDB::zero:
      throw std::runtime_error("Tried to get edge index when using zero heuristic.");
  }
  throw std::runtime_error("Failed to find edge_index type");
}


uint64_t Rubiks::get_edge_index(const uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices)
{
  uint8_t puzzle[12] = { UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX };
  uint8_t dual[12];
  uint8_t newdual[12];

  for (uint8_t i = 0; i < 12; ++i) {
    dual[i] = __cube_translations[state[edge_pos_indices_12[i]]];
  }

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

void Rubiks::restore_state_from_index(const size_t hash_index, uint8_t* state, const int size, const uint8_t* edges, const uint8_t* edge_rot_indices)
{
  uint8_t puzzle[12];
  uint8_t dual[12];
  const size_t pos_index = hash_index / (1i64 << size);
  mr::unrank(pos_index, puzzle, dual, size, 12);
  for (int i = 0; i < 20; ++i) {
    state[i] = 20ui8;
  }
  for (int i = 20; i < 40; ++i) {
    state[i] = 3ui8;
  }

  for (int i = 0; i < size; ++i) {
    state[edge_pos_indices_12[edges[i]]] = edge_pos_indices_12[dual[i]];
  }

  size_t rot_index = hash_index % (1i64 << size);
  for (int i = size - 1; i >= 0; --i)
  {
    state[edge_rot_indices[i]] = rot_index % 2;
    rot_index /= 2;
  }
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

void Rubiks::restore_new_edge_rot_index(const size_t index, uint8_t* state) {

}
void Rubiks::restore_new_edge_pos_index(const size_t index, uint8_t* state) {

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
    if (cube[i] != __goal[i]) {
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

  static std::vector<const uint8_t*> initialized_pdbs_locs;
  static std::vector<std::shared_ptr<PDBVectors>> initialized_pdbs;
  std::shared_ptr<PDBVectors>  vectors = nullptr;

  if (type == PDB::clear_state) {
    for (int i = 0; i < initialized_pdbs_locs.size(); ++i) {
      if (memcmp(initialized_pdbs_locs[i], start_state, 40) == 0) {
        initialized_pdbs_locs.erase(initialized_pdbs_locs.begin() + i);
        initialized_pdbs.erase(initialized_pdbs.begin() + i);
      }
    }
    return 0;
  }

  for (int i = 0; i < initialized_pdbs_locs.size(); ++i) {
    if (memcmp(initialized_pdbs_locs[i], start_state, 40) == 0) {
      vectors = initialized_pdbs[i];
      break;
    }
  }
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
      generate_pattern_database_multithreaded(corner_name, start_state, corner_max_count, inconsistent_max_depth, get_corner_index, restore_corner);
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
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_6_max_count, inconsistent_max_depth, get_edge_index6a, restore_index6a);
      }
    }
    else if (type == PDB::a888)
    {
      edge_name_a = "edge_db_8a_" + name + ".npy";

      if (!utility::test_file(edge_name_a))
      {
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_8_max_count, inconsistent_max_depth, get_edge_index8a, restore_index8a);
      }
    }
    else if (type == PDB::a12) {
      edge_name_a = "edge_db_12_pos_" + name + ".npy";
      if (!utility::test_file(edge_name_a))
      {
        generate_pattern_database_multithreaded(edge_name_a, start_state, edge_12_pos_max_count, edge_12_pos_max_depth, get_new_edge_pos_index, restore_new_edge_pos_index);
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
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_6_max_count, inconsistent_max_depth, get_edge_index6b, restore_index6b);
      }
    }
    else if (type == PDB::a888)
    {
      edge_name_b = "edge_db_8b_" + name + ".npy";

      if (!utility::test_file(edge_name_b))
      {
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_8_max_count, inconsistent_max_depth, get_edge_index8b, restore_index8b);
      }
    }
    else if (type == PDB::a12)
    {
      edge_name_b = "edge_db_12_rot_" + name + ".npy";

      if (!utility::test_file(edge_name_b))
      {
        generate_pattern_database_multithreaded(edge_name_b, start_state, edge_20_rot_max_count, edge_20_rot_max_depth, get_new_edge_rot_index, restore_new_edge_rot_index);
      }
    }
    npy::LoadArrayFromNumpy<uint8_t>(edge_name_b, shape, vectors->edge_b);
  }

  uint8_t best = vectors->corner_db[get_corner_index(state)];
  best = std::max(best, vectors->edge_a[get_edge_index(state, true, type)]);
  best = std::max(best, vectors->edge_b[get_edge_index(state, false, type)]);
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

void Rubiks::process_buffer(
  std::vector<uint8_t>& pattern_lookup,
  std::atomic_size_t& count,
  const std::vector<uint64_t>& local_results_buffer,
  const uint8_t id_depth,
  const size_t max_count)
{
  size_t divisor = 10000000;
  for (size_t i = 0; i < local_results_buffer.size(); ++i) {
    auto id = local_results_buffer[i];
    if (pattern_lookup[id] > id_depth) {
      bool new_value = pattern_lookup[id] == pdb_initialization_value;
      pattern_lookup[id] = id_depth;
      if (new_value) {
        count++;
        size_t remaining = max_count - count;
        if (remaining > 0) {
          while (remaining / divisor == 0) divisor /= 10;
          if (remaining % divisor == 0) {
            std::cout << remaining << '\n';
          }
        }
      }
      else {
        std::cout << "Not a new value\n";
      }
    }
  }
}

void Rubiks::pdb_expand_nodes(
  moodycamel::ConcurrentQueue<std::pair<uint64_t, uint64_t>>& input_queue,
  std::atomic_size_t& count,
  const size_t max_count,
  std::mutex& pattern_lookup_mutex,
  std::vector<uint8_t>& pattern_lookup,
  const std::function<size_t(const uint8_t* state)> lookup_func,
  const std::function<void(const size_t hash, uint8_t* state)> reverse_func,
  const uint8_t id_depth,
  const bool reverse_direction
)
{
  using namespace std::chrono_literals;
  const size_t buffer_size = 1024 * 1024;
  moodycamel::ConsumerToken ctok(input_queue);
  std::vector<uint64_t> local_results_buffer;

  std::tuple<uint64_t, uint64_t> pair;
  uint8_t tmp_state[40], tmp2[40];

  while (count < max_count) {

    while (input_queue.try_dequeue(ctok, pair) == false) {
      std::this_thread::sleep_for(10ms);
    }

    if (std::get<0>(pair) == UINT64_MAX && std::get<1>(pair) == UINT64_MAX) {
      break;
    }

    if (local_results_buffer.size() >= buffer_size) {
      pattern_lookup_mutex.lock();
      process_buffer(pattern_lookup, count, local_results_buffer, id_depth, max_count);
      pattern_lookup_mutex.unlock();
      local_results_buffer.clear();
    }

    for (auto hash = std::get<0>(pair), end = std::get<1>(pair); hash < end; ++hash) {
      reverse_func(hash, tmp_state);
      for (int face = 0; face < 6; ++face)
      {
        for (int rotation = 0; rotation < 3; ++rotation)
        {
          memcpy(tmp2, tmp_state, 40);
          rotate(tmp2, face, rotation);
          uint64_t tmp2_hash = lookup_func(tmp2);
          if (reverse_direction && pattern_lookup[tmp2_hash] == (id_depth - 1)) {
            local_results_buffer.emplace_back(hash);
          }
          else if (!reverse_direction) {
            local_results_buffer.emplace_back(tmp2_hash);
          }
        }
      }
    }
  }

  if (local_results_buffer.size() > 0) {
    pattern_lookup_mutex.lock();
    process_buffer(pattern_lookup, count, local_results_buffer, id_depth, max_count);
    pattern_lookup_mutex.unlock();
  }
}

void Rubiks::generate_pattern_database_multithreaded(
  const std::string filename,
  const uint8_t* state,
  const size_t max_count,
  const uint8_t max_depth,
  const std::function<size_t(const uint8_t* state)> lookup_func,
  const std::function<void(const size_t hash, uint8_t* state)> reverse_func
)
{
  using namespace std::chrono_literals;
  const unsigned int thread_count = std::thread::hardware_concurrency() - 1;


  std::cout << "Generating PDB\n";
  std::cout << "Count: " << max_count << "\n";
  std::vector<uint8_t> pattern_lookup(max_count);
  std::atomic_size_t count = 0;
  for (size_t i = 0; i < pattern_lookup.size(); ++i) {
    pattern_lookup[i] = pdb_initialization_value;
  }

  uint8_t reduced_starting_state[40];
  reverse_func(lookup_func(state), reduced_starting_state);
  if (lookup_func(state) != lookup_func(reduced_starting_state)) {
    std::cerr << "ERROR: lookup function is not reversible\n";
    exit(-1);
  }

  pattern_lookup[lookup_func(reduced_starting_state)] = 0;
  count += 1;

  assert(max_depth > 3);
  uint8_t id_depth = 2;
  moodycamel::ConcurrentQueue<std::pair<uint64_t, uint64_t>> input_queue(67108860); // 64 MB
  moodycamel::ProducerToken ptok(input_queue);
  std::thread* threads = new std::thread[thread_count];
  std::mutex pattern_mutex;

  //Initialize input queue by exploring down to depth 2 to create up to 252 (18*14) separate work inputs to divide amongst threads
  std::vector<RubiksIndex> initial_storage;
  for (int face = 0; face < 6; ++face)
  {
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      RubiksIndex ri = RubiksIndex(reduced_starting_state, 1, face, rotation);
      ri.index = lookup_func(ri.state);
      if (pattern_lookup[ri.index] == pdb_initialization_value) {
        pattern_lookup[ri.index] = 1;
        count += 1;
      }
    }
  }
  for (int face = 0; face < 6; ++face)
  {
    for (int rotation = 0; rotation < 3; ++rotation)
    {
      RubiksIndex ri = RubiksIndex(reduced_starting_state, 1, face, rotation);
      ri.index = lookup_func(ri.state);
      for (int face2 = 0; face2 < 6; ++face2)
      {
        if (Rubiks::skip_rotations(face, face2))
        {
          continue;
        }
        for (int rotation2 = 0; rotation2 < 3; ++rotation2)
        {
          RubiksIndex ri2 = RubiksIndex(ri.state, 2, face2, rotation2);
          ri2.index = lookup_func(ri2.state);
          if (pattern_lookup[ri2.index] == pdb_initialization_value) {
            pattern_lookup[ri2.index] = pattern_lookup[ri.index] + 1;
            count += 1;
          }
        }
      }
    }
  }

  while (count < max_count && id_depth < max_depth)
  {
    uint8_t target_val;
    bool reverse_direction;
    if (count < (max_count * .66)) {
      reverse_direction = false;
      target_val = id_depth;
    }
    else {
      reverse_direction = true;
      target_val = pdb_initialization_value;
    }

    id_depth += 1;
    std::cout << "Incrementing id-depth to " << unsigned int(id_depth) << "\n";
    //Start threads
    for (unsigned int i = 0; i < thread_count; ++i) {
      threads[i] = std::thread(Rubiks::pdb_expand_nodes, std::ref(input_queue), std::ref(count), max_count, std::ref(pattern_mutex), std::ref(pattern_lookup), lookup_func, reverse_func, id_depth, reverse_direction);
    }

    uint64_t start;
    bool set_start(false);
    for (size_t i = 0; i < max_count; ++i) {
      uint8_t val = pattern_lookup[i];
      if (!set_start && val == target_val) {
        set_start = true;
        start = i;
      }
      else if (set_start && val != target_val) {
        set_start = false;
        while (input_queue.try_enqueue(ptok, { start, i }) == false) {
          std::this_thread::sleep_for(10ms);
        }
      }
    }
    if (set_start) {
      while (input_queue.try_enqueue(ptok, { start, max_count }) == false) {
        std::this_thread::sleep_for(10ms);
      }
    }

    //Queue a termination notification for each thread
    for (unsigned int i = 0; i < thread_count; ++i) {
      while (input_queue.try_enqueue(ptok, { UINT64_MAX, UINT64_MAX }) == false) {
        std::this_thread::sleep_for(10ms);
      }
    }

    //Wait for all threads to terminate
    for (unsigned int i = 0; i < thread_count; ++i) {
      threads[i].join();
    }
  }
  for (size_t i = 0; i < max_count; ++i) {
    if (max_depth < pdb_initialization_value && pattern_lookup[i] == pdb_initialization_value) {
      pattern_lookup[i] = max_depth;
    }
    else if (pattern_lookup[i] == pdb_initialization_value) {
      std::cerr << "Error: index " << i << " was not set to a valid value." << std::endl;
      throw new std::exception("Error: index was not set to a valid value.");
    }
  }

  std::cout << "Saving data to " << filename << '\n';

  const uint64_t shape[] = { max_count };
  npy::SaveArrayAsNumpy<uint8_t>(filename, false, 1, shape, pattern_lookup);
}
