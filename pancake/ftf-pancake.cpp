#include "ftf-pancake.h"
#include <intrin.h>

  //static uint8_t gap_ftf(const hash_t hash_values[], std::unordered_map<hash_t, std::vector<const FTF_Pancake*>>& other_index)
  //{
  //  tsl::hopscotch_map<size_t, int> map_counter;
  //  uint8_t match = NUM_PANCAKES;
  //  for(int i = 1; i <= NUM_PANCAKES; ++i)
  //  {
  //    if(match < i) break; //Can add minimum g from other direction
  //    for(auto& other_pancake : other_index[hash_values[i]])
  //    {
  //      size_t ptr = reinterpret_cast<size_t>(other_pancake);
  //      int prev = map_counter.count(ptr);
  //      map_counter[ptr] = prev + 1;
  //      int tmp_match = NUM_PANCAKES - prev + 1 + other_pancake->g;
  //      if(tmp_match < match) match = tmp_match;
  //    }
  //  }
  //  return match;
  //}

  //static uint8_t gap_ftf2(const hash_t hash_values[], std::unordered_map<hash_t, std::vector<const FTF_Pancake*>>& other_index)
  //{
  //  uint8_t match = NUM_PANCAKES;
  //  const FTF_Pancake* ptr = nullptr;
  //  for(int i = 1; i <= NUM_PANCAKES; ++i)
  //  {
  //    if(match <= i) break;
  //    for(auto& other_pancake : other_index[hash_values[i]])
  //    {
  //      uint8_t tmp_match = 0;
  //      for(int hash_index = 1; hash_index <= NUM_PANCAKES; ++hash_index)
  //      {
  //        for(int other_hash_index = 1; other_hash_index <= NUM_PANCAKES; ++other_hash_index)
  //        {
  //          if(hash_values[hash_index] == other_pancake->hash_values[other_hash_index])
  //          {
  //            tmp_match += 1;
  //            break;
  //          }
  //        }
  //      }
  //      tmp_match = NUM_PANCAKES - tmp_match + other_pancake->g;
  //      if(tmp_match < match)
  //      {
  //        ptr = other_pancake;
  //        match = tmp_match;
  //        if(match <= i) break;
  //      }
  //    }
  //  }
  //  return match;
  //}

  //Copies pancake, applies a flip, and updates g/h/f values
__declspec(noinline) FTF_Pancake FTF_Pancake::apply_action(const int i, ftf_cudastructure& structure, ftf_matchstructure& structure2) const
{
  FTF_Pancake new_node(*this);
  #ifdef HISTORY
  new_node.actions.push_back(i);
  new_node.parent = this;
  #endif
  assert(i > 1 && i <= NUM_PANCAKES);
  new_node.apply_flip(i);
  //new_node.h = new_node.update_gap_lb(dir, i, new_node.h);
  /*if(i < NUM_PANCAKES)
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], new_node.source[i + 1]);
  else
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], NUM_PANCAKES + 1);*/
  for(int i = 1; i < NUM_PANCAKES; ++i)
  {
    new_node.hash_values[i] = hash_table::hash(new_node.source[i], new_node.source[i + 1]);
  }
  new_node.hash_values[NUM_PANCAKES] = hash_table::hash(new_node.source[NUM_PANCAKES], NUM_PANCAKES + 1);
  std::sort(std::begin(new_node.hash_values) + 1, std::end(new_node.hash_values));
  new_node.hash_64 = hash_table::compress(new_node.hash_values);
  auto h2 = structure2.match(&new_node);
  new_node.h = structure.match(&new_node);
  new_node.h = structure.match(&new_node);
  new_node.g = g + 1;
  new_node.f = new_node.g + new_node.h;
  //assert(new_node.f >= f); //Consistency check
  return new_node;
}

uint8_t FTF_Pancake::gap_lb(Direction dir) const
{
  unsigned char  LB;
  int            i;

  LB = 0;
  if(GAPX == -1) return(LB);

  if(dir == Direction::forward)
  {
    for(i = 2; i <= NUM_PANCAKES; i++)
    {
      if((source[i] <= GAPX) || (source[i - 1] <= GAPX)) continue;
      if(abs(source[i] - source[i - 1]) > 1) LB = LB + 1;
    }
    if((abs(NUM_PANCAKES + 1 - source[NUM_PANCAKES]) > 1) && (source[NUM_PANCAKES] > GAPX)) LB = LB + 1;
  }
  else if(Pancake::DUAL_SOURCE() != nullptr)
  {
    for(i = 2; i <= NUM_PANCAKES; i++)
    {
      if((source[i] <= GAPX) || (source[i - 1] <= GAPX)) continue;
      if(abs(Pancake::DUAL_SOURCE()[source[i]] - Pancake::DUAL_SOURCE()[source[i - 1]]) > 1) LB = LB + 1;
    }
    if((abs(NUM_PANCAKES + 1 - Pancake::DUAL_SOURCE()[source[NUM_PANCAKES]]) > 1) && (source[NUM_PANCAKES] > GAPX)) LB = LB + 1;
  }

  return(LB);
}

uint8_t FTF_Pancake::update_gap_lb(Direction dir, int i, uint8_t LB) const
{
  int            inv_p1, inv_pi, inv_pi1, p1, pi, pi1;

  if(GAPX == -1) return(0);

  assert((1 <= i) && (i <= NUM_PANCAKES));

  if(dir == Direction::forward)
  {
    p1 = source[1];
    pi = source[i];
    if(i < NUM_PANCAKES)
      pi1 = source[i + 1];
    else
      pi1 = NUM_PANCAKES + 1;

    if((pi <= GAPX) || (pi1 <= GAPX) || (abs(pi1 - pi) <= 1)) LB = LB + 1;
    if((p1 <= GAPX) || (pi1 <= GAPX) || (abs(pi1 - p1) <= 1)) LB = LB - 1;
  }
  else if(Pancake::DUAL_SOURCE() != nullptr)
  {
    p1 = source[1];
    pi = source[i];
    inv_p1 = Pancake::DUAL_SOURCE()[p1];
    inv_pi = Pancake::DUAL_SOURCE()[pi];
    if(i < NUM_PANCAKES)
    {
      pi1 = source[i + 1];
      inv_pi1 = Pancake::DUAL_SOURCE()[source[i + 1]];
    }
    else
    {
      pi1 = NUM_PANCAKES + 1;
      inv_pi1 = NUM_PANCAKES + 1;
    }
    if((pi <= GAPX) || (pi1 <= GAPX) || (abs(inv_pi1 - inv_pi) <= 1)) LB = LB + 1;
    if((p1 <= GAPX) || (pi1 <= GAPX) || (abs(inv_pi1 - inv_p1) <= 1)) LB = LB - 1;
  }

  return(LB);
}

bool FTF_Less::operator()(const FTF_Pancake* lhs, const FTF_Pancake* rhs) const
{
  int cmp = memcmp(lhs->source, rhs->source, NUM_PANCAKES + 1);
  if(cmp == 0)
  {
    return false;
  }
  else if(lhs->g == rhs->g)
  {
    return cmp < 0;
  }
  else
  {
    return lhs->g < rhs->g;
  }
}

__declspec(noinline) uint32_t ftf_cudastructure::match(const FTF_Pancake* val)
{
  //size_t m_rows, size_t n_cols, float alpha, float* A, float* x, float beta, float* y
  if(!valid_device_cache)
  {
    cuda.set_matrix(opposite_hash_values.size(), MAX_VAL, (float*)opposite_hash_values.data(), g_values.data());
    valid_device_cache = true;
  }
  float hash_vals[MAX_VAL];
  to_hash_array(val, hash_vals);
  return round(cuda.min_vector_matrix(
    opposite_hash_values.size(),
    MAX_VAL,
    hash_vals));
}

uint32_t ftf_matchstructure::match(const FTF_Pancake* val)
{
  static size_t counter = 0;
  uint8_t match = NUM_PANCAKES;
  const FTF_Pancake* ptr = nullptr;

  std::array<dataset_t*, NUM_PANCAKES> set;
  for(size_t i = 1; i <= NUM_PANCAKES; ++i)
  {
    set[i - 1] = &dataset[val->hash_values[i]];
  }
  std::sort(set.begin(), set.end(), [](dataset_t* a, dataset_t* b) {
    return a->size() < b->size();
  });

  //The largest value of match possible is NUM_PANCAKES - i
  for(size_t i = 0; i < NUM_PANCAKES; ++i)
  {
    if(match <= i) break;
    for(auto begin = set[i]->begin(), end = set[i]->end(); begin != end; ++begin)
    {
      const FTF_Pancake* other_pancake = (*begin);
      uint8_t tmp_match;
      if constexpr(SORTED_DATASET)
      {
        if(match <= i + other_pancake->g) break;
      }

      /*uint8_t pop_match = __popcnt64(val->hash_64 & other_pancake->hash_64);
      if(pop_match > NUM_PANCAKES - i)
      {
        continue;
      }*/

      uint8_t hash_index1 = 1;
      uint8_t hash_index2 = 1;

      tmp_match = 0;
      while(hash_index1 <= NUM_PANCAKES && hash_index2 <= NUM_PANCAKES)
      {
        if(val->hash_values[hash_index1] < other_pancake->hash_values[hash_index2])
        {
          hash_index1 += 1;
        }
        else if(val->hash_values[hash_index1] > other_pancake->hash_values[hash_index2])
        {
          hash_index2 += 1;
        }
        else
        {
          tmp_match += 1;
          hash_index1 += 1;
          hash_index2 += 1;
        }
      }

     /* uint8_t miss = 0;
      for(int hash_index = 1; hash_index <= NUM_PANCAKES; ++hash_index)
      {
        if(other_pancake->g + miss >= match)
        {
          break;
        }
        for(int other_hash_index = 1; other_hash_index <= NUM_PANCAKES; ++other_hash_index)
        {
          if(val->hash_values[hash_index] == other_pancake->hash_values[other_hash_index])
          {
            tmp_match += 1;
            break;
          }
          else if(other_hash_index == NUM_PANCAKES) miss += 1;
        }
      }*/
      //uint64_t combined = val->hash_64 & other_pancake->hash_64;
      //tmp_match = (uint8_t)__popcnt64(combined);
      counter += 1;
      tmp_match = NUM_PANCAKES - tmp_match + other_pancake->g;
      if(tmp_match < match)
      {
        ptr = other_pancake;
        match = tmp_match;
      }
    }
  }
  //std::cout << counter << "\n";
  return match;
}
