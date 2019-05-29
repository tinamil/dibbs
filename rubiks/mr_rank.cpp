#include "mr_rank.h"

void mr::_mr_unrank1(uint64_t rank, int n, uint8_t *vec)
{
  uint64_t q, r;
  if (n < 1)
    return;

  q = rank / n;
  r = rank % n;
  uint8_t tmp = vec[r];
  vec[r] = vec[n - 1];
  vec[n - 1] = tmp;
  _mr_unrank1(q, n - 1, vec);
}

uint64_t mr::_mr_rank1(int n, uint8_t *vec, uint8_t *inv)
{
  uint64_t s;
  if (n < 2)
    return 0;

  s = vec[n - 1];

  uint8_t tmp = vec[n - 1];
  vec[n - 1] = vec[inv[n - 1]];
  vec[inv[n - 1]] = tmp;

  tmp = inv[s];
  inv[s] = inv[n - 1];
  inv[n - 1] = tmp;

  return s + n * _mr_rank1(n - 1, vec, inv);
}

/* Fill the integer array <vec> (of size <n>) with the
 * permutation at rank <rank>.
 */
void mr::get_permutation(uint64_t rank, int n, uint8_t *vec)
{
  uint8_t i;
  for (i = 0; i < n; ++i)
    vec[i] = i;
  _mr_unrank1(rank, n, vec);
}

/* Return the rank of the current permutation of array <vec>
 * (of size <n>).
 */
uint64_t mr::get_rank(int n, uint8_t *vec)
{
  uint8_t i;
  uint64_t r;
  #pragma warning(suppress: 6255)
  uint8_t* v = (uint8_t*)_alloca(n);
  #pragma warning(suppress: 6255)
  uint8_t* inv = (uint8_t*)_alloca(n);

  for (i = 0; i < n; ++i)
  {
    v[i] = vec[i];
    inv[vec[i]] = i;
  }
  r = _mr_rank1(n, v, inv);
  return r;
}

uint64_t mr::k_rank(uint8_t *locs, uint8_t *dual, const int distinctSize, const int puzzleSize)
{
  uint64_t result2 = 0;
  uint64_t multiplier = 1;
  for (int i = 0; i < distinctSize; i++)
  {
    uint64_t tmp = dual[i];
    uint8_t tmp2 = locs[i];

    result2 += (tmp - i) * multiplier;
    multiplier *= (uint64_t(puzzleSize) - i);

    if (tmp2 < puzzleSize)
    {
      swap(locs[i], locs[dual[i]]);
      swap(dual[tmp2], dual[i]);
    }
  }

  return result2;
}
