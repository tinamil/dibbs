#include "mr_rank.h"

void mr::_mr_unrank1(uint64_t rank, int n, uint8_t *vec) {
    uint64_t t, q, r;
    if (n < 1) return;

    q = rank / n;
    r = rank % n;
    _MR_SWAP(vec[r], vec[n-1]);
    _mr_unrank1(q, n-1, vec);
}

uint64_t mr::_mr_rank1(int n, uint8_t *vec, uint8_t *inv) {
    uint64_t s, t;
    if (n < 2) return 0;

    s = vec[n-1];
    _MR_SWAP(vec[n-1], vec[inv[n-1]]);
    _MR_SWAP(inv[s], inv[n-1]);
    return s + n * _mr_rank1(n-1, vec, inv);
}

/* Fill the integer array <vec> (of size <n>) with the
 * permutation at rank <rank>.
 */
void mr::get_permutation(uint64_t rank, int n, uint8_t *vec) {
    uint8_t i;
    for (i = 0; i < n; ++i) vec[i] = i;
    _mr_unrank1(rank, n, vec);
}

/* Return the rank of the current permutation of array <vec>
 * (of size <n>).
 */
uint64_t mr::get_rank(int n, uint8_t *vec) {
    uint8_t i;
    uint64_t r;
    uint8_t v [n];
    uint8_t inv [n];

    for (i = 0; i < n; ++i) {
        v[i] = vec[i];
        inv[vec[i]] = i;
    }
    r = _mr_rank1(n, v, inv);
    return r;
}

