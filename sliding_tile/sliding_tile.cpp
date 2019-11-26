#include "sliding_tile.h"

uint8_t SlidingTile::distances[NUM_TILES][NUM_TILES];
uint8_t SlidingTile::moves[NUM_TILES][5];
uint8_t SlidingTile::starting[NUM_TILES];