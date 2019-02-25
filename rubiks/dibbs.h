#pragma once
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <queue>
#include "node.h"
#include "rubiks.h"

namespace search
{
void dibbs (const uint8_t state[]);
}
