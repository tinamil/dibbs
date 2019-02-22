#pragma once
#include <stdint.h>
#include <stack>
#include <limits>
#include <queue>
#include "node.h"
#include "rubiks.h"
#include <unordered_set>

namespace search
{
void a_star (const uint8_t state[]);
void dibbs (const uint8_t state[]);
}
