#pragma once

enum class Direction { forward, backward };

inline Direction OppositeDirection(Direction d) {
  if (d == Direction::forward) return Direction::backward;
  else return Direction::forward;
}