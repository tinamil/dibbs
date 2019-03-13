#pragma once
#include <vector>
#include "rubiks.h"
#include "utility.h"

class RubiksLoader
{
public:
  static std::vector<uint8_t*> load_cubes (std::string file);

private:
  enum Face
  {
    front = 0,
    up = 1,
    left = 2,
    back = 3,
    down = 4,
    right = 5
  };

  enum Rotation
  {
    clockwise = 0,
    counterclockwise = 1,
    half = 2
  };
  struct Cubes
  {
    int size;
    uint8_t** data;
  };

  struct Move
  {
    int face;
    int rotation;

    Move () : face(-1), rotation(-1) {}
    Move (Face _face, Rotation _rotation) : face ( (int) _face), rotation ( (int) _rotation) {}
  };

  static uint8_t* scramble (std::string notation);
  static Move convert (std::string move);
  static Face translate_face (char face);
};
