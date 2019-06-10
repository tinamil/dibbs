#pragma once
#include <vector>
#include "rubiks.h"
#include "utility.h"

class RubiksLoader
{
public:
  static std::vector<uint8_t*> load_cubes(std::string file);
  static uint8_t* RubiksLoader::scramble(std::string notation, const uint8_t* start_state = Rubiks::__goal);

private:
  struct Cubes
  {
    int size;
    uint8_t** data;
  };

  struct Move
  {
    int face;
    int rotation;

    Move() : face(-1), rotation(-1) {}
    Move(Rubiks::Face _face, Rubiks::Rotation _rotation) : face((int)_face), rotation((int)_rotation) {}
  };

  static Move convert(std::string move);
  static Rubiks::Face translate_face(char face);
};
