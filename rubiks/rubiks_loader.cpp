#include "rubiks_loader.h"


std::vector<uint8_t*> RubiksLoader::load_cubes(std::string file)
{
  std::ifstream infile(file);
  std::string line;
  std::vector<uint8_t*> cubes;
  while (std::getline(infile, line))
  {
    if (line.at(0) == '#') continue;
    uint8_t* cube = scramble(line);
    cubes.push_back(cube);
  }
  return cubes;
}


uint8_t* RubiksLoader::scramble(std::string notation, const uint8_t* start_state)
{

  std::vector<std::string> moves = utility::tokenizer(notation, ' ');
  RubiksLoader::Move move = convert(moves[0]);
  uint8_t* state = new uint8_t[40];
  memcpy(state, start_state, 40);

  for (size_t i = 0; i < moves.size(); ++i)
  {
    RubiksLoader::Move move = convert(moves[i]);
    Rubiks::rotate(state, move.face, move.rotation);
  }
  return state;
}



RubiksLoader::Move RubiksLoader::convert(std::string notation)
{
  Move move;
  if (notation.length() == 1)
  {
    move = Move(translate_face(notation[0]), Rubiks::Rotation::clockwise);
  }
  else
  {
    Rubiks::Face face = translate_face(notation[0]);
    if (notation[1] == '\'')
      move = Move(face, Rubiks::Rotation::counterclockwise);
    else if (notation[1] == '2')
      move = Move(face, Rubiks::Rotation::half);
  }
  return move;
}


Rubiks::Face RubiksLoader::translate_face(char input_face)
{
  char face = toupper(input_face);
  switch (face)
  {
  case 'U':
    return Rubiks::Face::up;
  case 'D':
    return Rubiks::Face::down;
  case 'F':
    return Rubiks::Face::front;
  case 'B':
    return Rubiks::Face::back;
  case 'R':
    return Rubiks::Face::right;
  case 'L':
    return Rubiks::Face::left;
  }
  throw std::runtime_error("Failed to identify face character");
}

