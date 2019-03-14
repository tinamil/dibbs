#include "rubiks_loader.h"


std::vector<uint8_t*> RubiksLoader::load_cubes(std::string file)
{
  std::ifstream infile(file);
  std::string line;
  std::vector<uint8_t*> cubes;
  while (std::getline(infile, line))
  {
    uint8_t* cube = scramble(line);
    cubes.push_back(cube);
  }
  return cubes;
}


uint8_t* RubiksLoader::scramble(std::string notation)
{
  uint8_t* state = new uint8_t[40];
  memcpy(state, Rubiks::__goal, 40);

  std::vector<std::string> moves = utility::tokenizer(notation, ' ');

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
    move = Move(translate_face(notation[0]), Rotation::clockwise);
  }
  else
  {
    Face face = translate_face(notation[0]);
    if (notation[1] == '\'')
      move = Move(face, Rotation::counterclockwise);
    else if (notation[1] == '2')
      move = Move(face, Rotation::half);
  }
  return move;
}


RubiksLoader::Face RubiksLoader::translate_face(char input_face)
{
  char face = toupper(input_face);
  switch (face)
  {
  case 'U':
    return Face::up;
  case 'D':
    return Face::down;
  case 'F':
    return Face::front;
  case 'B':
    return Face::back;
  case 'R':
    return Face::right;
  case 'L':
    return Face::left;
  }
  throw std::runtime_error("Failed to identify face character");
}

