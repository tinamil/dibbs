#include "utility.h"

std::vector<std::string> utility::tokenizer(const std::string& p_pcstStr, char delim)
{
  std::vector<std::string> tokens;
  std::stringstream   mySstream(p_pcstStr);
  std::string         temp;

  while (getline(mySstream, temp, delim))
  {
    if (temp.length() == 0 || (temp.length() == 1 && temp[0] == delim))
    {
      continue;
    }
    tokens.push_back(temp);
  }

  return tokens;
}

