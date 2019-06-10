#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

namespace utility
{
  std::vector<std::string> tokenizer(const std::string& p_pcstStr, char delim);
  inline bool test_file(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
  }
}
