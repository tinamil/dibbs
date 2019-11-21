#pragma once


#include <vector>

template <typename T>
class StackArray {
  const static size_t SIZE = 1000000;
  std::vector<T*> storage;
  std::vector<size_t> sizes;

public:
  StackArray() {
    storage.push_back(new T[SIZE]);
    sizes.push_back(0);
  }

  ~StackArray() {
    for (auto& x : storage) {
      delete[] x;
    }
  }

  T* push_back(T type) {
    if (*sizes.rbegin() < SIZE) {
      storage.back()[sizes.back()++] = type;
    }
    else {
      storage.push_back(new T[SIZE]);
      sizes.push_back(1);
      (*storage.rbegin())[0] = type;
    }
    return storage.back() + sizes.back() - 1;
  }
};