#pragma once

#include <fstream>
#include <vector>

template <typename T>
class StackArray {
  constexpr static inline size_t SIZE = 16384;
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

  void Serialize(std::ofstream& stream) const {
    size_t size_of_sizes = sizes.size();
    stream.write(reinterpret_cast<const char*>(&size_of_sizes), sizeof(size_of_sizes));
    stream.write(reinterpret_cast<const char*>(sizes.data()), sizeof(size_t) * size_of_sizes);
    for (size_t i = 0, end = storage.size(); i < end; ++i) {
      stream.write(reinterpret_cast<const char*>(storage[i]), sizeof(T) * sizes[i]);
    }
  }

  void Deserialize(std::ifstream& stream) {
    for (auto& x : storage) {
      delete[] x;
    }
    storage.clear();
    sizes.clear();

    size_t size_of_sizes = 0;
    stream.read(reinterpret_cast<char*>(&size_of_sizes), sizeof(size_of_sizes));
    sizes.resize(size_of_sizes);
    stream.read(reinterpret_cast<char*>(sizes.data()), sizeof(size_t) * size_of_sizes);
    for (size_t i = 0; i < size_of_sizes; ++i) {
      auto ptr = new T[SIZE];
      storage.push_back(ptr);
      stream.read(reinterpret_cast<char*>(ptr), sizeof(T) * sizes[i]);
    }
  }

  void clear() {
    for (auto& x : storage) {
      delete[] x;
    }
    storage.clear();
    sizes.clear();

    storage.push_back(new T[SIZE]);
    sizes.push_back(0);
  }

  size_t size() const {
    return (storage.size() - 1) * SIZE + sizes.back();
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

  T& operator[](std::size_t index) {
    size_t large_index = index / SIZE;
    size_t small_index = index % SIZE;
    return storage[large_index][small_index];
  }
  const T& operator[](std::size_t index) const {
    size_t large_index = index / SIZE;
    size_t small_index = index % SIZE;
    return storage[large_index][small_index];
  }

  class Iterator {
  private:
    StackArray* arch_ptr;
    size_t index;

  public:
    Iterator(StackArray* _arch_ptr, bool end = false) : arch_ptr(_arch_ptr), index(0) {
      assert(arch_ptr != nullptr);
      if (end) {
        index = arch_ptr->size();
      }
    }
    Iterator& operator++() {
      ++index;
      return *this;
    }

    bool operator==(Iterator other) const {
      return index == other.index;
    }
    bool operator!=(Iterator other) const {
      return !(*this == other);
    }
    T& operator*() {
      return (*arch_ptr)[index];
    }
  };

  Iterator begin() {
    return Iterator(this);
  }

  Iterator end() {
    return Iterator(this, true);
  }
};