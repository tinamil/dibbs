#pragma once

#include <vector>

template <typename T>
class StackArray {
  constexpr static size_t SIZE = 1000000;
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

  size_t size() {
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

  bool pop_back() {
    if (sizes.size() > 1 && sizes.back() == 0) {
      sizes.pop_back();
      delete[] storage[sizes.size()];
    }
    if (sizes.back() == 0 && sizes.size() == 1) {
      return false;
    }
    assert(sizes.back() != 0);
    --sizes.back();
    return true;
  }

  const T& top(const uint32_t optional = 0) {
    return this->operator[](size() - 1);
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
    T operator*() {
      return arch_ptr[index];
    }
  };

  Iterator begin() {
    return Iterator(this);
  }

  Iterator end() {
    return Iterator(this, true);
  }
};