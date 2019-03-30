#pragma once
#include <vector>
#include <omp.h>

template <typename T>
class thread_safe_stack
{
public:
  thread_safe_stack()
  {
    omp_init_lock(&omp_lock);
  }

  void copy(thread_safe_stack<T> &other) {
    lock();
    other.lock();
    storage.clear();
    for (size_t i = 0; i < other.storage.size(); ++i) {
      storage.push_back(other.storage[i]);
    }
    other.unlock();
    unlock();
  }

  void push(T value) {
    lock();
    storage.push_back(value);
    unlock();
  }

  std::pair<bool, T> pop() {
    lock();
    T value;
    bool success = !storage.empty();
    if (success) {
      value = storage.back();
      storage.pop_back();
    }
    unlock();
    return std::pair<bool, T>(success, value);
  }


private:
  std::vector<T> storage;
  omp_lock_t omp_lock;

  void lock() {
    omp_set_lock(&omp_lock);
  }
  void unlock() {
    omp_unset_lock(&omp_lock);
  }
};

