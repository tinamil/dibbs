#pragma once
#include <vector>
#include <mutex>

template <typename T>
class thread_safe_stack
{
public:
  thread_safe_stack()
  {
    
  }

  void copy(thread_safe_stack<T>& other) {
    lock.lock();
    other.lock.lock();
    storage.clear();
    for (size_t i = 0; i < other.storage.size(); ++i) {
      storage.push_back(other.storage[i]);
    }
    other.lock.unlock();
    lock.unlock();
  }

  void push(const T& value) {
    lock.lock();
    storage.push_back(value);
    lock.unlock();
  }

  std::pair<bool, T> pop() {
    lock.lock();
    T value;
    bool success = !storage.empty();
    if (success) {
      value = storage.back();
      storage.pop_back();
    }
    lock.unlock();
    return std::pair<bool, T>(success, value);
  }

  T top() {
    return storage.back();
  }

  size_t size() {
    return storage.size();
  }

  size_t empty() {
    return storage.empty();
  }

private:
  std::vector<T> storage;
  std::mutex lock;
};

