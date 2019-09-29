#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <shared_mutex>
#include <unordered_set>

/**
Hashes each input value and divides them into some number of buckets.  Each bucket corresponds to a file.
*/
template<class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key> >
class DiskHash {
public:
  DiskHash(const std::string& name);
  ~DiskHash();
  void open();
  void close();
  void insert(const Key& data);
  std::vector<std::pair<Key, Key>> compare_hash(const DiskHash& other) const;

  uint64_t size() {
    return insertion_count;
  }

  uint64_t disk_size() {
    uint64_t size = 0;
    for (size_t i = 0; i < bucket_count; ++i) {
      std::ifstream file(file_name(i), std::ifstream::ate | std::ios::binary);
      if (!file.is_open())
      {
        throw new std::exception("failed to open file");
      }
      size += file.tellg();
    }
    return size;
  }

private:
  const unsigned int thread_count;
  const size_t bucket_count;
  std::ofstream* open_files;
  std::mutex* locks;
  std::string name;
  std::atomic_uint64_t insertion_count;

  std::string file_name(size_t bucket) const {
    return name + "_" + std::to_string(bucket) + ".tmp";
  }
  std::unordered_set<Key, Hash, KeyEqual> load_bucket(size_t bucket) const;
  std::vector<Key> load_vector(size_t bucket) const;
};

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::DiskHash(const std::string& name) : name(name), thread_count(std::thread::hardware_concurrency()), bucket_count(480) {
  _setmaxstdio(8192);
  open_files = new std::ofstream[bucket_count];
  locks = new std::mutex[bucket_count];
  open();
}

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::~DiskHash() {
  close();
  delete[] open_files;
  delete[] locks;
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::open() {
  insertion_count = 0;
  std::thread* thread_array = new std::thread[thread_count];
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([thread_number, this]() {
      for (size_t i = thread_number; i < bucket_count; i += thread_count) {
        open_files[i] = std::ofstream(file_name(i), std::ios::binary);
      }
      });
  }
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::close() {
  for (size_t i = 0; i < bucket_count; ++i) {
    open_files[i].close();
  }
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::insert(const Key& data) {
  ++insertion_count;
  size_t hash_val = Hash{}(data);
  size_t bucket = hash_val % bucket_count;
  locks[bucket].lock();
  open_files[bucket].write(reinterpret_cast<const char*>(&data), sizeof data);
  locks[bucket].unlock();
}

template<class Key, class Hash, class KeyEqual>
std::unordered_set<Key, Hash, KeyEqual> DiskHash<Key, Hash, KeyEqual>::load_bucket(size_t bucket) const {
  auto data = load_vector(bucket);
  std::unordered_set<Key, Hash, KeyEqual> set;
  set.reserve(data.size());
  if (data.size() > 0) {
    set.insert(data.begin(), data.end());
  }
  return set;
}

template<class Key, class Hash, class KeyEqual>
std::vector<Key> DiskHash<Key, Hash, KeyEqual>::load_vector(size_t bucket) const {
  std::vector<Key> data;
  std::ifstream input(file_name(bucket), std::ios::binary | std::ios::ate);
  std::streamoff pos = input.tellg();
  input.seekg(0, std::ios::beg);
  size_t size = pos / sizeof Key;
  data.resize(size);
  input.read(reinterpret_cast<char*> (data.data()), sizeof(Key) * size);
  return data;
}


template<class Key, class Hash, class KeyEqual>
std::vector<std::pair<Key, Key>> DiskHash<Key, Hash, KeyEqual>::compare_hash(const DiskHash& other) const
{
  std::vector<std::pair<Key, Key>> intersection;
  std::thread* thread_array = new std::thread[thread_count];
  std::mutex result_lock;
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([&intersection, &result_lock, thread_number, this, &other]() {
      for (size_t i = thread_number; i < bucket_count; i += thread_count) {
        std::unordered_set<Key, Hash, KeyEqual> set = load_bucket(i);
        std::vector<Key> other_vec = other.load_vector(i);
        for (size_t j = 0; j < other_vec.size(); ++j) {
          auto it = set.find(other_vec[j]);
          if (it != set.end()) {
            result_lock.lock();
            intersection.push_back(std::make_pair(*it, other_vec[j]));
            result_lock.unlock();
          }
        }
      }
      });
  }
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;
  return intersection;
}