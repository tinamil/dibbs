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
  uint64_t size() {
    return count;
  }
  std::vector<std::pair<Key, Key>> compare_hash(const DiskHash& other) const;

private:
  static constexpr size_t NUM_BUCKETS = 500;
  std::ofstream open_files[NUM_BUCKETS];
  std::mutex locks[NUM_BUCKETS];
  std::string name;
  std::atomic_uint64_t count;

  std::string file_name(size_t bucket) const {
    return name + "_" + std::to_string(bucket) + ".tmp";
  }
  std::unordered_set<Key, Hash, KeyEqual> load_bucket(size_t bucket) const;
  std::vector<Key> load_vector(size_t bucket) const;
};

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::DiskHash(const std::string& name) : name(name) {
  open();
}

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::~DiskHash() {
  close();
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::open() {
  count = 0;
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    open_files[i] = std::ofstream(file_name(i), std::ios::binary);
  }
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::close() {
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    open_files[i].close();
  }
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::insert(const Key& data) {
  ++count;
  size_t hash_val = Hash{}(data);
  size_t bucket = hash_val % NUM_BUCKETS;
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
  std::cout << "Comparing " << name << " against " << other.name << "\n";
  std::vector<std::pair<Key, Key>> intersection;
  const unsigned int thread_count = std::thread::hardware_concurrency();
  std::thread* thread_array = new std::thread[thread_count];
  std::mutex result_lock;
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([&intersection, &result_lock, thread_number, this, &other, thread_count]() {
      for (size_t i = thread_number; i < NUM_BUCKETS; i += thread_count) {
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
  std::cout << "Finished comparing " << name << " against " << other.name << ", found " << std::to_string(intersection.size()) << " intersection results\n";
  return intersection;
}