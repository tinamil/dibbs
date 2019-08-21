#pragma once
#include <fstream>
#include <vector>
#include <string>
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
  std::pair<bool, Key> compare_hash(const DiskHash& other) const;

private:
  static constexpr size_t NUM_BUCKETS = 100;
  std::vector<std::ofstream> open_files;
  std::string name;

  std::string file_name(size_t bucket) const {
    return name + "_" + std::to_string(bucket) + ".tmp";
  }
  std::unordered_set<Key, Hash, KeyEqual> load_bucket(size_t bucket) const;
  std::vector<Key> load_vector(size_t bucket) const;
};

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::DiskHash(const std::string& name) : open_files(), name(name) {
  open_files.resize(NUM_BUCKETS);
  open();
}

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::~DiskHash() {
  close();
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::open() {
  for (size_t i = 0; i < open_files.size(); ++i) {
    open_files[i] = std::ofstream(file_name(i), std::ios::binary);
  }
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::close() {
  for (size_t i = 0; i < open_files.size(); ++i) {
    open_files[i].close();
  }
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::insert(const Key& data) {
  size_t hash_val = Hash{}(data);
  size_t bucket = hash_val % NUM_BUCKETS;
  open_files[bucket].write(reinterpret_cast<const char*>(&data), sizeof data);
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
  ifstream input = ifstream(file_name(bucket), std::ios::binary | ios::ate);
  std::streamoff pos = input.tellg();
  input.seekg(0, ios::beg);
  size_t size = pos / sizeof Key;
  std::vector<Key> data;
  data.resize(size);
  input.read(reinterpret_cast<char*> (data.data()), sizeof(Key) * size);
  return data;
}


template<class Key, class Hash, class KeyEqual>
std::pair<bool, Key> DiskHash<Key, Hash, KeyEqual>::compare_hash(const DiskHash& other) const
{
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    std::unordered_set<Key, Hash, KeyEqual> set = load_bucket(i);
    std::vector<Key> other_vec = other.load_vector(i);
    for (size_t j = 0; j < other_vec.size(); ++j) {
      auto it = set.find(other_vec[j]);
      if (it != set.end()) {
        return std::make_pair(true, *it);
      }
    }
  }
  return std::make_pair(false, Key());
}