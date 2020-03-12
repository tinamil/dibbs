#pragma once
#include <fstream>
#include <vector>
#include <string>
#include "tsl/hopscotch_set.h"

/**
Hashes each input value and divides them into some number of buckets.  Each bucket corresponds to a file.
*/

constexpr int DISKHASH_BUFFER_LIMIT = 1<<12;
template<class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key> >
class DiskHash {
public:
  DiskHash(const std::string& name, const size_t init_bucket_count = 480);
  ~DiskHash();
  void open();
  void close();
  void insert(const Key& data);
  void insert(const Key& data, const size_t bucket);
  std::vector<std::pair<Key, Key>> compare_hash(const DiskHash& other) const;
  void load_bucket(size_t bucket, tsl::hopscotch_set<Key, Hash, KeyEqual>& set) const;
  std::vector<Key> load_vector(size_t bucket) const;
  void load_vector(size_t bucket, std::vector<Key>& destination) const;

  uint64_t size() {
    return 0;//insertion_count;
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
  std::vector<Key>* buffers;
  //std::atomic_uint64_t insertion_count;

  std::string file_name(size_t bucket) const {
    return name + "_" + std::to_string(bucket) + ".tmp";
  }
};

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::DiskHash(const std::string& name, const size_t init_bucket_count) : name(name), thread_count(std::thread::hardware_concurrency()), bucket_count(init_bucket_count) {
  _setmaxstdio(8192);
  open_files = new std::ofstream[bucket_count];
  locks = new std::mutex[bucket_count];
  buffers = new std::vector<uint64_t>[bucket_count];
  for (int i = 0; i < bucket_count; ++i) {
    buffers[i].reserve(DISKHASH_BUFFER_LIMIT);
  }
  open();
}

template<class Key, class Hash, class KeyEqual>
DiskHash<Key, Hash, KeyEqual>::~DiskHash() {
  close();
  delete[] open_files;
  delete[] locks;
  delete[] buffers;
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::open() {
  //insertion_count = 0;
  std::thread* thread_array = new std::thread[thread_count];
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([thread_number, this]() {
      for (size_t i = thread_number; i < bucket_count; i += thread_count) {
        open_files[i] = std::ofstream(file_name(i), std::ios::binary | std::ios::out);
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

  std::thread* thread_array = new std::thread[thread_count];
  for (size_t thread_number = 0; thread_number < thread_count; ++thread_number) {
    thread_array[thread_number] = std::thread([thread_number, this]() {
      for (size_t i = thread_number; i < bucket_count; i += thread_count) {
        if (buffers[i].size() > 0) {
          open_files[i].write(reinterpret_cast<char*>(buffers[i].data()), buffers[i].size() * sizeof(Key));
          buffers[i].clear();
        }
        open_files[i].close();
      }
      });
  }
  for (size_t i = 0; i < thread_count; ++i) {
    thread_array[i].join();
  }
  delete[] thread_array;
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::insert(const Key& data) {
  //++insertion_count;
  size_t hash_val = Hash{}(data);
  size_t bucket = hash_val % bucket_count;
  insert(data, bucket);
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::insert(const Key& data, const size_t bucket) {
  //++insertion_count;
  locks[bucket].lock();
  buffers[bucket].push_back(data);
  if (buffers[bucket].size() >= DISKHASH_BUFFER_LIMIT) {
    open_files[bucket].write(reinterpret_cast<char*>(buffers[bucket].data()), buffers[bucket].size() * sizeof(Key));
    buffers[bucket].clear();
  }
  locks[bucket].unlock();
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::load_bucket(size_t bucket, tsl::hopscotch_set<Key, Hash, KeyEqual>& set) const {
  std::vector<Key> data;
  set.clear();
  load_vector(bucket, data);
  if (data.size() > 0) {
    set.insert(data.begin(), data.end());
  }
}

template<class Key, class Hash, class KeyEqual>
std::vector<Key> DiskHash<Key, Hash, KeyEqual>::load_vector(size_t bucket) const {
  std::vector<Key> data;
  load_vector(bucket, data);
  return data;
}

template<class Key, class Hash, class KeyEqual>
void DiskHash<Key, Hash, KeyEqual>::load_vector(size_t bucket, std::vector<Key>& destination) const {
  std::ifstream input(file_name(bucket), std::ios::binary | std::ios::ate | std::ios::in);
  std::streamoff pos = input.tellg();
  input.seekg(0, std::ios::beg);
  size_t size = pos / sizeof Key;
  destination.resize(size);
  input.read(reinterpret_cast<char*> (destination.data()), sizeof(Key)* size);
}


