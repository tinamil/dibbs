#pragma once
#include "Constants.h"
#include "mycuda.h"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <unordered_map>


#pragma warning(suppress : 4996)
static constexpr const char* type_str[] = {"USA", "CTR", "W", "E", "LKS", "CAL", "NE", "NW", "FLA", "COL", "BAY", "NY", "USA"};
enum Type { USA, CTR, W, E, LKS, CALI, NE, NW, FLA, COL, BAY, NY, USAMIRROR };
enum ARC { DISTANCE = 0, TRANSIT = 1 };

template <typename T>
constexpr inline T square(T l)
{
  return l * l;
}

uint32_t inline perimeterDistance(double lat1, double lng1, double lat2, double lng2)
{
  auto earthCyclePerimeter = EARTH_PERIMETER * cos((lat1 + lat2) / 2.0);
  auto dx = (lng1 - lng2) * earthCyclePerimeter * RAD_TO_DEG_I360;
  auto dy = AVERAGE_DIAMETER_OF_EARTH_M * PI * (lat1 - lat2) * RAD_TO_DEG_I360;

  return static_cast<uint32_t>(floor(0.95 * sqrt(square(dx) + square(dy))));
}

uint32_t inline flatDistance(double lat1, double lng1, double lat2, double lng2)
{
  auto a = HALF_PI - lat1;
  auto b = HALF_PI - lat2;
  auto u = a * a + b * b;
  auto v = -2 * a * b * cos(lng2 - lng1);
  auto c = sqrt(abs(u + v));
  return static_cast<uint32_t>(floor(AVERAGE_DIAMETER_OF_EARTH_M / 2 * c * 0.8));
}

uint32_t inline haversineDistance(double lat1, double lng1, double lat2, double lng2)
{
  double latDistance = sin((lat1 - lat2) / 2);
  double lngDistance = sin((lng1 - lng2) / 2);
  double a = square(latDistance) + cos(lat1) * cos(lat2) * square(lngDistance);
  double c = AVERAGE_DIAMETER_OF_EARTH_M * asin(sqrt(a));
  return static_cast<uint32_t>(floor(c));
}

struct Edge
{
  uint32_t other;
  uint32_t cost;
};

class Road
{
private:
  static inline std::vector<std::vector<Edge>> adj_vertices;
  static inline std::vector<Coordinate> coordinates;
  static inline uint32_t epsilon;

public:

  static uint32_t num_neighbors(uint32_t node)
  {
    return static_cast<uint32_t>(adj_vertices[node].size());
  }

  static const Edge& get_edge(uint32_t node, uint32_t edge_index)
  {
    return adj_vertices[node][edge_index];
  }

  static uint32_t heuristic(uint32_t left, uint32_t right)
  {
    //uint32_t p = perimeterDistance(coordinates[left].lat, coordinates[left].lng, coordinates[right].lat, coordinates[right].lng);
    //uint32_t f = flatDistance(coordinates[left].lat, coordinates[left].lng, coordinates[right].lat, coordinates[right].lng);
    uint32_t h = haversineDistance(coordinates[left].lat, coordinates[left].lng, coordinates[right].lat, coordinates[right].lng);
    return h;
  }

  static void MirrorGraph()
  {
    assert(coordinates.size() > 0);
    double minLng = coordinates[0].lng, maxLng = coordinates[0].lng;
    double minLat = coordinates[0].lat, maxLat = coordinates[0].lat;
    for(const auto& c : coordinates) {
      minLng = std::min(minLng, c.lng);
      maxLng = std::max(maxLng, c.lng);

      minLat = std::min(minLat, c.lat);
      maxLat = std::max(maxLat, c.lat);
    }

    double mirror_position = maxLat - (maxLat - minLat) * 0.25;
    std::unordered_map<size_t, size_t> new_map;
    for(size_t i = 0; i < coordinates.size(); ++i) {
      const Coordinate c = coordinates[i];
      if(c.lat < mirror_position) {
        new_map[i] = coordinates.size();
        coordinates.push_back(Coordinate{c.lng, 2 * mirror_position - c.lat});
      }
    }
    size_t prev_size = adj_vertices.size();
    size_t crossoverl = 0, crossoverr = 0;
    adj_vertices.resize(coordinates.size());
    for(size_t i = 0; i < prev_size; ++i) {
      //For any nodes that aren't mirrored, mirror edges that cross over to mirrored nodes
      if(!new_map.contains(i)) {
        for(const auto& e : adj_vertices[i]) {
          if(new_map.contains(e.other)) {
            adj_vertices[i].push_back(Edge{static_cast<uint32_t>(new_map[e.other]), heuristic(new_map[e.other], i)});
          }
        }
      }
      //For any nodes that are mirrored, add all their edges
      else {
        uint32_t new_id = new_map[i];
        for(const auto& e : adj_vertices[i]) {
          if(new_map.contains(e.other)) {
            adj_vertices[new_id].push_back(Edge{static_cast<uint32_t>(new_map[e.other]), e.cost});
          }
          else {
            adj_vertices[new_id].push_back(Edge{e.other, heuristic(e.other, new_id)});
          }
        }
      }
    }
    std::cout << "Final size = " << adj_vertices.size() << "\n";
    uint64_t arcsize = 0;
    for(size_t i = 0; i < adj_vertices.size(); ++i) {
      arcsize += adj_vertices[i].size();
    }
    std::cout << "Final arcs = " << arcsize << "\n";
  }

  static int LoadGraph(Type t, bool transit = false)
  {
    std::string coordinate_file = "data/USA-road-d.";
    coordinate_file += type_str[t];
    epsilon = UINT32_MAX;

    std::ifstream cofile(coordinate_file + ".co");
    for(std::string line; std::getline(cofile, line); )
    {
      if(line.size() == 0 || line[0] == 'c') continue;
      if(line[0] == 'p') {
        std::stringstream stream(line.substr(12));
        uint32_t nodes = 0;
        stream >> nodes;
        assert(nodes > 0);
        coordinates.resize(nodes);
      }
      if(line[0] == 'v') {
        std::stringstream stream(line.substr(1));
        uint32_t index;
        coordinate_t x, y;
        stream >> index >> x >> y;
        coordinates[static_cast<size_t>(index) - 1] = {DEG_TO_RAD * x, DEG_TO_RAD * y};
      }
    }

    std::string graph_input_file = "data/USA-road-";
    if(transit) graph_input_file += "t.";
    else graph_input_file += "d.";
    graph_input_file += type_str[t];
    std::ifstream grfile(graph_input_file + ".gr");
    for(std::string line; std::getline(grfile, line); )
    {
      if(line.size() == 0 || line[0] == 'c') continue;

      if(line[0] == 'p') {
        std::stringstream stream(line.substr(4));
        uint32_t nodes = 0, edges = 0;
        stream >> nodes >> edges;
        assert(nodes > 0);
        adj_vertices.clear();
        adj_vertices.resize(nodes);
      }

      if(line[0] == 'a') {
        std::stringstream stream(line.substr(1));
        uint32_t left, right;
        uint32_t cost;
        stream >> left >> right >> cost;
        adj_vertices[left - 1].push_back({right - 1, cost});
        epsilon = std::min(epsilon, cost);
      }
    }

    if(t == Type::USAMIRROR) {
      MirrorGraph();
    }

    assert(coordinates.size() == adj_vertices.size());

    mycuda::LoadCoordinates(coordinates);
    return static_cast<uint32_t>(coordinates.size());
  }
};