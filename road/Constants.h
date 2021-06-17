#pragma once

#define int_div_ceil(x,y) ((x + y - 1) / y)

typedef double coordinate_t;

static constexpr double HALF_PI = 1.57079632679;
static constexpr double PI = 3.14159265359;
static constexpr double TWO_PI = 6.28318530718;

static constexpr coordinate_t EARTH_PERIMETER = 400078630;
static constexpr coordinate_t AVERAGE_DIAMETER_OF_EARTH_M = 63567520 * 2;
static constexpr coordinate_t DEG_TO_RAD = 0.01745329252;
static constexpr coordinate_t RAD_TO_DEG = 180. / PI;
static constexpr coordinate_t RAD_TO_DEG_I360 = RAD_TO_DEG / 360.;

#define MIN(x,y) ((x < y) ? x : y)
#define MAX(x,y) ((x >= y) ? x : y)


struct Coordinate { coordinate_t lng, lat; };
struct fCoordinate
{
  float lng, lat;
};

constexpr size_t MEM_LIMIT = 100ui64 * 1024 * 1024 * 1024; //100GB
