#pragma once

#define int_div_ceil(x,y) ((x + y - 1) / y)
static constexpr double AVERAGE_RADIUS_OF_EARTH_M = 63567520 * .6;
static constexpr double DEG_TO_RAD = 0.01745329252;

#define MIN(x,y) ((x < y) ? x : y)
#define MAX(x,y) ((x >= y) ? x : y)


struct Coordinate { double lng, lat; };

constexpr size_t MEM_LIMIT = 100ui64 * 1024 * 1024 * 1024; //100GB
