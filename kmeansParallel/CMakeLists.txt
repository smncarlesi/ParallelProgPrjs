cmake_minimum_required(VERSION 3.29)
project(kmeansFINAL)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3 -march=native -fopenmp -fopt-info-vec-optimized -ftree-vectorize")

add_executable(kmeansFINAL
    kmeansFINAL.cpp)