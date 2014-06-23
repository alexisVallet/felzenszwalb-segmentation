cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

project(felzseg)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(WeightedGraph src/WeightedGraph.cpp)
add_library(DisjointSet src/DisjointSet.cpp)
add_library(Utils src/Utils.cpp)
add_library(ImageGraphs src/ImageGraphs.cpp)
add_library(felzseg SHARED src/Felzenszwalb.cpp)
target_link_libraries(felzseg WeightedGraph DisjointSet Utils ImageGraphs ${OpenCV_LIBS})
