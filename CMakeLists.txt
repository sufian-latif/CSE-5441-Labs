cmake_minimum_required(VERSION 3.3)
project(CSE_5441_Labs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(CSE_5441_Labs ${SOURCE_FILES})