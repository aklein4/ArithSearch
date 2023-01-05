
# boiler plate CMAKE
cmake_minimum_required(VERSION 3.19)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(DivSearch)

include_directories(${PROJECT_SOURCE_DIR}/extern)
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")

# add_library(libdivsearch ${cpu_source_files})

add_executable(DivTest "${CMAKE_SOURCE_DIR}/test.cpp")

# target_link_libraries(DivTest libdivsearch)