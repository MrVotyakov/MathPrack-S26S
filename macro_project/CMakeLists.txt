cmake_minimum_required(VERSION 3.14)
project(macroproject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(macroproject
    src/main.cpp
    src/core/grid.cpp
    src/core/solver.cpp
    src/scenes/laser_pulse.cpp
    src/export/vtk_writer.cpp
    src/export/probe_writer.cpp
)

target_include_directories(macroproject PRIVATE src)