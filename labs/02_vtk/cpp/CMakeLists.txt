cmake_minimum_required(VERSION 3.11)
project(vtkdemo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #для clangd

add_executable(cubic2d cubic2d.cpp)
add_executable(cubic3d cubic3d.cpp)
add_executable(tetr3d tetr3d.cpp)

# --- VTK ---
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# --- Gmsh ---
find_path(GMSH_INCLUDE_DIR
    NAMES gmsh.h
    HINTS
        /opt/homebrew/include
        /usr/local/include
)

find_library(GMSH_LIBRARY
    NAMES gmsh
    HINTS
        /opt/homebrew/lib
        /usr/local/lib
)

if(NOT GMSH_INCLUDE_DIR)
    message(FATAL_ERROR "gmsh.h not found")
endif()

if(NOT GMSH_LIBRARY)
    message(FATAL_ERROR "libgmsh not found")
endif()

message(STATUS "GMSH include dir: ${GMSH_INCLUDE_DIR}")
message(STATUS "GMSH library: ${GMSH_LIBRARY}")

target_include_directories(tetr3d PRIVATE ${GMSH_INCLUDE_DIR})

target_link_libraries(cubic2d PRIVATE ${VTK_LIBRARIES})
target_link_libraries(cubic3d PRIVATE ${VTK_LIBRARIES})
target_link_libraries(tetr3d PRIVATE ${VTK_LIBRARIES} ${GMSH_LIBRARY})

get_filename_component(GMSH_LIB_DIR ${GMSH_LIBRARY} DIRECTORY)
set_target_properties(tetr3d PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)