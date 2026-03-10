cmake_minimum_required(VERSION 3.11)

project(gmshdemo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sample01 sample01.cpp)
add_executable(sample02 sample02.cpp)
add_executable(sample03 sample03.cpp)
add_executable(sample04 sample04.cpp)

# --- Ищем Gmsh ---

find_path(GMSH_INCLUDE_DIR
    NAMES gmsh.h
    HINTS
        /opt/homebrew/include
        /usr/local/include
        /usr/include
)

find_library(GMSH_LIBRARY
    NAMES gmsh
    HINTS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
)

if(NOT GMSH_INCLUDE_DIR)
    message(FATAL_ERROR "gmsh.h not found")
endif()

if(NOT GMSH_LIBRARY)
    message(FATAL_ERROR "libgmsh not found")
endif()

message(STATUS "GMSH include dir: ${GMSH_INCLUDE_DIR}")
message(STATUS "GMSH library: ${GMSH_LIBRARY}")

target_include_directories(sample01 PRIVATE ${GMSH_INCLUDE_DIR})
target_include_directories(sample02 PRIVATE ${GMSH_INCLUDE_DIR})
target_include_directories(sample03 PRIVATE ${GMSH_INCLUDE_DIR})
target_include_directories(sample04 PRIVATE ${GMSH_INCLUDE_DIR})

target_link_libraries(sample01 PRIVATE ${GMSH_LIBRARY})
target_link_libraries(sample02 PRIVATE ${GMSH_LIBRARY})
target_link_libraries(sample03 PRIVATE ${GMSH_LIBRARY})
target_link_libraries(sample04 PRIVATE ${GMSH_LIBRARY})

# Чтобы macOS находил динамическую библиотеку при запуске
get_filename_component(GMSH_LIB_DIR ${GMSH_LIBRARY} DIRECTORY)

set_target_properties(sample01 PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)

set_target_properties(sample02 PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)

set_target_properties(sample03 PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)

set_target_properties(sample04 PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)