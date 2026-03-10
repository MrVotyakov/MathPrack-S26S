cmake_minimum_required(VERSION 3.11)
project(vtkdemo)

set(CMAKE_CXX_STANDARD 11)
cmake_policy(SET CMP0167 NEW)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #для clangd

add_executable(gen_mesh gen_mesh.cpp)
add_executable(sim sim.cpp)


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

#if(NOT GMSH_INCLUDE_DIR)
#    message(FATAL_ERROR "gmsh.h not found")
#endif()
#if(NOT GMSH_LIBRARY)
#    message(FATAL_ERROR "libgmsh not found")
#endif()

target_include_directories(gen_mesh PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(gen_mesh PRIVATE ${VTK_LIBRARIES} ${GMSH_LIBRARY})
target_include_directories(sim PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(sim PRIVATE ${VTK_LIBRARIES} ${GMSH_LIBRARY})

get_filename_component(GMSH_LIB_DIR ${GMSH_LIBRARY} DIRECTORY)
set_target_properties(gen_mesh PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)
set_target_properties(sim PROPERTIES
    BUILD_RPATH "${GMSH_LIB_DIR}"
    INSTALL_RPATH "${GMSH_LIB_DIR}"
)
