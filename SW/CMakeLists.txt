cmake_minimum_required (VERSION 2.6)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

add_definitions(-DON_BOARD=1)

project (XilinxHardwareCompetition)

include_directories(include)
include_directories(include_hls)

file(GLOB SOURCES "src/*.cpp" "src/*.c")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake_includes")
include(compiler_options)

add_executable(main ${SOURCES})
target_link_libraries(main "stdc++fs" "X11")
