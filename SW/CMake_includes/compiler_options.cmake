# Compiler and linker options for gcc
if(CMAKE_COMPILER_IS_GNUCXX)

  # C code compiler
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}")
  
  # C++ code compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -std=c++17")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -std=c++17")



endif(CMAKE_COMPILER_IS_GNUCXX)
