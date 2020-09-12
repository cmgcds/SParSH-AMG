cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

## Specify C and C++ Compiler path in CMAKE_C_COMPILER and CMAKE_CXX_COMPILER variable
set(CMAKE_C_COMPILER   "/opt/intel/compilers_and_libraries/linux/bin/intel64/icc")
set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries/linux/bin/intel64/icpc")

## Set the LANGUAGES for CMAKE Project
project(AMG LANGUAGES CXX CUDA)

## Set CMAKE CXX_BASIC_FLAGS
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CXX_BASIC_FLAGS "-g" "-O3" "-fopenmp" "-Wno-cpp")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("/opt/intel/mkl/include")
include_directories("/usr/local/cuda/include")

add_subdirectory(src)

add_executable(main ${CMAKE_SOURCE_DIR}/main.cpp)

#target_include_directories(main 
#    PRIVATE
#        ${CMAKE_SOURCE_DIR}/include
#        /opt/intel/mkl/include
#        /usr/local/cuda/include
#    )
    
link_directories("/opt/intel/mkl/lib/intel64")
link_directories("/opt/intel/lib/intel64")
#link_directories("/usr/local/cuda-10.0/lib64")
link_directories("/usr/local/lib")

set(CMAKE_EXE_LINKER_FLAGS "-ldl -lpthread -lcusparse -lculibos -lcublas -lnvToolsExt")     

target_link_libraries(main
    PUBLIC
#     amg m mkl_intel_lp64 mkl_intel_thread mkl_core irc svml iomp5)
      amg m mkl_rt irc svml iomp5)
