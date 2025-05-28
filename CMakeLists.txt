cmake_minimum_required(VERSION 3.20)
project(threads-laba-5)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(threads-laba-5 main.cpp)

if(OpenMP_CXX_FOUND)
    target_compile_options(threads-laba-5 PUBLIC -fopenmp)
    target_link_libraries(threads-laba-5 PUBLIC -fopenmp)
endif()
