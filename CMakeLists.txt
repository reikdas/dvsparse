cmake_minimum_required(VERSION 3.13)
project(DvSparse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(OPTIMIZATION_FLAGS "-O3 -march=native -funroll-all-loops -mprefer-vector-width=512 -mavx -ffast-math")

option(ENABLE_OPENMP "Enable a OPENMP" OFF)
if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS} -fopenmp -DOPENMP")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}")
endif()

find_package(MKL CONFIG REQUIRED)

include_directories(include)

add_executable(dense_spmv src/dense_spmv.cpp)
add_executable(naive_spmv src/naive_spmv.cpp)

add_executable(mkl_spmv src/mkl_spmv.cpp)
target_compile_options(mkl_spmv PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(mkl_spmv PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(mkl_spmv PUBLIC $<LINK_ONLY:MKL::MKL>)
