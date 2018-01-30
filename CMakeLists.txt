cmake_minimum_required(VERSION 2.8)

# define by input
set(INCLUDE_DIR "" CACHE PATH "3rdparty include")
set(LIBRARIES "" CACHE PATH "3rdparty libraries")

project(SimpleVO)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

find_package(OpenCV)
find_package(Eigen3 QUIET)
find_package(Sophus QUIET)
find_package(Ceres QUIET)

include_directories(
${INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
)

add_subdirectory(src)
