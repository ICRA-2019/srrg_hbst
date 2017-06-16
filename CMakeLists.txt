cmake_minimum_required(VERSION 2.8.3)
project(srrg_hbst)

#ds determine build type, default build type Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("build type: ${CMAKE_BUILD_TYPE}")

#ds add c++11 support, mediocre optimization, all warnings
add_definitions(-Ofast -march=native -Wall -pedantic -std=c++11)

#ds locate external packages: Eigen (currently not required)
find_package(Eigen3 QUIET)
if(NOT "${EIGEN3_INCLUDE_DIR}" STREQUAL "")
  message("found Eigen version: '3' (${EIGEN3_INCLUDE_DIR}) - building Eigen example code")
  
  #ds enable OpenCV usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_EIGEN true)
  include_directories(${EIGEN3_INCLUDE_DIR})
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
endif()

#ds check if opencv is present
set(SRRG_HBST_HAS_OPENCV false)
find_package(OpenCV QUIET)
if(NOT "${OpenCV_INCLUDE_DIRS}" STREQUAL "")
  message("found OpenCV version: '${OpenCV_VERSION}' (${OpenCV_DIR}) - building OpenCV example code")
  
  #ds enable OpenCV usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_OPENCV true)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
endif()

#ds define the catkin package
find_package(catkin)
catkin_package(INCLUDE_DIRS src)

#ds example build
include_directories(${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)

#ds add sub cmakelists
add_subdirectory(examples)
