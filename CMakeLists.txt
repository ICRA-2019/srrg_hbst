cmake_minimum_required(VERSION 2.8.3)
project(srrg_hbst)

#ds determine build type, default build type Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("srrg_hbst|build type: ${CMAKE_BUILD_TYPE}")

#ds add c++11 support, mediocre optimization, all warnings
add_definitions(-Ofast -march=native -Wall -pedantic -std=c++11)

#ds debug flags
#add_definitions(-O0 -march=native -Wall -pedantic -std=c++11 -g)

#ds locate external packages: Eigen (currently not required)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  message("srrg_hbst|found Eigen version: '3' (${EIGEN3_INCLUDE_DIR}) - building Eigen example code")

  #ds enable OpenCV usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_EIGEN true)
  include_directories(${EIGEN3_INCLUDE_DIR})
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
endif()

#ds check if opencv is present
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  message("srrg_hbst|found OpenCV version: '${OpenCV_VERSION}' (${OpenCV_DIR}) - building OpenCV example code")
  
  #ds enable OpenCV usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_OPENCV true)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
endif()

#ds define the catkin package
find_package(catkin)
catkin_package(INCLUDE_DIRS src)

#ds configure build for examples
include_directories(${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
add_subdirectory(examples)
