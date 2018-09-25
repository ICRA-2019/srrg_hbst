cmake_minimum_required(VERSION 2.8.3)
project(srrg_hbst)

#-------------------------------------------------------------------------------------------------------------------------------------
#ds HBST generic compilation flags - ADD THEM IN YOUR PROJECT AS WELL TO ENABLE THEM (HBST is header-only)

# - SRRG_MERGE_DESCRIPTORS: HBST checks for identical descriptors stemming from multiple images and represents them with a single entity
#two descriptors are merged if their distance is within a certain threshold
#the threshold can be controlled with the static variable maximum_distance_for_merge
#a value of maximum_distance_for_merge=0 means they have to be identical
#the higher this value, the more compact and efficient becomes the tree at the cost of matching accuracy
#this flag is recommended for large-scale datasets and lifelong recognition applications
add_definitions(-DSRRG_MERGE_DESCRIPTORS)
#-------------------------------------------------------------------------------------------------------------------------------------



#ds determine build type, default build type RELEASE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

#ds check flags for release build
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  add_definitions(-Ofast -march=native -Wall -pedantic -std=c++11 -DNDEBUG)
  message("srrg_hbst|adding flags for build type '${CMAKE_BUILD_TYPE}': '-Ofast -march=native -Wall -pedantic -std=c++11 -DNDEBUG'")
else()
  add_definitions(-O0 -march=native -Wall -pedantic -std=c++11 -g)
  message("srrg_hbst|adding flags for build type '${CMAKE_BUILD_TYPE}': '-O0 -march=native -Wall -pedantic -std=c++11 -g'")
endif()

#ds cmake modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#ds locate external packages: Eigen (currently not required)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND OR EIGEN3_INCLUDE_DIR)
  message("srrg_hbst|found Eigen version: '3' (${EIGEN3_INCLUDE_DIR}) - building Eigen example code")

  #ds enable Eigen usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_EIGEN true)
  include_directories(${EIGEN3_INCLUDE_DIR})
  
  #ds ADD THIS IN YOUR PROJECT TO BUILD EIGEN DEPENDENT CODE (HBST is header-only)
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
endif()

#ds check if ROS is present
if("$ENV{ROS_DISTRO}" STREQUAL "kinetic" OR "$ENV{ROS_DISTRO}" STREQUAL "indigo" OR "$ENV{ROS_DISTRO}" STREQUAL "melodic")
  message("srrg_hbst|found ROS version: '$ENV{ROS_DISTRO}' - building ROS example nodes")
  
  #ds check for required packages
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    cv_bridge
    image_geometry
  )
  
  #ds switch to ROS opencv libraries without calling find_package to avoid conflicts with other existing OpenCV installations
  set(OpenCV_DIR "ROS")
  set(OpenCV_FOUND true)
  set(OpenCV_LIBS ${catkin_LIBRARIES})

  #ds enable ROS usage
  set(SRRG_HBST_HAS_ROS true)
  
  #ds ADD THIS IN YOUR PROJECT TO BUILD ROS DEPENDENT CODE (HBST is header-only)
  add_definitions(-DSRRG_HBST_HAS_ROS)
endif()

#ds if opencv is not set yet
if(NOT OpenCV_FOUND)
  find_package(OpenCV QUIET)
endif()
if(OpenCV_FOUND)
  message("srrg_hbst|found OpenCV version: '${OpenCV_VERSION}' (${OpenCV_DIR}) - building OpenCV example code")

  #ds enable OpenCV usage for the complete project (wrappers, example code)
  set(SRRG_HBST_HAS_OPENCV true)
  include_directories(${OpenCV_INCLUDE_DIRS})

  #ds ADD THIS IN YOUR PROJECT TO BUILD OPENCV DEPENDENT CODE (HBST is header-only)
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
endif()

#ds define the catkin package
find_package(catkin)
catkin_package(INCLUDE_DIRS src)

#ds configure build for examples
include_directories(${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
add_subdirectory(examples)
