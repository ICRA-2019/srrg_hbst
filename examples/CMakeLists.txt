#ds build standalone examples
add_executable(match match.cpp)
add_executable(match_incremental match_incremental.cpp)

#ds if we found an opencv installation
if(SRRG_HBST_HAS_OPENCV)
  target_link_libraries(match ${OpenCV_LIBS})
  target_link_libraries(match_incremental ${OpenCV_LIBS})
  add_executable(match_opencv_indices match_opencv_indices.cpp)
  target_link_libraries(match_opencv_indices ${OpenCV_LIBS})
  add_executable(match_opencv_pointers match_opencv_pointers.cpp)
  target_link_libraries(match_opencv_pointers ${OpenCV_LIBS})
  add_executable(match_incremental_opencv_pointers match_incremental_opencv_pointers.cpp)
  target_link_libraries(match_incremental_opencv_pointers ${OpenCV_LIBS})
  add_executable(tracker tracker.cpp)
  target_link_libraries(tracker ${OpenCV_LIBS})
endif()

#ds if we found an eigen installation
if(SRRG_HBST_HAS_EIGEN)
  add_executable(match_probabilistic match_probabilistic.cpp)
  
  #ds and an opencv installation
  if(SRRG_HBST_HAS_OPENCV)
    target_link_libraries(match_probabilistic ${OpenCV_LIBS})
  endif()
endif()
