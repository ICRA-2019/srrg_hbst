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
  
  add_executable(match_incremental_opencv_score match_incremental_opencv_score.cpp)
  target_link_libraries(match_incremental_opencv_score ${OpenCV_LIBS})
  
  add_executable(tracker tracker.cpp)
  target_link_libraries(tracker ${OpenCV_LIBS})
  
    add_executable(recognizer recognizer.cpp)
  target_link_libraries(recognizer ${OpenCV_LIBS})
endif()

#ds if we found an eigen installation
if(SRRG_HBST_HAS_EIGEN)
  add_executable(match_probabilistic match_probabilistic.cpp)
  
  #ds and an opencv installation
  if(SRRG_HBST_HAS_OPENCV)
    target_link_libraries(match_probabilistic ${OpenCV_LIBS})
    
    #ds smoother example also has GUI utilities - check dependencies
    find_package(QGLViewer QUIET)
    if(QGLVIEWER_FOUND OR QGLVIEWER_INCLUDE_DIR)
      message("srrg_hbst|found QGLViewer version - building smoother example code")
      include_directories(${QT_INCLUDE_DIRS} ${QGLVIEWER_INCLUDE_DIR})
      
      #ds monocular smoother example - only opencv3 supported - TODO support for opencv2
      if ("${OpenCV_VERSION_MAJOR}" STREQUAL "3")
        add_executable(smoother_monocular smoother_monocular.cpp)
        target_link_libraries(smoother_monocular 
          ${OpenCV_LIBS} 
          ${QGLVIEWER_LIBRARY} 
          ${QT_LIBRARIES}
          -lglut -lGL
        )
      endif()
      
      #ds binocular smoother example - opencv2 and opencv3 supported
      add_executable(smoother_binocular smoother_binocular.cpp)
      target_link_libraries(smoother_binocular 
        ${OpenCV_LIBS} 
        ${QGLVIEWER_LIBRARY} 
        ${QT_LIBRARIES}
        -lglut -lGL
      )
    endif()
  endif()
endif()
