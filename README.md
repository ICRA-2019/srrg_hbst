    HBST: Hamming Binary Search Tree Header-only library
Contributors: Dominik Schlegel, Giorgio Grisetti
## [* check out the wiki for videos and experimental results! *](https://gitlab.com/srrg-software/srrg_hbst/wikis/home)

Supported platforms:
- Linux: Ubuntu 14.04 LTS, Ubuntu 16.04 LTS, Ubuntu 18.04 LTS

Minimum requirements:
- [CMake](https://cmake.org) 2.8.3 or higher
- [C++ 11](http://en.cppreference.com) or higher
- [GCC](https://gcc.gnu.org) 5 or higher

Optional:
- [Eigen3](http://eigen.tuxfamily.org/) for probabilisticly enhanced search access (add the definition `-DSRRG_HBST_HAS_EIGEN` in your cmake project).
- [OpenCV2/3](http://opencv.org/) for the automatic build of wrapped constructors and OpenCV related example code (add the definition `-DSRRG_HBST_HAS_OPENCV` in your cmake project).
- [libQGLViewer](http://libqglviewer.com/) for visual odometry examples ([viewers](examples))
- [catkin Command Line Tools](https://catkin-tools.readthedocs.io/en/latest/) for easy CMake project integration
- [ROS Indigo/Kinetic/Melodic](http://wiki.ros.org/ROS/Installation) for live ROS nodes (make sure you have a sane OpenCV installation)

## [* example code *](examples) (catkin ready!)
Out of source CMake build sequence for example code (in project root):

    mkdir build
    cd build
    cmake ..
    make

A simple example program with visuals can be called with (from the project root, when OpenCV is installed):

    build/examples/match_opencv_indices examples/test_images

Showing the HBST matching performance for a sequence of 10 images using `indexed` Matchables. <br>
The example sequence of 10 images is part of the repository and can be found under `examples/test_images`. <br>
All images are courtesy of the [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

---
Alternative catkin build:

    catkin build srrg_hbst
    
An example using the incremental HBST can be run from the project root with:

    rosrun srrg_hbst match_incremental_opencv_pointers examples/test_images


## Build your own Descriptor/Node types!
The 2 base classes: `BinaryNode` and `BinaryMatchable` (see `src/srrg_hbst_types`) can easily be inherited. <br>
Users might specify their own, augmented binary descriptor and node classes with specific leaf spawning. <br>
Two variants of subclassing are already provided in `src/srrg_hbst_types_probabilistic`.

## It doesn't work?
[Open an issue](https://gitlab.com/srrg-software/srrg_hbst/issues) or contact the maintainer (see package.xml)

## Related publications
Please cite our most recent article when using the HBST library: <br>

    @article{2018-schlegel-hbst, 
      author  = {D. Schlegel and G. Grisetti}, 
      journal = {IEEE Robotics and Automation Letters}, 
      title   = {{HBST: A Hamming Distance Embedding Binary Search Tree for Feature-Based Visual Place Recognition}}, 
      year    = {2018}, 
      volume  = {3}, 
      number  = {4}, 
      pages   = {3741-3748}
    }

> RA-L 2018 'HBST: A Hamming Distance Embedding Binary Search Tree for Feature-Based Visual Place Recognition' <br>
> https://ieeexplore.ieee.org/document/8411466/ (DOI: 10.1109/LRA.2018.2856542)

Prior works:

    @inproceedings{2016-schlegel-hbst, 
      author    = {D. Schlegel and G. Grisetti}, 
      booktitle = {2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
      title     = {Visual localization and loop closing using decision trees and binary features}, 
      year      = {2016}, 
      pages     = {4616-4623}, 
    }

> IROS 2016 'Visual localization and loop closing using decision trees and binary features' <br>
> http://ieeexplore.ieee.org/document/7759679/ (DOI: 10.1109/IROS.2016.7759679)
