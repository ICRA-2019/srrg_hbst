    HBST: Hamming Binary Search Tree Header-only library
Contributors: Dominik Schlegel, Giorgio Grisetti
## [* check out the wiki for videos and experimental results! *](https://gitlab.com/srrg-software/srrg_hbst/wikis/home)
Supported platforms:
- UNIX x86/x64 (tested: Ubuntu 14.04 x64, Ubuntu 16.04 x64)
- Windows x86/x64 (untested)

Minimum requirements:
- CMake 2.8.3+ (https://cmake.org/)
- C++ 11 STL libraries
- GCC 5 or higher

Optionals:
- Eigen3 (http://eigen.tuxfamily.org/) for probabilisticly enhanced search access (add the definition `-DSRRG_HBST_HAS_EIGEN` in your cmake project).
- OpenCV2/3 (http://opencv.org/) for the automatic build of wrapped constructors and OpenCV related example code (add the definition `-DSRRG_HBST_HAS_OPENCV` in your cmake project).
- Catkin Command Line Tools (https://catkin-tools.readthedocs.io/en/latest/) for easy integration

## Example code (catkin ready!)
Out of source CMake build sequence for example code (in project root):

    mkdir build
    cd build
    cmake ..
    make

A simple example program with visuals can be called with (from the project root, when OpenCV is installed):

    build/examples/match_opencv_indices examples/test_images

Showing the HBST matching performance for a sequence of 10 images using `indexed` Matchables. <br>
The example sequence of 10 images is part of the repository and can be found under `examples/test_images`. <br>
The example images are courtesy of the [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

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
Please cite our most recent publication when using the HBST library: <br>

    @article{2018-schlegel-hbst,
      author        = {Dominik Schlegel and Giorgio Grisetti},
      title         = {{HBST: A Hamming Distance embedding Binary Search Tree for Visual Place Recognition}},
      journal       = {CoRR},
      volume        = {abs/1802.09261},
      year          = {2018},
      url           = {http://arxiv.org/abs/1802.09261},
      archivePrefix = {arXiv},
      eprint        = {1802.09261}
    }

> arXiv 2018 - HBST: A Hamming Distance embedding Binary Search Tree for Visual Place Recognition: https://arxiv.org/abs/1802.09261 (arXiv: 1802.09261)

    @inproceedings{2016-schlegel-hbst,
      title        = {Visual localization and loop closing using decision trees and binary features},
      author       = {Schlegel, Dominik and Grisetti, Giorgio},
      booktitle    = {Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference on},
      pages        = {4616--4623},
      year         = {2016},
      organization = {IEEE}
    }
    
> IROS 2016 - Visual localization and loop closing using decision trees and binary features: http://ieeexplore.ieee.org/document/7759679/ (DOI: 10.1109/IROS.2016.7759679)
