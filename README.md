# HBST: Hamming Binary Search Tree Header-only library

Contributors: Dominik Schlegel, Giorgio Grisetti <br/>
<br/>

Supported platforms:
- UNIX x86/x64
- Windows x86/x64 (untested) <br/>

Dependencies:
- CMake 2.8.3+ (https://cmake.org/)
- C++ 11 STL libraries <br/>

Optionals:
- Eigen3 (http://eigen.tuxfamily.org/) for probabilisticly enhanced search access (add the definition `-DSRRG_HBST_HAS_EIGEN` in your cmake project)
- OpenCV2/3 (http://opencv.org/) for the automatic build of wrapped constructors and OpenCV related example code (add the definition `-DSRRG_HBST_HAS_OPENCV` in your cmake project) <br/>

---
## Example code (catkin ready!)
CMake build sequence for example code (in project root):

    mkdir build
    cd build
    cmake ..
    make

A simple example program can be called with (while still in the build folder):

    examples/srrg_hbst_search_opencv_indices ../examples/test_images

Showing the HBST matching performance for a sequence of 10 images using `indexed` Matchables <br/>
The example sequence of 10 images is part of the repository and can be found under `examples/test_images`

---
## Build your own Descriptor/Node types!
The 2 base classes: `BinaryNode` and `BinaryMatchable` (see `src/srrg_hbst_types`) can easily be inherited <br/>
Users might specify their own, augmented binary descriptor and node classes with specific leaf spawning <br>
Two variants of subclassing are already provided in `src/srrg_hbst_types_probabilistic`

---
## Related publications
IROS 2016 - Visual localization and loop closing using decision trees and binary features: http://ieeexplore.ieee.org/document/7759679/ (DOI: 10.1109/IROS.2016.7759679)

    @inproceedings{schlegel2016visual,
      title={Visual localization and loop closing using decision trees and binary features},
      author={Schlegel, Dominik and Grisetti, Giorgio},
      booktitle={Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference on},
      pages={4616--4623},
      year={2016},
      organization={IEEE}
    }
