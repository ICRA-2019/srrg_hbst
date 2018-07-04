    HBST examples
## Standalone:
The following example shows the core functionality provided when using an *individual tree* to describe *each image*:

    match
    
Alternatively one can build only *a single tree* that is incrementally grown for each image:
    
    match_incremental
    
Note that the above executables required to be run from the `build` folder or alternatively via `rosrun srrg_hbst`. <br>
None of the above examples require OpenCV or Eigen.
## OpenCV:
The HBST library already shipps with OpenCV2/3 wrappers. <br>
The following 4 examples demonstrate the descriptor matching performance for a set of 10 test images. <br>
In each example we compute a set of input descriptors using standard OpenCV feature detectors and descriptor extractors. <br>
To visualize the obtained correspondences we link them with our Matchables using `indices` or `pointers`.

For the individual trees we have an example with indices:

    match_opencv_indices /test_images/
    
and one using pointers:
    
    match_opencv_pointers /test_images/
    
In the case of the incremental tree check out:
    
    match_incremental_opencv_pointers  /test_images/
    
Additionally we provide a straightforward feature tracking application based on HBST:   
  
    tracker /test_images/
    
Note that with this tracking algorithm we can obtain multiple correspondences for image registration or loop closing at no time! <br>
Again the file paths need to be adjusted or the executables run with `rosrun srrg_hbst`.
    
## Eigen:
More robust descriptor track based tree construction can be inspected in:

    match_probabilistic

## Eigen, OpenCV and QGLViewer:
We implemented a simple monocular visual odometry system based on HBST data association:

    rosrun srrg_hbst smoother_monocular image_0/000000.png calib.txt

For now, only processing of the raw [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) is supported. <br>
Note that for building these examples, the [libQGLViewer](http://libqglviewer.com/) library is required.
