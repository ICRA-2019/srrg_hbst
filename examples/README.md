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

    match_opencv_indices test_images/
    
and one using pointers:
    
    match_opencv_pointers test_images/
    
In the case of the incremental tree check out:
    
    match_incremental_opencv_pointers  test_images/
    
Who is interested in only image scores finds a simple example in:

    match_incremental_opencv_score test_images/

Additionally we provide a straightforward feature tracking application based on HBST:   
  
    tracker -images test_images/
    
Alternatively, the tracker can also process video files:   
  
    tracker -video video.mp4
    
Note that with this tracking algorithm we can obtain multiple correspondences for image registration or loop closing at no time!

Furthermore we provide a live visual place recognition application:

    recognizer -video video.mp4 -space 100
    
Which displays computed features for the current image (blue) and highlights them (green) in case they have been matched against a feature from a database image. <br>
The parameter `-space` controls the minimum number of images that have to lie between the current image and the database image (e.g. 100).

## Eigen:
More robust descriptor track based tree construction can be inspected in:

    match_probabilistic

## Eigen, OpenCV and QGLViewer:
For now, only processing of the raw [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) is supported. <br>
Note that for building these examples, the [libQGLViewer](http://libqglviewer.com/) library is required.

An example dataset sequence is available at: [kitti_sequence_00](https://drive.google.com/open?id=1KPay-nqVXvj5Ht6lfF0KdILQpN97AbRX) (2.3GB, courtesy of the [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php))

We implemented a very simple <b>monocular</b> visual odometry system based on HBST data association, including a 3D viewer in a single source file (700 lines):

    smoother_monocular kitti_sequence_00/image_0/000000.png kitti_sequence_00/calib.txt
    
Where `kitti_sequence_00/image_0/000000.png` defines the single camera input image stream corresponding to the camera calibration of `kitti_sequence_00/calib.txt`.

As an initial motion guess we dissect the essential matrix for feature matches between the current and a previous image. <br>
In case of insufficient conditions the motion from a previous is utilized (constant velocity motion model). <br>
From this motion guess we obtain an initial guess for the features 3D point positions in the first camera frame (SfM via midpoint triangulation). <br>
Based on the 3D camera positions and the measured 2D image positions of the features we refine the motion guess using a Projective ICP approach. <br>
The system experiences significant drift, resulting from the dead reckoning motion estimation scheme,
as well as the scale ambiguity that is present in a monocular system.

We also implemented a very simple <b>binocular</b> visual odometry system based on HBST data association, including a 3D viewer in a single source file (600 lines):

    smoother_binocular kitti_sequence_00/image_0/000000.png kitti_sequence_00/image_1/000000.png kitti_sequence_00/calib.txt

Where `kitti_sequence_00/image_0/000000.png` defines the left camera input image stream, <br>
and `kitti_sequence_00/image_1/000000.png` defines the right camera input image stream corresponding to the stereo camera calibration of `kitti_sequence_00/calib.txt`.

As an initial motion guess the motion from a previous is utilized (constant velocity motion model). <br>
The features 3D point positions are obtained from stereopsis, defined by a rigid stereo camera configuration. <br>
Based on the 3D camera positions and the measured 2D image positions of the features we refine the motion guess using a Stereo Projective ICP approach. <br>
The system experiences tolerable drift, resulting from the dead reckoning motion estimation scheme. <br>
The scale is estimated correctly thanks to the available rigid stereo triangulation (assuming a majority of correct triangulation pairs).

### ROS (in combination with catkin)
We provide our example applications also in the form of ROS nodes, enabling easy integration.
The available nodes are:

    rosun srrg_hbst recognizer_node -camera <ROS/camera/topic> -space <integer>
    
The ROS `recognizer_node` is fully interactive, hence by pressing [C] you can reset the HBST database at any time! <br>
To obtain a camera image stream on your machine (i.e. `<ROS/camera/topic>`) you can use a regular ROS camera driver node such as [usb_cam](http://wiki.ros.org/usb_cam)
