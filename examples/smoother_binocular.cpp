#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <qapplication.h>
#include <QKeyEvent>
#include <QGLViewer/qglviewer.h>
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif
#if (((QGLVIEWER_VERSION & 0xff0000) >> 16) >= 2 && ((QGLVIEWER_VERSION & 0x00ff00) >> 8) >= 6)
#define qglv_real qreal
#else
#define qglv_real float
#endif

//ds current HBST configuration - 256 bit
typedef srrg_hbst::BinaryTree256 Tree;
typedef srrg_hbst::BinaryTree256::Matchable Matchable;
typedef srrg_hbst::BinaryTree256::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree256::Match Match;
typedef srrg_hbst::BinaryTree256::MatchVector MatchVector;
typedef srrg_hbst::BinaryTree256::MatchVectorMap MatchVectorMap;

//ds aligned eigen types
typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > IsometryVector;



//ds map objects
struct Landmark {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d coordinates_in_world;
};

struct Framepoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Framepoint(const cv::KeyPoint& keypoint_left_,
             const cv::KeyPoint& keypoint_right_,
             const cv::Mat& descriptor_left_,
             const cv::Mat& descriptor_right_,
             const Eigen::Vector3d& coordinates_in_camera_): keypoint_left(keypoint_left_),
                                                             keypoint_right(keypoint_right_),
                                                             descriptor_left(descriptor_left_),
                                                             descriptor_right(descriptor_right_),
                                                             coordinates_in_camera(coordinates_in_camera_) {}

  //ds measured
  const cv::KeyPoint keypoint_left;
  const cv::KeyPoint keypoint_right;
  const cv::Mat descriptor_left;
  const cv::Mat descriptor_right;

  //ds computed
  Framepoint* previous  = 0;
  uint32_t track_length = 0;
  Eigen::Vector3d coordinates_in_camera;
  Landmark* landmark    = 0;
};

//ds simple trajectory viewer
class Viewer: public QGLViewer {
public:
  Viewer(QWidget* parent = 0): QGLViewer(parent) {setWindowTitle("trajectory viewer [OpenGL]");}
  void init();
  virtual void draw();
  void addPose(const Eigen::Isometry3d& pose_) {_poses.push_back(pose_);}
  void setLandmarks(std::shared_ptr<std::vector<Landmark*> > landmarks_) {_landmarks = landmarks_;}

  //ds usable camera
  class StandardCamera: public qglviewer::Camera {
  public:
    StandardCamera() {}
    qglv_real zNear() const {return qglv_real(_z_near);}
    qglv_real zFar() const {return qglv_real(_z_far);}
  protected:
    float _z_near = 0.001f;
    float _z_far  = 10000.0f;
  };

private:
  IsometryVector _poses;
  std::shared_ptr<std::vector<Landmark*> > _landmarks = 0;
  StandardCamera* _camera                             = 0;
  Eigen::Isometry3d _camera_to_robot                  = Eigen::Isometry3d::Identity();
};



//ds helpers
Eigen::Matrix3d getCameraCalibrationMatrixKITTI(const std::string& file_name_calibration_, Eigen::Vector3d& baseline_pixels_);
Eigen::Isometry3d v2t(const Eigen::Matrix<double, 6, 1>& t);
Eigen::Matrix3d skew(const Eigen::Vector3d& p);
std::vector<Framepoint*> getPointsFromStereo(const Eigen::Matrix3d& camera_calibration_matrix_,
                                             const Eigen::Vector3d& offset_to_camera_right_,
                                             const std::vector<cv::KeyPoint>& keypoints_left_,
                                             const std::vector<cv::KeyPoint>& keypoints_right_,
                                             const cv::Mat& descriptors_left_,
                                             const cv::Mat& descriptors_right_,
                                             const uint32_t& maximum_descriptor_distance_);



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 4) {
    std::cerr << "ERROR: invalid call - please use: ./smoother file_name_initial_image_LEFT file_name_initial_image_RIGHT calib.txt" << std::endl;
    return 0;
  }

  //ds configuration - TODO input validation
  const std::string file_name_initial_image_left  = argv_[1];
  const std::string file_name_initial_image_right = argv_[2];
  const std::string file_name_calibration         = argv_[3];
  std::cerr << "initial image LEFT: " << file_name_initial_image_left << std::endl;
  std::cerr << "initial image RIGHT: " << file_name_initial_image_right << std::endl;
  std::cerr << "calibration: " << file_name_calibration << std::endl;

  //ds load camera matrix and stereo configuration (offset to camera right)
  Eigen::Vector3d baseline_pixels_(Eigen::Vector3d::Zero());
  const Eigen::Matrix3d camera_calibration_matrix(getCameraCalibrationMatrixKITTI(file_name_calibration, baseline_pixels_));

  //ds parse image extension from left image
  const std::size_t index_delimiter_extension = file_name_initial_image_left.find_last_of('.');
  const std::string extension                 = file_name_initial_image_left.substr(index_delimiter_extension, file_name_initial_image_left.length()-index_delimiter_extension);
  std::cerr << "using image extension: '" << extension << "'" << std::endl;

  //ds parse folders - UNIX only
  const std::size_t index_delimiter_directory_left = file_name_initial_image_left.find_last_of('/');
  const std::string directory_name_images_left     = file_name_initial_image_left.substr(0, index_delimiter_directory_left+1);
  std::cerr << "reading images from directory: '" << directory_name_images_left << "'" << std::endl;
  const std::size_t index_delimiter_directory_right = file_name_initial_image_right.find_last_of('/');
  const std::string directory_name_images_right     = file_name_initial_image_right.substr(0, index_delimiter_directory_right+1);
  std::cerr << "reading images from directory: '" << directory_name_images_right << "'" << std::endl;

  //ds parse enumeration characters range (as file name of images)
  const uint32_t number_of_enumeration_characters = index_delimiter_extension-index_delimiter_directory_left-1;
  std::cerr << "using enumeration pattern: '";
  for (uint32_t u = 0; u < number_of_enumeration_characters; ++u) {
    std::cerr << u;
  }
  std::cerr << extension << "'" << std::endl;

  //ds miscellaneous configuration
  const uint32_t maximum_descriptor_distance = 25; //ds number of mismatching bits
  const uint32_t maximum_keypoint_distance   = 50; //ds pixels

  //ds feature handling
  cv::Ptr<cv::FastFeatureDetector> keypoint_detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
#if CV_MAJOR_VERSION == 2
  keypoint_detector    = new cv::FastFeatureDetector(25);
  descriptor_extractor = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
  keypoint_detector    = cv::FastFeatureDetector::create(25);
  descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#endif

  //ds 'map' gets populated during processing
  std::shared_ptr<std::vector<Landmark*> > landmarks(std::make_shared<std::vector<Landmark*> >(0));
  std::vector<std::vector<Framepoint*> > framepoints_per_image(0);

  //ds estimated trajectory
  IsometryVector estimated_camera_poses(1, Eigen::Isometry3d::Identity());

//  //ds initialize viewer
//  QApplication ui_server(argc_, argv_);
//  Viewer viewer;
//  viewer.setLandmarks(landmarks);
//  viewer.show();
//  viewer.updateGL();

  //ds constant velocity motion model bookkeeping
  Eigen::Isometry3d motion_previous(Eigen::Isometry3d::Identity());

  //ds allocate an empty tree
  Tree hbst_tree;

  //ds start processing images (forced grayscale)
  uint32_t number_of_processed_images = 0;
  std::string file_name_image_current_left  = file_name_initial_image_left;
  std::string file_name_image_current_right = file_name_initial_image_right;
  cv::Mat image_current_left                = cv::imread(file_name_image_current_left, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat image_current_right               = cv::imread(file_name_image_current_right, CV_LOAD_IMAGE_GRAYSCALE);
  while (image_current_left.rows != 0 && image_current_left.cols != 0 && number_of_processed_images < 100) {

// FEATURE EXTRACTION AND STRUCTURE COMPUTATION ---------------------------------------------------------------------------------------------------------------------------
    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;
    keypoint_detector->detect(image_current_left, keypoints_left);
    keypoint_detector->detect(image_current_right, keypoints_right);

    //ds compute BRIEF descriptors
    cv::Mat descriptors_left;
    cv::Mat descriptors_right;
    descriptor_extractor->compute(image_current_left, keypoints_left, descriptors_left);
    descriptor_extractor->compute(image_current_right, keypoints_right, descriptors_right);
    uint32_t number_of_keypoints = keypoints_left.size();

    //ds obtain 3D points from rigid stereo
    std::vector<Framepoint*> current_points(getPointsFromStereo(camera_calibration_matrix,
                                                                baseline_pixels_,
                                                                keypoints_left,
                                                                keypoints_right,
                                                                descriptors_left,
                                                                descriptors_right,
                                                                maximum_descriptor_distance));
    framepoints_per_image.push_back(current_points);
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// HBST FEATURE MATCHING ----------------------------------------------------------------------------------------------------------------------------------------------------
    //ds obtain linked matchables for the LEFT image
    MatchableVector matchables(current_points.size());
    for (uint32_t u = 0; u < current_points.size(); ++u) {
      matchables[u] = new Matchable(current_points[u], current_points[u]->descriptor_left, number_of_processed_images);
    }

    //ds obtain matches against all inserted matchables (i.e. images so far)
    MatchVectorMap matches_per_image;
    hbst_tree.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //ds info
    cv::Mat image_display(image_current_left);
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

    //ds if we obtained matches
    uint32_t number_of_measurements = 0;
    if (number_of_processed_images > 0) {

// MATCH PRUNING AND BOOKKEEPING --------------------------------------------------------------------------------------------------------------------------------------------
      //ds grab matches from previous image only (currently ignoring matches to images which lay further in the past)
      const MatchVector& matches(matches_per_image[number_of_processed_images-1]);

      //ds ignorantly filter outlier matches in a brutal loop
      double average_track_length  = 0;
      std::set<Framepoint*> linked_previous;
      for (uint32_t u = 0; u < matches.size(); ++u) {

        //ds mother of all EVIL casts in 'c++' TODO maybe adjust library
        Framepoint* current_point  = const_cast<Framepoint*>(static_cast<const Framepoint*>(matches[u].pointer_query));
        Framepoint* previous_point = const_cast<Framepoint*>(static_cast<const Framepoint*>(matches[u].pointer_reference));

        //ds cross-check if we not already matched against this previous point TODO enable cross-check in HBST
        if (linked_previous.count(previous_point)) {

          //ds rudely ignore the track
          continue;
        }

        //ds check maximum tracking distance
        if (cv::norm(cv::Mat(current_point->keypoint_left.pt), cv::Mat(previous_point->keypoint_left.pt)) < maximum_keypoint_distance) {

          //ds establish track between framepoints
          current_point->landmark     = previous_point->landmark;
          current_point->previous     = previous_point;
          current_point->track_length = previous_point->track_length+1;
          average_track_length += current_point->track_length;
          linked_previous.insert(previous_point);

          //ds record match
          current_points[number_of_measurements] = current_point;

          //ds initial guess statistic
          ++number_of_measurements;
        }
      }
      current_points.resize(number_of_measurements);
      average_track_length /= number_of_measurements;
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

      std::cerr << "read images: " << file_name_image_current_left << " | " << file_name_image_current_right
                << " with triangulated points: " << current_points.size() << "/" << number_of_keypoints
                << " inlier tracks: " << number_of_measurements
                << " average track length: " << average_track_length << std::endl;

// IMAGE REGISTRATION (INITIAL POSE GUESS OPTIMIZATION) ---------------------------------------------------------------------------------------------------------------------
      //ds always use the constant velocity motion model for pose motion guess
      Eigen::Isometry3d previous_to_current = motion_previous;

      //ds initialize LS (stereo projective error minimization)
      const uint32_t maximum_number_of_iterations = 100;
      double total_error_squared_previous         = 0;
      const double maximum_error_squared          = 10*10;
      Eigen::Matrix<double, 6, 6> H(Eigen::Matrix<double, 6, 6>::Zero());
      Eigen::Matrix<double, 6, 1> b(Eigen::Matrix<double, 6, 1>::Zero());
      Eigen::Matrix3d omega(Eigen::Matrix3d::Identity());
      uint32_t number_of_iterations               = 0;

      //ds damping
      Eigen::Matrix<double, 6, 6> damping(Eigen::Matrix<double, 6, 6>::Identity());
      damping *= number_of_measurements;

      //ds start iterations
      for (uint32_t u = 0; u < maximum_number_of_iterations; ++u) {
        H.setZero();
        b.setZero();
        double total_error_squared = 0;

        //ds linearize all measurements
        for (uint32_t index_measurement = 0; index_measurement < number_of_measurements; ++index_measurement) {
          omega.setIdentity();
          const Framepoint* current_point  = current_points[index_measurement];
          const Framepoint* previous_point = current_point->previous;

          //ds transform point from previous frame into current
          const Eigen::Vector3d point_in_camera_left(previous_to_current*previous_point->coordinates_in_camera);

          //ds check for invalid depth
          if (point_in_camera_left.z() <= 0) {
            continue;
          }

          //ds project past point in current frame
          const Eigen::Vector3d homogeneous_coordinates_left  = camera_calibration_matrix*point_in_camera_left;
          const Eigen::Vector3d homogeneous_coordinates_right = homogeneous_coordinates_left+baseline_pixels_;
          const Eigen::Vector2d image_coordinates_moving_left(homogeneous_coordinates_left.x()/homogeneous_coordinates_left.z(),
                                                              homogeneous_coordinates_left.y()/homogeneous_coordinates_left.z());
          const Eigen::Vector2d image_coordinates_moving_right(homogeneous_coordinates_right.x()/homogeneous_coordinates_right.z(),
                                                               homogeneous_coordinates_right.y()/homogeneous_coordinates_right.z());

          //ds check if out of range
          if (image_coordinates_moving_left.x() < 0
            || image_coordinates_moving_left.y() < 0
            || image_coordinates_moving_left.x() > image_current_left.cols
            || image_coordinates_moving_left.y() > image_current_left.rows
            || image_coordinates_moving_right.x() < 0
            || image_coordinates_moving_right.y() < 0
            || image_coordinates_moving_right.x() > image_current_right.cols
            || image_coordinates_moving_right.y() > image_current_right.rows ) {
            continue;
          }

          //ds compute error - only once for the vertical component since we're assuming rectified images with horizontal stereopsis
          const Eigen::Vector3d error(image_coordinates_moving_left.x() - current_point->keypoint_left.pt.x,
                                      image_coordinates_moving_left.y() - current_point->keypoint_left.pt.y,
                                      image_coordinates_moving_right.x() - current_point->keypoint_right.pt.x);

          //ds weight horizontal error twice (since in fact the error stems from two measurements)
          omega(1,1) *= 2;

          //ds squared error
          const double error_squared = error.transpose()*omega*error;

          //ds kernelize
          if (error_squared > maximum_error_squared) {
            omega *= maximum_error_squared/error_squared;
          }

          //ds update error
          total_error_squared += error_squared;

          //ds precompute homogeneous divisions in both camera image planes
          const double inverse_sampled_c_left          = 1/homogeneous_coordinates_left.z();
          const double inverse_sampled_c_squared_left  = inverse_sampled_c_left*inverse_sampled_c_left;
          const double inverse_sampled_c_right         = 1/homogeneous_coordinates_right.z();
          const double inverse_sampled_c_squared_right = inverse_sampled_c_right*inverse_sampled_c_right;

          //ds jacobian
          Eigen::Matrix<double, 3, 6> jacobian(Eigen::Matrix<double, 3, 6>::Zero());

          //ds jacobian of the transform
          Eigen::Matrix<double, 3, 6> jacobian_transform;
          jacobian_transform.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
          jacobian_transform.block<3,3>(0,3) = -2*skew(point_in_camera_left);

          //ds precompute parts of the final jacobian
          const Eigen::Matrix<double, 3, 6> jacobian_camera_matrix_transform(camera_calibration_matrix*jacobian_transform);

          //ds jacobian parts of the homogeneous division: left
          Eigen::Matrix<double, 2, 3> jacobian_homogeneous_left;
          jacobian_homogeneous_left << inverse_sampled_c_left, 0, -homogeneous_coordinates_left.x()*inverse_sampled_c_squared_left,
                                       0, inverse_sampled_c_left, -homogeneous_coordinates_left.y()*inverse_sampled_c_squared_left;

          //ds jacobian parts of the homogeneous division: right, we compute only the contribution for the horizontal error
          Eigen::Matrix<double, 1, 3> jacobian_homogeneous_right;
          jacobian_homogeneous_right << inverse_sampled_c_right, 0, -homogeneous_coordinates_right.x()*inverse_sampled_c_squared_right;

          //ds we have to compute the full block
          jacobian.block<2,6>(0,0) = jacobian_homogeneous_left*jacobian_camera_matrix_transform;

          //ds we only have to compute the horizontal block
          jacobian.block<1,6>(2,0) = jacobian_homogeneous_right*jacobian_camera_matrix_transform;

          //ds precompute
          const Eigen::Matrix<double, 6, 3> jacobian_transposed = jacobian.transpose();

          //ds sum up
          H += jacobian_transposed*omega*jacobian;
          b += jacobian_transposed*omega*error;
        }

        //ds compute current perturbation
        const Eigen::VectorXd dx = (H+damping).fullPivLu().solve(-b);

        //ds apply perturbation to pose
        previous_to_current = v2t(dx)*previous_to_current;

        //ds enforce proper rotation matrix
        const Eigen::Matrix3d rotation      = previous_to_current.linear();
        Eigen::Matrix3d rotation_squared    = rotation.transpose() * rotation;
        rotation_squared.diagonal().array() -= 1;
        previous_to_current.linear()        -= 0.5*rotation*rotation_squared;

        //ds check for convergence and terminate if so
        if (std::fabs(total_error_squared-total_error_squared_previous) < 1e-5) {
          break;
        }
        total_error_squared_previous = total_error_squared;
        ++number_of_iterations;
      }
      std::cerr << "final motion estimate (average squared pixel error: " << total_error_squared_previous/number_of_measurements
                << ", iterations: " << number_of_iterations << "): \n" << previous_to_current.matrix() << std::endl;

      //ds update motion model
      motion_previous = previous_to_current;

      //ds update trajectory
      estimated_camera_poses.push_back(estimated_camera_poses[number_of_processed_images-1]*previous_to_current.inverse());

      //ds refine triangulated point positions in current frame based on transform and positions in previous frame
      uint32_t number_of_created_landmarks = 0;
      for (Framepoint* framepoint: current_points) {
        Framepoint* framepoint_iterator                  = framepoint;
        uint32_t index_pose                              = estimated_camera_poses.size()-1;
        Eigen::Vector3d coordinates_in_world_accumulated = Eigen::Vector3d::Zero();
        while(framepoint_iterator) {
          const Eigen::Isometry3d& camera_to_world = estimated_camera_poses[index_pose];
          coordinates_in_world_accumulated        += camera_to_world*framepoint_iterator->coordinates_in_camera;
          --index_pose;
          framepoint_iterator = framepoint_iterator->previous;
        }

        //ds if there is no landmark yet (track length 1)
        if (!framepoint->landmark) {

          //ds create landmark and set it to the minimal track
          framepoint->landmark           = new Landmark();
          framepoint->previous->landmark = framepoint->landmark;
          landmarks->push_back(framepoint->landmark);
          ++number_of_created_landmarks;
        }

        //ds update landmark position
        framepoint->landmark->coordinates_in_world = coordinates_in_world_accumulated/(framepoint->track_length+1);
      }
      std::cerr << "created new landmarks: " << number_of_created_landmarks << "/" << current_points.size() << std::endl;
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

      //ds draw tracks on image
      for (const Framepoint* current_point: current_points) {
        cv::line(image_display, current_point->keypoint_left.pt, current_point->previous->keypoint_left.pt, cv::Scalar(0, 255, 0));
      }
    } else {
      std::cerr << "read image: " << file_name_image_current_left << " | " << file_name_image_current_right
                << " with triangulated points: " << current_points.size()
                << " initial image " << std::endl;
    }

//    //ds draw currently detected points
//    for (Framepoint* framepoint: current_points) {
//      cv::circle(image_display, framepoint->keypoint_left.pt, 2, cv::Scalar(255, 0, 0), -1);
//    }
//    cv::imshow("current image", image_display);
//    cv::waitKey(1);

//    //ds update viewer
//    viewer.addPose(estimated_camera_poses[number_of_processed_images]);
//    viewer.updateGL();
//    ui_server.processEvents();

    //ds compute file name for next images
    ++number_of_processed_images;
    std::string file_name_tail             = std::to_string(number_of_processed_images);
    const uint32_t number_of_padding_zeros = number_of_enumeration_characters-file_name_tail.length();
    for (uint32_t u = 0; u < number_of_padding_zeros; ++u) {
      file_name_tail = "0"+file_name_tail;
    }
    file_name_image_current_left  = directory_name_images_left+file_name_tail+extension;
    file_name_image_current_right = directory_name_images_right+file_name_tail+extension;

    //ds read next images
    image_current_left  = cv::imread(file_name_image_current_left, CV_LOAD_IMAGE_GRAYSCALE);
    image_current_right = cv::imread(file_name_image_current_right, CV_LOAD_IMAGE_GRAYSCALE);
  }
  std::cerr << "stopped at image: " << file_name_image_current_left << " - not found" << std::endl;
  std::cerr << "stopped at image: " << file_name_image_current_right << " - not found" << std::endl;

  //ds free map objects
  for (const Landmark* landmark: *landmarks) {
    delete landmark;
  }
  for (const std::vector<Framepoint*>& framepoints: framepoints_per_image) {
    for (const Framepoint* framepoint: framepoints) {
      delete framepoint;
    }
  }
  return 0;
}

void Viewer::init() {
  QGLViewer::init();

  //ds transform robot to camera (for visualization purposes only)
  _camera_to_robot.linear() << 0,  0,  1,
                              -1,  0,  0,
                               0, -1,  0;
  _poses.clear();

  //ds mouse bindings.
  setMouseBinding(Qt::NoModifier, Qt::RightButton, CAMERA, ZOOM);
  setMouseBinding(Qt::NoModifier, Qt::MidButton, CAMERA, TRANSLATE);
  setMouseBinding(Qt::ControlModifier, Qt::LeftButton, RAP_FROM_PIXEL);

  //ds set flags
  glClearColor(1.0f,1.0f,1.0f,1.0f);
  glEnable(GL_LINE_SMOOTH);
  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glShadeModel(GL_FLAT);

  //ds set custom camera and free old one
  if (_camera) {delete _camera;}
  _camera = new StandardCamera();
  _camera->setPosition(qglviewer::Vec(0.0f, 0.0f, 1000.0f));
  _camera->lookAt(qglviewer::Vec(0.0f, 0.0f, 0.0f));
  qglviewer::Camera* old_camera = camera();
  setCamera(_camera);
  delete old_camera;
}

void Viewer::draw() {
  glPointSize(1);
  glLineWidth(1);

  glPushMatrix();
  glMultMatrixf(_camera_to_robot.cast<float>().data());
  glColor3f(0.0, 0.0, 0.0);
  drawAxis(10);

  //ds draw camera poses in blue
  glBegin(GL_LINES);
  glColor3f(0, 0, 1);
  Eigen::Isometry3d previous_pose(Eigen::Isometry3d::Identity());
  for (const Eigen::Isometry3d& pose: _poses) {
    glVertex3f(previous_pose.translation().x(), previous_pose.translation().y(), previous_pose.translation().z());
    glVertex3f(pose.translation().x(), pose.translation().y(), pose.translation().z());
    previous_pose = pose;
  }
  glEnd();

  //ds draw points in grey
  glBegin(GL_POINTS);
  glColor3f(0.75, 0.75, 0.75);
  for (const Landmark* landmark: *_landmarks) {
    glVertex3f(landmark->coordinates_in_world.x(), landmark->coordinates_in_world.y(), landmark->coordinates_in_world.z());
  }
  glEnd();

  glPopMatrix();
}

Eigen::Matrix3d getCameraCalibrationMatrixKITTI(const std::string& file_name_calibration_, Eigen::Vector3d& baseline_pixels_) {

  //ds load camera matrix - for now only KITTI parsing
  std::ifstream file_calibration(file_name_calibration_, std::ifstream::in);
  std::string line_buffer("");
  std::getline(file_calibration, line_buffer);
  if (line_buffer.empty()) {
    throw std::runtime_error("invalid camera calibration file provided");
  }
  std::istringstream stream_left(line_buffer);
  Eigen::Matrix3d camera_calibration_matrix(Eigen::Matrix3d::Identity());
  baseline_pixels_.setZero();

  //ds parse in fixed order
  std::string filler(""); stream_left >> filler;
  stream_left >> camera_calibration_matrix(0,0);
  stream_left >> filler;
  stream_left >> camera_calibration_matrix(0,2);
  stream_left >> filler; stream_left >> filler;
  stream_left >> camera_calibration_matrix(1,1);
  stream_left >> camera_calibration_matrix(1,2);

  //ds read second projection matrix to obtain the horizontal offset
  std::getline(file_calibration, line_buffer);
  std::istringstream stream_right(line_buffer);
  stream_right >> filler; stream_right >> filler; stream_right >> filler; stream_right >> filler;
  stream_right >> baseline_pixels_(0);
  file_calibration.close();
  std::cerr << "loaded camera calibration matrix: \n" << camera_calibration_matrix << std::endl;
  std::cerr << "with baseline (pixels): " << baseline_pixels_.transpose() << std::endl;
  return camera_calibration_matrix;
}

Eigen::Isometry3d v2t(const Eigen::Matrix<double, 6, 1>& t) {
  Eigen::Isometry3d T;
  T.setIdentity();
  T.translation()=t.head<3>();
  float w=t.block<3,1>(3,0).squaredNorm();
  if (w<1) {
    w=sqrt(1-w);
    T.linear()=Eigen::Quaterniond(w, t(3), t(4), t(5)).toRotationMatrix();
  } else {
    Eigen::Vector3d q=t.block<3,1>(3,0);
    q.normalize();
    T.linear()=Eigen::Quaterniond(0, q(0), q(1), q(2)).toRotationMatrix();
  }
  return T;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& p){
  Eigen::Matrix3d s;
  s <<
    0,  -p.z(), p.y(),
    p.z(), 0,  -p.x(),
    -p.y(), p.x(), 0;
  return s;
}

std::vector<Framepoint*> getPointsFromStereo(const Eigen::Matrix3d& camera_calibration_matrix_,
                                             const Eigen::Vector3d& offset_to_camera_right_,
                                             const std::vector<cv::KeyPoint>& keypoints_left_,
                                             const std::vector<cv::KeyPoint>& keypoints_right_,
                                             const cv::Mat& descriptors_left_,
                                             const cv::Mat& descriptors_right_,
                                             const uint32_t& maximum_descriptor_distance_) {
  std::vector<Framepoint*> points(descriptors_left_.rows);
  uint32_t number_of_triangulated_points = 0;

  //ds configuration
  const double b_u = -offset_to_camera_right_(0);
  const double f_u = camera_calibration_matrix_(0, 0);
  const double c_u = camera_calibration_matrix_(0, 2);
  const double c_v = camera_calibration_matrix_(1, 2);

  //ds perform stereo matching using a temporary HBST - the left descriptors are the reference
  Tree tree_left(descriptors_left_);
  MatchVector stereo_matches;
  tree_left.match(descriptors_right_, stereo_matches, maximum_descriptor_distance_);

  //ds for each stereo match
  for (const Match& match: stereo_matches) {
    const uint32_t& index_left  = match.identifier_reference;
    const uint32_t& index_right = match.identifier_query;

    const cv::Point2d& point_in_image_left  = keypoints_left_[index_left].pt;
    const cv::Point2d& point_in_image_right = keypoints_right_[index_right].pt;
    const double disparity_pixels = point_in_image_left.x-point_in_image_right.x;

    //ds if disparity is sufficient
    if (disparity_pixels > 0) {

      //ds compute depth from disparity
      const float depth_meters = b_u/disparity_pixels;

      //ds compute full point
      const float depth_meters_per_pixel = depth_meters/f_u;
      const Eigen::Vector3d point_in_camera(depth_meters_per_pixel*(point_in_image_left.x-c_u),
                                            depth_meters_per_pixel*(point_in_image_left.y-c_v),
                                            depth_meters);

      //ds add the point
      points[number_of_triangulated_points] = new Framepoint(keypoints_left_[index_left],
                                                             keypoints_right_[index_right],
                                                             descriptors_left_.row(index_left),
                                                             descriptors_right_.row(index_right),
                                                             point_in_camera);
      ++number_of_triangulated_points;
    }
  }
  points.resize(number_of_triangulated_points);
  return points;
}
