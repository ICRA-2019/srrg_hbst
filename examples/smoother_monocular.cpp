#include <iostream>
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
typedef srrg_hbst::BinaryTree256::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree256::Match Match;
typedef srrg_hbst::BinaryTree256::MatchVector MatchVector;
typedef srrg_hbst::BinaryTree256::MatchVectorMap MatchVectorMap;

//ds aligned eigen types
typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > IsometryVector;



//ds our 'map' object
struct Framepoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Framepoint(const cv::KeyPoint& keypoint_,
             const cv::Mat& descriptor_): keypoint(keypoint_),
                                          descriptor(descriptor_) {}
  Framepoint() = delete;

  //ds measured
  cv::KeyPoint keypoint;
  cv::Mat descriptor;

  //ds computed
  Framepoint* previous                  = 0;
  uint32_t track_length                 = 0;
  Eigen::Vector3d coordinates_in_camera = Eigen::Vector3d::Zero();
};

//ds simple trajectory viewer
class Viewer: public QGLViewer {
public:
  Viewer(QWidget* parent = 0): QGLViewer(parent) {setWindowTitle("trajectory viewer [OpenGL]");}
  void init();
  virtual void draw();
  void addPose(const Eigen::Isometry3d& pose_) {_poses.push_back(pose_);}

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
  StandardCamera* _camera = 0;
};



//ds helpers
Eigen::Matrix3d getCameraCalibrationMatrixKITTI(const std::string& file_name_calibration_);
Eigen::Isometry3d v2t(const Eigen::Matrix<double, 6, 1>& t);
Eigen::Matrix3d skew(const Eigen::Vector3d& p);
void estimatePointsInCamerasFromMotion(std::vector<Framepoint*>& current_points_,
                                       const Eigen::Isometry3d& motion_previous_to_current_,
                                       const Eigen::Matrix3d& camera_calibration_matrix_);
Eigen::Isometry3d getMotionFromEssentialMatrix(const std::vector<Framepoint*>& current_points_,
                                               const Eigen::Matrix3d& camera_calibration_matrix_);
std::pair<Eigen::Vector3d, Eigen::Vector3d> getPointsInCameras(const cv::Point2d image_point_previous_,
                                                               const cv::Point2d image_point_current_,
                                                               const Eigen::Isometry3d& camera_previous_to_current_,
                                                               const Eigen::Matrix3d& camera_calibration_matrix_);



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 3) {
    std::cerr << "ERROR: invalid call - please use: ./smoother file_name_initial_image calib.txt" << std::endl;
    return 0;
  }

  //ds configuration - TODO input validation
  const std::string file_name_initial_image = argv_[1];
  const std::string file_name_calibration   = argv_[2];
  std::cerr << "initial image: " << file_name_initial_image << std::endl;
  std::cerr << "calibration: " << file_name_calibration << std::endl;

  //ds load camera matrix
  const Eigen::Matrix3d camera_calibration_matrix(getCameraCalibrationMatrixKITTI(file_name_calibration));

  //ds parse image extension
  const std::size_t index_delimiter_extension = file_name_initial_image.find_last_of('.');
  const std::string extension                 = file_name_initial_image.substr(index_delimiter_extension, file_name_initial_image.length()-index_delimiter_extension);
  std::cerr << "using image extension: '" << extension << "'" << std::endl;

  //ds parse folder - UNIX only
  const std::size_t index_delimiter_directory = file_name_initial_image.find_last_of('/');
  const std::string directory_name_images     = file_name_initial_image.substr(0, index_delimiter_directory+1);
  std::cerr << "reading images from directory: '" << directory_name_images << "'" << std::endl;

  //ds parse enumeration characters range (as file name of images)
  const uint32_t number_of_enumeration_characters = index_delimiter_extension-index_delimiter_directory-1;
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
  keypoint_detector    = new cv::FastFeatureDetector(50);
  descriptor_extractor = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
  keypoint_detector    = cv::FastFeatureDetector::create(50);
  descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#endif

  //ds 'map'
  std::vector<std::vector<Framepoint*>> framepoints_per_image(0);

  //ds estimated trajectory
  IsometryVector estimated_poses(1, Eigen::Isometry3d::Identity());

  //ds initialize viewer
  QApplication ui_server(argc_, argv_);
  Viewer viewer;
  viewer.show();
  viewer.updateGL();

  //ds transform robot to camera (for visualization purposes only)
  Eigen::Isometry3d robot_to_camera(Eigen::Isometry3d::Identity());
  robot_to_camera.linear() << 0,  0,  1,
                             -1,  0,  0,
                              0, -1,  0;

  //ds constant velocity motion model bookkeeping
  Eigen::Isometry3d motion_previous(Eigen::Isometry3d::Identity());

  //ds allocate an empty tree
  Tree hbst_tree;

  //ds start processing images (forced grayscale)
  uint32_t number_of_processed_images = 0;
  std::string file_name_image_current = file_name_initial_image;
  cv::Mat image_current               = cv::imread(file_name_image_current, CV_LOAD_IMAGE_GRAYSCALE);
  while (image_current.rows != 0 && image_current.cols != 0) {

// FEATURE EXTRACTION -----------------------------------------------------------------------------------------------------------------------------------------------------
    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image_current, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image_current, keypoints, descriptors);

    //ds allocate framepoints for current detections
    std::vector<Framepoint*> framepoints(keypoints.size());
    for (size_t u = 0; u < keypoints.size(); ++u) {
      framepoints[u] = new Framepoint(keypoints[u], descriptors.row(u));
    }
    framepoints_per_image.push_back(framepoints);
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// HBST FEATURE MATCHING ----------------------------------------------------------------------------------------------------------------------------------------------------
    //ds obtain linked matchables
    const MatchableVector matchables(Tree::getMatchablesWithPointer<Framepoint*>(descriptors, framepoints, number_of_processed_images));

    //ds obtain matches against all inserted matchables (i.e. images so far)
    MatchVectorMap matches_per_image;
    hbst_tree.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //ds info
    cv::Mat image_display(image_current);
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

    //ds if we obtained matches
    uint32_t number_of_measurements = 0;
    if (matches_per_image.size() > 0) {

// MATCH PRUNING AND BOOKKEEPING --------------------------------------------------------------------------------------------------------------------------------------------
      //ds grab matches from previous image only (currently ignoring matches to images which lay further in the past)
      const MatchVector& matches(matches_per_image[number_of_processed_images-1]);
      std::vector<Framepoint*> current_points(matches.size());

      //ds ignorantly filter outlier matches in a brutal loop
      double average_track_length  = 0;
      std::set<Framepoint*> linked_previous;
      for (uint32_t u = 0; u < current_points.size(); ++u) {

        //ds mother of all EVIL casts in 'c++' TODO maybe adjust library
        Framepoint* current_point  = const_cast<Framepoint*>(static_cast<const Framepoint*>(matches[u].pointer_query));
        Framepoint* previous_point = const_cast<Framepoint*>(static_cast<const Framepoint*>(matches[u].pointer_reference));

        //ds cross-check if we not already matched against this previous point TODO enable cross-check in HBST
        if (linked_previous.count(previous_point)) {

          //ds rudely ignore the track
          continue;
        }

        //ds check maximum tracking distance
        if (cv::norm(cv::Mat(current_point->keypoint.pt), cv::Mat(previous_point->keypoint.pt)) < maximum_keypoint_distance) {

          //ds establish track between framepoints
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

      std::cerr << "read image: " << file_name_image_current
                << " with keypoints/descriptors: " << descriptors.rows
                << " inlier tracks: " << number_of_measurements
                << " average track length: " << average_track_length << std::endl;

// STRUCTURE FROM MOTION GUESS ------------------------------------------------------------------------------------------------------------------------------------------------
      //ds state - transform from previous to current image
      Eigen::Isometry3d previous_to_current = motion_previous;

      //ds if we have a sufficient number of tracks
      if (number_of_measurements > 100) {

        //ds extract motion guess from essential matrix
        previous_to_current = getMotionFromEssentialMatrix(current_points, camera_calibration_matrix);
      }
      std::cerr << "initial motion guess: \n" << previous_to_current.matrix() << std::endl;

      //ds estimate point positions, removing invalid ones
      estimatePointsInCamerasFromMotion(current_points, previous_to_current, camera_calibration_matrix);
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// IMAGE REGISTRATION (INITIAL POSE GUESS OPTIMIZATION) -----------------------------------------------------------------------------------------
      //ds always use the constant velocity motion model for pose motion guess
      previous_to_current = motion_previous;

      //ds initialize LS
      const uint32_t maximum_number_of_iterations = 100;
      double total_error_squared_previous         = 0;
      const double maximum_error_squared          = 10*10;
      Eigen::Matrix<double, 6, 6> H(Eigen::Matrix<double, 6, 6>::Zero());
      Eigen::Matrix<double, 6, 1> b(Eigen::Matrix<double, 6, 1>::Zero());
      Eigen::Matrix2d omega(Eigen::Matrix2d::Identity());
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
          const cv::KeyPoint& keypoint_in_current_image = current_point->keypoint;

          //ds measurements
          const Eigen::Vector2d image_coordinates_fixed(keypoint_in_current_image.pt.x, keypoint_in_current_image.pt.y);

          //ds transform point from previous frame into current
          const Eigen::Vector3d point_in_current_camera(previous_to_current*previous_point->coordinates_in_camera);

          //ds check for invalid depth
          if (point_in_current_camera.z() <= 0) {
            continue;
          }

          //ds project past point in current frame
          const Eigen::Vector3d homogeneous_coordinates = camera_calibration_matrix*point_in_current_camera;
          const Eigen::Vector2d image_coordinates_moving(homogeneous_coordinates.x()/homogeneous_coordinates.z(),
                                                         homogeneous_coordinates.y()/homogeneous_coordinates.z());

          //ds check if out of range
          if (image_coordinates_moving.x() < 0
            || image_coordinates_moving.y() < 0
            || image_coordinates_moving.x() > image_current.cols
            || image_coordinates_moving.y() > image_current.rows) {
            continue;
          }

          //ds compute error
          const Eigen::Vector2d error(image_coordinates_moving-image_coordinates_fixed);
          const double error_squared = error.transpose()*omega*error;

          //ds kernelize
          if (error_squared > maximum_error_squared) {
            omega *= maximum_error_squared/error_squared;
          }

          //ds update error
          total_error_squared += error_squared;

          //ds precompute
          const double inverse_sampled_c         = 1/homogeneous_coordinates.z();
          const double inverse_sampled_c_squared = inverse_sampled_c*inverse_sampled_c;

          //ds jacobians
          Eigen::Matrix<double, 2, 6> jacobian(Eigen::Matrix<double, 2, 6>::Zero());

          //ds jacobian of the transform
          Eigen::Matrix<double, 3, 6> jacobian_transform;
          jacobian_transform.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
          jacobian_transform.block<3,3>(0,3) = -2*skew(point_in_current_camera);

          //ds jacobian parts of the homogeneous division of the projection in the current image
          Eigen::Matrix<double, 2, 3> jacobian_homogeneous;
          jacobian_homogeneous << inverse_sampled_c, 0, -homogeneous_coordinates.x()*inverse_sampled_c_squared,
                                  0, inverse_sampled_c, -homogeneous_coordinates.y()*inverse_sampled_c_squared;

          //ds assemble full jacobian - pose part
          jacobian = jacobian_homogeneous*camera_calibration_matrix*jacobian_transform;

          //ds precompute
          const Eigen::Matrix<double, 6, 2> jacobian_transposed = jacobian.transpose();

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

      //ds update framepoints of current camera for the next image registration
      for (Framepoint* current_point: current_points) {
        current_point->coordinates_in_camera = previous_to_current*current_point->previous->coordinates_in_camera;
      }

      //ds update motion model
      motion_previous = previous_to_current;

      //ds update trajectory
      estimated_poses.push_back(estimated_poses[number_of_processed_images-1]*previous_to_current.inverse());
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

      //ds draw tracks on image
      for (const Framepoint* current_point: current_points) {
        cv::line(image_display, current_point->keypoint.pt, current_point->previous->keypoint.pt, cv::Scalar(0, 255, 0));
      }
    } else {
      std::cerr << "read image: " << file_name_image_current
                << " with keypoints/descriptors: " << descriptors.rows
                << " initial image " << std::endl;
    }

    //ds draw currently detected points
    for (Framepoint* framepoint: framepoints) {
      cv::circle(image_display, framepoint->keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    cv::imshow("current image", image_display);
    cv::waitKey(1);

    //ds update viewer
    viewer.addPose(robot_to_camera*estimated_poses[number_of_processed_images]);
    viewer.updateGL();
    ui_server.processEvents();

    //ds compute file name for next image
    ++number_of_processed_images;
    std::string file_name_tail             = std::to_string(number_of_processed_images);
    const uint32_t number_of_padding_zeros = number_of_enumeration_characters-file_name_tail.length();
    for (uint32_t u = 0; u < number_of_padding_zeros; ++u) {
      file_name_tail = "0"+file_name_tail;
    }
    file_name_image_current = directory_name_images+file_name_tail+extension;

    //ds read next image
    image_current = cv::imread(file_name_image_current, CV_LOAD_IMAGE_GRAYSCALE);
  }
  std::cerr << "stopped at image: " << file_name_image_current << " - not found" << std::endl;

  //ds free remaining structures
  for (const std::vector<Framepoint*>& framepoints: framepoints_per_image) {
    for (const Framepoint* framepoint: framepoints) {
      delete framepoint;
    }
  }
  return 0;
}

void Viewer::init() {
  QGLViewer::init();

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
  glColor3f(0.0, 0.0, 0.0);
  drawAxis(10);

  glBegin(GL_LINES);
  glColor3f(0, 0, 1);
  Eigen::Isometry3d previous_pose(Eigen::Isometry3d::Identity());
  for (const Eigen::Isometry3d& pose: _poses) {
    glVertex3f(previous_pose.translation().x(), previous_pose.translation().y(), previous_pose.translation().z());
    glVertex3f(pose.translation().x(), pose.translation().y(), pose.translation().z());
    previous_pose = pose;
  }
  glEnd();
  glPopMatrix();
}

Eigen::Matrix3d getCameraCalibrationMatrixKITTI(const std::string& file_name_calibration_) {

  //ds load camera matrix - for now only KITTI parsing
  std::ifstream file_calibration(file_name_calibration_, std::ifstream::in);
  std::string line_buffer("");
  std::getline(file_calibration, line_buffer);
  if (line_buffer.empty()) {
    throw std::runtime_error("invalid camera calibration file provided");
  }
  std::istringstream stream(line_buffer);
  Eigen::Matrix3d camera_calibration_matrix(Eigen::Matrix3d::Identity());

  //ds parse in fixed order TODO microfunction
  std::string filler(""); stream >> filler;
  stream >> camera_calibration_matrix(0,0);
  stream >> filler;
  stream >> camera_calibration_matrix(0,2);
  stream >> filler; stream >> filler;
  stream >> camera_calibration_matrix(1,1);
  stream >> camera_calibration_matrix(1,2);
  file_calibration.close();
  stream.clear();
  std::cerr << "loaded camera calibration matrix: \n" << camera_calibration_matrix << std::endl;
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

void estimatePointsInCamerasFromMotion(std::vector<Framepoint*>& current_points_,
                             const Eigen::Isometry3d& motion_previous_to_current_,
                             const Eigen::Matrix3d& camera_calibration_matrix_) {

  //ds estimate point positions, removing invalid ones
  uint32_t number_of_measurements   = 0;
  uint32_t number_of_removed_points = 0;
  uint32_t number_of_new_points     = 0;
  for (uint32_t u = 0; u < current_points_.size(); ++u) {
    Framepoint* point_current  = current_points_[u];
    Framepoint* point_previous = point_current->previous;

    //ds obtain point triangulation candidates
    const std::pair<Eigen::Vector3d, Eigen::Vector3d> candidates = getPointsInCameras(point_previous->keypoint.pt,
                                                                                      point_current->keypoint.pt,
                                                                                      motion_previous_to_current_,
                                                                                      camera_calibration_matrix_);

    //ds use midpoint as initial guess
    const Eigen::Vector3d midpoint_in_camera_previous = (candidates.first+motion_previous_to_current_.inverse()*candidates.second)/2;

    //ds drop invalid guesses
    if (midpoint_in_camera_previous.z() <= 0) {
      ++number_of_removed_points;
      continue;
    }

    //ds keep point
    current_points_[number_of_measurements] = point_current;

    //ds if there's no previous estimation
    if (!point_previous->previous) {

      //ds set initial guess
      point_previous->coordinates_in_camera = midpoint_in_camera_previous;
      ++number_of_new_points;
    } else {

      //ds average previous guesses including the new one
      point_previous->coordinates_in_camera = (point_current->track_length*point_previous->coordinates_in_camera+midpoint_in_camera_previous)/(1+point_current->track_length);
    }
    ++number_of_measurements;
  }
  current_points_.resize(number_of_measurements);
  std::cerr << "number of new points: " << number_of_new_points << "/" << number_of_measurements << std::endl;
  std::cerr << "number of removed points: " << number_of_removed_points << "/" << number_of_measurements << std::endl;
}

Eigen::Isometry3d getMotionFromEssentialMatrix(const std::vector<Framepoint*>& current_points_,
                                               const Eigen::Matrix3d& camera_calibration_matrix_) {
  std::vector<cv::Point2d> image_points_current(current_points_.size());
  std::vector<cv::Point2d> image_points_previous(current_points_.size());
  for (uint32_t u = 0; u < current_points_.size(); ++u) {
    image_points_current[u]  = current_points_[u]->keypoint.pt;
    image_points_previous[u] = current_points_[u]->previous->keypoint.pt;
  }

  //ds input validation
  if (camera_calibration_matrix_(0,0) != camera_calibration_matrix_(1,1)) {
    throw std::runtime_error("invalid camera matrix, symmetric focal length required");
  }
  if (image_points_previous.size() != image_points_current.size()) {
    throw std::runtime_error("mismatching number of image point correspondences");
  }

  const double& focal_length = camera_calibration_matrix_(0,0);
  const cv::Point2d principal_point(camera_calibration_matrix_(0,2),
                                    camera_calibration_matrix_(1,2));

  //ds retrieve essential matrix for previous and current keypoint associations
  const cv::Mat essential_matrix = cv::findEssentialMat(image_points_previous,
                                                        image_points_current,
                                                        focal_length,
                                                        principal_point);

  //ds extract pose from essential matrix
  cv::Mat rotation_previous_to_current;
  cv::Mat translation_previous_to_current;
  const uint32_t number_of_essential_inliers = cv::recoverPose(essential_matrix,
                                                               image_points_previous,
                                                               image_points_current,
                                                               rotation_previous_to_current,
                                                               translation_previous_to_current,
                                                               focal_length,
                                                               principal_point);

  std::cerr << "motion estimation inliers: " << number_of_essential_inliers << "/" << image_points_previous.size() << std::endl;

  //ds convert to eigen space
  Eigen::Isometry3d previous_to_current(Eigen::Isometry3d::Identity());
  for (uint32_t r = 0; r < 3; ++r) {
    for (uint32_t c = 0; c < 3; ++c) {
      previous_to_current.linear()(r,c) = rotation_previous_to_current.at<double>(r,c);
    }
  }
  previous_to_current.translation().x() = translation_previous_to_current.at<double>(0);
  previous_to_current.translation().y() = translation_previous_to_current.at<double>(1);
  previous_to_current.translation().z() = translation_previous_to_current.at<double>(2);
  return previous_to_current;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> getPointsInCameras(const cv::Point2d image_point_previous_,
                                                               const cv::Point2d image_point_current_,
                                                               const Eigen::Isometry3d& camera_previous_to_current_,
                                                               const Eigen::Matrix3d& camera_calibration_matrix_) {

  //ds readability
  const Eigen::Matrix<double, 1, 3>& r_1 = camera_previous_to_current_.linear().block<1,3>(0,0);
  const Eigen::Matrix<double, 1, 3>& r_2 = camera_previous_to_current_.linear().block<1,3>(1,0);
  const Eigen::Matrix<double, 1, 3>& r_3 = camera_previous_to_current_.linear().block<1,3>(2,0);

  //ds precomputations
  const double a_0 = (image_point_previous_.x-camera_calibration_matrix_(0,2))/camera_calibration_matrix_(0,0);
  const double b_0 = (image_point_previous_.y-camera_calibration_matrix_(1,2))/camera_calibration_matrix_(1,1);
  const double a_1 = (image_point_current_.x-camera_calibration_matrix_(0,2))/camera_calibration_matrix_(0,0);
  const double b_1 = (image_point_current_.y-camera_calibration_matrix_(1,2))/camera_calibration_matrix_(1,1);
  const Eigen::Vector3d x_0(a_0, b_0, 1);
  const Eigen::Vector3d x_1(a_1, b_1, 1);

  //ds build constraint matrix
  Eigen::Matrix<double, 3, 2> A(Eigen::Matrix<double, 3, 2>::Zero());
  A << -r_1*x_0, a_1,
       -r_2*x_0, b_1,
       -r_3*x_0, 1;

  //ds minimize squared error for both image points
  const Eigen::Vector2d z = A.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(camera_previous_to_current_.translation());

  //ds return points
  return std::make_pair(x_0*z(0), x_1*z(1));
}
