#include <iostream>
#include <thread>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif



//ds readability
typedef srrg_hbst::BinaryTree256 Tree;

//ds nasty global buffers
cv::Ptr<cv::FeatureDetector> keypoint_detector;
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
Tree tree;

//ds configuration
uint32_t number_of_processed_images   = 0;
uint64_t number_of_stored_descriptors = 0;
double image_display_scale            = 1;
double maximum_descriptor_distance    = 25;
uint32_t number_of_images_interspace  = 100;



void callbackImage(const sensor_msgs::ImageConstPtr& image_);

int32_t main(int32_t argc_, char** argv_) {
  std::cerr << "--------------------------------------------------------------------------------" << std::endl;
  std::cerr << "CONTROLS: press [0] / [+] to shrink / grow the displayed image" << std::endl;
  std::cerr << "          press [ESC] to terminate processing" << std::endl;
  std::cerr << "          press [C] for clean the current HBST database" << std::endl;
  std::cerr << "--------------------------------------------------------------------------------" << std::endl;

  //ds validate input
  if (argc_ != 5) {
    std::cerr << "ERROR: invalid call - please use: rosrun recognizer_node -camera <ROS/camera/topic> -space <integer>" << std::endl;
    return 0;
  }

  //ds parse image folder/video file
  const std::string topic_camera = argv_[2];
  std::cerr << "camera topic: '" << topic_camera << "'" << std::endl;

  //ds matching interspace
  number_of_images_interspace = std::stoi(argv_[4]);
  std::cerr << "image interspace: " << number_of_images_interspace << std::endl;

  //ds matching threshold
  maximum_descriptor_distance = 75;
  std::cerr << "maximum descriptor distance: " << maximum_descriptor_distance << std::endl;

#ifdef SRRG_MERGE_DESCRIPTORS
  //ds configure the tree
  Tree::maximum_distance_for_merge = 0;
#endif

  //ds feature handling
#if CV_MAJOR_VERSION == 2
  keypoint_detector    = new cv::ORB(1000);
  descriptor_extractor = new cv::ORB();
#elif CV_MAJOR_VERSION == 3
  keypoint_detector    = cv::ORB::create(1000);
  descriptor_extractor = cv::ORB::create();
#endif

  //ds initialize roscpp
  ros::init(argc_, argv_, "recognizer_node");

  //ds start node
  ros::NodeHandle node;

  //ds subscribe to camera info topics
  ros::Subscriber subscriber_camera_info_left  = node.subscribe(topic_camera, 1, callbackImage);

  //ds start processing loop
  std::cerr << "starting to spin" << std::endl;
  while (ros::ok()) {

    //ds trigger callbacks
    ros::spinOnce();

    //ds breathe
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::cerr << "terminating" << std::endl;
  return 0;
}

void callbackImage(const sensor_msgs::ImageConstPtr& image_) {
  try {

    //ds obtain cv image pointer for the left image and set it (intensity only)
    cv_bridge::CvImagePtr image_pointer = cv_bridge::toCvCopy(image_, sensor_msgs::image_encodings::MONO8);
    cv::Mat image = image_pointer->image;

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image, keypoints, descriptors);

    //ds allocate keypoints manually in memory (we will directly link to them with our HBST matchables, avoiding any auxiliary bookkeeping)
    std::vector<const cv::KeyPoint*> keypoints_addresses(keypoints.size());
    for (uint32_t u = 0; u < keypoints.size(); ++u) {
      keypoints_addresses[u] = &keypoints[u];
    }

    //ds obtain linked matchables
    const Tree::MatchableVector matchables(Tree::getMatchablesWithPointer<const cv::KeyPoint*>(descriptors, keypoints_addresses, number_of_processed_images));

    //ds obtain matches against all inserted matchables (i.e. images so far) and integrate them simultaneously
    Tree::MatchVectorMap matches_per_image;
    std::chrono::time_point<std::chrono::system_clock> time_begin(std::chrono::system_clock::now());
    tree.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
    const double processing_duration_seconds = std::chrono::duration<double>(std::chrono::system_clock::now()-time_begin).count();
    number_of_stored_descriptors += descriptors.rows;
    ++number_of_processed_images;

    //ds info display
    cv::Mat image_display(image);
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

    //ds draw current keypoints in blue
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
    }

    //ds for each match vector (i.e. matching results for each past image) of ALL past images
    if (number_of_processed_images > number_of_images_interspace) {
      for (uint32_t image_number_reference = 0; image_number_reference < number_of_processed_images-number_of_images_interspace-1; ++image_number_reference) {

        //ds if we have sufficient matches
        const uint32_t number_of_matches = matches_per_image[image_number_reference].size();
        const double matching_ratio      = static_cast<double>(number_of_matches)/keypoints.size();
        if (matching_ratio > 0.1 && number_of_matches > 100) {

          //ds draw matched descriptors
          for (const Tree::Match& match: matches_per_image[image_number_reference]) {
            const cv::KeyPoint* point_query     = static_cast<const cv::KeyPoint*>(match.pointer_query);
            cv::circle(image_display, point_query->pt, 2, cv::Scalar(0, 255, 0), -1);
          }
        }
      }
    }

    //ds display image
    const cv::Size current_size(image_display_scale*image_display.cols, image_display_scale*image_display.rows);
    cv::resize(image_display, image_display, current_size);
    cv::imshow("callbackImage|current image", image_display);
    const int32_t key = cv::waitKey(1);

    //ds stats
    std::printf("callbackImage|processed images: %6u descriptors: %5d (total: %9lu) processing time(s): %4.3f\r",
                number_of_processed_images, descriptors.rows, number_of_stored_descriptors, processing_duration_seconds);
    std::fflush(stdout);

    //ds check if image shrinking [+] or growing [-] is desired TODO softcode
    if (key == 45) {
      image_display_scale /= 2;
    }
    if (key == 43) {
      image_display_scale *= 2;
    }

    //ds check for database clearing [C]
    if (key == 99) {
      std::cerr << "\ncallbackImage|clearing database" << std::endl;
      tree.clear(true);
      number_of_processed_images   = 0;
      number_of_stored_descriptors = 0;
    }

    //ds termination [ESC] TODO softcode
    if (key == 27) {
      std::cerr << "\ncallbackImage|termination requested" << std::endl;
      ros::shutdown();
    }
  }
  catch (const cv_bridge::Exception& exception_) {
    std::cerr << "\ncallbackImage|exception: " << exception_.what() << " (skipping image)" << std::endl;
  }
}
