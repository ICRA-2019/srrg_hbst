#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

//ds readability
typedef srrg_hbst::BinaryTree256 Tree;
typedef srrg_hbst::BinaryTree256::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree256::Match Match;
typedef srrg_hbst::BinaryTree256::MatchVectorMap MatchVectorMap;

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 2) {
    std::cerr << "ERROR: invalid call - please use: ./tracker /path/to/srrg_hbst/examples/test_images" << std::endl;
    return 0;
  }

  //ds feature handling
  cv::Ptr<cv::FastFeatureDetector> keypoint_detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
#if CV_MAJOR_VERSION == 2
  keypoint_detector    = new cv::FastFeatureDetector();
  descriptor_extractor = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
  keypoint_detector    = cv::FastFeatureDetector::create();
  descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#endif

  //ds keypoint buffer (to keep keypoint information for multiple images)
  std::vector<std::vector<const cv::KeyPoint*>> keypoints_per_image(0);

  //ds configuration
  const uint32_t maximum_matching_distance = 25;

  //ds get test image path
  const std::string test_images_folder = argv_[1];

  //ds number of test images
  const uint32_t number_of_images = 10;

  //ds track colorization for each of the 10 images (BGR)
  std::vector<cv::Scalar> color_per_image
  = {cv::Scalar(0, 255, 0),
     cv::Scalar(0, 0, 255),
     cv::Scalar(255, 255, 0),
     cv::Scalar(0, 255, 255),
     cv::Scalar(255, 0, 255),
     cv::Scalar(255, 255, 255),
     cv::Scalar(150, 150, 50),
     cv::Scalar(150, 50, 150),
     cv::Scalar(50, 150, 150)
  };

  //ds allocate an empty tree
  std::shared_ptr<Tree> hbst_tree = std::make_shared<Tree>();

  //ds process images
  std::printf("------------[ press any key to step ]------------\n");
  for (uint32_t image_number = 0; image_number < number_of_images; ++image_number) {

    //ds build full image file name string
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", image_number);
    const std::string file_name_image = test_images_folder + "/" + buffer;

    //ds load image (project root folder)
    const cv::Mat image = cv::imread(file_name_image, CV_LOAD_IMAGE_GRAYSCALE);

    //ds sanity check
    if (image.rows == 0 || image.cols == 0) {
      std::cerr << "ERROR: invalid test image path provided: " << test_images_folder << std::endl;
      return 0;
    }

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image, keypoints, descriptors);
    std::cerr << "loaded image: " << file_name_image << " with keypoints/descriptors: " << descriptors.rows << std::endl;

    //ds allocate keypoints manually in memory (we will directly link to them with our HBST matchables, avoiding any index bookkeeping)
    std::vector<const cv::KeyPoint*> dynamic_keypoints(0);
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::KeyPoint* dynamic_keypoint = new cv::KeyPoint(keypoint);
      dynamic_keypoints.push_back(dynamic_keypoint);
    }
    keypoints_per_image.push_back(dynamic_keypoints);

    //ds obtain linked matchables
    const MatchableVector matchables(Tree::getMatchablesWithPointer<const cv::KeyPoint*>(descriptors, dynamic_keypoints, image_number));

    //ds obtain matches against all inserted matchables (i.e. images so far)
    MatchVectorMap matches_per_image;
    hbst_tree->matchAndAdd(matchables, matches_per_image, maximum_matching_distance);

    //ds info
    cv::Mat image_display(image);
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

    //ds draw currently detected points
    for (const cv::KeyPoint* keypoint: dynamic_keypoints) {
      cv::circle(image_display, keypoint->pt, 2, cv::Scalar(255, 0, 0), -1);
    }

    //ds for each match vector (i.e. matching results for each past image)
    std::cerr << "-------------------------------------------------" << std::endl;
    for (uint32_t image_number_reference = 0; image_number_reference < image_number; ++image_number_reference) {
      std::cerr << "matches from image [" << image_number << "] to image [" << image_number_reference << "]: "
                <<  matches_per_image[image_number_reference].size() << std::endl;

      //ds draw tracks on image
      for (const Match& match: matches_per_image[image_number_reference]) {
        const cv::KeyPoint* point_query     = static_cast<const cv::KeyPoint*>(match.pointer_query);
        const cv::KeyPoint* point_reference = static_cast<const cv::KeyPoint*>(match.pointer_reference);

        //ds on the fly cutoff kernel
        if (cv::norm(cv::Mat(point_query->pt), cv::Mat(point_reference->pt)) < 75) {
          cv::line(image_display, point_query->pt, point_reference->pt, color_per_image[image_number_reference]);
        }
      }
    }
    std::cerr << "-------------------------------------------------" << std::endl;
    cv::imshow("current frame", image_display);
    cv::waitKey(0);
  }

  //ds free linked structures to matchables (the matchables themselves get freed by the tree)
  for (const std::vector<const cv::KeyPoint*>& keypoints: keypoints_per_image) {
    for (const cv::KeyPoint* keypoint: keypoints) {
      delete keypoint;
    }
  }
  return 0;
}
