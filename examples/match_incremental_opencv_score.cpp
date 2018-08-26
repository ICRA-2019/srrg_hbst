#include <iostream>
#include <chrono>
#include <ctime>
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

//ds current setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryNode<Matchable>::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree<Node> Tree;



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 2) {
    std::cerr << "invalid call - please use: ./match_incremental_opencv_score /path/to/srrg_hbst/examples/test_images" << std::endl;
    return 0;
  }

  //ds feature handling
#if CV_MAJOR_VERSION == 2
  cv::Ptr<cv::FastFeatureDetector> keypoint_detector    = new cv::FastFeatureDetector();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::FastFeatureDetector> keypoint_detector    = cv::FastFeatureDetector::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#endif

  //ds measurements
  std::chrono::time_point<std::chrono::system_clock> time_begin;
  std::chrono::duration<double> duration_construction(0);
  const uint32_t maximum_matching_distance = 25;

  //ds get test image path
  const std::string test_images_folder = argv_[1];

  //ds number of test images
  const uint32_t number_of_images = 10;

  //ds our HBST database
  Tree database;

  //ds for each image
  std::printf("------------[ press any key to step ]------------\n");
  for (uint32_t index_image = 0; index_image < number_of_images; ++index_image) {

    //ds compute image file name and load it from disk
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", index_image);
    const std::string file_test_image = test_images_folder + "/" + buffer;
    const cv::Mat image = cv::imread(file_test_image, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image, keypoints, descriptors);

    //ds show the matching in an image pair
    cv::Mat image_display;
    cv::cvtColor(image, image_display, CV_GRAY2RGB);

    //ds draw query point in lower image
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(0, 0, 255));
    }
    cv::imshow("detected keypoints with descriptors", image_display);
    cv::waitKey(0);

    //ds obtain matchables for descriptors and image
    const MatchableVector matchables(Tree::getMatchablesWithIndex(descriptors, index_image));

    //ds add descriptors to the tree
    std::printf("\nadding image [%02u] to database (descriptors: %5d) |", index_image, descriptors.rows);
    time_begin = std::chrono::system_clock::now();
    database.add(matchables, srrg_hbst::SplittingStrategy::SplitEven);
    std::chrono::duration<double> duration_construction = std::chrono::system_clock::now()-time_begin;
    std::printf(" duration (s): %6.4f\n", duration_construction.count());

    //ds query for matching ratio
    std::printf("matching ratios for image [%02u] |", index_image);
    time_begin = std::chrono::system_clock::now();
    Tree::ScoreVector image_scores(database.getScorePerImage(matchables, true, maximum_matching_distance));
    std::chrono::duration<double> duration_query = std::chrono::system_clock::now()-time_begin;

    //ds print sorted scores
    std::printf(" duration (s): %6.4f\n", duration_query.count());
    std::cerr << "------------------------------------------------" << std::endl;
    for (const Tree::Score& score: image_scores) {
      std::printf("matching score for for QUERY [%02u] to REFERENCE [%02lu]: %5.3f (total matches: %4lu)\n",
                  index_image, score.identifier_reference, score.matching_ratio, score.number_of_matches);
    }
    std::cerr << "------------------------------------------------" << std::endl;
  }
  return 0;
}
