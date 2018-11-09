#include <iostream>
#include <chrono>
#include <ctime>
#include "srrg_hbst_types/binary_tree.hpp"

//ds HBST setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<uint64_t, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 2) {
    std::cerr << "invalid call - please use: ./stress_test /path/to/srrg_hbst/examples/test_images" << std::endl;
    return 0;
  }

  //ds feature handling
#if CV_MAJOR_VERSION == 2
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = new cv::FastFeatureDetector();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = cv::FastFeatureDetector::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create();
#endif

  //ds configuration
  const uint32_t maximum_matching_distance = 25;
  const std::string test_images_folder     = argv_[1];
  const uint32_t number_of_images          = 1000000;
  uint64_t number_of_stored_descriptors    = 0;

  //ds load the test image - we will process always the same image
  const cv::Mat image = cv::imread(test_images_folder + "/image_00.pgm", CV_LOAD_IMAGE_GRAYSCALE);

  //ds our HBST database
  std::printf("allocating empty tree\n");
  Tree database;

  //ds for the number of samples
  for (uint64_t index_image = 0; index_image < number_of_images; ++index_image) {

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image, keypoints, descriptors);

    //ds obtain matchables for each descriptor with continuous indexing
    std::vector<uint64_t> indices(descriptors.rows, 0);
    std::for_each(indices.begin(), indices.end(), [](uint64_t &index){++index;});
    const Tree::MatchableVector matchables(Tree::getMatchables(descriptors, indices, index_image));

    //ds query for matching ratio
    std::chrono::time_point<std::chrono::system_clock> time_begin = std::chrono::system_clock::now();
    Tree::ScoreVector image_scores(database.getScorePerImage(matchables, true, maximum_matching_distance));
    const double duration_seconds_score = std::chrono::duration<double>(std::chrono::system_clock::now()-time_begin).count();

    //ds show extracted descriptors
    cv::Mat image_display;
    cv::cvtColor(image, image_display, CV_GRAY2RGB);
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(0, 0, 255));
    }
    cv::imshow("detected keypoints with descriptors (repeated)", image_display);
    cv::waitKey(1);

    //ds add descriptors to the tree
    time_begin = std::chrono::system_clock::now();
    database.add(matchables, srrg_hbst::SplittingStrategy::SplitEven);
    const double duration_seconds_add = std::chrono::duration<double>(std::chrono::system_clock::now()-time_begin).count();
    number_of_stored_descriptors += matchables.size();
    std::printf("scored and added image [%lu] (descriptors: %d) to database | total images stored: %lu | total descriptors stored: %lu"
                "| duration SCORE (s): %f | duration ADD (s): %f\r",
                index_image, descriptors.rows, database.size(), number_of_stored_descriptors,
                duration_seconds_score, duration_seconds_add);
  }
  return 0;
}
