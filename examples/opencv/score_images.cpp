#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"

//ds HBST setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<uint64_t, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 2) {
    std::cerr << "invalid call - please use: ./score_images /path/to/srrg_hbst/examples/test_images" << std::endl;
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
  const uint32_t number_of_images          = 10;
  uint64_t number_of_stored_descriptors    = 0;

  //ds our HBST database
  std::printf("allocating empty tree\n");
  Tree database;

  //ds for each image
  std::printf("------------[ press any key to step ]------------\n");
  for (uint32_t index_image = 0; index_image < number_of_images; ++index_image) {

    //ds compute image file name and load it from disk
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", index_image);
    const cv::Mat image = cv::imread(test_images_folder + "/" + buffer, CV_LOAD_IMAGE_GRAYSCALE);

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
    std::printf("matching ratios for image [%02u] with %lu images in database\n", index_image, database.size());
    Tree::ScoreVector image_scores(database.getScorePerImage(matchables, true, maximum_matching_distance));

    //ds print sorted scores
    std::cerr << "-------------------------------------------------" << std::endl;
    for (const Tree::Score& score: image_scores) {
      std::printf(" > matching score for for QUERY [%02u] to REFERENCE [%02lu]: %5.3f (total matches: %4lu)\n",
                  index_image, score.identifier_reference, score.matching_ratio, score.number_of_matches);
    }
    std::cerr << "-------------------------------------------------" << std::endl;

    //ds show the matching in an image pair
    cv::Mat image_display;
    cv::cvtColor(image, image_display, CV_GRAY2RGB);

    //ds draw query point in lower image
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(0, 0, 255));
    }
    cv::imshow("detected keypoints with descriptors", image_display);
    cv::waitKey(0);

    //ds add descriptors to the tree
    std::printf("adding image [%02u] to database (descriptors: %5d) | total descriptors stored: %lu\n",
                index_image, descriptors.rows, number_of_stored_descriptors);
    database.add(matchables, srrg_hbst::SplittingStrategy::SplitEven);
    number_of_stored_descriptors += matchables.size();
    std::cerr << "-------------------------------------------------" << std::endl;
  }
  return 0;
}
