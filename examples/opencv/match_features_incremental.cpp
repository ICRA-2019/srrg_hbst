#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"

//ds HBST setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<cv::KeyPoint, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;



int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ != 2) {
    std::cerr << "invalid call - please use: ./match_features_incremental /path/to/srrg_hbst/examples/test_images" << std::endl;
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

  //ds create HBST database
  std::printf("allocating empty tree\n");
  Tree database;

  //ds bookkeeping for visualization/stats
  std::vector<cv::Mat> images(number_of_images);
  std::vector<Tree::MatchableVector> matchables_per_image(number_of_images);

  //ds check each image against each other and itself (100% ratio)
  std::printf("starting matching and adding with maximum distance: %u \n", maximum_matching_distance);
  std::printf("------------[ press any key to step ]------------\n");
  for (uint32_t index_image_query = 0; index_image_query < number_of_images; ++index_image_query) {

    //ds load image from disk
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", index_image_query);
    const cv::Mat image_query = cv::imread(test_images_folder + "/" + buffer, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image_query, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image_query, keypoints, descriptors);
    std::cerr << "loaded image: " << index_image_query << " with keypoints/descriptors: " << descriptors.rows << std::endl;

    //ds for each keypoint - descriptor pair we allocate a matchable to put into the tree
    Tree::MatchableVector matchables_query(descriptors.rows);
    for (uint64_t u = 0; u < static_cast<uint64_t>(descriptors.rows); ++u) {
      matchables_query[u] = new Tree::Matchable(keypoints[u], descriptors.row(u), index_image_query);
    }

    //ds query HBST with current image and add the descriptors subsequently
    Tree::MatchVectorMap matches_per_reference_image;
    database.matchAndAdd(matchables_query, matches_per_reference_image, maximum_matching_distance);

    //ds check all match vectors in the map
    for (const Tree::MatchVectorMapElement& match_vector: matches_per_reference_image) {
      const uint64_t& index_image_reference     = match_vector.first;
      const Tree::MatchVector& matches = match_vector.second;
      const uint64_t number_of_matches          = matches.size();

      //ds compute matching ratio/score for this reference image
      const double score = static_cast<double>(number_of_matches)/matchables_per_image[index_image_reference].size();
      std::printf("matches for QUERY [%02u] to REFERENCE [%02lu]: %4lu (matching ratio: %5.3f)\n",
                  index_image_query, index_image_reference, number_of_matches, score);

      //ds show the matching in an image pair
      cv::Mat image_display;
      cv::vconcat(image_query, images[index_image_reference], image_display);
      cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

      //ds shift to lower image
      const cv::Point2f shift(0, image_query.rows);

      //ds draw correspondences - for each match
      for (const Tree::Match& match: matches) {

        //ds draw correspondence line between images
        cv::line(image_display, match.object_query.pt, match.object_reference.pt+shift, cv::Scalar(0, 255, 0));

        //ds draw query point in upper image
        cv::circle(image_display, match.object_query.pt, 2, cv::Scalar(255, 0, 0));

        //ds draw reference point in lower image
        cv::circle(image_display, match.object_reference.pt+shift, 2, cv::Scalar(0, 0, 255));
      }
      cv::imshow("matching (top: QUERY, bot: REFERENCE)", image_display);
      cv::waitKey(0);
    }

    //ds bookkeep image and matchables for display
    images[index_image_query]               = image_query;
    matchables_per_image[index_image_query] = matchables_query;
  }
  return 0;
}
