#include <iostream>
#include <chrono>
#include <ctime>

//ds resolve opencv includes
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

#if CV_MAJOR_VERSION == 2
  //ds no specifics
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

#include "srrg_hbst_types/binary_tree.hpp"
using namespace srrg_hbst;

//ds feature handling
#if CV_MAJOR_VERSION == 2
cv::FeatureDetector* feature_detector                 = new cv::FastFeatureDetector();
const cv::DescriptorExtractor* descriptor_extractor   = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
cv::Ptr<cv::FastFeatureDetector> feature_detector     = cv::FastFeatureDetector::create();
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#else
#error OpenCV version not supported
#endif

//ds visualization buffers
std::vector<cv::Mat> images(10);

//ds buffer keypoints - otherwise the memory pointers get broken by the scope
std::vector<std::vector<cv::KeyPoint>> buffer_keypoints(10);

//ds retrieves HBST matchables from an opencv image stored at the provided location
const BinaryTree256b::MatchableVector getMatchables(const std::string& filename_image_, const uint64_t& identifier_tree_);

int32_t main(int32_t argc, char** argv) {

  //ds validate input
  if (argc != 2) {
    std::cerr << "invalid call - please use: ./srrg_hbst_search_opencv_indices /path/to/srrg_hbst/examples/test_images" << std::endl;
    return 0;
  }

  //ds measurements
  std::chrono::time_point<std::chrono::system_clock> time_begin;
  std::chrono::duration<double> duration_construction(0);
  std::chrono::duration<double> duration_query(0);
  double matching_ratio = 0;

  //ds get test image path
  const std::string test_images_folder = argv[1];

  //ds matchables vector
  std::vector<BinaryTree256b::MatchableVector> matchables_per_image(10);
  for (uint32_t u = 0; u < 10; ++u) {

    //ds generate file name
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", u);
    const std::string file_test_image = test_images_folder + "/" + buffer;

    matchables_per_image[u] = getMatchables(file_test_image, u);
  }

  //ds create HBSTs
  std::vector<BinaryTree256b*> trees(10, 0);
  for (uint32_t u = 0; u < 10; ++u) {
    std::printf("building tree with image [%02u]\n", u);
    time_begin = std::chrono::system_clock::now();
    trees[u] = new BinaryTree256b(u, matchables_per_image[u]);
    duration_construction += std::chrono::system_clock::now()-time_begin;
  }

  //ds check each image against each other and itself (100% ratio)
  for (uint32_t index_query = 0; index_query < 10; ++index_query) {

    //ds check reference
    for (uint32_t index_reference = 0; index_reference < 10; ++index_reference) {

      //ds query reference with matchables
      BinaryTree256b::MatchVector matches;
      time_begin = std::chrono::system_clock::now();
      trees[index_reference]->match(matchables_per_image[index_query], matches);
      duration_query += std::chrono::system_clock::now()-time_begin;

      std::printf("matches for QUERY [%02u] to REFERENCE [%02u]: %4lu (matching ratio: %5.3f)\n", index_query, index_reference, matches.size(),
                  static_cast<double>(matches.size())/matchables_per_image[index_reference].size());
      matching_ratio += static_cast<double>(matches.size())/matchables_per_image[index_reference].size();

      //ds show the matching in an image pair
      cv::Mat image_display;
      cv::vconcat(images[index_reference], images[index_query], image_display);
      cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

      //ds shift to lower image
      cv::Point2f shift(0, images[index_reference].rows);

      //ds draw correspondences
      for (const BinaryTree256b::Match& match: matches) {

        //ds draw correspondence
        cv::line(image_display, buffer_keypoints[index_reference][match.identifier_reference].pt, buffer_keypoints[index_query][match.identifier_query].pt+shift, cv::Scalar(0, 255, 0));

        //ds draw reference point in upper image
        cv::circle(image_display, buffer_keypoints[index_reference][match.identifier_reference].pt, 2, cv::Scalar(0, 0, 255));

        //ds draw query point in lower image
        cv::circle(image_display, buffer_keypoints[index_query][match.identifier_query].pt+shift, 2, cv::Scalar(255, 0, 0));
      }

      cv::imshow("matching (top: reference, bot: query)", image_display);
      cv::waitKey(0);
    }
  }

  //ds statistics summary
  std::cerr << "------------------------------------------------" << std::endl;
  std::printf("   construction duration (s): %6.4f\n", duration_construction.count());
  std::printf("  average query duration (s): %6.4f\n", duration_query.count()/10);
  std::printf("average query matching ratio: %6.4f\n", matching_ratio/(10*10));
  std::cerr << "------------------------------------------------" << std::endl;

#if CV_MAJOR_VERSION == 2
  delete feature_detector;
  delete descriptor_extractor;
#endif

  //ds fight memory leaks!
  for (const BinaryTree256b* tree: trees) {
    delete tree;
  }
  matchables_per_image.clear();

  //ds done
  return 0;
}

const BinaryTree256b::MatchableVector getMatchables(const std::string& filename_image_, const uint64_t& identifier_tree_) {

  //ds allocate empty matchable vector
  BinaryTree256b::MatchableVector matchables(0);

  //ds load image (project root folder)
  images[identifier_tree_] = cv::imread(filename_image_, CV_LOAD_IMAGE_GRAYSCALE);

  //ds detect FAST keypoints
  feature_detector->detect(images[identifier_tree_], buffer_keypoints[identifier_tree_]);

  //ds compute BRIEF descriptors
  cv::Mat descriptors;
  descriptor_extractor->compute(images[identifier_tree_], buffer_keypoints[identifier_tree_], descriptors);
  std::cerr << "loaded image: " << filename_image_ << " with keypoints/descriptors: " << descriptors.rows << std::endl;

  //ds get descriptors to HBST format
  return BinaryTree256b::getMatchablesWithIndex(descriptors);
}
