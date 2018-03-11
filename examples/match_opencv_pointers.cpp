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

using namespace srrg_hbst;

//ds feature handling
#if CV_MAJOR_VERSION == 2
cv::FeatureDetector* keypoint_detector                 = new cv::FastFeatureDetector();
const cv::DescriptorExtractor* descriptor_extractor   = new cv::BriefDescriptorExtractor(32);
#elif CV_MAJOR_VERSION == 3
cv::Ptr<cv::FastFeatureDetector> keypoint_detector     = cv::FastFeatureDetector::create();
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
#else
#error OpenCV version not supported
#endif

//ds buffers
std::vector<cv::Mat> images(10);
std::vector<std::vector<const cv::KeyPoint*>> buffer_keypoints(10);

//ds retrieves HBST matchables from an opencv image stored at the provided location
const BinaryTree256::MatchableVector getMatchables(const std::string& filename_image_, const uint64_t& identifier_tree_);

int32_t main(int32_t argc, char** argv) {

  //ds validate input
  if (argc != 2) {
    std::cerr << "invalid call - please use: ./srrg_hbst_search_opencv_pointers /path/to/srrg_hbst/examples/test_images" << std::endl;
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
  std::vector<BinaryTree256::MatchableVector> matchables_per_image(10);
  for (uint32_t u = 0; u < 10; ++u) {

    //ds generate file name
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", u);
    const std::string file_test_image = test_images_folder + "/" + buffer;

    matchables_per_image[u] = getMatchables(file_test_image, u);
  }

  //ds create HBSTs
  std::vector<BinaryTree256*> trees(10, 0);
  for (uint32_t u = 0; u < 10; ++u) {
    std::printf("building tree with image [%02u]\n", u);
    time_begin = std::chrono::system_clock::now();
    trees[u] = new BinaryTree256(u, matchables_per_image[u]);
    duration_construction += std::chrono::system_clock::now()-time_begin;
  }

  //ds check each image against each other and itself (100% ratio)
  for (uint32_t index_query = 0; index_query < 10; ++index_query) {

    //ds check reference
    for (uint32_t index_reference = 0; index_reference < 10; ++index_reference) {

      //ds query reference with matchables
      BinaryTree256::MatchVector matches;
      time_begin = std::chrono::system_clock::now();
      trees[index_reference]->matchLazy(matchables_per_image[index_query], matches);
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
      for (const BinaryTree256::Match& match: matches) {

        //ds directly get the keypoint objects
        const cv::KeyPoint* keypoint_query     = static_cast<const cv::KeyPoint*>(match.pointer_reference);
        const cv::KeyPoint* keypoint_reference = static_cast<const cv::KeyPoint*>(match.pointer_query);

        //ds draw correspondence
        cv::line(image_display, keypoint_query->pt, keypoint_reference->pt+shift, cv::Scalar(0, 255, 0));

        //ds draw reference point in upper image
        cv::circle(image_display, keypoint_query->pt, 2, cv::Scalar(0, 0, 255));

        //ds draw query point in lower image
        cv::circle(image_display, keypoint_reference->pt+shift, 2, cv::Scalar(255, 0, 0));
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
  delete keypoint_detector;
  delete descriptor_extractor;
#endif

  //ds fight memory leaks!
  for (const BinaryTree256* tree: trees) {
    delete tree;
  }
  matchables_per_image.clear();
  for (const std::vector<const cv::KeyPoint*>& keypoints: buffer_keypoints) {
    for (const cv::KeyPoint* keypoint: keypoints) {
      delete keypoint;
    }
  }

  //ds done
  return 0;
}

const BinaryTree256::MatchableVector getMatchables(const std::string& filename_image_, const uint64_t& identifier_tree_) {

  //ds allocate empty matchable vector
  BinaryTree256::MatchableVector matchables(0);

  //ds load image (project root folder)
  images[identifier_tree_] = cv::imread(filename_image_, CV_LOAD_IMAGE_GRAYSCALE);

  //ds detect FAST keypoints
  std::vector<cv::KeyPoint> keypoints;
  keypoint_detector->detect(images[identifier_tree_], keypoints);

  //ds compute BRIEF descriptors
  cv::Mat descriptors;
  descriptor_extractor->compute(images[identifier_tree_], keypoints, descriptors);
  std::cerr << "loaded image: " << filename_image_ << " with keypoints/descriptors: " << descriptors.rows << std::endl;
  for (const cv::KeyPoint& keypoint: keypoints) {
    cv::KeyPoint* linked_keypoint = new cv::KeyPoint(keypoint);
    buffer_keypoints[identifier_tree_].push_back(linked_keypoint);
  }

  //ds get descriptors to HBST format
  return BinaryTree256::getMatchablesWithPointer<const cv::KeyPoint*>(descriptors, buffer_keypoints[identifier_tree_]);
}
