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
    std::cerr << "invalid call - please use: ./match_opencv /path/to/srrg_hbst/examples/test_images" << std::endl;
    return 0;
  }

  //ds initialize feature handling
#if CV_MAJOR_VERSION == 2
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = new cv::FastFeatureDetector();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = cv::FastFeatureDetector::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create();
#endif

  //ds get test image path
  const std::string test_images_folder = argv_[1];

  //ds image storage
  std::vector<cv::Mat> images(10);

  //ds matchables vector
  std::vector<Tree::MatchableVector> matchables_per_image(10);
  for (uint32_t image_number = 0; image_number < 10; ++image_number) {

    //ds generate file name
    char buffer[32];
    std::sprintf(buffer, "image_%02u.pgm", image_number);
    const std::string file_test_image = test_images_folder + "/" + buffer;

    //ds load image (project root folder)
    images[image_number] = cv::imread(file_test_image, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(images[image_number], keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(images[image_number], keypoints, descriptors);
    std::cerr << "loaded image: " << file_test_image << " with keypoints/descriptors: " << descriptors.rows << std::endl;

    //ds connect each descriptor and keypoint in an HBST matchable and store the vector
    matchables_per_image[image_number] = Tree::getMatchables(descriptors, keypoints);
  }

  //ds create HBSTs
  std::vector<Tree*> trees(10, 0);
  for (uint32_t u = 0; u < 10; ++u) {
    std::printf("building tree with image [%02u]\n", u);
    trees[u] = new Tree(u, matchables_per_image[u]);
  }

  //ds check each image against each other and itself (100% ratio)
  std::printf("------------[ press any key to step ]------------\n");
  for (uint32_t index_query = 0; index_query < 10; ++index_query) {

    //ds check reference
    for (uint32_t index_reference = 0; index_reference < 10; ++index_reference) {

      //ds query reference with matchables
      Tree::MatchVector matches;
      trees[index_reference]->matchLazy(matchables_per_image[index_query], matches);

      std::printf("matches for QUERY [%02u] to REFERENCE [%02u]: %4lu (matching ratio: %5.3f)\n", index_query, index_reference, matches.size(),
                  static_cast<double>(matches.size())/matchables_per_image[index_reference].size());

      //ds show the matching in an image pair
      cv::Mat image_display;
      cv::vconcat(images[index_reference], images[index_query], image_display);
      cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

      //ds shift to lower image
      cv::Point2f shift(0, images[index_reference].rows);

      //ds draw correspondences
      for (const Tree::Match& match: matches) {

        //ds draw correspondence
        cv::line(image_display, match.object_query.pt, match.object_reference.pt+shift, cv::Scalar(0, 255, 0));

        //ds draw reference point in upper image
        cv::circle(image_display, match.object_query.pt, 2, cv::Scalar(0, 0, 255));

        //ds draw query point in lower image
        cv::circle(image_display, match.object_reference.pt+shift, 2, cv::Scalar(255, 0, 0));
      }

      cv::imshow("matching (top: reference, bot: query)", image_display);
      cv::waitKey(0);
    }
  }

  //ds fight memory leaks!
  for (const Tree* tree: trees) {
    delete tree;
  }
  matchables_per_image.clear();

  //ds done
  return 0;
}
