#include <iostream>
#include <chrono>
#include "dirent.h" //ds potential compatibility break
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

//ds HBST setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<const cv::KeyPoint*, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;

//ds processing modes
enum ProcessingMode {
  ImageFolder,
  Video
};



int32_t main(int32_t argc_, char** argv_) {
  std::cerr << "--------------------------------------------------------------------------------" << std::endl;
  std::cerr << "CONTROLS: press [0] / [+] to shrink / grow the displayed image" << std::endl;
  std::cerr << "          press any key (except [ESC] and [SPACE]) to start processing" << std::endl;
  std::cerr << "          press [ESC] to terminate processing" << std::endl;
  std::cerr << "          press [SPACE] for stepwise processing" << std::endl;
  std::cerr << "--------------------------------------------------------------------------------" << std::endl;

  //ds validate input
  if (argc_ != 3) {
    std::cerr << "ERROR: invalid call - please use: ./tracker -images </path/to/srrg_hbst/examples/test_images>" << std::endl;
    return 0;
  }

  //ds parse image folder/video file
  const std::string input_source = argv_[2];

  //ds evaluate desired processing mode
  ProcessingMode processing_mode;
  if (std::string(argv_[1]) == "-images") {
    std::cerr << "processing images: '" << input_source << "'" << std::endl;
    processing_mode = ProcessingMode::ImageFolder;
  } else if (std::string(argv_[1]) == "-video") {
    std::cerr << "processing video: '" << input_source << "'" << std::endl;
    processing_mode = ProcessingMode::Video;
  } else {
    std::cerr << "ERROR: invalid processing mode, chose -images <image_folder> OR -video <video_file>" << std::endl;
    return 0;
  }

  //ds image/video processing handles
  cv::VideoCapture video_player;
  double image_display_scale = 1;
  std::vector<std::string> image_file_paths(0);

  //ds validate input
  switch(processing_mode) {
    case ProcessingMode::ImageFolder: {

      //ds parse the image directory
      DIR* handle_directory = nullptr;
      dirent* iterator      = nullptr;
      if ((handle_directory = opendir(input_source.c_str()))) {
        while ((iterator = readdir(handle_directory))) {

          //ds buffer file name and construct full UNIX file path
          if (iterator->d_name[0] != '.') {
            image_file_paths.push_back(input_source+"/"+iterator->d_name);
          }
        }
      }
      std::sort(image_file_paths.begin(), image_file_paths.end());
      std::cerr << "loaded images: " << image_file_paths.size() << std::endl;
      break;
    }
    case ProcessingMode::Video: {
      if (!video_player.open(input_source)) {
        std::cerr << "ERROR: invalid test image path provided: " << input_source << std::endl;
      }
      break;
    }
    default: {
      std::cerr << "ERROR: invalid processing mode, chose -images <image_folder> OR -video <video_file>" << std::endl;
      return 0;
    }
  }

  //ds feature handling
#if CV_MAJOR_VERSION == 2
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = new cv::FastFeatureDetector();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::FeatureDetector> keypoint_detector        = cv::FastFeatureDetector::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create(0);
#endif

  //ds keypoint buffer (to keep keypoint information for multiple images)
  std::vector<std::vector<const cv::KeyPoint*>> keypoints_per_image(0);

  //ds matching threshold
  const uint32_t maximum_descriptor_distance = 10;

  //ds instantiate an empty tree
  Tree tree;

  //ds start processing - specializing only image loading depending on chosen parsing mode
  uint32_t number_of_processed_images   = 0;
  uint64_t number_of_stored_descriptors = 0;
  std::chrono::time_point<std::chrono::system_clock> time_begin;
  double duration_current_seconds       = 0;
  uint32_t number_of_processed_images_current_window = 0;

  //ds trust me on this loop TODO softcode key stroke
  int32_t key = 32;
  while (true) {
    time_begin = std::chrono::system_clock::now();

    //ds load next image from disk (from a video or a folder)
    cv::Mat image;
    switch(processing_mode) {
      case ProcessingMode::ImageFolder: {
        if (number_of_processed_images < image_file_paths.size()) {
          image = cv::imread(image_file_paths[number_of_processed_images]);
        }
        break;
      }
      case ProcessingMode::Video: {
        video_player.read(image);
        break;
      }
      default: {
        std::cerr << "ERROR: invalid processing mode, chose -images <image_folder> OR -video <video_file>" << std::endl;
        return 0;
      }
    }

    //ds check image and escape on issue
    if (image.rows <= 0 || image.cols <= 0) {
      std::cerr << "WARNING: failed to load image from disk, terminating" << std::endl;
      break;
    }

    //ds preprocess image
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    keypoint_detector->detect(image, keypoints);

    //ds compute BRIEF descriptors
    cv::Mat descriptors;
    descriptor_extractor->compute(image, keypoints, descriptors);

    //ds allocate keypoints manually in memory (we will directly link to them with our HBST matchables, avoiding any auxiliary bookkeeping)
    std::vector<const cv::KeyPoint*> dynamic_keypoints(0);
    for (const cv::KeyPoint& keypoint: keypoints) {
      dynamic_keypoints.push_back(new cv::KeyPoint(keypoint));
    }
    keypoints_per_image.push_back(dynamic_keypoints);

    //ds obtain linked matchables
    const Tree::MatchableVector matchables(Tree::getMatchables(descriptors, dynamic_keypoints, number_of_processed_images));

    //ds obtain matches against all inserted matchables (i.e. images so far) and integrate them simultaneously
    Tree::MatchVectorMap matches_per_image;
    tree.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
    number_of_stored_descriptors += matchables.size();

    //ds info
    std::cerr << "processed images: " << number_of_processed_images << " | total descriptors stored: " << number_of_stored_descriptors
              << " | current fps: " << number_of_processed_images_current_window/duration_current_seconds << std::endl;

    //ds timing - reset measurement window for every 100 frames
    if (number_of_processed_images_current_window > 100) {
      number_of_processed_images_current_window = 0;
      duration_current_seconds                  = 0;
    }

    //ds draw current keypoints
    cv::Mat image_display(image);
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
    for (const cv::KeyPoint* keypoint: dynamic_keypoints) {
      cv::circle(image_display, keypoint->pt, 2, cv::Scalar(255, 0, 0), -1);
    }

    //ds for each match vector (i.e. matching results for each past image) of ALL past images
    for (uint32_t image_number_reference = 0; image_number_reference < number_of_processed_images; ++image_number_reference) {

      //ds if we have sufficient matches
      if (matches_per_image[image_number_reference].size() > 100) {
        std::cerr << "matches from image [" << image_number_reference << "] to image [" << number_of_processed_images << "]: "
                  <<  matches_per_image[image_number_reference].size()
                  << " (ratio: " << static_cast<double>(matches_per_image[image_number_reference].size())/dynamic_keypoints.size() << ")" << std::endl;

        //ds draw tracks on image
        for (const Tree::Match& match: matches_per_image[image_number_reference]) {
          const cv::KeyPoint* point_query     = match.object_query;
          const cv::KeyPoint* point_reference = match.object_reference;

          //ds on the fly cutoff kernel
          if (cv::norm(cv::Mat(point_query->pt), cv::Mat(point_reference->pt)) < 100) {
            cv::line(image_display, point_query->pt, point_reference->pt, cv::Scalar(rand()%255, rand()%255, rand()%255));
          }
        }
      }
    }
    const cv::Size current_size(image_display_scale*image_display.cols, image_display_scale*image_display.rows);
    cv::resize(image_display, image_display, current_size);
    cv::imshow("current image", image_display);

    //ds check if we have to pause processing (SPACE pressed) TODO softcode
    if (key == 32) {
      key = cv::waitKey(0);
    } else {
      key = cv::waitKey(1);
    }

    //ds check if image shrinking or growing is desired TODO softcode
    if (key == 45) {
      image_display_scale /= 2;
    }
    if (key == 43) {
      image_display_scale *= 2;
    }

    //ds termination (ESC key) TODO softcode
    if (key == 27) {
      std::cerr << "termination requested" << std::endl;
      break;
    }

    //ds timing
    duration_current_seconds += std::chrono::duration<double>(std::chrono::system_clock::now()-time_begin).count();

    //ds done
    ++number_of_processed_images;
    ++number_of_processed_images_current_window;
  }

  //ds free linked structures to matchables (the matchables themselves get freed by the tree)
  for (const std::vector<const cv::KeyPoint*>& keypoints: keypoints_per_image) {
    for (const cv::KeyPoint* keypoint: keypoints) {
      delete keypoint;
    }
  }
  image_file_paths.clear();
  return 0;
}
