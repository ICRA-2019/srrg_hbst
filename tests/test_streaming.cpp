#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "srrg_hbst/types/binary_tree.hpp"

using namespace srrg_hbst;
typedef BinaryTree256<size_t> Tree;

// ds test fixture
class Streaming : public ::testing::Test {
protected:
  void SetUp() override {
    Tree::Node::maximum_partitioning = 0.45;            // ds noisy, synthetic case
    random_number_generator          = std::mt19937(0); // ds locked seed for reproducibility
    generateMatchables(matchables_train_per_image, number_of_images_train, 0);
    generateMatchables(matchables_query_per_image, number_of_images_query, number_of_images_train);
  }

  void TearDown() override {
    // ds training matchables are already freed by the tree
    for (const Tree::MatchableVector& matchables : matchables_query_per_image) {
      for (const Tree::Matchable* matchable : matchables) {
        delete matchable;
      }
    }
  }

  void generateMatchables(std::vector<Tree::MatchableVector>& matchables_per_image_,
                          const size_t& number_of_images_,
                          const size_t& image_number_start_) {
    matchables_per_image_.clear();
    matchables_per_image_.reserve(number_of_images_);

    // ds populate matchable vectors
    for (size_t index_image = 0; index_image < number_of_images_; ++index_image) {
      Tree::MatchableVector matchables;
      matchables.reserve(number_of_matchables_per_image);
      for (size_t index_descriptor = 0; index_descriptor < number_of_matchables_per_image;
           ++index_descriptor) {
        Tree::Descriptor descriptor;
        ASSERT_EQ(descriptor.size(), Tree::Matchable::descriptor_size_bits);
        for (size_t flips = 0; flips < number_of_bits_to_flip; ++flips) {
          std::uniform_int_distribution<uint32_t> bit_index_to_flip(
            0, Tree::Matchable::descriptor_size_bits - 1);
          descriptor.set(bit_index_to_flip(random_number_generator));
        }
        matchables.emplace_back(
          new Tree::Matchable(index_descriptor, descriptor, index_image + image_number_start_));
      }
      matchables_per_image_.emplace_back(matchables);
    }
  }

  //! dummy query and reference (= database) matchables
  std::vector<Tree::MatchableVector> matchables_query_per_image;
  std::vector<Tree::MatchableVector> matchables_train_per_image;

  //! configuration
  static constexpr size_t number_of_images_query         = 1;
  static constexpr size_t number_of_images_train         = 10;
  static constexpr size_t number_of_matchables_per_image = 1000;

  //! random number generator used to generate binary descriptors
  static std::mt19937 random_number_generator;
  static constexpr size_t number_of_bits_to_flip = 18;

  //! "ground truth" data
  static std::vector<size_t> identifiers_query;
  static std::vector<size_t> identifiers_train;
  static std::vector<Tree::real_type> matching_distances;
};

// ds come on c++11
constexpr size_t Streaming::number_of_images_query;
constexpr size_t Streaming::number_of_images_train;
constexpr size_t Streaming::number_of_matchables_per_image;
std::mt19937 Streaming::random_number_generator;
constexpr size_t Streaming::number_of_bits_to_flip;

int main(int argc_, char** argv_) {
  testing::InitGoogleTest(&argc_, argv_);
  return RUN_ALL_TESTS();
}

TEST_F(Streaming, Write) {
  // ds populate the database
  Tree database;
  for (Tree::MatchableVector& matchables_train : matchables_train_per_image) {
    database.add(matchables_train, SplittingStrategy::SplitEven);
  }
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds verify matching before serializing
  for (Tree::MatchableVector& matchables_query : matchables_query_per_image) {
    Tree::MatchVectorMap match_vectors;
    database.match(matchables_query, match_vectors);
    ASSERT_EQ(match_vectors.size(), static_cast<size_t>(10));
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_query.size());
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_train.size());
    ASSERT_EQ(match_vectors.at(0).size(), matching_distances.size());
    for (size_t i = 0; i < match_vectors.at(0).size(); ++i) {
      // ds check match against hardcoded sampling ground truth
      const Tree::Match& match = match_vectors.at(0)[i];
      ASSERT_EQ(match.object_query, identifiers_query[i]);
      ASSERT_GE(match.object_references.size(), static_cast<size_t>(1));
      ASSERT_EQ(match.object_references[0], identifiers_train[i]);
      ASSERT_EQ(match.distance, matching_distances[i]);
    }
  }

  // ds save database to disk (this operation will not clean dynamic memory)
  ASSERT_TRUE(database.write("database.hbst"));

  // ds clear database
  database.clear(true);
}

TEST_F(Streaming, Read) {
  // ds load database from disk
  Tree database;
  ASSERT_EQ(database.size(), static_cast<size_t>(0));
  ASSERT_TRUE(database.read("database.hbst"));
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds verify matching after deserializing
  for (Tree::MatchableVector& matchables_query : matchables_query_per_image) {
    Tree::MatchVectorMap match_vectors;
    database.match(matchables_query, match_vectors);
    ASSERT_EQ(match_vectors.size(), static_cast<size_t>(10));
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_query.size());
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_train.size());
    ASSERT_EQ(match_vectors.at(0).size(), matching_distances.size());
    for (size_t i = 0; i < match_vectors.at(0).size(); ++i) {
      // ds check match against hardcoded sampling ground truth
      const Tree::Match& match = match_vectors.at(0)[i];
      ASSERT_EQ(match.object_query, identifiers_query[i]);
      ASSERT_GE(match.object_references.size(), static_cast<size_t>(1));
      ASSERT_EQ(match.object_references[0], identifiers_train[i]);
      ASSERT_EQ(match.distance, matching_distances[i]);
    }
  }

  // ds clear database
  database.clear(true);
}

// ds "ground truth" assuming consistent random sampling on testing architectures
std::vector<size_t> Streaming::identifiers_query = {
  58,  78,  91,  108, 109, 116, 122, 127, 144, 146, 149, 150, 151, 159, 183, 187,
  190, 192, 200, 212, 217, 241, 247, 258, 259, 266, 309, 312, 320, 340, 349, 360,
  372, 377, 412, 421, 443, 449, 464, 466, 480, 492, 497, 506, 509, 573, 575, 586,
  598, 621, 626, 627, 641, 666, 680, 681, 705, 707, 737, 755, 763, 765, 766, 776,
  783, 815, 852, 855, 859, 881, 888, 895, 903, 907, 921, 951, 952, 955};
std::vector<size_t> Streaming::identifiers_train = {
  512, 896, 874, 621, 495, 998, 289, 321, 965, 615, 658, 359, 251, 99,  965, 679,
  751, 270, 407, 471, 615, 110, 932, 193, 270, 719, 373, 939, 675, 474, 867, 648,
  342, 300, 811, 771, 182, 180, 464, 865, 275, 877, 707, 618, 366, 320, 577, 722,
  563, 235, 699, 507, 91,  37,  468, 52,  947, 559, 765, 427, 852, 125, 279, 754,
  392, 904, 807, 380, 311, 662, 839, 83,  212, 96,  546, 46,  354, 872};
std::vector<Tree::real_type> Streaming::matching_distances = {
  22, 23, 24, 24, 24, 24, 24, 22, 24, 22, 24, 24, 24, 24, 23, 24, 24, 24, 24, 23,
  24, 24, 24, 23, 24, 23, 24, 23, 24, 24, 22, 23, 24, 23, 23, 23, 24, 24, 24, 24,
  24, 24, 23, 24, 24, 22, 24, 23, 23, 23, 23, 24, 24, 23, 24, 24, 20, 24, 23, 22,
  24, 20, 24, 20, 24, 22, 23, 24, 24, 23, 24, 24, 23, 24, 23, 24, 23, 24};
