#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"
#include "srrg_hbst_types_probabilistic/probabilistic_node.hpp"

//ds current setup
#define MAXIMUM_DISTANCE_HAMMING 25
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::ProbabilisticMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::ProbabilisticNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;
const double RAND_MAX_AS_DOUBLE = static_cast<double>(RAND_MAX);

//ds dummy descriptor generation
Tree::MatchableVector getDummyMatchables(const uint64_t& number_of_matchables_) {

  //ds preallocate vector
  Tree::MatchableVector matchables(number_of_matchables_);

  //ds set values
  for(uint64_t identifier = 0; identifier < number_of_matchables_; ++identifier) {

    //ds generate random probabilities
    Matchable::BitStatisticsVector bit_probabilities(Matchable::BitStatisticsVector::Zero());
    for (uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u) {
      bit_probabilities(u) = rand()/RAND_MAX_AS_DOUBLE;
    }

    //ds same counts
    Matchable::BitStatisticsVector bit_permanences(Matchable::BitStatisticsVector::Ones());

    //ds generate a "random" descriptor by flipping bits
    Matchable::Descriptor descriptor;
    for (uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS/3; ++u) {
      if (rand() % 2) {
        descriptor.flip(u);
      }
    }

    //ds set matchable
    matchables[identifier] = new Matchable(identifier, descriptor, bit_probabilities, bit_permanences);
  }

  //ds done
  return matchables;
}

int32_t main() {

  //ds train descriptor pool
  const Tree::MatchableVector matchables_reference = getDummyMatchables(10000);

  //ds allocate a BTree object on these descriptors (no shared pointer passed as the tree will have its own constant copy of the train descriptors)
  const Tree tree(0, matchables_reference);

  //ds query descriptor pool
  const Tree::MatchableVector matchables_query = getDummyMatchables(5000);

  //ds matching results
  Tree::MatchVector matches;

  //ds query tree for matches (exhaustive search in leafs) with maximum distance 25
  tree.match(matchables_query, matches, 25);

//  hbst_tree.match(*matchables_query, matches1);
//
//  //ds get matches directly
//  const std::shared_ptr<const Tree::MatchVector> matches2 = hbst_tree.getMatches(matchables_query);
//
//  //ds match count functions (FAST)
//  const uint64_t number_of_matches      = hbst_tree.getNumberOfMatches(matchables_query);
//  const double matching_ratio           = hbst_tree.getMatchingRatio(matchables_query);
//  const uint64_t number_of_matches_lazy = hbst_tree.getNumberOfMatchesLazyEvaluation(matchables_query);
//  const double matching_ratio_lazy      = hbst_tree.getMatchingRatioLazyEvaluation(matchables_query);
//
//  std::cerr << "matches for distance of 25: " << number_of_matches << std::endl;
//  std::cerr << "                     ratio: " << matching_ratio << std::endl;
//  std::cerr << "lazy matches for distance of 25: " << number_of_matches_lazy << std::endl;
//  std::cerr << "                          ratio: " << matching_ratio_lazy << std::endl;
//
//
//
//
//  //ds matches must be identical: check if number of elements differ
//  if(matches1.size() != matches2->size()) {
//    std::cerr << "received inconsistent matching returns" << std::endl;
//    return -1;
//  }
//
//  //ds check each element
//  for (uint64_t index_match = 0; index_match < matches1.size( ); ++index_match) {
//
//    //ds check if not matching
//    if(matches1[index_match].identifier_query     != matches2->at(index_match).identifier_query    ||
//       matches1[index_match].identifier_reference != matches2->at(index_match).identifier_reference||
//       matches1[index_match].identifier_tree      != matches2->at(index_match).identifier_tree     ||
//       matches1[index_match].distance             != matches2->at(index_match).distance            ) {
//        std::cerr << "received inconsistent matching returns" << std::endl;
//        return -1;
//    }
//  }

  //ds fight memory leaks!
  for(const Matchable* matchable: matchables_query) {
    delete matchable;
  }

  //ds done
  std::cout << "successfully matched descriptors: " << matches.size() << std::endl;
  return 0;
}
