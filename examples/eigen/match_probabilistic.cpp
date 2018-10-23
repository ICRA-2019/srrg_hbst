#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"
#include "srrg_hbst_types_probabilistic/probabilistic_node.hpp"

//ds current setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::ProbabilisticMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::ProbabilisticNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;
const double RAND_MAX_AS_DOUBLE = static_cast<double>(RAND_MAX);

//ds dummy descriptor generation
Tree::MatchableVector getDummyMatchables(const uint64_t& number_of_matchables_);



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
  std::cerr << "successfully matched descriptors: " << matches.size() << std::endl;

  //ds query tree for matching ratio
  const double matching_ratio = tree.getMatchingRatio(matchables_query, 25);
  std::cerr << "matching ratio: " << matching_ratio << std::endl;

  //ds get lazy matching ratio
  const double matching_ratio_lazy = tree.getMatchingRatioLazy(matchables_query, 25);
  std::cerr << "matching ratio lazy: " << matching_ratio_lazy << std::endl;

  //ds fight memory leaks!
  for(const Matchable* matchable: matchables_query) {
    delete matchable;
  }
  return 0;
}

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
