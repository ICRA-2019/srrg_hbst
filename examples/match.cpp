#include <iostream>
#include "srrg_hbst_types/binary_tree.hpp"

//ds current setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;

//ds dummy descriptor generation
const std::shared_ptr<const Tree::MatchableVector> getDummyMatchables(const uint64_t& number_of_matchables_) {

  //ds preallocate vector
  Tree::MatchableVector matchables(number_of_matchables_);

  //ds set values
  for(uint64_t identifier = 0; identifier < number_of_matchables_; ++identifier) {

    //ds generate a "random" descriptor by flipping some bits
    Matchable::Descriptor descriptor;
    for (uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS/3; ++u) {
      if (rand() % 2) {
        descriptor.flip(u);
      }
    }

    //ds set matchable
    matchables[identifier] = new Matchable(identifier, descriptor);
  }

  //ds done
  return std::make_shared<const Tree::MatchableVector>(matchables);
}

int32_t main(int32_t argc, char** argv) {

  //ds train descriptor pool
  const std::shared_ptr<const Tree::MatchableVector> matchables_reference = getDummyMatchables(10000);

  //ds allocate a BTree object on these descriptors (no shared pointer passed as the tree will have its own constant copy of the train descriptors)
  const Tree hbst_tree(0, *matchables_reference);

  //ds query descriptor pool
  const std::shared_ptr<const Tree::MatchableVector> matchables_query = getDummyMatchables(5000);



  //ds get matches (opencv IN/OUT style)
  Tree::MatchVector matches1;
  hbst_tree.matchLazy(*matchables_query, matches1);

  //ds get matches directly
  const std::shared_ptr<const Tree::MatchVector> matches2 = hbst_tree.getMatchesLazy(matchables_query);

  //ds match count functions (FAST)
  const uint64_t number_of_matches      = hbst_tree.getNumberOfMatches(matchables_query);
  const double matching_ratio           = hbst_tree.getMatchingRatio(matchables_query);
  const uint64_t number_of_matches_lazy = hbst_tree.getNumberOfMatchesLazy(matchables_query);
  const double matching_ratio_lazy      = hbst_tree.getMatchingRatioLazy(matchables_query);

  std::cerr << "matches: " << number_of_matches << "/" << matchables_query->size() <<  std::endl;
  std::cerr << "  ratio: " << matching_ratio << std::endl;
  std::cerr << "lazy matches: " << number_of_matches_lazy << "/" << matchables_query->size() <<  std::endl;
  std::cerr << "       ratio: " << matching_ratio_lazy << std::endl;

  //ds matches must be identical: check if number of elements differ
  if(matches1.size() != matches2->size()) {
    std::cerr << "received inconsistent matching returns" << std::endl;
    return -1;
  }

  //ds check each element
  for (uint64_t index_match = 0; index_match < matches1.size( ); ++index_match) {

    //ds check if not matching
    if(matches1[index_match].identifier_query     != matches2->at(index_match).identifier_query    ||
       matches1[index_match].identifier_reference != matches2->at(index_match).identifier_reference||
       matches1[index_match].distance             != matches2->at(index_match).distance            ) {
        std::cerr << "received inconsistent matching returns" << std::endl;
        return -1;
    }
  }

  //ds fight memory leaks!
  for(const Matchable* matchable: *matchables_query) {
    delete matchable;
  }

  //ds done
  std::cout << "successfully matched descriptors: " << matches1.size( ) << std::endl;
  return 0;
}
