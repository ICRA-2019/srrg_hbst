#pragma once
#include <memory>
#include "srrg_hbst_types_probabilistic/probabilistic_node.hpp"

namespace srrg_hbst {

template<typename ProbabilisticNodeType_, uint32_t maximum_distance_hamming_ = 25>
class ProbabilisticTree
{

//ds template forwarding
public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename ProbabilisticNodeType_::Matchable Matchable;
  typedef typename ProbabilisticNodeType_::MatchableVector MatchableVector;
  typedef typename ProbabilisticNodeType_::Match Match;
  typedef std::vector<Match> MatchVector;

//ds ctor/dtor
public:

  //ds construct tree upon allocation on filtered descriptors
  ProbabilisticTree(const uint64_t& identifier_,
                    const MatchableVector& matchables_): identifier(identifier_),
                                                         root(new ProbabilisticNodeType_(matchables_)) {
    std::cerr << "ProbabilisticTree" << std::endl;
    assert(0 != root);
  }

#ifdef SRRG_HBST_HAS_OPENCV
  //ds wrapped constructors - only available if OpenCV is present on building system
  ProbabilisticTree(const uint64_t& identifier_,
                    const cv::Mat& matchables_): identifier(identifier_),
                                                 root(new ProbabilisticNodeType_(getMatchablesWithIndex(matchables_))) {
    assert(0 != root);
  }
#endif

  //ds free all nodes in the tree
  ~ProbabilisticTree() {

    //ds erase all nodes (and descriptors)
    _displant();
  }

  //ds disable default construction
  ProbabilisticTree() = delete;

//ds control fields
public:

  //ds tree ID
  const uint64_t identifier;

private:

  //ds root node
  const ProbabilisticNodeType_* root;

//ds access (shared pointer wrappings)
public:

  const uint64_t getNumberOfMatches(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return getNumberOfMatches(*matchables_query_);
  }

  const typename ProbabilisticNodeType_::precision getMatchingRatio(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return static_cast<typename ProbabilisticNodeType_::precision>(getNumberOfMatches(*matchables_query_))/matchables_query_->size();
  }

//ds access
public:

  //ds TODO port matching full functionality for VBST
  const uint64_t getNumberOfMatches(const MatchableVector& matchables_query_) const {
    uint64_t number_of_matches = 0;

    //ds for each descriptor
    for (const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const ProbabilisticNodeType_* node_current = root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if(node_current->has_leaves) {

          //ds check the split bit and go deeper
          if(matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = static_cast<const ProbabilisticNodeType_*>(node_current->leaf_ones);
          } else {
            node_current = static_cast<const ProbabilisticNodeType_*>(node_current->leaf_zeros);
          }
        } else {

          //ds check current descriptors in this node and exit
          for (const Matchable* matchable_reference: node_current->matchables) {
            if (maximum_distance_hamming_ > matchable_query->distanceHamming(matchable_reference)) {
              ++number_of_matches;
              break;
            }
          }
          break;
        }
      }
    }
    return number_of_matches;
  }

  const typename ProbabilisticNodeType_::precision getMatchingRatio(const MatchableVector& matchables_query_) const {
    return static_cast<typename ProbabilisticNodeType_::precision>(getNumberOfMatches(matchables_query_))/matchables_query_.size();
  }

  //ds grow the tree
  void plant(const MatchableVector& matchables_) {

    //ds grow tree on root
    root = new ProbabilisticNodeType_(matchables_);
  }

#ifdef SRRG_HBST_HAS_OPENCV
  //ds creates a matchable vector (indexed) from opencv descriptors - only available if OpenCV is present on building system
  static const MatchableVector getMatchablesWithIndex(const cv::Mat& descriptors_cv_) {
    MatchableVector matchables(descriptors_cv_.rows);

    //ds copy raw data
    for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows; ++index_descriptor) {
      matchables[index_descriptor] = new Matchable(index_descriptor, descriptors_cv_.row(index_descriptor));
    }
    return matchables;
  }

  //ds creates a matchable vector (pointers) from opencv descriptors - only available if OpenCV is present on building system
  template<typename Type>
  static const MatchableVector getMatchablesWithPointers(const cv::Mat& descriptors_cv_, const std::vector<Type>& pointers_) {
    MatchableVector matchables(descriptors_cv_.rows);

    //ds copy raw data
    for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows; ++index_descriptor) {
      matchables[index_descriptor] = new Matchable(reinterpret_cast<const void*>(&pointers_[index_descriptor]), descriptors_cv_.row(index_descriptor));
    }
    return matchables;
  }
#endif

//ds helpers
private:

  //ds delete tree
  void _displant() {

    //ds nodes holder
    std::vector<const ProbabilisticNodeType_*> nodes_collection;

    //ds set vector
    _setNodesRecursive(root, nodes_collection);

    //ds free nodes
    for (const ProbabilisticNodeType_* node: nodes_collection) {
      delete node;
    }

    //ds clear nodes vector
    nodes_collection.clear();
  }

  void _setNodesRecursive(const ProbabilisticNodeType_* node_, std::vector<const ProbabilisticNodeType_*>& nodes_collection_) const {

    //ds must not be zero
    assert(0 != node_);

    //ds add the current node
    nodes_collection_.push_back(node_);

    //ds check if there are leafs
    if(node_->has_leaves) {

      //ds add leafs and so on
      _setNodesRecursive(static_cast<const ProbabilisticNodeType_*>(node_->leaf_ones), nodes_collection_);
      _setNodesRecursive(static_cast<const ProbabilisticNodeType_*>(node_->leaf_zeros), nodes_collection_);
    }
  }
};
}
