#pragma once
#include <memory>
#include "binary_node.hpp"

namespace srrg_hbst {

//! @class the binary tree class, consisting of binary nodes holding binary descriptors
template<typename BinaryNodeType_>
class BinaryTree
{

//ds template forwarding
public:

  //! @brief directly exported types (readability)
  typedef BinaryNodeType_ Node;
  typedef typename Node::Matchable Matchable;
  typedef typename Node::MatchableVector MatchableVector;
  typedef typename Node::Descriptor Descriptor;
  typedef typename Node::Match Match;
  typedef typename Node::real_type real_type;
  typedef std::vector<Match> MatchVector;
  typedef std::map<uint64_t, std::vector<Match>> MatchVectorMap;
  typedef std::pair<uint64_t, std::vector<Match>> MatchVectorMapElement;

  //! @brief component object used for tree training
  struct Trainable {
    Node* node;
    const Matchable* matchable;
    bool spawn_leaves;
  };

//ds ctor/dtor
public:

  //ds construct tree upon allocation on filtered descriptors
  BinaryTree(const MatchableVector& matchables_): identifier(0),
                                                  _root(new Node(matchables_)) {
    _matchables.clear();
    _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
  }

  //ds construct tree upon allocation on filtered descriptors
  BinaryTree(const uint64_t& identifier_,
             const MatchableVector& matchables_): identifier(identifier_),
                                                  _root(new Node(matchables_)) {
    _matchables.clear();
    _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
  }

  //ds construct tree upon allocation on filtered descriptors: with bit mask
  BinaryTree(const uint64_t& identifier_,
             const MatchableVector& matchables_,
             Descriptor bit_mask_): identifier(identifier_),
                                    _root(new Node(matchables_, bit_mask_)) {
    _matchables.clear();
    _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
  }

  //ds construct tree with fixed split order
  BinaryTree(const uint64_t& identifier_,
             const MatchableVector& matchables_,
             std::vector<uint32_t> split_order_): identifier(identifier_),
                                                  _root(new Node(matchables_, split_order_)) {
    _matchables.clear();
    _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
  }

#ifdef SRRG_HBST_HAS_OPENCV

  //ds wrapped constructors - only available if OpenCV is present on building system
  BinaryTree(const uint64_t& identifier_,
             const cv::Mat& matchables_): BinaryTree(identifier_, getMatchablesWithIndex(matchables_)) {}
  BinaryTree(const uint64_t& identifier_,
             const cv::Mat& matchables_,
             Descriptor bit_mask_): identifier(identifier_, getMatchablesWithIndex(matchables_), bit_mask_) {}
  BinaryTree(const uint64_t& identifier_,
             const cv::Mat& matchables_,
             std::vector<uint32_t> split_order_): identifier(identifier_, getMatchablesWithIndex(matchables_), split_order_) {}

#endif

  //ds free all nodes in the tree
  virtual ~BinaryTree() {

    //ds erase all nodes and matchables contained in the tree
    clearNodes();
    clearMatchables();
  }

  //ds disable default construction
  BinaryTree() = delete;

//ds control fields
public:

  //! @brief identifier for this tree
  const uint64_t identifier;

//ds attributes
protected:

  //! @brief root node (e.g. starting point for similarity search)
  Node* _root;

  //! @brief bookkeeping: all matchables contained in the tree
  MatchableVector _matchables;

//ds access (shared pointer wrappings)
public:

  const uint64_t getNumberOfMatches(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return getNumberOfMatches(*matchables_query_);
  }

  const typename Node::real_type getMatchingRatio(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatches(*matchables_query_))/matchables_query_->size();
  }

  const uint64_t getNumberOfMatchesLazyEvaluation(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return getNumberOfMatchesLazyEvaluation(*matchables_query_);
  }

  const typename Node::real_type getMatchingRatioLazyEvaluation(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatchesLazyEvaluation(*matchables_query_))/matchables_query_->size();
  }

//ds access
public:

  const uint64_t getNumberOfMatches(const MatchableVector& matchables_query_, const uint32_t& maximum_distance_ = 25) const {
    uint64_t number_of_matches = 0;

    //ds for each descriptor
    for (const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if(node_current->has_leaves) {

          //ds check the split bit and go deeper
          if(matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = static_cast<const Node*>(node_current->leaf_ones);
          } else {
            node_current = static_cast<const Node*>(node_current->leaf_zeros);
          }
        } else {

          //ds check current descriptors in this node and exit
          for (const Matchable* matchable_reference: node_current->matchables) {
            if (maximum_distance_ > matchable_query->distanceHamming(matchable_reference)) {
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

  const typename Node::real_type getMatchingRatio(const MatchableVector& matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatches(matchables_query_))/matchables_query_.size();
  }

  const uint64_t getNumberOfMatchesLazyEvaluation(const MatchableVector& matchables_query_, const uint32_t& maximum_distance_ = 25) const {
    uint64_t number_of_matches = 0;

    //ds for each descriptor
    for (const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leaves) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = static_cast<const Node*>(node_current->leaf_ones);
          } else {
            node_current = static_cast<const Node*>(node_current->leaf_zeros);
          }
        }
        else
        {
          //ds check current descriptors in this node and exit
          if(maximum_distance_ > matchable_query->distanceHamming(node_current->matchables.front())) {
            ++number_of_matches;
          }
          break;
        }
      }
    }
    return number_of_matches;
  }

  const typename Node::real_type getMatchingRatioLazyEvaluation(const MatchableVector& matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatchesLazyEvaluation(matchables_query_))/matchables_query_.size();
  }

  //ds direct matching function on this tree
  virtual void match(const MatchableVector& matchables_query_, MatchVector& matches_, const uint32_t& maximum_distance_ = 25) const {

    //ds for each descriptor
    for(const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leaves) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = static_cast<const Node*>(node_current->leaf_ones);
          } else {
            node_current = static_cast<const Node*>(node_current->leaf_zeros);
          }
        } else {

          //ds check current descriptors in this node and exit
          for (const Matchable* matchable_reference: node_current->matchables) {
            if (maximum_distance_ > matchable_query->distanceHamming(matchable_reference)) {
              matches_.push_back(Match(matchable_query, matchable_reference, maximum_distance_));
              break;
            }
          }
          break;
        }
      }
    }
  }

  //ds return matches directly
  const std::shared_ptr<const MatchVector> getMatches(const std::shared_ptr<const MatchableVector> matchables_query_, const uint32_t& maximum_distance_ = 25) const {

    //ds match vector to be filled
    MatchVector matches;

    //ds for each QUERY descriptor
    for(const Matchable* matchable_query: *matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leaves) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = static_cast<const Node*>(node_current->leaf_ones);
          } else {
            node_current = static_cast<const Node*>(node_current->leaf_zeros);
          }
        } else {

          //ds check current descriptors in this node and exit
          for (const Matchable* matchable_reference: node_current->matchables) {
            if (maximum_distance_ > matchable_query->distanceHamming(matchable_reference)) {
              matches.push_back(Match(matchable_query, matchable_reference, maximum_distance_));
              break;
            }
          }
          break;
        }
      }
    }

    //ds return findings
    return std::make_shared<const MatchVector>(matches);
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
  template<typename PointerType_>
  static const MatchableVector getMatchablesWithPointer(const cv::Mat& descriptors_cv_, const std::vector<PointerType_>& pointers_) {
    MatchableVector matchables(descriptors_cv_.rows);

    //ds copy raw data
    for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows; ++index_descriptor) {
      matchables[index_descriptor] = new Matchable(reinterpret_cast<const void*>(&pointers_[index_descriptor]), descriptors_cv_.row(index_descriptor));
    }
    return matchables;
  }

  //ds creates a matchable vector (indexed) from opencv descriptors - only available if OpenCV is present on building system
  static const MatchableVector getMatchablesWithIndex(const cv::Mat& descriptors_cv_, const uint64_t& identifier_tree_) {
    MatchableVector matchables(descriptors_cv_.rows);

    //ds copy raw data
    for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows; ++index_descriptor) {
      matchables[index_descriptor] = new Matchable(index_descriptor, descriptors_cv_.row(index_descriptor), identifier_tree_);
    }
    return matchables;
  }

  //ds creates a matchable vector (pointers) from opencv descriptors - only available if OpenCV is present on building system
  template<typename PointerType_>
  static const MatchableVector getMatchablesWithPointer(const cv::Mat& descriptors_cv_, const std::vector<PointerType_>& pointers_, const uint64_t& identifier_tree_) {
    MatchableVector matchables(descriptors_cv_.rows);

    //ds copy raw data
    for (int64_t index_descriptor = 0; index_descriptor < descriptors_cv_.rows; ++index_descriptor) {
      matchables[index_descriptor] = new Matchable(reinterpret_cast<const void*>(&pointers_[index_descriptor]), descriptors_cv_.row(index_descriptor), identifier_tree_);
    }
    return matchables;
  }

#endif

  //! @brief free all tree nodes (destructor)
  void clearNodes() {

    //ds nodes holder
    std::vector<const Node*> nodes_collection;

    //ds set vector
    _setNodesRecursive(_root, nodes_collection);

    //ds free nodes
    for (const Node* node: nodes_collection) {
      delete node;
    }
    nodes_collection.clear();
  }

  //! @brief free all matchables contained in the tree (destructor)
  void clearMatchables() {
    for (const BinaryMatchable256b* matchable: _matchables) {
      delete matchable;
    }
    _matchables.clear();
  }

//ds helpers
protected:

  //! @brief recursively computes all nodes in the tree - used for memory management only (destructor)
  //! @param[in] node_ the previous node from which the function was spawned
  //! @param[in] nodes_collection_ all nodes collected so far
  void _setNodesRecursive(const Node* node_, std::vector<const Node*>& nodes_collection_) const {

    //ds must not be zero
    assert(0 != node_);

    //ds add the current node
    nodes_collection_.push_back(node_);

    //ds check if there are leafs
    if(node_->has_leaves) {

      //ds add leafs and so on
      _setNodesRecursive(static_cast<const Node*>(node_->leaf_ones), nodes_collection_);
      _setNodesRecursive(static_cast<const Node*>(node_->leaf_zeros), nodes_collection_);
    }
  }
};

typedef BinaryTree<BinaryNode512b> BinaryTree512b;
typedef BinaryTree<BinaryNode256b> BinaryTree256b;
typedef BinaryTree<BinaryNode128b> BinaryTree128b;
}
