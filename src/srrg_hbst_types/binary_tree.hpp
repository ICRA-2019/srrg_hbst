#pragma once
#include <memory>
#include "binary_node.hpp"

namespace srrg_hbst {

  template<typename BinaryNodeType_, uint32_t maximum_distance_hamming_ = 25>
  class BinaryTree
  {

  //ds template forwarding
  public:

    typedef typename BinaryNodeType_::Matchable Matchable;
    typedef typename BinaryNodeType_::MatchableVector MatchableVector;
    typedef typename BinaryNodeType_::Descriptor Descriptor;
    typedef typename BinaryNodeType_::Match Match;
    typedef std::vector<Match> MatchVector;

  //ds ctor/dtor
  public:

    //ds construct tree upon allocation on filtered descriptors
    BinaryTree(const uint64_t& identifier_,
               const MatchableVector& matchables_): identifier(identifier_),
                                                    root(new BinaryNodeType_(matchables_)) {
      assert(0 != root);
    }

    //ds construct tree upon allocation on filtered descriptors: with bit mask
    BinaryTree(const uint64_t& identifier_,
               const MatchableVector& matchables_,
               Descriptor bit_mask_): identifier(identifier_),
                                      root(new BinaryNodeType_(matchables_, bit_mask_)) {
      assert(0 != root);
    }

    //ds construct tree with fixed split order
    BinaryTree(const uint64_t& identifier_,
               const MatchableVector& matchables_,
               std::vector<uint32_t> split_order_): identifier(identifier_),
                                                    root(new BinaryNodeType_(matchables_, split_order_)) {
      assert(0 != root);
    }

#ifdef SRRG_HBST_HAS_OPENCV

    //ds wrapped constructors - only available if OpenCV is present on building system
    BinaryTree(const uint64_t& identifier_,
               const cv::Mat& matchables_): identifier(identifier_),
                                            root(new BinaryNodeType_(getMatchablesWithIndex(matchables_))) {
      assert(0 != root);
    }

    BinaryTree(const uint64_t& identifier_,
               const cv::Mat& matchables_,
               Descriptor bit_mask_): identifier(identifier_),
                                      root(new BinaryNodeType_(getMatchablesWithIndex(matchables_), bit_mask_)) {
      assert(0 != root);
    }

    BinaryTree(const uint64_t& identifier_,
               const cv::Mat& matchables_,
               std::vector<uint32_t> split_order_): identifier(identifier_),
                                                    root(new BinaryNodeType_(getMatchablesWithIndex(matchables_), split_order_)) {
      assert(0 != root);
    }

#endif

    //ds free all nodes in the tree
    virtual ~BinaryTree() {

      //ds erase all nodes (and descriptors)
      _displant();
    }

    //ds disable default construction
    BinaryTree() = delete;

  //ds control fields
  public:

    //ds tree ID
    const uint64_t identifier;

  protected:

    //ds root node
    BinaryNodeType_* root;

  //ds access (shared pointer wrappings)
  public:

    const uint64_t getNumberOfMatches(const std::shared_ptr<const MatchableVector> matchables_query_) const {
      return getNumberOfMatches(*matchables_query_);
    }

    const typename BinaryNodeType_::precision getMatchingRatio(const std::shared_ptr<const MatchableVector> matchables_query_) const {
      return static_cast<typename BinaryNodeType_::precision>(getNumberOfMatches(*matchables_query_))/matchables_query_->size();
    }

    const uint64_t getNumberOfMatchesLazyEvaluation(const std::shared_ptr<const MatchableVector> matchables_query_) const {
      return getNumberOfMatchesLazyEvaluation(*matchables_query_);
    }

    const typename BinaryNodeType_::precision getMatchingRatioLazyEvaluation(const std::shared_ptr<const MatchableVector> matchables_query_) const {
      return static_cast<typename BinaryNodeType_::precision>(getNumberOfMatchesLazyEvaluation(*matchables_query_))/matchables_query_->size();
    }

  //ds access
  public:

    const uint64_t getNumberOfMatches(const MatchableVector& matchables_query_) const {
      uint64_t number_of_matches = 0;

      //ds for each descriptor
      for (const Matchable* matchable_query: matchables_query_) {

        //ds traverse tree to find this descriptor
        const BinaryNodeType_* node_current = root;
        while (node_current) {

          //ds if this node has leaves (is splittable)
          if(node_current->has_leaves) {

            //ds check the split bit and go deeper
            if(matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_ones);
            } else {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_zeros);
            }
          } else {

            //ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference: node_current->matchables) {
              if (maximum_distance_hamming_ > matchable_query->distance(matchable_reference)) {
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

    const typename BinaryNodeType_::precision getMatchingRatio(const MatchableVector& matchables_query_) const {
      return static_cast<typename BinaryNodeType_::precision>(getNumberOfMatches(matchables_query_))/matchables_query_.size();
    }

    const uint64_t getNumberOfMatchesLazyEvaluation(const MatchableVector& matchables_query_) const {
      uint64_t number_of_matches = 0;

      //ds for each descriptor
      for (const Matchable* matchable_query: matchables_query_) {

        //ds traverse tree to find this descriptor
        const BinaryNodeType_* node_current = root;
        while (node_current) {

          //ds if this node has leaves (is splittable)
          if (node_current->has_leaves) {

            //ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_ones);
            } else {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_zeros);
            }
          }
          else
          {
            //ds check current descriptors in this node and exit
            if(maximum_distance_hamming_ > matchable_query->distance(node_current->matchables.front())) {
              ++number_of_matches;
            }
            break;
          }
        }
      }
      return number_of_matches;
    }

    const typename BinaryNodeType_::precision getMatchingRatioLazyEvaluation(const MatchableVector& matchables_query_) const {
      return static_cast<typename BinaryNodeType_::precision>(getNumberOfMatchesLazyEvaluation(matchables_query_))/matchables_query_.size();
    }

    //ds direct matching function on this tree
    virtual void match(const MatchableVector& matchables_query_, MatchVector& matches_) const {

      //ds for each descriptor
      for(const Matchable* matchable_query: matchables_query_) {

        //ds traverse tree to find this descriptor
        const BinaryNodeType_* node_current = root;
        while (node_current) {

          //ds if this node has leaves (is splittable)
          if (node_current->has_leaves) {

            //ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_ones);
            } else {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_zeros);
            }
          } else {

            //ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference: node_current->matchables) {
              if (maximum_distance_hamming_ > matchable_query->distance(matchable_reference)) {
                matches_.push_back(Match(matchable_query, matchable_reference, maximum_distance_hamming_));
                break;
              }
            }
            break;
          }
        }
      }
    }

    //ds return matches directly
    const std::shared_ptr<const MatchVector> getMatches(const std::shared_ptr<const MatchableVector> matchables_query_) const {

      //ds match vector to be filled
      MatchVector matches;

      //ds for each QUERY descriptor
      for(const Matchable* matchable_query: *matchables_query_) {

        //ds traverse tree to find this descriptor
        const BinaryNodeType_* node_current = root;
        while (node_current) {

          //ds if this node has leaves (is splittable)
          if (node_current->has_leaves) {

            //ds check the split bit and go deeper
            if (matchable_query->descriptor[node_current->index_split_bit]) {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_ones);
            } else {
              node_current = static_cast<const BinaryNodeType_*>(node_current->leaf_zeros);
            }
          } else {

            //ds check current descriptors in this node and exit
            for (const Matchable* matchable_reference: node_current->matchables) {
              if (maximum_distance_hamming_ > matchable_query->distance(matchable_reference)) {
                matches.push_back(Match(matchable_query, matchable_reference, maximum_distance_hamming_));
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

    //ds grow the tree
    void plant(const MatchableVector& matchables_) {

      //ds grow tree on root
      root = new BinaryNodeType_(matchables_);
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
    static const MatchableVector getMatchablesWithPointer(const cv::Mat& descriptors_cv_, const std::vector<Type>& pointers_) {
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
      std::vector<const BinaryNodeType_*> nodes_collection;

      //ds set vector
      _setNodesRecursive(root, nodes_collection);

      //ds free nodes
      for (const BinaryNodeType_* node: nodes_collection) {
        delete node;
      }

      //ds clear nodes vector
      nodes_collection.clear();
    }

    void _setNodesRecursive(const BinaryNodeType_* node_, std::vector<const BinaryNodeType_*>& nodes_collection_) const {

      //ds must not be zero
      assert(0 != node_);

      //ds add the current node
      nodes_collection_.push_back(node_);

      //ds check if there are leafs
      if(node_->has_leaves) {

        //ds add leafs and so on
        _setNodesRecursive(static_cast<const BinaryNodeType_*>(node_->leaf_ones), nodes_collection_);
        _setNodesRecursive(static_cast<const BinaryNodeType_*>(node_->leaf_zeros), nodes_collection_);
      }
    }
  };

  typedef BinaryTree<BinaryNode512b> BinaryTree512b;
  typedef BinaryTree<BinaryNode256b> BinaryTree256b;
  typedef BinaryTree<BinaryNode128b> BinaryTree128b;
}
