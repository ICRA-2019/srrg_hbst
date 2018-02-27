#pragma once
#include <memory>
#include "binary_node.hpp"

namespace srrg_hbst {

//ds DEVELOPMENT ONLY TODO purge
class ViewerBonsai;

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
    bool spawn_leafs;
  };

//ds ctor/dtor
public:

  //ds empty tree instantiation
  BinaryTree(): identifier(0),
                _root(0) {
    _matchables.clear();
    _matchables_to_train.clear();
    _added_identifiers_train.clear();
    _trainables.clear();
  }

  //ds empty tree instantiation with specific identifier
  BinaryTree(const uint64_t& identifier_): identifier(identifier_),
                                           _root(0) {
    _matchables.clear();
    _matchables_to_train.clear();
    _added_identifiers_train.clear();
    _trainables.clear();
  }

  //ds construct tree upon allocation on filtered descriptors
  BinaryTree(const MatchableVector& matchables_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven): identifier(0),
                                                                     _root(new Node(matchables_, train_mode_)) {
    _matchables.clear();
    _matchables_to_train.clear();
    _matchables_to_train.insert(_matchables_to_train.end(), matchables_.begin(), matchables_.end());
    _added_identifiers_train.clear();
    _added_identifiers_train.insert(identifier);
    _trainables.clear();
  }

  //ds construct tree upon allocation on filtered descriptors
  BinaryTree(const uint64_t& identifier_,
             const MatchableVector& matchables_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven): identifier(identifier_),
                                                                     _root(new Node(matchables_, train_mode_)) {
    _matchables.clear();
    _matchables_to_train.clear();
    _matchables_to_train.insert(_matchables_to_train.end(), matchables_.begin(), matchables_.end());
    _added_identifiers_train.clear();
    _added_identifiers_train.insert(identifier);
    _trainables.clear();
  }

  //ds construct tree upon allocation on filtered descriptors: with bit mask
  BinaryTree(const uint64_t& identifier_,
             const MatchableVector& matchables_,
             Descriptor bit_mask_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven): identifier(identifier_),
                                                                     _root(new Node(matchables_, bit_mask_, train_mode_)) {
    _matchables.clear();
    _matchables_to_train.clear();
    _matchables_to_train.insert(_matchables_to_train.end(), matchables_.begin(), matchables_.end());
    _added_identifiers_train.clear();
    _added_identifiers_train.insert(identifier);
    _trainables.clear();
  }

#ifdef SRRG_HBST_HAS_OPENCV

  //ds wrapped constructors - only available if OpenCV is present on building system
  BinaryTree(const cv::Mat& matchables_): BinaryTree(getMatchablesWithIndex(matchables_)) {}
  BinaryTree(const uint64_t& identifier_,
             const cv::Mat& matchables_): BinaryTree(identifier_, getMatchablesWithIndex(matchables_)) {}
  BinaryTree(const uint64_t& identifier_,
             const cv::Mat& matchables_,
             Descriptor bit_mask_): identifier(identifier_, getMatchablesWithIndex(matchables_), bit_mask_) {}

#endif

  //ds free all nodes in the tree
  virtual ~BinaryTree() {
    clear();
  }

//ds accessible attributes
public:

  //! @brief identifier for this tree
  const uint64_t identifier = 0;

  //! @brief const accessor to root node
  const Node* root() const {return _root;}

#ifdef ENABLE_SRRG_HBST_ANALYTICS

  //! @brief 1-number of new leaves/number of train descriptors
  double current_aggregation_ratio = 0;

#endif

//ds attributes
protected:

  //! @brief root node (e.g. starting point for similarity search)
  Node* _root = 0;

  //! @brief bookkeeping: all matchables contained in the tree
  MatchableVector _matchables;
  MatchableVector _matchables_to_train;

  //! @brief bookkeeping: integrated matchable train identifiers (unique)
  std::set<uint64_t> _added_identifiers_train;

  //! @brief bookkeeping: trainable matchables resulting from last matchAndAdd call
  std::vector<Trainable> _trainables;

//ds access (shared pointer wrappings)
public:

  const uint64_t getNumberOfMatches(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return getNumberOfMatches(*matchables_query_);
  }

  const typename Node::real_type getMatchingRatio(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatches(*matchables_query_))/matchables_query_->size();
  }

  const uint64_t getNumberOfMatchesLazy(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return getNumberOfMatchesLazy(*matchables_query_);
  }

  const typename Node::real_type getMatchingRatioLazy(const std::shared_ptr<const MatchableVector> matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatchesLazy(*matchables_query_))/matchables_query_->size();
  }

//ds access
public:

  //! @brief train tree with current _trainable_matchables according to selected mode
  //! @param[in] train_mode_ desired training mode
  virtual void train(const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {

    //ds if random splitting is chosen
    if (train_mode_ == SplittingStrategy::SplitRandomUniform) {

      //ds initialize random number generator with new seed
      std::random_device random_device;
      Node::random_number_generator = std::mt19937(random_device());
    }

    //ds return for no training (redundant call)
    if (train_mode_ == SplittingStrategy::DoNothing) {
      return;
    }

#ifdef ENABLE_SRRG_HBST_ANALYTICS
    uint64_t number_of_new_leafs = 0;
#endif

    //ds check if we have to build an initial tree first (no training afterwards)
    if (!_root) {
       _root = new Node(_matchables_to_train, train_mode_);
      _matchables.insert(_matchables.end(), _root->matchables.begin(), _root->matchables.end());
      _matchables_to_train.clear();
#ifdef ENABLE_SRRG_HBST_ANALYTICS
      current_aggregation_ratio = 1;
#endif
      return;
    }

    //ds nodes to update after the addition of matchables to leafs
    std::set<Node*> nodes_to_update;

    //ds for each new descriptor
    for (const Matchable* matchable_to_insert: _matchables_to_train) {

      //ds traverse tree to find a leaf for this descriptor
      Node* node = _root;
      while (node) {

        //ds if this node has leaves (is splittable)
        if (node->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_to_insert->descriptor[node->index_split_bit]) {
            node = node->right;
          } else {
            node = node->left;
          }
        } else {

          //ds place the descriptor into the leaf
          node->matchables.push_back(matchable_to_insert);
          nodes_to_update.insert(node);
          break;
        }
      }
    }

    //ds spawn leaves if requested
    for (Node* node: nodes_to_update) {
      if (node->spawnLeafs(train_mode_)) {
#ifdef ENABLE_SRRG_HBST_ANALYTICS
        ++number_of_new_leafs;
#endif
      }
    }

#ifdef ENABLE_SRRG_HBST_ANALYTICS
    current_aggregation_ratio = 1-static_cast<double>(number_of_new_leafs)/_matchables_to_train.size();
#endif
    _matchables.insert(_matchables.end(), _matchables_to_train.begin(), _matchables_to_train.end());
    _matchables_to_train.clear();
  }

  const uint64_t getNumberOfMatches(const MatchableVector& matchables_query_, const uint32_t& maximum_distance_ = 25) const {
    uint64_t number_of_matches = 0;

    //ds for each descriptor
    for (const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if(node_current->has_leafs) {

          //ds check the split bit and go deeper
          if(matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
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

  const uint64_t getNumberOfMatchesLazy(const MatchableVector& matchables_query_, const uint32_t& maximum_distance_ = 25) const {
    uint64_t number_of_matches = 0;

    //ds for each descriptor
    for (const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
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

  const typename Node::real_type getMatchingRatioLazy(const MatchableVector& matchables_query_) const {
    return static_cast<typename Node::real_type>(getNumberOfMatchesLazyEvaluation(matchables_query_))/matchables_query_.size();
  }

  //ds direct matching function on this tree
  virtual void matchLazy(const MatchableVector& matchables_query_, MatchVector& matches_, const uint32_t& maximum_distance_ = 25) const {

    //ds for each descriptor
    for(const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
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

  //ds direct matching function on this tree
  virtual void match(const MatchableVector& matchables_query_,
                     MatchVector& matches_,
                     const uint32_t& maximum_distance_ = 25) const {

    //ds for each descriptor
    for(const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
          }
        } else {

          //ds current best (0 if none)
          const Matchable* matchable_reference_best = 0;
          uint32_t distance_best                    = maximum_distance_;

          //ds check current descriptors in this node and exit
          for (const Matchable* matchable_reference: node_current->matchables) {
            const uint32_t& distance = matchable_query->distanceHamming(matchable_reference);
            if (distance < distance_best) {
              matchable_reference_best = matchable_reference;
              distance_best = distance;
            }
          }

          //ds if a match was found
          if (matchable_reference_best) {
            matches_.push_back(Match(matchable_query, matchable_reference_best, maximum_distance_));
          }
          break;
        }
      }
    }
  }

  //ds return matches directly
  const std::shared_ptr<const MatchVector> getMatchesLazy(const std::shared_ptr<const MatchableVector> matchables_query_, const uint32_t& maximum_distance_ = 25) const {

    //ds wrap the call
    MatchVector matches;
    matchLazy(*matchables_query_, matches, maximum_distance_);
    return std::make_shared<const MatchVector>(matches);
  }

  //! @brief knn multi-matching function
  //! @param[in] matchables_query_ query matchables
  //! @param[out] matches_ output matching results: contains all available matches for all training images added to the tree
  //! @param[in] maximum_distance_ the maximum distance allowed for a positive match response
  virtual void match(const MatchableVector& matchables_query_,
                     MatchVectorMap& matches_,
                     const uint32_t& maximum_distance_matching_ = 25) const {

    //ds prepare match vector map for all ids in the tree
    matches_.clear();
    for (const uint64_t identifier_tree: _added_identifiers_train) {
      matches_.insert(std::make_pair(identifier_tree, MatchVector()));

      //ds preallocate space to speed up match addition
      matches_.at(identifier_tree).reserve(matchables_query_.size());
    }

    //ds for each descriptor
    for(const Matchable* matchable_query: matchables_query_) {

      //ds traverse tree to find this descriptor
      const Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
          }
        } else {

          //ds obtain best matches in the current leaf via brute-force search
          std::map<uint64_t, Match> best_matches;
          _matchExhaustive(matchable_query, node_current->matchables, maximum_distance_matching_, best_matches);

          //ds register all matches in the output structure
          for (const std::pair<uint64_t, Match> best_match: best_matches) {
            matches_.at(best_match.first).push_back(best_match.second);
          }
          break;
        }
      }
    }
  }

  //! @brief incrementally grows the tree
  //! @param[in] matchables_ new input matchables to integrate into the current tree (transferring the ownership!)
  //! @param[in] train_mode_ train_mode_
  void add(const MatchableVector& matchables_,
           const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {

    //ds store arguments
    _added_identifiers_train.insert(matchables_.front()->identifier_tree);
    _matchables_to_train.insert(_matchables_to_train.end(), matchables_.begin(), matchables_.end());

    //ds train based on set matchables (no effect for do nothing train mode)
    train(train_mode_);
  }

  //! @brief knn multi-matching function with simultaneous adding
  //! @param[in] matchables_ query matchables, which will also automatically be added to the tree (transferring the ownership!)
  //! @param[out] matches_ output matching results: contains all available matches for all training images added to the tree
  //! @param[in] maximum_distance_matching_ the maximum distance allowed for a positive match response
  virtual void matchAndAdd(const MatchableVector& matchables_,
                           MatchVectorMap& matches_,
                           const uint32_t maximum_distance_matching_ = 25,
                           const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {

    //ds check if we have to build an initial tree first
    if (!_root) {
      _root = new Node(matchables_);
      _matchables.insert(_matchables.end(), _root->matchables.begin(), _root->matchables.end());
      _added_identifiers_train.insert(matchables_.front()->identifier_tree);
#ifdef ENABLE_SRRG_HBST_ANALYTICS
      current_aggregation_ratio = 1;
#endif
      return;
    }

    //ds prepare match vector map for all ids in the tree
    matches_.clear();
    for (const uint64_t identifier_tree: _added_identifiers_train) {
      matches_.insert(std::make_pair(identifier_tree, MatchVector()));

      //ds preallocate space to speed up match addition
      matches_.at(identifier_tree).reserve(matchables_.size());
    }

    //ds prepare node/matchable list to integrate
    _trainables.resize(matchables_.size());

    //ds for each descriptor
    for(uint64_t index_trainable = 0; index_trainable < _trainables.size(); ++index_trainable) {
      const Matchable* matchable_query  = matchables_[index_trainable];
      _trainables[index_trainable].node = 0;

      //ds traverse tree to find this descriptor
      Node* node_current = _root;
      while (node_current) {

        //ds if this node has leaves (is splittable)
        if (node_current->has_leafs) {

          //ds check the split bit and go deeper
          if (matchable_query->descriptor[node_current->index_split_bit]) {
            node_current = node_current->right;
          } else {
            node_current = node_current->left;
          }
        } else {

          //ds obtain best matches in the current leaf via brute-force search
          std::map<uint64_t, Match> best_matches;
          _matchExhaustive(matchable_query, node_current->matchables, maximum_distance_matching_, best_matches);

          //ds register all matches in the output structure
          for (const std::pair<uint64_t, Match> best_match: best_matches) {
            matches_.at(best_match.first).push_back(best_match.second);
          }

          //ds bookkeep matchable for addition
          _trainables[index_trainable].node        = node_current;
          _trainables[index_trainable].matchable   = matchable_query;

          //ds done
          break;
        }
      }
    }

    //ds perform matchable addition and spawn leaves if requested
    std::set<Node*> nodes_to_update;
    for (const Trainable& trainable: _trainables) {
      if (trainable.node) {
        trainable.node->matchables.push_back(trainable.matchable);
        nodes_to_update.insert(trainable.node);
      }
    }
    for (Node* node: nodes_to_update) {
      node->spawnLeafs(train_mode_);
    }

    //ds insert identifier of the integrated matchables
    _added_identifiers_train.insert(matchables_.front()->identifier_tree);
    _matchables.insert(_matchables.end(), matchables_.begin(), matchables_.end());
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

  //ds wrapped
  virtual void add(const cv::Mat& matchables_train_,
                   const uint64_t& identifier_tree_) {
    add(getMatchablesWithIndex(matchables_train_, identifier_tree_));
  }

  //ds wrapped
  virtual void match(const cv::Mat& matchables_query_,
                     const uint64_t& identifier_tree_,
                     MatchVectorMap& matches_,
                     const uint32_t maximum_distance_matching_ = 25,
                     const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {
    match(getMatchablesWithIndex(matchables_query_, identifier_tree_), matches_, maximum_distance_matching_);
  }

  //ds wrapped
  virtual void matchAndAdd(const cv::Mat& matchables_query_,
                           const uint64_t& identifier_tree_,
                           MatchVectorMap& matches_,
                           const uint32_t maximum_distance_matching_ = 25,
                           const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven) {
    matchAndAdd(getMatchablesWithIndex(matchables_query_, identifier_tree_), matches_, maximum_distance_matching_, train_mode_);
  }

#endif

  //! @brief clears complete structure (corresponds to empty construction)
  virtual void clear(const bool& delete_matchables_ = true) {

    //ds own structures
    clearNodes();
    _added_identifiers_train.clear();
    _trainables.clear();
    _root = 0;

    //ds ownership dependent
    if (delete_matchables_) {
      deleteMatchables();
    }
    _matchables.clear();
    _matchables_to_train.clear();
  }

  //! @brief free all tree nodes (destructor)
  virtual void clearNodes() {
    if (!_root) {return;}

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
  virtual void deleteMatchables() {
    for (const Matchable* matchable: _matchables) {
      delete matchable;
    }
    for (const Matchable* matchable: _matchables_to_train) {
      delete matchable;
    }
  }

  //! @brief recursively counts all leafs and descriptors stored in the tree (expensive)
  //! @param[out] number_of_leafs_
  //! @param[out] number_of_matchables_
  //! @param[out] maximum_depth_
  //! @param[in] starting node (only subtree will be evaluated)
  void count(uint64_t& number_of_leafs_,
             uint64_t& number_of_matchables_,
             std::vector<uint32_t>& leaf_depths_,
             Node* node_ = 0) {

    //ds call on empty tree
    if (_root == 0) {
      number_of_leafs_      = 0;
      number_of_matchables_ = 0;
      return;
    }

    //ds start from root if not specified otherwise
    if (node_ == 0) {
      node_ = _root;
    }

    //ds if we are in a leaf
    if (!node_->left) {

      //ds update statistics and terminate
      ++number_of_leafs_;
      number_of_matchables_ += node_->matchables.size();
      leaf_depths_.push_back(node_->depth);
    } else {

      //ds look deeper on left and right subtree
      count(number_of_leafs_, number_of_matchables_, leaf_depths_, node_->left);
      count(number_of_leafs_, number_of_matchables_, leaf_depths_, node_->right);
    }
  }

//ds helpers
protected:

  //! @brief recursively computes all nodes in the tree - used for memory deallocation only (destructor)
  //! @param[in] node_ the previous node from which the function was spawned
  //! @param[in] nodes_collection_ all nodes collected so far
  void _setNodesRecursive(const Node* node_, std::vector<const Node*>& nodes_collection_) const {

    //ds must not be zero
    assert(0 != node_);

    //ds add the current node
    nodes_collection_.push_back(node_);

    //ds check if there are leafs
    if(node_->has_leafs) {

      //ds add leafs and so on
      _setNodesRecursive(node_->left, nodes_collection_);
      _setNodesRecursive(node_->right, nodes_collection_);
    }
  }

  //! @brief retrieves best matches (BF search) for provided matchables for all image indices
  //! @param[in] matchable_query_
  //! @param[in] matchables_reference_
  //! @param[in] maximum_distance_matching_
  //! @param[in,out] best_matches_ best match search storage: image id, match candidate
  void _matchExhaustive(const Matchable* matchable_query_,
                        const MatchableVector& matchables_reference_,
                        const uint32_t& maximum_distance_matching_,
                        std::map<uint64_t, Match>& best_matches_) const {

    //ds check current descriptors in this node
    for (const Matchable* matchable_reference: matchables_reference_) {
      const uint64_t& identifer_tree_reference = matchable_reference->identifier_tree;

      //ds compute the descriptor distance
      const uint32_t distance = matchable_query_->distanceHamming(matchable_reference);

      //ds if matching distance is within the threshold
      if (distance < maximum_distance_matching_) {

        //ds check if there already is a match for this identifier
        try {

          //ds update match if current is better
          if (distance < best_matches_.at(identifer_tree_reference).distance) {
            best_matches_.at(identifer_tree_reference).matchable_reference  = matchable_reference;
            best_matches_.at(identifer_tree_reference).identifier_reference = matchable_reference->identifier;
            best_matches_.at(identifer_tree_reference).pointer_reference    = matchable_reference->pointer;
            best_matches_.at(identifer_tree_reference).distance             = distance;
          }
        } catch (const std::out_of_range& /*exception*/) {

          //ds add a new match
          best_matches_.insert(std::make_pair(identifer_tree_reference, Match(matchable_query_, matchable_reference, distance)));
        }
      }
    }
  }

  //ds DEVELOPMENT ONLY TODO purge
  friend ViewerBonsai;
};

typedef BinaryTree<BinaryNode512> BinaryTree512;
typedef BinaryTree<BinaryNode256> BinaryTree256;
typedef BinaryTree<BinaryNode128> BinaryTree128;
}
