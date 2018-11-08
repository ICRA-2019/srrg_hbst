#pragma once
#include <cmath>
#include <random>
#include "binary_match.hpp"

namespace srrg_hbst {

  //! @brief leaf spawning modes
  enum SplittingStrategy {
    DoNothing,
    SplitEven,
    SplitUneven,
    SplitRandomUniform
  };

template<typename BinaryMatchableType_, typename real_type_ = double>
class BinaryNode
{

  //ds readability
  using Node = BinaryNode<BinaryMatchableType_, real_type_>;

//ds template forwarding
public:

  typedef Node BaseNode;
  typedef BinaryMatchableType_ Matchable;
  typedef std::vector<Matchable*> MatchableVector;
  typedef typename Matchable::Descriptor Descriptor;
  typedef real_type_ real_type;
  typedef BinaryMatch<Matchable, real_type> Match;

//ds ctor/dtor
public:

  //ds access only through this constructor: no mask provided
  BinaryNode(const MatchableVector& matchables_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven): Node(0, 0, matchables_, Descriptor().set(), train_mode_) {}

  //ds access only through this constructor: mask provided
  BinaryNode(const MatchableVector& matchables_,
             Descriptor bit_mask_,
             const SplittingStrategy& train_mode_ = SplittingStrategy::SplitEven): Node(0, 0, matchables_, bit_mask_, train_mode_) {}

  //ds the default constructor is triggered by subclasses - the responsibility of attribute initialization is left to the subclass
  //ds this is required, since we do not want to trigger the automatic leaf spawning of the baseclass in a subclass
  BinaryNode() {}

  //ds destructor: recursive destruction of child nodes (risky but readable)
  virtual ~BinaryNode() {
    delete left;
    delete right;
  }

//ds access
public:

  //ds create leafs (external use intented)
  virtual const bool spawnLeafs(const SplittingStrategy& train_mode_) {
    assert(!has_leafs);

    //ds exit if maximum depth is reached
    if (depth == maximum_depth) {
      return false;
    }

    //ds exit if we have insufficient data
    if (matchables.size() < maximum_leaf_size) {
      return false;
    }

    //ds affirm initial situation
    index_split_bit         = -1;
    number_of_on_bits_total = 0;
    partitioning            = maximum_partitioning;

    //ds for balanced splitting
    switch (train_mode_) {
      case SplittingStrategy::SplitEven: {

        //ds we have to find the split for this node - scan all indices
        for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {

          //ds if this index is available in the mask
          if (bit_mask[bit_index]) {

            //ds temporary set bit count
            uint64_t number_of_set_bits = 0;

            //ds compute distance for this index (0.0 is perfect)
            const double partitioning_current = std::fabs(0.5-_getSetBitFraction(bit_index, matchables, number_of_set_bits));

            //ds if better
            if (partitioning_current < partitioning) {
              partitioning            = partitioning_current;
              number_of_on_bits_total = number_of_set_bits;
              index_split_bit         = bit_index;

              //ds finalize loop if maximum target is reached
              if (partitioning == 0) break;
            }
          }
        }
        break;
      }
      case SplittingStrategy::SplitUneven: {
        partitioning = 0;

        //ds we have to find the split for this node - scan all indices
        for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {

          //ds if this index is available in the mask
          if (bit_mask[bit_index]) {

            //ds temporary set bit count
            uint64_t number_of_set_bits = 0;

            //ds compute distance for this index (0.0 is perfect)
            const double partitioning_current = std::fabs(0.5-_getSetBitFraction(bit_index, matchables, number_of_set_bits));

            //ds if worse
            if (partitioning_current > partitioning) {
              partitioning            = partitioning_current;
              number_of_on_bits_total = number_of_set_bits;
              index_split_bit         = bit_index;

              //ds finalize loop if maximum target is reached
              if (partitioning == 0.5) break;
            }
          }
        }
        break;
      }
      case SplittingStrategy::SplitRandomUniform: {

        //ds compute available bits
        std::vector<uint32_t> available_bits;
        for (uint32_t bit_index = 0; bit_index < Matchable::descriptor_size_bits; ++bit_index) {

          //ds if this index is available in the mask
          if (bit_mask[bit_index]) {
            available_bits.push_back(bit_index);
          }
        }

        //ds if bits are available
        if (available_bits.size() > 0) {
          std::uniform_int_distribution<uint32_t> available_indices(0, available_bits.size()-1);

          //ds sample uniformly at random
          index_split_bit = available_bits[available_indices(Node::random_number_generator)];

          //ds compute distance for this index (0.0 is perfect)
          partitioning = std::fabs(0.5-_getSetBitFraction(index_split_bit, matchables, number_of_on_bits_total));
        }
        break;
      }
      default: {
        throw std::runtime_error("invalid leaf spawning mode");
      }
    }

    //ds if best was found and the partitioning is sufficient (0 to 0.5) - we can spawn leaves
    if (index_split_bit != -1 && partitioning < maximum_partitioning) {

      //ds enabled
      has_leafs = true;

      //ds get a mask copy
      Descriptor bit_mask_previous(bit_mask);

      //ds update mask for leafs
      bit_mask_previous[index_split_bit] = 0;

      //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
      MatchableVector matchables_ones(number_of_on_bits_total);
      MatchableVector matchables_zeros(matchables.size()-number_of_on_bits_total);

      //ds loop over all descriptors and assigning them to the new vectors based on bit status
      uint64_t index_ones  = 0;
      uint64_t index_zeros = 0;
      for (Matchable* matchable: matchables) {
        if (matchable->descriptor[index_split_bit]) {
          matchables_ones[index_ones] = matchable;
          ++index_ones;
        } else {
          matchables_zeros[index_zeros] = matchable;
          ++index_zeros;
        }
      }
      assert(matchables_ones.size() == index_ones);
      assert(matchables_zeros.size() == index_zeros);

      //ds if there are elements for leaves
      assert(0 < matchables_ones.size());
      right = new Node(this, depth+1, matchables_ones, bit_mask_previous, train_mode_);

      assert(0 < matchables_zeros.size());
      left = new Node(this, depth+1, matchables_zeros, bit_mask_previous, train_mode_);

      //ds success
      return true;
    } else {

      //ds failed to spawn leaf - terminate recursion
      return false;
    }
  }

//ds inner constructors (used for recursive tree building)
protected:

  //ds only internally called: default for single matchables
  BinaryNode(Node* parent_,
             const uint64_t& depth_,
             const MatchableVector& matchables_,
             Descriptor bit_mask_,
             const SplittingStrategy& train_mode_): depth(depth_),
                                                    matchables(matchables_),
                                                    bit_mask(bit_mask_),
                                                    parent(parent_) {
    spawnLeafs(train_mode_);
  }

//ds helpers
protected:

  const real_type _getSetBitFraction(const uint32_t& index_split_bit_,
                                     const MatchableVector& matchables_,
                                     uint64_t& number_of_set_bits_total_) const {
    assert(0 < matchables_.size());

    //ds count
    uint64_t number_of_set_bits = 0;

    //ds just add the bits up (a one counts automatically as one)
    for (const Matchable* matchable: matchables_) {
      number_of_set_bits += matchable->descriptor[index_split_bit_];
    }

    //ds set total
    number_of_set_bits_total_ = number_of_set_bits;
    assert(number_of_set_bits_total_ <= matchables_.size());

    //ds return ratio
    return (static_cast<real_type>(number_of_set_bits)/matchables_.size());
  }

//ds fields TODO encapsulate
public:

  //! @brief depth of this node (number of splits passed)
  uint64_t depth = 0;

  //! @brief matchables contained in this node
  MatchableVector matchables;

  //! @brief minimum number of matchables in a node before splitting
  static uint64_t maximum_leaf_size;

  //! @brief the split bit diving potential leafs of this node
  int32_t index_split_bit = -1;

  //! @brief number of bits with value on
  uint64_t number_of_on_bits_total = 0;

  //! @brief flag set if the current node has 2 leafs
  bool has_leafs = false;

  //! @brief achieved descriptor group partitioning using the index_split_bit
  real_type partitioning = 1;

  //! @brief maximum achieved descriptor group partitioning using the index_split_bit
  static real_type maximum_partitioning;

  //! @brief bit splitting mask considered before choosing index_split_bit
  Descriptor bit_mask;

  //! @brief leaf containing all unset bits
  Node* left = nullptr;

  //! @brief leaf containing all set bits
  Node* right = nullptr;

  //! @brief parent node (if any, for root:parent=0)
  Node* parent = nullptr;

  //! @brief maximum tree depth (leaf spwaning blocks if reached, default: descriptor dimension)
  static uint32_t maximum_depth;

  //ds random number generator, used for random splitting (for all nodes)
  static std::mt19937 random_number_generator;
};

//ds default configuration
template<typename BinaryMatchableType_, typename real_type_>
uint64_t BinaryNode<BinaryMatchableType_, real_type_>::maximum_leaf_size      = 100;
template<typename BinaryMatchableType_, typename real_type_>
real_type_ BinaryNode<BinaryMatchableType_, real_type_>::maximum_partitioning = 0.1;
template<typename BinaryMatchableType_, typename real_type_>
uint32_t BinaryNode<BinaryMatchableType_, real_type_>::maximum_depth          = BinaryMatchableType_::descriptor_size_bits;
template<typename BinaryMatchableType_, typename real_type_>
std::mt19937 BinaryNode<BinaryMatchableType_, real_type_>::random_number_generator;
}
