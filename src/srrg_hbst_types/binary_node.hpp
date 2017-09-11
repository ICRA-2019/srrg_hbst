#pragma once
#include <vector>
#include <cmath>
#include "binary_match.hpp"

namespace srrg_hbst {

template<typename BinaryMatchableType_, typename real_type_ = double>
class BinaryNode
{

  //ds readability
  using Node = BinaryNode<BinaryMatchableType_, real_type_>;

//ds template forwarding
public:

  typedef BinaryMatchableType_ Matchable;
  typedef std::vector<const BinaryMatchableType_*> MatchableVector;
  typedef typename BinaryMatchableType_::Descriptor Descriptor;
  typedef BinaryMatch<BinaryMatchableType_, real_type_> Match;
  typedef real_type_ real_type;

//ds ctor/dtor
public:

  //ds access only through this constructor: no mask provided
  BinaryNode(const MatchableVector& matchables_): Node(0, matchables_, _getMaskClean()) {}

  //ds access only through this constructor: mask provided
  BinaryNode(const MatchableVector& matchables_, Descriptor bit_mask_): Node(0, matchables_, bit_mask_) {}

  //ds access only through this constructor: split order provided (bit mask ignored)
  BinaryNode(const MatchableVector& matchables_, std::vector<uint32_t> split_order_): Node(0, matchables_, split_order_) {}

  //ds the default constructor is triggered by subclasses - the responsibility of attribute initialization is left to the subclass
  //ds this is required, since we do not want to trigger the automatic leaf spawning of the baseclass in a subclass
  BinaryNode() {}

  //ds destructor: nothing to do (the leafs will be freed by the tree)
  virtual ~BinaryNode() {}

//ds access
public:

  //ds create leafs (external use intented)
  virtual const bool spawnLeafs() {

    //ds if there are at least 2 descriptors (minimal split)
    if (matchables.size() > 1) {
      assert(!has_leaves);

      //ds affirm initial situation
      index_split_bit          = -1;
      number_of_ones_total = 0;
      partitioning             = 1;

      //ds we have to find the split for this node - scan all index
      for (uint32_t bit_index = 0; bit_index < BinaryMatchableType_::descriptor_size_bits; ++bit_index) {

        //ds if this index is available in the mask
        if (bit_mask[bit_index]) {

          //ds temporary set bit count
          uint64_t number_of_set_bits = 0;

          //ds compute distance for this index (0.0 is perfect)
          const double partitioning_current = std::fabs(0.5-_getOnesFraction(bit_index, matchables, number_of_set_bits));

          //ds if better
          if (partitioning > partitioning_current) {
            partitioning  = partitioning_current;
            number_of_ones_total     = number_of_set_bits;
            index_split_bit = bit_index;

            //ds finalize loop if maximum target is reached
            if (partitioning == 0) break;
          }
        }
      }

      //ds if best was found - we can spawn leaves
      if (index_split_bit != -1) {

        //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
        if (0 < number_of_ones_total && 0.5 > partitioning) {

          //ds enabled
          has_leaves = true;

          //ds get a mask copy
          Descriptor bit_mask_previous(bit_mask);

          //ds update mask for leafs
          bit_mask_previous[index_split_bit] = 0;

          //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
          MatchableVector matchables_ones;
          matchables_ones.reserve(number_of_ones_total);
          MatchableVector matchables_zeros;
          matchables_zeros.reserve(matchables.size( )-number_of_ones_total);

          //ds loop over all descriptors and assigning them to the new vectors based on bit status
          for (const BinaryMatchableType_* matchable: matchables) {
            if ( matchable->descriptor[index_split_bit] ) {
              matchables_ones.push_back(matchable);
            } else {
              matchables_zeros.push_back(matchable);
            }
          }

          //ds if there are elements for leaves
          assert(0 < matchables_ones.size());
          leaf_ones = new Node(depth+1, matchables_ones, bit_mask_previous);

          assert(0 < matchables_zeros.size());
          leaf_zeros = new Node(depth+1, matchables_zeros, bit_mask_previous);

          //ds success
          return true;
        }
        else {
          return false;
        }
      }
      else {
        return false;
      }
    }
    else {
      return false;
    }
  }

  //ds create leafs following the set split order
  virtual const bool spawnLeafs(std::vector<uint32_t> split_order_) {

    //ds if there are at least 2 descriptors (minimal split)
    if (1 < matchables.size()) {
      assert(!has_leaves);

      //ds affirm initial situation
      index_split_bit          = -1;
      number_of_ones_total = 0;
      partitioning             = 1;

      //uint32_t uShift = 0;
      //uint32_t uBitOptimal = p_vecSplitOrder[uDepth];

      //ds try a selection of available bit splits
      for (uint32_t depth_trial = depth; depth_trial < 2*depth+1; ++depth_trial) {
        uint64_t number_of_ones_total_current = 0;

        //ds compute distance for this index (0.0 is perfect)
        const real_type partitioning_current = std::fabs(0.5-_getOnesFraction(split_order_[depth_trial], matchables, number_of_ones_total_current));

        if (partitioning > partitioning_current) {

          //ds buffer found bit
          const uint32_t split_bit_best = split_order_[depth_trial];

          //ds shift the last best index to the chosen depth in a later step
          split_order_[depth_trial] = index_split_bit;

          partitioning             = partitioning_current;
          number_of_ones_total = number_of_ones_total_current;
          index_split_bit          = split_bit_best;

          //ds update the split order vector to the current bit
          split_order_[depth] = split_bit_best;

          //uShift = uDepthTrial-uDepth;

          //ds finalize loop if maximum target is reached
          if (0 == partitioning) break;
        }
      }

      //ds if best was found - we can spawn leaves
      if (-1 != index_split_bit) {

        //ds check if we have enough data to split
        if(0 < number_of_ones_total && 0.5 > partitioning) {

          //ds enabled
          has_leaves = true;

          //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
          MatchableVector matchables_ones;
          matchables_ones.reserve(number_of_ones_total);
          MatchableVector matchable_zeros;
          matchable_zeros.reserve(matchables.size()-number_of_ones_total);

          //ds loop over all descriptors and assing them to the new vectors
          for (const BinaryMatchableType_* matchable: matchables) {
            if (matchable->descriptor[index_split_bit]) {
              matchables_ones.push_back(matchable);
            } else {
              matchable_zeros.push_back(matchable);
            }
          }

          //ds if there are elements for leaves
          assert(0 < matchables_ones.size());
          leaf_ones = new Node(depth+1, matchables_ones, split_order_);

          assert(0 < matchable_zeros.size());
          leaf_zeros = new Node(depth+1, matchable_zeros, split_order_);

          //ds worked
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

//ds inner constructors (used for recursive tree building)
protected:

  //ds only internally called: without split order
  BinaryNode(const uint64_t& depth_,
             const MatchableVector& matchables_,
             Descriptor bit_mask_): depth(depth_), matchables(matchables_), has_leaves(false), bit_mask(bit_mask_), leaf_ones(0), leaf_zeros(0) {

    //ds call recursive leaf spawner
    spawnLeafs();
  }

  //ds only internally called: with split order (bit mask ignored)
  BinaryNode(const uint64_t& depth_,
             const MatchableVector& matchables_,
             std::vector< uint32_t > split_order_): depth(depth_), matchables(matchables_), has_leaves(false), leaf_ones(0), leaf_zeros(0) {

    //ds call recursive leaf spawner
    spawnLeafs(split_order_);
  }

//ds fields
public:

  //! @brief depth of this node (number of splits passed)
  uint64_t depth;

  //! @brief matchables contained in this node
  MatchableVector matchables;

  //! @brief the split bit diving potential leafs of this node
  int32_t index_split_bit;

  //! @brief number of bits with value one
  uint64_t number_of_ones_total;

  //! @brief flag set if the current node has 2 leaves
  bool has_leaves;

  //! @brief achieved descriptor group partitioning using the index_split_bit
  real_type partitioning;

  //! @brief bit splitting mask considered before choosing index_split_bit
  Descriptor bit_mask;

  //! @brief leaf containing all one bits
  Node* leaf_ones;

  //! @brief leaf containing all zero bits
  Node* leaf_zeros;

//ds helpers
protected:

  //ds helpers
  const real_type _getOnesFraction(const uint32_t& index_split_bit_, const MatchableVector& matchables_, uint64_t& number_of_ones_total_) const {
    assert(0 < matchables_.size());

    //ds count
    uint64_t number_of_ones = 0;

    //ds just add the bits up (a one counts automatically as one)
    for (const BinaryMatchableType_* matchable: matchables_) {
        number_of_ones += matchable->descriptor[index_split_bit_];
    }

    //ds set total
    number_of_ones_total_ = number_of_ones;
    assert(number_of_ones_total_ <= matchables_.size());

    //ds return ratio
    return (static_cast<real_type>(number_of_ones)/matchables_.size());
  }

  //ds returns a bitset with all bits set to true
  Descriptor _getMaskClean( ) const {
    Descriptor bit_mask;
    bit_mask.set( );
    return bit_mask;
  }
};

typedef BinaryNode<BinaryMatchable512b> BinaryNode512b;
typedef BinaryNode<BinaryMatchable256b> BinaryNode256b;
typedef BinaryNode<BinaryMatchable128b> BinaryNode128b;
}
