#pragma once
#include "binary_matchable.hpp"

namespace srrg_hbst {

//! @class elementary match object: inspired by opencv cv::DMatch (docs.opencv.org/trunk/d4/de0/classcv_1_1DMatch.html)
//! @param MatchableType_ matchable type (class) for the match
//! @param real_precision_ matching distance precision
template<typename MatchableType_, typename real_type_ = double>
struct BinaryMatch {

  //! @brief constructor
  //! @param[in] query_ query matchable
  //! @param[in] reference_ reference matchable
  //! @param[in] distance_ Hamming distance for the given association
  //! @returns a match instance
  BinaryMatch(const MatchableType_* query_,
              const MatchableType_* reference_,
              const real_type_& distance_): matchable_query(query_),
                                            matchable_reference(reference_),
                                            identifier_query(query_->identifier),
                                            identifier_reference(reference_->identifier),
                                            pointer_query(query_->pointer),
                                            pointer_reference(reference_->pointer),
                                            distance(distance_) {}

  //! @brief copy constructor
  //! @param[in] match_ binary match object to be copied from
  //! @returns a binary match copy of the match_
  BinaryMatch(const BinaryMatch& match_): matchable_query(match_.matchable_query),
                                          matchable_reference(match_.matchable_reference),
                                          identifier_query(match_.identifier_query),
                                          identifier_reference(match_.identifier_reference),
                                          pointer_query(match_.pointer_query),
                                          pointer_reference(match_.pointer_reference),
                                          distance(match_.distance) {}

  //! @brief default destructor: nothing to do
  ~BinaryMatch() {}

  //! @brief prohibit default construction
  BinaryMatch() = delete;

  //! @brief attributes
  const MatchableType_* matchable_query;
  const MatchableType_* matchable_reference;
  const uint64_t identifier_query;
  const uint64_t identifier_reference;
  const void* pointer_query;
  const void* pointer_reference;
  const real_type_ distance;
};
}
