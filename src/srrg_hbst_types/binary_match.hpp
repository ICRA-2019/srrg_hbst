#pragma once
#include "binary_matchable.hpp"

namespace srrg_hbst {

//! @class elementary match object: inspired by opencv cv::DMatch (docs.opencv.org/trunk/d4/de0/classcv_1_1DMatch.html)
//! @param MatchableType_ matchable type (class) for the match
//! @param real_precision_ matching distance precision
template<typename MatchableType_, typename real_type_ = double>
struct BinaryMatch {

  //! @brief default constructor for an uninitialized match
  //! @returns an uninitialized match
  BinaryMatch(): matchable_query(nullptr),
                 matchable_reference(nullptr),
                 identifier_query(0),
                 identifier_reference(0),
                 pointer_query(nullptr),
                 pointer_reference(nullptr),
                 distance(0) {}

  //! @brief default constructor for an uninitialized match
  //! @returns a fully initialized match
  BinaryMatch(const MatchableType_* matchable_query_,
              const MatchableType_* matchable_reference_,
              const uint64_t& identifier_query_,
              const uint64_t& identifier_reference_,
              const void* pointer_query_,
              const void* pointer_reference_,
              const real_type_& distance_): matchable_query(matchable_query_),
                                            matchable_reference(matchable_reference_),
                                            identifier_query(identifier_query_),
                                            identifier_reference(identifier_reference_),
                                            pointer_query(pointer_query_),
                                            pointer_reference(pointer_reference_),
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
  //BinaryMatch() = delete; uncommented 2018-06-21

  //! @brief attributes
  const MatchableType_* matchable_query;
  const MatchableType_* matchable_reference;
  uint64_t identifier_query;
  uint64_t identifier_reference;
  const void* pointer_query;
  const void* pointer_reference;
  real_type_ distance;
};
}
