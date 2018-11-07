#pragma once
#include "binary_matchable.hpp"

namespace srrg_hbst {

//! @class elementary match object: inspired by opencv cv::DMatch (docs.opencv.org/trunk/d4/de0/classcv_1_1DMatch.html)
//! @param MatchableType_ matchable type (class) for the match
//! @param real_precision_ matching distance precision
template<typename BinaryMatchableType_, typename real_type_ = double>
struct BinaryMatch {

  typedef BinaryMatchableType_ Matchable;
  typedef typename Matchable::ObjectType ObjectType;
  typedef real_type_ real_type;

  //! @brief default constructor for an uninitialized match
  //! @returns an uninitialized match
  BinaryMatch(): matchable_query(nullptr),
                 matchable_reference(nullptr),
                 distance(0) {}

  //! @brief default constructor for an uninitialized match
  //! @returns a fully initialized match
  BinaryMatch(const Matchable* matchable_query_,
              const Matchable* matchable_reference_,
              ObjectType pointer_query_,
              ObjectType pointer_reference_,
              const real_type_& distance_): matchable_query(matchable_query_),
                                            matchable_reference(matchable_reference_),
                                            object_query(pointer_query_),
                                            object_reference(pointer_reference_),
                                            distance(distance_) {}

  //! @brief copy constructor
  //! @param[in] match_ binary match object to be copied from
  //! @returns a binary match copy of the match_
  BinaryMatch(const BinaryMatch& match_): matchable_query(match_.matchable_query),
                                          matchable_reference(match_.matchable_reference),
                                          object_query(match_.object_query),
                                          object_reference(match_.object_reference),
                                          distance(match_.distance) {}

  //! @brief default destructor: nothing to do
  ~BinaryMatch() {}

  //! @brief prohibit default construction
  //BinaryMatch() = delete; uncommented 2018-06-21

  //! @brief attributes
  const Matchable* matchable_query;
  const Matchable* matchable_reference;
  ObjectType object_query;
  ObjectType object_reference;
  real_type distance;
};
}
