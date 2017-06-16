#pragma once
#include "binary_matchable.hpp"

namespace srrg_hbst {

  //ds elementary match object: inspired by opencv cv::DMatch (docs.opencv.org/trunk/d4/de0/classcv_1_1DMatch.html)
  template<typename MatchableType_, typename real_precision_ = double>
  struct BinaryMatch
  {
    BinaryMatch(const MatchableType_* query_,
                const MatchableType_* reference_,
                const real_precision_& distance_): identifier_query(query_->identifier),
                                                  identifier_reference(reference_->identifier),
                                                  pointer_query(query_->pointer),
                                                  pointer_reference(reference_->pointer),
                                                  distance(distance_) {}

    ~BinaryMatch(){}

    //ds disable default construction
    BinaryMatch() = delete;

    const uint64_t identifier_query;
    const uint64_t identifier_reference;
    const void* pointer_query;
    const void* pointer_reference;
    const real_precision_ distance;
  };
}
