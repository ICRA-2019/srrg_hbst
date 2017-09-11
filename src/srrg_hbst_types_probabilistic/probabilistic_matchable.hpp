#pragma once
#include <Eigen/Core>
#include "srrg_hbst_types/binary_matchable.hpp"

namespace srrg_hbst {

template<uint32_t descriptor_size_bits_ = 256, typename real_precision_ = double>
class ProbabilisticMatchable: public BinaryMatchable< descriptor_size_bits_>
{

//ds template forwarding
public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename BinaryMatchable<descriptor_size_bits_>::Descriptor Descriptor;
  typedef Eigen::Matrix<real_precision_, descriptor_size_bits_, 1> BitStatisticsVector;

//ds ctor/dtor
public:

  ProbabilisticMatchable(const uint64_t& identifier_,
                         const Descriptor& descriptor_,
                         const BitStatisticsVector& bit_probabilities_,
                         const BitStatisticsVector& bit_volatility_ ): BinaryMatchable<descriptor_size_bits_>(identifier_, descriptor_),
                                                                       bit_probabilities(bit_probabilities_),
                                                                       bit_volatility(bit_volatility_) {}

  ~ProbabilisticMatchable() {}

//ds attributes
public:

  //ds statistical data: bit probabilities
  const BitStatisticsVector bit_probabilities;

  //ds statistical data: bit volatity info, maximum number of appearances where the bit was stable - currently not used
  const BitStatisticsVector bit_volatility;
};
}
