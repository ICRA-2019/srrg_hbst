#pragma once
#include <stdint.h>
#include <bitset>
#include <assert.h>

//ds if opencv is present on building system
#ifdef SRRG_HBST_HAS_OPENCV
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#endif

namespace srrg_hbst {

//! @class default matching object (wraps the input descriptors and more)
//! @param descriptor_size_bits_ number of bits for the native descriptor
template<uint32_t descriptor_size_bits_ = 256>
class BinaryMatchable {

//ds template forwarding
public:

  //! @brief descriptor type
  typedef std::bitset<descriptor_size_bits_> Descriptor;

//ds shared properties
public:

  //! @brief descriptor size in bits (for all matchables)
  static const uint32_t descriptor_size_bits = descriptor_size_bits_;

  //! @brief descriptor size in bytes (for all matchables)
  static const uint32_t descriptor_size_bytes = descriptor_size_bits_/8;

//ds ctor/dtor
public:

  //! @brief default constructor: DISABLED
  BinaryMatchable() = delete;

  //! @brief constructor with index for association
  //! @param[in] identifier_ association index
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier
  BinaryMatchable(const uint64_t& identifier_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(identifier_),
                                                                       pointer(0),
                                                                       descriptor(descriptor_),
                                                                       identifier_tree(identifier_tree_) {}

  //! @brief constructor with an object pointer for association
  //! @param[in] pointer_ associated object
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier
  BinaryMatchable(const void* pointer_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(0),
                                                                       pointer(pointer_),
                                                                       descriptor(descriptor_),
                                                                       identifier_tree(identifier_tree_) {}

  //! @brief constructor with index and object pointer for association
  //! @param[in] identifier_ association index
  //! @param[in] pointer_ associated object
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier
  BinaryMatchable(const uint64_t& identifier_,
                  const void* pointer_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(identifier_),
                                                                       pointer(pointer_),
                                                                       descriptor(descriptor_),
                                                                       identifier_tree(identifier_tree_) {}

//ds wrapped constructors - only available if OpenCV is present on building system
#ifdef SRRG_HBST_HAS_OPENCV

  BinaryMatchable(const uint64_t& identifier_,
                  const cv::Mat& descriptor_,
                  const uint64_t& identifier_tree_ = 0): BinaryMatchable(identifier_,
                                                         getDescriptor(descriptor_),
                                                         identifier_tree_) {}

  BinaryMatchable(const void* pointer_,
                  const cv::Mat& descriptor_,
                  const uint64_t& identifier_tree_ = 0): BinaryMatchable(pointer_,
                                                         getDescriptor(descriptor_),
                                                         identifier_tree_) {}

  BinaryMatchable(const uint64_t& identifier_,
                  const void* pointer_,
                  const cv::Mat& descriptor_,
                  const uint64_t& identifier_tree_ = 0): BinaryMatchable(identifier_,
                                                         pointer_,
                                                         getDescriptor(descriptor_),
                                                         identifier_tree_) {}

#endif

  //! @brief default destructor: nothing to do
  ~BinaryMatchable() {}

//ds functionality
public:

  //! @brief computes the classic Hamming descriptor distance between this and another matchable
  //! @param[in] matchable_query_ the matchable to compare this against
  //! @returns the matching distance as integer
  inline const uint32_t distanceHamming(const BinaryMatchable<descriptor_size_bits_>* matchable_query_) const {
    return (matchable_query_->descriptor^this->descriptor).count();
  }

#ifdef SRRG_HBST_HAS_OPENCV

  //! @brief descriptor wrapping - only available if OpenCV is present on building system
  //! @param[in] descriptor_cv_ opencv descriptor to convert into HBST format
  static inline Descriptor getDescriptor(const cv::Mat& descriptor_cv_) {
    Descriptor binary_descriptor(descriptor_size_bits_);
    for (uint64_t byte_index = 0; byte_index < descriptor_size_bytes; ++byte_index) {

      //ds get minimal datafrom cv::mat
      const uchar value = descriptor_cv_.at<uchar>(byte_index);

      //ds get bitstring
      for (uint8_t v = 0; v < 8; ++v) {
        binary_descriptor[byte_index*8+v] = (value >> v) & 1;
      }
    }
    return binary_descriptor;
  }

#endif

//ds attributes
public:

  //! @brief unique descriptor identifier
  const uint64_t identifier;

  //! @brief a connected object for correspondence handling (optional) - when using this field one must ensure the permanence of the referenced object!
  const void* pointer;

  //! @brief descriptor data string vector
  const Descriptor descriptor;

  //! @brief tree reference index to which this matchable is belonging
  const uint64_t identifier_tree;
};

typedef BinaryMatchable<512> BinaryMatchable512b;
typedef BinaryMatchable<256> BinaryMatchable256b;
typedef BinaryMatchable<128> BinaryMatchable128b;
}
