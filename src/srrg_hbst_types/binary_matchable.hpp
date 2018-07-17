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

  //! @brief descriptor type (extended by augmented bits, no effect if zero)
  typedef std::bitset<descriptor_size_bits_> Descriptor;

//ds shared properties
public:

  //! @brief descriptor size in bits (for all matchables)
  static constexpr uint32_t descriptor_size_bits = descriptor_size_bits_;

  //! @brief descriptor size in bytes (for all matchables, size after the number of augmented bits)
  static constexpr uint32_t raw_descriptor_size_bytes = descriptor_size_bits_/8;

//ds ctor/dtor
public:

  //! @brief default constructor: DISABLED
  BinaryMatchable() = delete;

  //! @brief constructor with index for association
  //! @param[in] identifier_ association index
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier (optional)
  BinaryMatchable(const uint64_t& identifier_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(identifier_),
                                                         pointer(0),
                                                         descriptor(descriptor_),
                                                         identifier_reference(identifier_tree_) {}

  //! @brief constructor with an object pointer for association
  //! @param[in] pointer_ associated object
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier (optional)
  BinaryMatchable(const void* pointer_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(0),
                                                         pointer(pointer_),
                                                         descriptor(descriptor_),
                                                         identifier_reference(identifier_tree_) {}

  //! @brief constructor with index and object pointer for association
  //! @param[in] identifier_ association index
  //! @param[in] pointer_ associated object
  //! @param[in] descriptor_ HBST descriptor
  //! @param[in] identifier_tree_ HBST tree identifier (optional)
  //! @param[in] augmentation_ HBST augmentation vector (optional)
  BinaryMatchable(const uint64_t& identifier_,
                  const void* pointer_,
                  const Descriptor& descriptor_,
                  const uint64_t& identifier_tree_ = 0): identifier(identifier_),
                                                         pointer(pointer_),
                                                         descriptor(descriptor_),
                                                         identifier_reference(identifier_tree_) {}

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

    //ds buffer
    Descriptor binary_descriptor(descriptor_size_bits_);

    //ds set original descriptor string after augmentation
    for (uint64_t byte_index = 0; byte_index < raw_descriptor_size_bytes; ++byte_index) {
      const uint32_t bit_index_start = byte_index*8;

      //ds grab a byte and convert it to a bitset so we can access the single bits
      const std::bitset<8>descriptor_byte(descriptor_cv_.at<uchar>(byte_index));

      //ds set bitstring
      for (uint8_t v = 0; v < 8; ++v) {
        binary_descriptor[bit_index_start+v] = descriptor_byte[v];
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

  //! @brief image reference index to which this matchable is belonging
  const uint64_t identifier_reference;
};

typedef BinaryMatchable<512> BinaryMatchable512;
typedef BinaryMatchable<256> BinaryMatchable256;
typedef BinaryMatchable<128> BinaryMatchable128;
}
