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

  template<uint32_t descriptor_size_bits_ = 256>
  class BinaryMatchable {

  //ds template forwarding
  public:

    typedef std::bitset<descriptor_size_bits_> Descriptor;

  //ds shared properties
  public:

    //ds descriptor size is set identically for all instances in the current setup
    static const uint32_t descriptor_size_bits  = descriptor_size_bits_;
    static const uint32_t descriptor_size_bytes = descriptor_size_bits_/8;

  //ds ctor/dtor
  public:

    BinaryMatchable(const uint64_t& identifier_,
                    const Descriptor& descriptor_): identifier(identifier_),
                                                    pointer(0),
                                                    descriptor(descriptor_) {}

    BinaryMatchable(const void* pointer_,
                    const Descriptor& descriptor_): identifier(0),
                                                    pointer(pointer_),
                                                    descriptor(descriptor_) {}

    BinaryMatchable(const uint64_t& identifier_,
                    const void* pointer_,
                    const Descriptor& descriptor_): identifier(identifier_),
                                                    pointer(pointer_),
                                                    descriptor(descriptor_) {}

#ifdef SRRG_HBST_HAS_OPENCV

    //ds wrapped constructors - only available if OpenCV is present on building system
    BinaryMatchable(const uint64_t& identifier_,
                    const cv::Mat& descriptor_): identifier(identifier_),
                                                 pointer(0),
                                                 descriptor(getDescriptor(descriptor_)) {}

    BinaryMatchable(const void* pointer_,
                    const cv::Mat& descriptor_): identifier(0),
                                                 pointer(pointer_),
                                                 descriptor(getDescriptor(descriptor_)) {}

    BinaryMatchable(const uint64_t& identifier_,
                    const void* pointer_,
                    const cv::Mat& descriptor_): identifier(identifier_),
                                                 pointer(pointer_),
                                                 descriptor(getDescriptor(descriptor_)) {}

#endif

    ~BinaryMatchable() {}

    //ds disable default construction
    BinaryMatchable() = delete;

  //ds functionality
  public:

    //ds computes the distance to another matchable
    inline const uint32_t distance(const BinaryMatchable<descriptor_size_bits_>* matchable_query_) const {
      return (matchable_query_->descriptor^this->descriptor).count();
    }

#ifdef SRRG_HBST_HAS_OPENCV

    //ds descriptor wrapping - only available if OpenCV is present on building system
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

    //ds unique descriptor identifier
    const uint64_t identifier;

    //ds a connected object for correspondence handling (optional) - when using this field one must ensure the permanence of the referenced object!
    const void* pointer;

    //ds descriptor data string vector
    const Descriptor descriptor;
  };

  typedef BinaryMatchable<512> BinaryMatchable512b;
  typedef BinaryMatchable<256> BinaryMatchable256b;
  typedef BinaryMatchable<128> BinaryMatchable128b;
}
