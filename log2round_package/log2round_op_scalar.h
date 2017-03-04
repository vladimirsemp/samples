#ifndef LOG2ROUND_OP_SCALAR_H_
#define LOG2ROUND_OP_SCALAR_H_

#include <cmath>
#include <functional>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace Eigen {
namespace internal {

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
_Pragma("GCC diagnostic push")         \
_Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif
    
template <typename Scalar>
struct scalar_log2round_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar
  operator()(const Scalar& x) const {
    EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex),
                        NUMERIC_TYPE_MUST_BE_REAL)

#define MAX_ROUND Scalar(1024)
#define MIN_ROUND Scalar(0.00006103515)
#define LOG2E_ROUND Scalar(1.442695)

    if(x == Scalar(0)) return x;
    else if(x > Scalar(0)){
        if(x >= MAX_ROUND) return MAX_ROUND;
        if(x <= MIN_ROUND) return MIN_ROUND;
        const Scalar degree = Eigen::numext::floor(Eigen::numext::log(x) * LOG2E_ROUND);
        const Scalar round_val = Eigen::numext::pow(Scalar(2), degree);
        if(Scalar(2) * (x - round_val) >= round_val) return Scalar(2) * round_val;
        else return round_val;
    }
    else{
        if(x <= -MAX_ROUND) return -MAX_ROUND;
        if(x >= -MIN_ROUND) return -MIN_ROUND;
        const Scalar degree = Eigen::numext::floor(Eigen::numext::log(- x) * LOG2E_ROUND);
        const Scalar round_val = - Eigen::numext::pow(Scalar(2), degree);
        if(Scalar(2) * (round_val - x) >= - round_val) return Scalar(2) * round_val;
        else return round_val;
    }

#undef LOG2E_ROUND
#undef MAX_ROUND
#undef MIN_ROUND

  }
};

template <typename Scalar>
struct functor_traits<scalar_log2round_op <Scalar> > {
  enum {
    PacketAccess = false,
    Cost =
    ((sizeof(Scalar) == 4)
     // float: 7 pmadd, 6 pmul, 4 padd/psub, 10 other
     ? (21 * NumTraits<Scalar>::AddCost + 13 * NumTraits<Scalar>::MulCost)
     // double: 7 pmadd, 5 pmul, 3 padd/psub, 13 other
     : (23 * NumTraits<Scalar>::AddCost + 12 * NumTraits<Scalar>::MulCost))
  };
};

#undef ENABLE_FLOAT_EQUALITY_WARNING
#undef DISABLE_FLOAT_EQUALITY_WARNING

}  // end namespace internal
}  // end namespace Eigen

#endif  // LOG2ROUND_OP_SCALAR_H_

