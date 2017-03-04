#ifndef LOG2ROUND_OP_FUNCTOR_H_
#define LOG2ROUND_OP_FUNCTOR_H_
// Functor definition for Log2roundOp, must be compilable by nvcc.

#include "log2round_op_scalar.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by Log2roundOp to do the computations.
template <typename Device, typename T>
struct Log2round {
  // Computes Log2round activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) = features.unaryExpr(Eigen::internal::scalar_log2round_op<T>());
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // LOG2ROUND_OP_FUNCTOR_H_
