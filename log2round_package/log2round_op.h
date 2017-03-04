#ifndef LOG2ROUND_OP_H_
#define LOG2ROUND_OP_H_

#define EIGEN_USE_THREADS

#include "log2round_op_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

template <typename Device, typename T>
class Log2roundOp : public UnaryElementWiseOp<T, Log2roundOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, Log2roundOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Log2round<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // LOG2ROUND_OP_H_
