#define EIGEN_USE_THREADS

#include "log2round_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
    
    REGISTER_OP("Log2round")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
         Rounds input features to the nearest power of two; doesn't change gradient during backpropagation step. 
         )doc");

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_LOG2ROUND_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Log2round").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      Log2roundOp<CPUDevice, type>);                                       

TF_CALL_REAL_NUMBER_TYPES(REGISTER_LOG2ROUND_KERNELS);
#undef REGISTER_LOG2ROUND_KERNELS

#if GOOGLE_CUDA
    // Forward declarations of the functor specializations for GPU.
    namespace functor {
        
#define DECLARE_GPU_SPEC(T)                                              \
template <>                                                              \
void Log2round<GPUDevice, T>::operator()(                                     \
const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
typename TTypes<T>::Tensor activations);                                 \
extern template struct Log2round<GPUDevice, T>;

        TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
    }  // namespace functor
    
    // Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                              \
REGISTER_KERNEL_BUILDER(                                        \
Name("Log2round").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
Log2roundOp<GPUDevice, type>);

    TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
    
#undef REGISTER_GPU_KERNELS
    
#endif  // GOOGLE_CUDA
    
}  // namespace tensorflow
