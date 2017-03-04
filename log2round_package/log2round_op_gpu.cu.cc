#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "log2round_op_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
    
    typedef Eigen::GpuDevice GPUDevice;
    
    // Definition of the GPU implementations declared in log2round_op.cc.
#define DEFINE_GPU_KERNELS(T)                       \
template struct functor::Log2round<GPUDevice, T>;
    
    TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
    
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
