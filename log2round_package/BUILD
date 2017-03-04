package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cuda_tests_tags")

tf_custom_op_library(
    name =  "log2round_op.so",
    srcs = ["log2round_op.cc",
            "log2round_op.h",
            "log2round_op_functor.h",
            "log2round_op_scalar.h"
           ],
    gpu_srcs = ["log2round_op_gpu.cu.cc"],
)

py_library(
    name = "log2round_module",
    srcs = ["log2round_module.py"],
    data = [":log2round_op.so"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "log2round_grad",
    srcs = ["log2round_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":log2round_module",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

py_test(
    name = "log2round_test",
    size = "small",
    srcs = ["log2round_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":log2round_module",
        ":log2round_grad",
        "//tensorflow:tensorflow_py",
    ],
)