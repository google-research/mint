# Library of util function for training multi_modal model.
package(
    default_visibility = ["//third_party/py/mint:mint_friends"],
)

licenses(["notice"])

py_library(
    name = "config_util",
    srcs = ["config_util.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/mint/protos:pipeline_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "inputs_util",
    srcs = ["inputs_util.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/mint/protos:dataset_py_pb2",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_test(
    name = "inputs_util_test",
    srcs = ["inputs_util_test.py"],
    python_version = "PY3",
    deps = [
        ":inputs_util",
        "//testing/pybase",
        "//third_party/py/mint/protos:dataset_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

