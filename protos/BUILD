package(
    default_visibility = ["//third_party/py/mint:mint_friends"],
)

licenses(["notice"])

proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "model_py_pb2",
    api_version = 2,
    deps = [":model_proto"],
)

proto_library(
    name = "pipeline_proto",
    srcs = ["pipeline.proto"],
    cc_api_version = 2,
    deps = [
        ":dataset_proto",
        ":eval_proto",
        ":model_proto",
        ":train_proto",
    ],
)

py_proto_library(
    name = "pipeline_py_pb2",
    api_version = 2,
    deps = [":pipeline_proto"],
)

proto_library(
    name = "dataset_proto",
    srcs = ["dataset.proto"],
    cc_api_version = 2,
    deps = [
        ":preprocessor_proto",
    ],
)

py_proto_library(
    name = "dataset_py_pb2",
    api_version = 2,
    deps = [
        ":dataset_proto",
    ],
)

proto_library(
    name = "eval_proto",
    srcs = ["eval.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "eval_py_pb2",
    api_version = 2,
    deps = [":eval_proto"],
)

proto_library(
    name = "train_proto",
    srcs = ["train.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "train_py_pb2",
    api_version = 2,
    deps = [":train_proto"],
)

proto_library(
    name = "preprocessor_proto",
    srcs = ["preprocessor.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "preprocessor_py_pb2",
    api_version = 2,
    deps = [":preprocessor_proto"],
)
