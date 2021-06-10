package(
    default_visibility = ["//third_party/py/mint:mint_friends"],
)

licenses(["notice"])

py_library(
    name = "base_models",
    srcs = ["base_models.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_util",
        "//third_party/py/einops",
        "//third_party/py/mint/protos:model_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "base_model_util",
    srcs = ["base_model_util.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "multi_modal_model_util",
    srcs = ["multi_modal_model_util.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_util",
        ":base_models",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "base_models_test",
    size = "large",
    timeout = "long",
    srcs = ["base_models_test.py"],
    python_version = "PY3",
    deps = [
        ":base_models",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_util",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_graphics/geometry/transformation",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    deps = [
        ":metrics",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "fact_model",
    srcs = ["fact_model.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_util",
        ":base_models",
        ":metrics",
        ":multi_modal_model",
        ":multi_modal_model_util",
        "//third_party/py/absl/logging",
        "//third_party/py/mint/protos:dataset_py_pb2",
        "//third_party/py/mint/protos:model_py_pb2",
        "//third_party/py/mint/utils:inputs_util",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "fact_model_test",
    size = "large",
    timeout = "long",
    srcs = ["fact_model_test.py"],
    python_version = "PY3",
    deps = [
        ":fact_model",
        "//third_party/py/mint/protos:model_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":fact_model",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons",
    ],
)

py_library(
    name = "multi_modal_model",
    srcs = ["multi_modal_model.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs.py"],
    srcs_version = "PY3",
    deps = [
        "//file/colossus/public:cns",
        "//file/placer",
        "//pyglib:gfile",
        "//third_party/py/mint/utils:inputs_util",
        "//third_party/py/mint/utils:text_util",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "learning_schedules",
    srcs = ["learning_schedules.py"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "learning_schedules_test",
    srcs = ["learning_schedules_test.py"],
    python_version = "PY3",
    deps = [
        ":learning_schedules",
        "//third_party/py/tensorflow",
    ],
)
