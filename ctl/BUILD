package(
    default_visibility = ["//third_party/py/mint:mint_friends"],
)

licenses(["notice"])

py_library(
    name = "single_task_evaluator",
    srcs = ["single_task_evaluator.py"],
    deps = [
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "single_task_evaluator_test",
    srcs = ["single_task_evaluator_test.py"],
    python_version = "PY3",
    deps = [
        ":single_task_evaluator",
        ":single_task_trainer",
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
    ],
)

py_library(
    name = "single_task_trainer",
    srcs = ["single_task_trainer.py"],
    deps = [
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "single_task_trainer_test",
    srcs = ["single_task_trainer_test.py"],
    python_version = "PY3",
    deps = [
        ":single_task_trainer",
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
    ],
)
