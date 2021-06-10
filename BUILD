"""Mint is a multi-modal video content understanding and creation library
with basic transformer building blocks and the SOTA human motion generation
model."""

package(default_visibility = [":mint_friends"])

package_group(
    name = "mint_friends",
    packages = [
        "//research/vision/couchpotato/...",
        "//third_party/py/mint/...",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    deps = [
        "//file/placer",
        "//perftools/gputools/profiler:gpuprof_lib",  # for GPU jobs
        "//perftools/gputools/profiler:xprofilez_with_server",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/mint/core:inputs",
        "//third_party/py/mint/core:learning_schedules",
        "//third_party/py/mint/core:model_builder",
        "//third_party/py/mint/ctl:single_task_trainer",
        "//third_party/py/mint/utils:config_util",
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    python_version = "PY3",
    deps = [
        "//file/placer",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/mint/core:inputs",
        "//third_party/py/mint/core:model_builder",
        "//third_party/py/mint/ctl:single_task_evaluator",
        "//third_party/py/mint/utils:config_util",
        "//third_party/py/orbit",
        "//third_party/py/tensorflow",
    ],
)
