package(
    default_visibility = ["//third_party/py/mint:mint_friends"],
)

licenses(["notice"])

py_binary(
    name = "convert_sstable_to_tfrecordio_beam",
    srcs = ["convert_sstable_to_tfrecordio_beam.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//pipeline/flume/py:runner",
        "//pipeline/flume/py/io:sstableio",
        "//pipeline/flume/py/io:tfrecordio",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/apache_beam",
        "//third_party/py/tensorflow",
    ],
)
