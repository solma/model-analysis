# Usage
# bazel test notebook_integration_test  --test_output=errors \
# --test_env JUPYTER_NOTEBOOK_LIST="Currently running servers: http://localhost:8888/?token=... :: /jupyter/notebook/running/path" \
# --test_env DEBUG_OUTPUT_DIR="/debug/output/dir"
# --test_env TEST_DATA_DIR="/test/data/dir"

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_webtesting//web:py.bzl", "py_web_test_suite")

py_web_test_suite(
    name = "notebook_integration_test",
    srcs = ["notebook_integration_test.py"],
    browsers = [
        "@io_bazel_rules_webtesting//browsers:chromium-local",
    ],
    flaky = 1,
    local = True,
    deps = ["@io_bazel_rules_webtesting//testing/web"],
)
