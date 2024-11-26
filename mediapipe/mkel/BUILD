# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

# load("@rules_proto_grpc//grpc:defs.bzl", "cc_grpc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cc_grpc_library")

package(default_visibility=["//visibility:public"])

selects.config_setting_group(
    name="apple_or_disable_gpu",
    match_any=[
        "//mediapipe/gpu:disable_gpu",
        "//mediapipe:apple",
    ],
)

mediapipe_proto_library(
    name="gestures_proto",
    srcs=["gestures.proto"],
    deps=[
    ],
)

cc_grpc_library(
    name="gestures_grpc",
    srcs=["gesture_server.cc"],
    # Set to True if you only want gRPC (no protobuf messages)
    grpc_only=False,
    deps=[
        # "@com_github_grpc_grpc//:grpc++",
    ],
    protos=["gestures.proto"],
)

cc_library(
    name="types",
    srcs=["hand_landmarks.cc"],
    hdrs=["hand_landmarks.h"],
    deps=[
        "//mediapipe/framework/formats:landmark_cc_proto",
    ],
)

cc_library(
    name="gesture_detector_calculator",
    srcs=[
        "gesture_detector_calculator.cc",
        "proto_writer.cc",
        "gesture_detector.cc",
    ],
    hdrs=[
        "gesture_detector_calculator.h",
        "proto_writer.h",
        "gesture_detector.h",
    ],
    deps=[
        ":types",
        ":gestures_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/util/tflite:config",
        "//mediapipe/util/tflite:tflite_model_loader",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        # "@org_tensorflow//tensorflow/lite:ops",
        # "@org_tensorflow//tensorflow/lite:kernels",
    ],
    alwayslink=1,
)

mediapipe_simple_subgraph(
    name="gesture_processor",
    graph="gesture_processor.pbtxt",
    register_as="GestureProcessorSubgraph",
    deps=[
        ":gesture_detector_calculator",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:labels_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

cc_binary(
    name="hello_bazel",
    srcs=["main.cpp"],
    deps=[":test_lib", "@luxonis_depthai//:luxonis_depthai"],
)

cc_import(
    name="test_lib",
    hdrs=["sample_lib.h"],
    includes=["."],
    static_library="libmylib.a",
    # interface_library = "libmylib.dylib",
)

# cc_library(
#     name="dcam_lib_with_includes",
#     deps=[":dcam_lib"],
#     # Specify the base path here
#     includes=[
#         "/Users/michaelkeller/Documents/code/luxonis_cpp2/depthai-core/include"],
#     # Optional: specify headers if they are needed
#     # hdrs=glob(
#     #    ["/Users/michaelkeller/Documents/code/luxonis_cpp2/depthai-core/include/**/*.h"]),
# )

# cc_import(
#     name="dcam_lib",
#     # hdrs = ["sample_lib.h"],
#     includes=["."],
#     static_library="libdepthai-core.a",
#     # alwayslink=True,
#     # interface_library = "libmylib.dylib",
# )
