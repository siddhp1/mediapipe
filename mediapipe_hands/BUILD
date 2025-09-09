cc_library(
    name = "mediapipe_hands_c_lib",
    srcs = ["mediapipe_hands_c.cc"],
    hdrs = ["mediapipe_hands_c.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1
)

cc_binary(
    name = "mediapipe_hands_c",
    deps = [":mediapipe_hands_c_lib"],
    linkshared = 1,
)
