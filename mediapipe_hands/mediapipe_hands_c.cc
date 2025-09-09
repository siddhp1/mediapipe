#include "mediapipe_hands/mediapipe_hands_c.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mediapipe;

struct MP_Context {
  CalculatorGraph graph;
  std::unique_ptr<OutputStreamPoller> poller_landmarks;
  std::unique_ptr<OutputStreamPoller> poller_handedness;
  int64_t frame_timestamp;
};

int mp_hands_create(void **ctx_out) {
  if (!ctx_out)
    return -1;

  auto ctx = std::make_unique<MP_Context>();
  ctx->frame_timestamp = 0;

  // Load the official desktop CPU graph
  const std::string graph_path =
      "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";

  CalculatorGraphConfig config;
  std::string contents;
  auto status = file::GetContents(graph_path, &contents);
  if (!status.ok() ||
      !ParseTextProto<CalculatorGraphConfig>(contents, &config)) {
    return -2;
  }

  if (ctx->graph.Initialize(config) != absl::OkStatus()) {
    return -3;
  }

  auto poller1 = ctx->graph.AddOutputStreamPoller("multi_hand_landmarks");
  if (!poller1.ok())
    return -4;
  ctx->poller_landmarks =
      std::make_unique<OutputStreamPoller>(std::move(poller1.value()));

  auto poller2 = ctx->graph.AddOutputStreamPoller("multi_handedness");
  if (!poller2.ok())
    return -5;
  ctx->poller_handedness =
      std::make_unique<OutputStreamPoller>(std::move(poller2.value()));

  if (ctx->graph.StartRun({}) != absl::OkStatus()) {
    return -6;
  }

  *ctx_out = ctx.release();
  return 0;
}

int mp_hands_process(void *ctx_ptr, const unsigned char *bgr, int width,
                     int height, MP_HandsResult *out) {
  if (!ctx_ptr || !bgr || !out)
    return -1;

  auto ctx = reinterpret_cast<MP_Context *>(ctx_ptr);

  // Wrap frame
  cv::Mat input_mat(height, width, CV_8UC3, (void *)bgr);
  auto frame = std::make_unique<ImageFrame>(
      ImageFormat::SRGB, width, height, ImageFrame::kDefaultAlignmentBoundary);

  input_mat.copyTo(formats::MatView(frame.get()));

  auto ts = Timestamp(ctx->frame_timestamp++);
  if (ctx->graph.AddPacketToInputStream(
          "input_video", MakePacket<ImageFrame>(std::move(*frame)).At(ts)) !=
      absl::OkStatus()) {
    return -2;
  }

  Packet pkt_landmarks, pkt_handedness;
  if (!ctx->poller_landmarks->Next(&pkt_landmarks))
    return -3;
  if (!ctx->poller_handedness->Next(&pkt_handedness))
    return -4;

  const auto &landmarks_vec =
      pkt_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
  const auto &handedness_vec =
      pkt_handedness.Get<std::vector<mediapipe::ClassificationList>>();

  out->num_hands = std::min((int)landmarks_vec.size(), 2);
  for (int i = 0; i < out->num_hands; ++i) {
    MP_Hand &h = out->hands[i];
    h.present = 1;

    // Copy 21 landmarks
    for (int j = 0; j < landmarks_vec[i].landmark_size(); ++j) {
      auto &lm = landmarks_vec[i].landmark(j);
      h.landmarks[j].x = lm.x();
      h.landmarks[j].y = lm.y();
      h.landmarks[j].z = lm.z();
    }

    // Copy handedness
    if (handedness_vec.size() > i &&
        handedness_vec[i].classification_size() > 0) {
      auto &c = handedness_vec[i].classification(0);
      std::strncpy(h.label, c.label().c_str(), sizeof(h.label) - 1);
      h.label[sizeof(h.label) - 1] = '\0';
      h.score = c.score();
    } else {
      std::strncpy(h.label, "Unknown", sizeof(h.label) - 1);
      h.score = 0.f;
    }
  }
  return 0;
}

void mp_hands_destroy(void *ctx_ptr) {
  if (!ctx_ptr)
    return;
  auto ctx = reinterpret_cast<MP_Context *>(ctx_ptr);
  ctx->graph.CloseInputStream("input_video");
  ctx->graph.WaitUntilDone();
  delete ctx;
}
