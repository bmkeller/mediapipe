#include "mediapipe/mkel/gesture_detector_calculator.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mediapipe {
namespace {
constexpr char kMultiNormLandmarksTag[] = "MULTI_NORM_LANDMARKS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kPredictedGestureTag[] = "PREDICTED_GESTURE";
constexpr float kTouchDistanceThreshold = 0.10f;

constexpr char kProtoPath[] = "/tmp/proto_ipc";
constexpr int kMaxMessagesToKeep = 20;

}  // namespace

absl::Status GestureRecognizerCalculator::GetContract(CalculatorContract *cc) {
  cc->Inputs()
      .Tag(kMultiNormLandmarksTag)
      .Set<std::vector<NormalizedLandmarkList>>();
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kHandednessTag).Set<std::vector<ClassificationList>>();

  cc->Outputs()
      .Tag(kMultiNormLandmarksTag)
      .Set<std::vector<NormalizedLandmarkList>>();
  cc->Outputs().Tag(kPredictedGestureTag).Set<int>();

  return absl::OkStatus();
}

absl::Status GestureRecognizerCalculator::Open(CalculatorContext *cc) {
  cc->SetOffset(TimestampDiff(0));
  proto_writer_.Initialize(kProtoPath, kMaxMessagesToKeep);
  frame_index_ = 0;
  return absl::OkStatus();
}

absl::Status GestureRecognizerCalculator::Process(CalculatorContext *cc) {
  frame_index_++;
  gesture_.set_frame_index(frame_index_);

  if (cc->Inputs().Tag(kMultiNormLandmarksTag).IsEmpty()) {
    cc->Outputs()
        .Tag(kMultiNormLandmarksTag)
        .AddPacket(mediapipe::Adopt(new std::vector<NormalizedLandmarkList>())
                       .At(cc->InputTimestamp()));

    cc->Outputs()
        .Tag(kPredictedGestureTag)
        .AddPacket(mediapipe::Adopt(new int(-1)).At(cc->InputTimestamp()));

    return absl::OkStatus();
  }

  gesture_.set_timestamp(absl::FormatTime(absl::Now()));
  gesture_.set_hand_position(gestures::HandPositions::NONE);

  const auto landmarks = cc->Inputs()
                             .Tag(kMultiNormLandmarksTag)
                             .Get<std::vector<NormalizedLandmarkList>>();

  cc->Outputs()
      .Tag(kMultiNormLandmarksTag)
      .AddPacket(
          mediapipe::Adopt(new std::vector<NormalizedLandmarkList>(landmarks))
              .At(cc->InputTimestamp()));

  std::cout << "Found " << landmarks.size() << " hands" << std::endl;

  const auto &image = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  std::cout << "Image size: " << image.Width() << "x" << image.Height()
            << std::endl;

  const auto landmark_map = generateLandmarkMap(landmarks[0]);

  auto pt = landmark_map.at(LandmarkType::INDEX_FINGER_TIP);
  gesture_.mutable_index_finger_tip()->set_x(pt.x());
  gesture_.mutable_index_finger_tip()->set_y(pt.y());

  pt = landmark_map.at(LandmarkType::THUMB_TIP);
  gesture_.mutable_thumb_tip()->set_x(pt.x());
  gesture_.mutable_thumb_tip()->set_y(pt.y());

  pt = landmark_map.at(LandmarkType::WRIST);
  gesture_.mutable_wrist()->set_x(pt.x());
  gesture_.mutable_wrist()->set_y(pt.y());

  gesture_.set_hand_position(gesture_detector_.getHandPosition());
  gesture_.set_hand_position_probability(
      gesture_detector_.getHandPositionProbability());
  gesture_.set_hand_position_index(gesture_detector_.getHandPositionCounter());

  const std::optional<gestures::HandPositions> predicted_gesture =
      gesture_detector_.performInference(landmarks[0]);

  if (predicted_gesture) {
    cc->Outputs()
        .Tag(kPredictedGestureTag)
        .AddPacket(mediapipe::Adopt(new int(predicted_gesture.value()))
                       .At(cc->InputTimestamp()));
  }

  proto_writer_.Write(gesture_.SerializeAsString());

  return absl::OkStatus();
}

void GestureRecognizerCalculator::OnGestureDetected() {
  std::cout << "Gesture detected" << std::endl;
}

REGISTER_CALCULATOR(GestureRecognizerCalculator);

}  // namespace mediapipe
