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

  if (cc->Inputs().Tag(kMultiNormLandmarksTag).IsEmpty()) {
    std::cout << "MKEL: No hands found" << std::endl;

    cc->Outputs()
        .Tag(kMultiNormLandmarksTag)
        .AddPacket(mediapipe::Adopt(new std::vector<NormalizedLandmarkList>())
                       .At(cc->InputTimestamp()));

    return absl::OkStatus();
  }

  gesture_.set_index(123);
  gesture_.set_frame_index(frame_index_);
  gesture_.set_center_x(2.0f);
  gesture_.set_center_y(4.0f);
  gesture_.set_gesture_type(gestures::GestureType::NONE);

  // RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());

  // std::pair<int, int> image_size =
  //     cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

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

  // const auto &handedness =
  //     cc->Inputs().Tag(kHandednessTag).Get<std::vector<ClassificationList>>();
  // std::cout << "Handedness: " << handedness.size() << std::endl;
  // std::cout << "Handedness: " << handedness[0].classification_size()
  //           << std::endl;
  // std::cout << "Handedness: " << handedness[0].classification(0).label()
  //           << std::endl;

  // HandState hand_state;
  // hand_state.InitializeLandmarkMap(landmarks[0]);

  // const float touchDistance = hand_state.Distance(
  //     LandmarkType::THUMB_TIP, LandmarkType::INDEX_FINGER_TIP, true);

  // const bool touchDetected = touchDistance < kTouchDistanceThreshold;

  // if (touchDetected) {
  //   OnGestureDetected();
  // }

  // std::cout << "Distance: " << touchDistance
  //           << ", touch detected: " << touchDetected << std::endl;

  gesture_detection::HandPositions handResult =
      gesture_detector_.updateLeftHand(landmarks[0]);

  std::string serialized_gesture = gesture_.SerializeAsString();
  // std::cout << "Serialized gesture: " << serialized_gesture.size() <<
  // std::endl;

  proto_writer_.Write(serialized_gesture);

  return absl::OkStatus();
}

void GestureRecognizerCalculator::OnGestureDetected() {
  std::cout << "Gesture detected" << std::endl;
}

REGISTER_CALCULATOR(GestureRecognizerCalculator);

}  // namespace mediapipe
