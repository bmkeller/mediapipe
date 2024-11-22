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

enum class GestureTypes {
  UNKNOWN,
  CLICK,
};

float ComputeDistance(float dx, float dy) {
  float d = dx * dx + dy * dy;

  if (d != 0) {
    d = sqrt(d);
  }

  return d;
}

}  // namespace

struct HandState {
  NormalizedLandmarkList landmarks;
  std::string hand;
  std::unordered_map<LandmarkType, NormalizedLandmark> landmark_map;
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float max_y = std::numeric_limits<float>::min();

  void InitializeLandmarkMap(const NormalizedLandmarkList &landmarks_list) {
    landmarks = landmarks_list;

    for (int i = kFirstLandmarkIndex; i <= kLastLandmarkIndex; i++) {
      if (landmarks.landmark_size() > i) {
        const auto &landmark = landmarks.landmark(i);
        landmark_map[static_cast<LandmarkType>(i)] = landmark;

        min_x = std::min(min_x, landmark.x());
        min_y = std::min(min_y, landmark.y());
        max_x = std::max(max_x, landmark.x());
        max_y = std::max(max_y, landmark.y());
      }
    }
  }

  std::string DebugStriong(LandmarkType landmark_type) const {
    std::stringstream ss;
    ss << "Hand: " << hand << " " << static_cast<int>(landmark_type) << ": "
       << landmark_map.at(landmark_type).DebugString();
    return ss.str();
  }

  void PrintPoint(LandmarkType l) const {
    const auto &point = landmark_map.at(l);
    std::cout << "Point: " << point.x() << ", " << point.y() << std::endl;
  }

  float Distance(LandmarkType l1, LandmarkType l2, bool scaled) const {
    float dx = landmark_map.at(l1).x() - landmark_map.at(l2).x();
    float dy = landmark_map.at(l1).y() - landmark_map.at(l2).y();

    if (scaled) {
      dx /= (max_x - min_x);
      dy /= (max_y - min_y);
    }

    return ComputeDistance(dx, dy);
  }
};

GestureTypes GetGestureType(const HandState &hand_state) {
  const float touchDistance = hand_state.Distance(
      LandmarkType::THUMB_TIP, LandmarkType::INDEX_FINGER_TIP, true);

  const float pinkyDistance = hand_state.Distance(
      LandmarkType::PINKY_TIP, LandmarkType::INDEX_FINGER_TIP, true);

  std::cout << "Touch distance: " << touchDistance
            << ", Pinky distance: " << pinkyDistance << std::endl;

  if (touchDistance < 0.1f && pinkyDistance > 0.2f) {
    return GestureTypes::CLICK;
  }

  return GestureTypes::UNKNOWN;
}

absl::Status GestureRecognizerCalculator::GetContract(CalculatorContract *cc) {
  cc->Inputs()
      .Tag(kMultiNormLandmarksTag)
      .Set<std::vector<NormalizedLandmarkList>>();
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kHandednessTag).Set<std::vector<ClassificationList>>();
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
    return absl::OkStatus();
  }

  gesture_.set_index(frame_index_);
  gesture_.set_frame_index(frame_index_);
  gesture_.set_center_x(0.0f);
  gesture_.set_center_y(0.0f);
  gesture_.set_gesture_type(gestures::GestureType::NONE);

  // RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());

  // std::pair<int, int> image_size =
  //     cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  const auto landmarks = cc->Inputs()
                             .Tag(kMultiNormLandmarksTag)
                             .Get<std::vector<NormalizedLandmarkList>>();

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

  HandState hand_state;
  hand_state.InitializeLandmarkMap(landmarks[0]);

  const float touchDistance = hand_state.Distance(
      LandmarkType::THUMB_TIP, LandmarkType::INDEX_FINGER_TIP, true);

  const bool touchDetected = touchDistance < kTouchDistanceThreshold;

  if (touchDetected) {
    OnGestureDetected();
  }

  std::cout << "Distance: " << touchDistance
            << ", touch detected: " << touchDetected << std::endl;

  proto_writer_.Write(gesture_.SerializeAsString());

  return absl::OkStatus();
}

void GestureRecognizerCalculator::OnGestureDetected() {
  std::cout << "Gesture detected" << std::endl;
}

REGISTER_CALCULATOR(GestureRecognizerCalculator);

}  // namespace mediapipe
