#include "mediapipe/mkel/gesture_detector.h"

#include <limits>
#include <sstream>
#include <unordered_map>

namespace mediapipe {
namespace gesture_detection {
namespace {

TimePoint now() { return std::chrono::system_clock::now(); }

float computeDistance(float dx, float dy) {
  float d = dx * dx + dy * dy;

  if (d > 0) {
    d = sqrt(d);
  }

  return d;
}

BoundingBox computeBoundingBox(const NormalizedLandmarkList &landmarks) {
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float max_y = std::numeric_limits<float>::min();

  for (int i = kFirstLandmarkIndex; i <= kLastLandmarkIndex; i++) {
    if (landmarks.landmark_size() > i) {
      const auto &landmark = landmarks.landmark(i);

      min_x = std::min(min_x, landmark.x());
      min_y = std::min(min_y, landmark.y());
      max_x = std::max(max_x, landmark.x());
      max_y = std::max(max_y, landmark.y());
    }
  }

  return BoundingBox{min_x, min_y, max_x, max_y};
}

float distanceBetweenPoints(const HandState &hand, LandmarkType l1,
                            LandmarkType l2, bool scaled) {
  float dx = hand.landmark_map.at(l1).x() - hand.landmark_map.at(l2).x();
  float dy = hand.landmark_map.at(l1).y() - hand.landmark_map.at(l2).y();

  if (scaled) {
    dx /= (hand.bounding_box.max_x - hand.bounding_box.min_x);
    dy /= (hand.bounding_box.max_y - hand.bounding_box.min_y);
  }

  return computeDistance(dx, dy);
}

bool isCurled(const HandState &hand, LandmarkType tip, LandmarkType midpoint) {
  const float toTip =
      distanceBetweenPoints(hand, LandmarkType::WRIST, tip, true);
  const float toMidpoint =
      distanceBetweenPoints(hand, LandmarkType::WRIST, midpoint, true);

  return toTip <= toMidpoint;
}

void computeCurledFingers(HandState &hand) {
  hand.thumb_curled =
      isCurled(hand, LandmarkType::THUMB_TIP, LandmarkType::THUMB_IP);
  hand.index_curled = isCurled(hand, LandmarkType::INDEX_FINGER_TIP,
                               LandmarkType::INDEX_FINGER_PIP);
  hand.pinky_curled =
      isCurled(hand, LandmarkType::PINKY_TIP, LandmarkType::PINKY_PIP);
}

void updateHandState(HandState &hand, const NormalizedLandmarkList &landmarks) {
  hand.landmarks = landmarks;
  hand.bounding_box = computeBoundingBox(landmarks);

  for (int i = kFirstLandmarkIndex; i <= kLastLandmarkIndex; i++) {
    if (landmarks.landmark_size() > i) {
      const auto &landmark = landmarks.landmark(i);
      hand.landmark_map[static_cast<LandmarkType>(i)] = landmark;
    }
  }

  computeCurledFingers(hand);
}

}  // namespace

/*
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
*/

GestureDetector::GestureDetector() {
  leftHand_.hand = kLeft;
  rightHand_.hand = kRight;
}

// GestureTypes GetGestureType(const HandState &hand_state) {
//   const float touchDistance = hand_state.Distance(
//       LandmarkType::THUMB_TIP, LandmarkType::INDEX_FINGER_TIP, true);

//   const float pinkyDistance = hand_state.Distance(
//       LandmarkType::PINKY_TIP, LandmarkType::INDEX_FINGER_TIP, true);

//   std::cout << "Touch distance: " << touchDistance
//             << ", Pinky distance: " << pinkyDistance << std::endl;

//   if (touchDistance < 0.1f && pinkyDistance > 0.2f) {
//     return GestureTypes::CLICK;
//   }

//   return GestureTypes::UNKNOWN;
// }

// HandPositions GestureDetector::GetHandPosition(
//     const NormalizedLandmarkList &landmarks) {
//   HandState hand_state;
//   hand_state.InitializeLandmarkMap(landmarks);

//   return GetGestureType(hand_state);
// }

void GestureDetector::updateHand(HandState &hand,
                                 const NormalizedLandmarkList &landmarks) {
  updateHandState(hand, landmarks);
  hand.last_update_time = now();

  if (isFist(hand)) {
    hand.hand_position = HandPositions::FIST;
    hand.directionDegrees = std::nullopt;
    std::cout << "----> FIST" << std::endl;
    return;
  }

  hand.hand_position = HandPositions::UNKNOWN;
}

bool GestureDetector::isFist(const HandState &hand) {
  // First is defined as everything close together.
  constexpr float kDistanceScale = 0.80f;

  const float thumbWristDistance = distanceBetweenPoints(
      hand, LandmarkType::THUMB_TIP, LandmarkType::WRIST, true);

  const float thumbIndexDistance = distanceBetweenPoints(
      hand, LandmarkType::THUMB_TIP, LandmarkType::INDEX_FINGER_TIP, true);

  const float thumbPinkyDistance = distanceBetweenPoints(
      hand, LandmarkType::THUMB_TIP, LandmarkType::PINKY_TIP, true);

  const float maxAllowedDistance = thumbWristDistance * kDistanceScale;

  std::cout << "Checking for fist (threshold=" << maxAllowedDistance
            << ")--------" << std::endl;
  std::cout << "Thumb index distance: " << thumbIndexDistance << std::endl;
  std::cout << "Pinky index distance: " << thumbPinkyDistance << std::endl;

  std::cout << "Thumb curled: " << hand.thumb_curled
            << ", index curled: " << hand.index_curled
            << ", pinky curled: " << hand.pinky_curled << std::endl;

  return thumbIndexDistance < maxAllowedDistance &&
         thumbPinkyDistance < maxAllowedDistance;
}

std::optional<float> GestureDetector::isIndexPoint(const HandState &hand) {
  // Index finger must be sticking out and pinky must be curled.
  if (hand.index_curled || !hand.pinky_curled) {
    return std::nullopt;
  }

  // Thumb tip should be closer than

  return distanceBetweenPoints(hand, LandmarkType::INDEX_FINGER_TIP,
                               LandmarkType::INDEX_FINGER_PIP, true);
}

HandPositions GestureDetector::updateLeftHand(
    const NormalizedLandmarkList &landmarks) {
  updateHand(leftHand_, landmarks);
  return leftHand_.hand_position;
}

HandPositions GestureDetector::updateRightHand(
    const NormalizedLandmarkList &landmarks) {
  updateHand(rightHand_, landmarks);
  return rightHand_.hand_position;
}

}  // namespace gesture_detection
}  // namespace mediapipe
