#pragma once

#include <chrono>
#include <optional>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/mkel/hand_landmarks.h"

namespace mediapipe {
namespace gesture_detection {

using TimePoint = std::chrono::system_clock::time_point;

enum class HandPositions {
  UNKNOWN,
  FIST,
  CLICK,
  INDEX_POINT,
  THUMB_POINT,
};

struct BoundingBox {
  float min_x = 0;
  float min_y = 0;
  float max_x = 0;
  float max_y = 0;
};

struct HandState {
  std::string hand;
  NormalizedLandmarkList landmarks;
  std::unordered_map<LandmarkType, NormalizedLandmark> landmark_map;
  HandPositions hand_position;
  BoundingBox bounding_box;
  TimePoint last_update_time;
  std::optional<float> directionDegrees;

  bool thumb_curled = false;
  bool index_curled = false;
  bool pinky_curled = false;
};

class GestureDetector {
 public:
  static constexpr char kLeft[] = "LEFT";
  static constexpr char kRight[] = "RIGHT";

  GestureDetector();

  HandPositions updateLeftHand(const NormalizedLandmarkList &landmarks);
  HandPositions updateRightHand(const NormalizedLandmarkList &landmarks);

  const HandState &leftHand() const { return leftHand_; }
  const HandState &rightHand() const { return rightHand_; }

 private:
  void updateHand(HandState &hand, const NormalizedLandmarkList &landmarks);

  HandState leftHand_;
  HandState rightHand_;

  bool isFist(const HandState &hand);
  std::optional<float> isIndexPoint(const HandState &hand);
};

}  // namespace gesture_detection
}  // namespace mediapipe