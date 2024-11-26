#pragma once

#include <chrono>
#include <optional>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/mkel/hand_landmarks.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe {
namespace gesture_detection {

using TimePoint = std::chrono::system_clock::time_point;

enum class HandPositions {
  UNKNOWN,
  FIST,
  CLICK,
  INDEX_POINT,
  THUMB_POINT,
  TAP,
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

  std::optional<int> performInference(const NormalizedLandmarkList &landmarks);

  std::pair<float, float> getIndexFingerTip() const;

  std::pair<int, int> getGesture() const;

 private:
  void updateHand(HandState &hand, const NormalizedLandmarkList &landmarks);
  void loadTfliteModel();

  HandState leftHand_;
  HandState rightHand_;

  bool isFist(const HandState &hand);
  std::optional<float> isIndexPoint(const HandState &hand);

  // Model stuff
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  int input_batch_size_ = -1;
  int input_landmarks_ = -1;
  int output_classes_ = -1;

  int last_gesture_code_ = 0;
  int last_gesture_counter_ = 0;
};

}  // namespace gesture_detection
}  // namespace mediapipe
