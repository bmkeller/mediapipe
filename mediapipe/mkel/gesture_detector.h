#pragma once

#include <chrono>
#include <optional>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/mkel/gestures.pb.h"
#include "mediapipe/mkel/hand_landmarks.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe {
namespace gesture_detection {

class GestureDetector {
 public:
  GestureDetector();

  std::optional<gestures::HandPositions> performInference(
      const NormalizedLandmarkList &landmarks);

  float getHandPositionProbability() const {
    return last_hand_position_probability_;
  }

  gestures::HandPositions getHandPosition() const {
    return last_hand_position_;
  }

  int getHandPositionCounter() const { return last_hand_position_counter_; }

 private:
  void loadTfliteModel();

  // Model stuff
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  int input_batch_size_ = -1;
  int input_landmarks_ = -1;
  int num_output_classes_ = -1;

  gestures::HandPositions last_hand_position_ = gestures::HandPositions::NONE;
  float last_hand_position_probability_ = 0.0f;
  int last_hand_position_counter_ = 0;
};

}  // namespace gesture_detection
}  // namespace mediapipe
