#include "mediapipe/mkel/gesture_detector.h"

#include <array>
#include <filesystem>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace mediapipe {
namespace gesture_detection {
namespace {

constexpr char kFallbackModelPath[] = "./model.tflite";

constexpr char kModelPath[] =
    "/Users/michaelkeller/Documents/code/notebook/output/model.tflite";

// float computeDistance(float dx, float dy) {
//   float d = dx * dx + dy * dy;

//   if (d > 0) {
//     d = sqrt(d);
//   }

//   return d;
// }

// BoundingBox computeBoundingBox(const NormalizedLandmarkList &landmarks) {
//   float min_x = std::numeric_limits<float>::max();
//   float min_y = std::numeric_limits<float>::max();
//   float max_x = std::numeric_limits<float>::min();
//   float max_y = std::numeric_limits<float>::min();

//   for (int i = kFirstLandmarkIndex; i <= kLastLandmarkIndex; i++) {
//     if (landmarks.landmark_size() > i) {
//       const auto &landmark = landmarks.landmark(i);

//       min_x = std::min(min_x, landmark.x());
//       min_y = std::min(min_y, landmark.y());
//       max_x = std::max(max_x, landmark.x());
//       max_y = std::max(max_y, landmark.y());
//     }
//   }

//   return BoundingBox{min_x, min_y, max_x, max_y};
// }

// float distanceBetweenPoints(const HandState &hand, LandmarkType l1,
//                             LandmarkType l2, bool scaled) {
//   float dx = hand.landmark_map.at(l1).x() - hand.landmark_map.at(l2).x();
//   float dy = hand.landmark_map.at(l1).y() - hand.landmark_map.at(l2).y();

//   if (scaled) {
//     dx /= (hand.bounding_box.max_x - hand.bounding_box.min_x);
//     dy /= (hand.bounding_box.max_y - hand.bounding_box.min_y);
//   }

//   return computeDistance(dx, dy);
// }

}  // namespace

GestureDetector::GestureDetector() { loadTfliteModel(); }

void GestureDetector::loadTfliteModel() {
  // Load model
  // Check if the file exists
  std::array kAvailableModelPaths = {kModelPath, kFallbackModelPath};

  const char *model_path = nullptr;

  for (const auto &path : kAvailableModelPaths) {
    if (std::filesystem::exists(path)) {
      model_path = path;
      break;
    }
  }

  if (model_path == nullptr) {
    throw std::runtime_error("TFLite model file does not exist: " +
                             std::string(kModelPath));
  }

  model_ = tflite::FlatBufferModel::BuildFromFile(model_path);

  if (!model_) {
    throw std::runtime_error("Failed to load TFLite model from: " +
                             std::string(model_path));
  }

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);

  builder(&interpreter_);

  if (!interpreter_) {
    throw std::runtime_error("Failed to build TFLite interpreter");
  }

  // Allocate tensors
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("Failed to allocate tensors");
  }

  // Validate input tensor
  if (interpreter_->inputs().size() != 1) {
    throw std::runtime_error("Model must have exactly one input");
  }

  TfLiteTensor *input_tensor = interpreter_->input_tensor(0);
  if (input_tensor->type != kTfLiteFloat32) {
    throw std::runtime_error("Input tensor must be float32");
  }

  // Expected shape: [1, num_landmarks]
  const int expected_dims = 2;
  if (input_tensor->dims->size != expected_dims) {
    throw std::runtime_error(
        "Input tensor must have " + std::to_string(expected_dims) +
        " dimensions, got: " + std::to_string(input_tensor->dims->size));
  }

  input_batch_size_ = input_tensor->dims->data[0];
  input_landmarks_ = input_tensor->dims->data[1];

  // Validate output tensor
  if (interpreter_->outputs().size() != 1) {
    throw std::runtime_error("Model must have exactly one output");
  }

  TfLiteTensor *output_tensor = interpreter_->output_tensor(0);
  if (output_tensor->type != kTfLiteFloat32) {
    throw std::runtime_error("Output tensor must be float32");
  }

  // Expected shape: [1, num_classes]
  if (output_tensor->dims->size != 2) {
    throw std::runtime_error("Output tensor must have 2 dimensions, got: " +
                             std::to_string(output_tensor->dims->size));
  }

  num_output_classes_ = output_tensor->dims->data[1];

  std::cout << "TFLite model loaded successfully:" << std::endl
            << "  Input shape: [" << input_batch_size_ << ", "
            << input_landmarks_ << ", " << "]" << std::endl
            << "  Output classes: " << num_output_classes_ << std::endl;
}

std::optional<gestures::HandPositions> GestureDetector::performInference(
    const NormalizedLandmarkList &landmarks) {
  // Get input tensor pointer
  float *input_data = interpreter_->typed_input_tensor<float>(0);

  // Flatten landmarks into input tensor
  // For each landmark, add x, y coordinates
  for (int i = 0; i < landmarks.landmark_size() && i < input_landmarks_ / 3;
       i++) {
    const auto &landmark = landmarks.landmark(i);
    input_data[i * 3] = landmark.x();
    input_data[i * 3 + 1] = landmark.y();
    input_data[i * 3 + 2] = landmark.z();
  }

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    std::cerr << "Failed to invoke interpreter" << std::endl;
    return std::nullopt;
  }

  // Get output tensor
  float *output = interpreter_->typed_output_tensor<float>(0);

  // Find class with highest probability
  float max_prob = 0;
  int predicted_class = -1;

  for (int i = 0; i < num_output_classes_; i++) {
    if (output[i] > max_prob) {
      max_prob = output[i];
      predicted_class = i;
    }
  }

  const gestures::HandPositions hand_position =
      static_cast<gestures::HandPositions>(predicted_class);

  std::cout << "Predicted class: " << predicted_class
            << " with probability: " << max_prob << std::endl;

  last_hand_position_probability_ = max_prob;

  if (last_hand_position_ != hand_position) {
    last_hand_position_ = hand_position;
    last_hand_position_counter_++;
  }

  return hand_position;
}

}  // namespace gesture_detection
}  // namespace mediapipe
