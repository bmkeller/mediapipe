#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/mkel/gesture_detector.h"
#include "mediapipe/mkel/gestures.pb.h"
#include "mediapipe/mkel/hand_landmarks.h"
#include "mediapipe/mkel/proto_writer.h"

namespace mediapipe {

class GestureRecognizerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);

  absl::Status Open(CalculatorContext *cc) override;

  absl::Status Process(CalculatorContext *cc) override;

  // Triggers when a gesture is detected.
  void OnGestureDetected();

 private:
  gestures::Gestures gesture_;
  gestures::ProtoWriter proto_writer_;
  gesture_detection::GestureDetector gesture_detector_;
  int frame_index_ = 0;
};

}  // namespace mediapipe
