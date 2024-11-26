#include "mediapipe/mkel/hand_landmarks.h"

namespace mediapipe {

LandmarkMap generateLandmarkMap(const NormalizedLandmarkList &landmarks) {
  LandmarkMap landmark_map;
  for (int i = kFirstLandmarkIndex; i <= kLastLandmarkIndex; i++) {
    if (landmarks.landmark_size() > i) {
      landmark_map[static_cast<LandmarkType>(i)] = landmarks.landmark(i);
    }
  }
  return landmark_map;
}

}  // namespace mediapipe
