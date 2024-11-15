#pragma once

#include <chrono>
#include <vector>

#include "mediapipe/framework/port/opencv_imgproc_inc.h"

cv::Mat loadPlanarRGBToMat(const std::vector<uint8_t>& planarData, int width,
                           int height);

class FPSCounter {
 public:
  explicit FPSCounter(int window_size = 30);

  void update();
  double getFPS() const;
  void display() const;

 private:
  const int window_size_;
  int frame_count_;
  double fps_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};
