#pragma once

#include <chrono>
#include <filesystem>
#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

cv::Mat loadPlanarRGBToMat(const std::vector<uint8_t>& planarData, int width,
                           int height);

bool writeResultsToDisk(
    std::filesystem::path basePath, const cv::Mat& baseImage,
    const cv::Mat& overlayImage,
    const std::vector<mediapipe::NormalizedLandmarkList>& landmarks);

class FPSCounter {
 public:
  explicit FPSCounter(int window_size = 30);

  void update();
  double getFPS() const;
  std::string getFPSString() const;
  void display() const;

  int frame_count() const { return frame_count_; }

 private:
  const int window_size_;
  int frame_count_;
  double fps_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

class IntervalLogger {
 public:
  explicit IntervalLogger(std::chrono::duration<double> log_interval)
      : log_interval_(log_interval) {
    last_log_time_ = std::chrono::high_resolution_clock::now();
  }

  bool MaybeLog(int frame_count, double fps, int width, int height);

 private:
  const std::chrono::duration<double> log_interval_;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_log_time_;
};

template <typename K, typename V>
V get_with_default(const std::unordered_map<K, V>& map, const K& key,
                   const V& default_value) {
  auto it = map.find(key);
  if (it != map.end()) {
    return it->second;  // Key exists, return the value
  }
  return default_value;  // Key does not exist, return the default value
}