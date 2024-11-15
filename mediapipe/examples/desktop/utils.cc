#include "mediapipe/examples/desktop/utils.h"

#include <iomanip>
#include <iostream>

cv::Mat loadPlanarRGBToMat(const std::vector<uint8_t>& planarData, int width,
                           int height) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  if (planarData.size() != width * height * 3) {
    throw std::invalid_argument("Data size does not match dimensions");
  }

  // Allocate an interleaved Mat for storing the final image
  cv::Mat image(height, width, CV_8UC3);

  // Pointers to each plane in the planar data
  const uint8_t* redPlane = planarData.data();
  const uint8_t* greenPlane = redPlane + width * height;
  const uint8_t* bluePlane = greenPlane + width * height;

  // Populate the interleaved Mat with horizontally flipped pixels
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Calculate flipped x coordinate (width - 1 - x)
      int flipped_x = width - 1 - x;
      image.at<cv::Vec3b>(y, x)[2] = bluePlane[y * width + flipped_x];  // Blue
      image.at<cv::Vec3b>(y, x)[1] =
          greenPlane[y * width + flipped_x];                           // Green
      image.at<cv::Vec3b>(y, x)[0] = redPlane[y * width + flipped_x];  // Red
    }
  }

  //   auto end_time = std::chrono::high_resolution_clock::now();
  //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
  //       end_time - start_time);
  //   std::cout << "Planar RGB conversion took " << duration.count() << "ms"
  //             << std::endl;

  return image;
}

FPSCounter::FPSCounter(int window_size)
    : window_size_(window_size),
      frame_count_(0),
      fps_(0.0),
      start_time_(std::chrono::high_resolution_clock::now()) {}

void FPSCounter::update() {
  frame_count_++;
  if (frame_count_ % window_size_ == 0) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_diff =
        std::chrono::duration<double>(current_time - start_time_).count();
    fps_ = window_size_ / time_diff;
    start_time_ = current_time;
  }
}

double FPSCounter::getFPS() const { return fps_; }

void FPSCounter::display() const {
  std::cout << "[" << frame_count_ << "] FPS: " << std::fixed
            << std::setprecision(1) << fps_ << " Hz" << std::endl;
}
