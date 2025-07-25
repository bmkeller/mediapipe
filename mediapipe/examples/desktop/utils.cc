#include "mediapipe/examples/desktop/utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json

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

bool IntervalLogger::MaybeLog(int frame_count, double fps, int width,
                              int height) {
  auto now = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration<double>(now - last_log_time_).count();
  if (time_diff < log_interval_.count()) {
    return false;
  }

  last_log_time_ = now;

  std::cout << absl::StrFormat("[%d @ %1.1f Hz]: Res=%dx%d", frame_count, fps,
                               width, height)
            << std::endl;

  return true;
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

std::string FPSCounter::getFPSString() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << fps_ << " Hz";
  return oss.str();
}

std::string convertLandmarksToJson(
    const std::vector<mediapipe::NormalizedLandmarkList>& landmarks) {
  nlohmann::json json_landmarks;
  json_landmarks["num_hands"] = landmarks.size();
  json_landmarks["hands"] = nlohmann::json::array();

  for (size_t hand_idx = 0; hand_idx < landmarks.size(); ++hand_idx) {
    nlohmann::json hand;
    hand["hand_index"] = hand_idx;
    hand["landmarks"] = nlohmann::json::array();

    const auto& landmark_list = landmarks[hand_idx];
    for (int i = 0; i < landmark_list.landmark_size(); ++i) {
      const auto& landmark = landmark_list.landmark(i);
      nlohmann::json point = {
          {"x", landmark.x()},
          {"y", landmark.y()},
          {"z", landmark.z()},
          {"visibility",
           landmark.has_visibility() ? landmark.visibility() : 0.0},
          {"presence", landmark.has_presence() ? landmark.presence() : 0.0}};
      hand["landmarks"].push_back(point);
    }

    json_landmarks["hands"].push_back(hand);
  }

  return json_landmarks.dump(2);
}

bool writeResultsToDisk(
    std::filesystem::path basePath, const cv::Mat& baseImage,
    const cv::Mat& overlayImage,
    const std::vector<mediapipe::NormalizedLandmarkList>& landmarks) {
  if (landmarks.size() != 1) {
    return false;
  }
  // Get current time
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);

  // Format timestamp
  std::stringstream timestamp;
  std::tm* tm = std::localtime(&time_t_now);
  timestamp << std::put_time(tm, "%Y%m%d_%H%M%S");

  // Generate filenames with timestamp
  std::filesystem::path regular_filename =
      basePath / (timestamp.str() + ".jpg");
  std::filesystem::path overlay_filename =
      basePath / ("overlay_" + timestamp.str() + ".jpg");

  // Write images
  cv::imwrite(regular_filename.string(), baseImage);
  cv::imwrite(overlay_filename.string(), overlayImage);

  std::filesystem::path landmarks_filename =
      basePath / ("landmarks_" + timestamp.str() + ".json");

  // Write out landmarks
  std::ofstream landmarks_file(landmarks_filename.string());
  landmarks_file << convertLandmarksToJson(landmarks);

  std::cout << "Wrote landmarks to " << landmarks_filename.string()
            << std::endl;
  landmarks_file.close();
}
