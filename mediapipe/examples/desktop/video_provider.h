#pragma once

#include <memory>
#include <string>

#include "StreamFactory.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

class VideoProvider {
 public:
  VideoProvider(cv::VideoCapture& capture) : capture_(capture) {}

  bool LoadLuxonis(const std::string& stream_name, int width, int height);
  bool LoadWebcam();
  bool LoadVideo(const std::string& video_path);

  cv::Mat GetNextFrame();

  const std::string& SourceName() const { return sourceName_; }

 private:
  std::string sourceName_;
  std::unique_ptr<MyStreamInterface> luxonisStream_;

  cv::VideoCapture& capture_;
};
