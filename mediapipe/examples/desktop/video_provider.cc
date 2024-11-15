#include "video_provider.h"

#include "utils.h"

namespace {
constexpr char kLuxonisSource[] = "luxonis";
constexpr char kWebcamSource[] = "webcam";
constexpr char kVideoSource[] = "video";
}  // namespace

bool VideoProvider::LoadLuxonis(const std::string& stream_name, int width,
                                int height) {
  sourceName_ = kLuxonisSource;
  luxonisStream_ = StreamFactory::CreateStream(stream_name);
  return luxonisStream_->Connect(width, height);
}

bool VideoProvider::LoadWebcam() {
  sourceName_ = kWebcamSource;
  capture_.open(0);
  return capture_.isOpened();
}

bool VideoProvider::LoadVideo(const std::string& video_path) {
  sourceName_ = kVideoSource;
  capture_.open(video_path);
  return capture_.isOpened();
}

cv::Mat VideoProvider::GetNextFrame() {
  if (sourceName_ == kLuxonisSource) {
    auto luxFrame = luxonisStream_->GetFrame();
    return loadPlanarRGBToMat(luxFrame.data, luxFrame.width, luxFrame.height);
  } else {
    cv::Mat frame;
    capture_ >> frame;
    cv::flip(frame, frame, /*flipcode=HORIZONTAL*/ 1);
    return frame;
  }
}
