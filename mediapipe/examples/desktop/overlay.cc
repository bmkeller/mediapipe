#include "mediapipe/examples/desktop/overlay.h"

namespace mkel {

void drawTextOnImage(cv::Mat& image, const std::string& text,
                     const cv::Point& org, const cv::Scalar& color,
                     int font_face, double font_scale, int thickness) {
  cv::putText(image, text, org, font_face, font_scale, color, thickness);
}

cv::Size getTextSize(const std::string& text, int font_face, double font_scale,
                     int thickness) {
  int baseline = 0;
  return cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
}

void drawDetectedGesture(cv::Mat& image, const std::string& gesture) {
  // Get text size
  cv::Size text_size = getTextSize(gesture, cv::FONT_HERSHEY_SIMPLEX, 1, 2);

  // Calculate position to ensure text fits within image bounds
  int x = std::max(0, std::min(image.cols - text_size.width, image.cols - 200));

  drawTextOnImage(image, gesture, cv::Point(x, 30), cv::Scalar(0, 255, 0),
                  cv::FONT_HERSHEY_SIMPLEX, 1, 2);
}

}  // namespace mkel
