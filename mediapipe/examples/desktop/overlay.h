#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace mkel {

void drawTextOnImage(cv::Mat& image, const std::string& text,
                     const cv::Point& org, const cv::Scalar& color,
                     int font_face, double font_scale, int thickness);

void drawDetectedGesture(cv::Mat& image, const std::string& gesture);

}  // namespace mkel
