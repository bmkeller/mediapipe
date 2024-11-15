#pragma once

#include <vector>

#include "mediapipe/framework/port/opencv_imgproc_inc.h"

cv::Mat loadPlanarRGBToMat(const std::vector<uint8_t>& planarData, int width,
                           int height);
