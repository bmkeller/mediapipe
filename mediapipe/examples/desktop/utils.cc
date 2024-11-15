#include "mediapipe/examples/desktop/utils.h"

cv::Mat loadPlanarRGBToMat(const std::vector<uint8_t>& planarData, int width,
                           int height) {
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
      image.at<cv::Vec3b>(y, x)[0] = bluePlane[y * width + flipped_x];  // Blue
      image.at<cv::Vec3b>(y, x)[1] =
          greenPlane[y * width + flipped_x];                           // Green
      image.at<cv::Vec3b>(y, x)[2] = redPlane[y * width + flipped_x];  // Red
    }
  }

  return image;
}
