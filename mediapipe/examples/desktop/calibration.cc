#include "mediapipe/examples/desktop/calibration.h"

#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace {

cv::Mat createCheckerboard(cv::Size imageSize, cv::Size boardSize) {
  // Add a 1-square margin around the board for a "quiet zone"
  int squareWidth = imageSize.width / boardSize.width;
  int squareHeight = imageSize.height / boardSize.height;

  // Create a white background canvas
  cv::Mat board(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  // Color for the dark squares
  cv::Scalar black = cv::Scalar(255, 0, 0);

  // Loop through the board squares
  for (int r = 0; r < boardSize.height; ++r) {
    for (int c = 0; c < boardSize.width; ++c) {
      // Only draw the black squares
      if ((r + c) % 2 == 0) {
        // Define the top-left corner of the square
        cv::Point topLeft(c * squareWidth, r * squareHeight);
        cv::Point bottomRight(topLeft.x + squareWidth,
                              topLeft.y + squareHeight);

        // Draw the filled black square
        cv::rectangle(board, topLeft, bottomRight, black, cv::FILLED);
      }
    }
  }

  return board;
}

}  // namespace

void Calibration::ShowCheckerboardWindow() {
  ABSL_LOG(INFO) << "Showing checkerboard window.";

  // Create a full screen window.
  constexpr char kWindowName[] = "Checkerboard";
  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
  cv::setWindowProperty(kWindowName, cv::WND_PROP_FULLSCREEN,
                        cv::WINDOW_FULLSCREEN);

  // Wait for the window to be created.
  //   cv::waitKey(3000);
  cv::Size windowSize(1710, 1107);

  ABSL_LOG(INFO) << "Window size: " << windowSize;

  // Generate a checkerboard image.
  cv::Size boardDimensions(9, 6);
  cv::Mat checkerboard = createCheckerboard(windowSize, boardDimensions);

  // In a real scenario, you would draw your checkerboard here.
  cv::putText(checkerboard, "Calibration Screen - Press ESC to close",
              cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 255, 0), 2);

  cv::Scalar lineColor = cv::Scalar(0, 255, 0);
  int lineThickness = 10;

  // Line 1: Top-left to bottom-right
  cv::line(checkerboard, cv::Point(0, 0),
           cv::Point(checkerboard.cols, checkerboard.rows), lineColor,
           lineThickness);

  // Line 2: Top-right to bottom-left
  cv::line(checkerboard, cv::Point(checkerboard.cols, 0),
           cv::Point(0, checkerboard.rows), lineColor, lineThickness);

  cv::imshow(kWindowName, checkerboard);
}
