#include "mediapipe/examples/desktop/calibration.h"

#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace {

cv::Mat createCheckerboard(cv::Size boardSize, int squareSize) {
  // Add a 1-square margin around the board for a "quiet zone"
  int margin = squareSize * 0;
  cv::Size imageSize = cv::Size(boardSize.width * squareSize + margin * 2,
                                boardSize.height * squareSize + margin * 2);

  // Create a white canvas
  cv::Mat board(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  // Color for the dark squares
  cv::Scalar black = cv::Scalar(255, 0, 0);

  // Loop through the board squares
  for (int r = 0; r < boardSize.height; ++r) {
    for (int c = 0; c < boardSize.width; ++c) {
      // Only draw the black squares
      if ((r + c) % 2 == 0) {
        // Define the top-left corner of the square
        cv::Point topLeft(margin + c * squareSize, margin + r * squareSize);
        cv::Point bottomRight(topLeft.x + squareSize, topLeft.y + squareSize);

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

  // 1. Create a checkerboard image (or any other image)
  cv::Size boardDimensions(9, 6);
  int squareSize = 80;
  cv::Mat checkerboard = createCheckerboard(boardDimensions, squareSize);

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

  // 2. Create and display the temporary window
  std::string calibWindowName = "Checkerboard";
  cv::namedWindow(calibWindowName, cv::WINDOW_NORMAL);
  cv::setWindowProperty(calibWindowName, cv::WND_PROP_FULLSCREEN,
                        cv::WINDOW_FULLSCREEN);
  cv::imshow(calibWindowName, checkerboard);
}
