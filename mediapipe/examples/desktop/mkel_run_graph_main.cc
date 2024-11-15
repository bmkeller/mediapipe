// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <memory>

#include "StreamFactory.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

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

  // Populate the interleaved Mat
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.at<cv::Vec3b>(y, x)[0] = bluePlane[y * width + x];   // Blue
      image.at<cv::Vec3b>(y, x)[1] = greenPlane[y * width + x];  // Green
      image.at<cv::Vec3b>(y, x)[2] = redPlane[y * width + x];    // Red
    }
  }

  return image;
}

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  bool useLuxonis = false;

  std::unique_ptr<MyStreamInterface> luxStream;

  if (load_video) {
    const std::string input_video_path = absl::GetFlag(FLAGS_input_video_path);
    std::cout << "Using video file: " << input_video_path << std::endl;

    if (input_video_path == "lux") {
      std::cout << "Using Luxonis stream." << std::endl;
      useLuxonis = true;
      load_video = false;

      luxStream = StreamFactory::CreateStream("rgb");
      luxStream->Connect(1440, 1080);
    } else {
      capture.open(input_video_path);
    }
  } else {
    std::cout << "Using webcam." << std::endl;
    capture.open(0);
  }

  if (!useLuxonis) {
    RET_CHECK(capture.isOpened());
  }

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 768);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  int frame_count = 0;

  // Add FPS calculation variables using chrono
  double fps = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();
  const int FPS_WINDOW = 30;  // Calculate FPS over 30 frames

  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;

    if (useLuxonis) {
      auto luxFrame = luxStream->GetFrame();
      // camera_frame_raw = cv::Mat(luxFrame.height, luxFrame.width, CV_8UC3,
      //                            luxFrame.data.data());
      camera_frame_raw =
          loadPlanarRGBToMat(luxFrame.data, luxFrame.width, luxFrame.height);

      // std::cout << "Input size: " << luxFrame.width << "x" << luxFrame.height
      //           << std::endl;
      // std::cout << "Raw frame: " << camera_frame_raw.cols << "x"
      //           << camera_frame_raw.rows << std::endl;

    } else {
      capture >> camera_frame_raw;
    }

    // Calculate FPS every FPS_WINDOW frames
    if (frame_count % FPS_WINDOW == 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto time_diff =
          std::chrono::duration<double>(current_time - start_time).count();
      fps = FPS_WINDOW / time_diff;
      start_time = current_time;

      std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps
                << std::endl;
    }

    // std::cout << "[" << frame_count++ << "] " << camera_frame_raw.cols << "x"
    //           << camera_frame_raw.rows << std::endl;

    if (camera_frame_raw.empty()) {
      if (!load_video) {
        ABSL_LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;

    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // std::cout << "Final frame size: " << camera_frame.cols << "x"
    //           << camera_frame.rows << std::endl;

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Prepare video writer.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  ABSL_LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
