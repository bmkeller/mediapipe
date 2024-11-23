#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <unordered_map>
#include <vector>

#include "StreamFactory.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/examples/desktop/utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "utils.h"
#include "video_provider.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";
constexpr int kFpsWindowSize = 30;  // Calculate FPS over 30 frames
constexpr char kDataCollectionDir[] =
    "/Users/michaelkeller/data_collection/hand_poses/";
constexpr int kWindowWidth = 1024;
constexpr int kWindowHeight = 768;

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

ABSL_FLAG(std::string, video_source, "webcam", "'webcam', 'lux', 'video.mp4'");

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

  const std::unordered_map<char, std::string> collection_mappings = {
      {'1', "index_point"}, {'2', "thumbs_up"}, {'3', "fist"}};

  for (const auto& [_, value] : collection_mappings) {
    std::filesystem::path p = kDataCollectionDir;
    p /= value;
    if (!std::filesystem::exists(p)) {
      std::filesystem::create_directories(p);
    }
  }

  const std::string videoSource = absl::GetFlag(FLAGS_video_source);
  cv::VideoCapture capture;
  VideoProvider videoProvider(capture);
  bool loadSuccess = false;

  if (videoSource == "lux") {
    loadSuccess = videoProvider.LoadLuxonis("rgb", 1440, 1080);
  } else if (videoSource == "video") {
    loadSuccess =
        videoProvider.LoadVideo(absl::GetFlag(FLAGS_input_video_path));
  } else {
    loadSuccess = videoProvider.LoadWebcam();
  }

  if (!loadSuccess) {
    return absl::InvalidArgumentError("Failed to load video source: " +
                                      videoSource);
  }

  ABSL_LOG(INFO) << "Successfully load video source: " << videoSource;

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    capture.set(cv::CAP_PROP_FRAME_WIDTH, kWindowWidth);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, kWindowHeight);
    capture.set(cv::CAP_PROP_FPS, 30);
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller image_poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller multi_norm_landmarks_poller,
                      graph.AddOutputStreamPoller("multi_hand_landmarks"));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  int frame_count = 0;

  FPSCounter fps_counter(kFpsWindowSize);

  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame = videoProvider.GetNextFrame();

    fps_counter.update();
    fps_counter.display();

    if (camera_frame.empty()) {
      if (videoProvider.SourceName() == "webcam") {
        ABSL_LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }

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
    mediapipe::Packet image_packet;
    if (!image_poller.Next(&image_packet)) break;
    auto& output_frame = image_packet.Get<mediapipe::ImageFrame>();

    mediapipe::Packet multi_norm_landmarks_packet;
    if (!multi_norm_landmarks_poller.Next(&multi_norm_landmarks_packet)) {
      ABSL_LOG(ERROR) << "Failed to get multi norm landmarks packet";
    }
    auto multi_norm_landmarks =
        multi_norm_landmarks_packet
            .Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    // std::cout << "Multi norm landmarks: " << multi_norm_landmarks.size()
    //           << std::endl;

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
      const int pressed_key = cv::waitKey(5);

      if (pressed_key >= 0 && pressed_key != 255) {
        const char c = static_cast<char>(pressed_key);
        if (c == 'q' || c == 'Q') {
          grab_frames = false;
        } else {
          const std::string collection_name =
              get_with_default<char, std::string>(collection_mappings, c, "");
          if (!collection_name.empty()) {
            cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
            writeResultsToDisk(
                std::filesystem::path(kDataCollectionDir) / collection_name,
                camera_frame, output_frame_mat, multi_norm_landmarks);
          }
        }
      }
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
  const absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
