// #include <depthai.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include "StreamFactory.h"
// #include "third_party/OpenCV/core.hpp"  // IWYU pragma: keep

int blahShouldDoubleIt(int x);
int blahSquartsIt(int x);
int findCameras();

constexpr int kBaseNumber = 12;

int main() {
  std::cout << "Hello, Bazel!" << std::endl;

  std::cout << "V2: Squaring " << kBaseNumber << " gives "
            << blahSquartsIt(kBaseNumber) << std::endl;

  // int numCameras = findCameras();
  // std::cout << "Found " << numCameras << " cameras" << std::endl;

  auto stream = StreamFactory::CreateStream("rgb");
  const bool success = stream->Connect();

  std::cout << "Stream connected: " << success << std::endl;

  for (int i = 0; i < 10; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto frameSize = stream->GetFrame();

    std::cout << "Frame size: " << frameSize.first << "x" << frameSize.second
              << std::endl;
  }

  return 0;
}
