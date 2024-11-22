#include "mediapipe/mkel/proto_writer.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace {

constexpr char kExtension[] = ".bin";
void ensureDirectoryExists(std::filesystem::path path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  }
}

}

namespace gestures {

void ProtoWriter::Initialize(std::filesystem::path path, int maxMessagesToKeep) {
  path_ = path;
  maxMessagesToKeep_ = maxMessagesToKeep;
}

std::string ProtoWriter::generateFilename() {
  // Build a filename based on and append with a counter
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  
  std::stringstream filename;
  filename << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
  filename << "_" << std::setfill('0') << std::setw(3) << now_ms.count();
  filename << "_" << counter_++;
  filename << kExtension;

  return path_ / filename.str();
}

void ProtoWriter::Write(std::string_view contents) {
  ensureDirectoryExists(path_);

  auto filename = generateFilename();
  std::ofstream file(filename, std::ios::binary);
  file.write(contents.data(), contents.size());

  clearOldFiles();
}

size_t ProtoWriter::clearOldFiles() {
  std::vector<std::filesystem::path> matching_files;
  
  // 1. Collect all matching files
  for (const auto& entry : std::filesystem::directory_iterator(path_)) {
    if (entry.path().extension() == kExtension) {
      matching_files.push_back(entry.path());
    }
  }

  // 2. Sort files by name (timestamp-based naming ensures chronological order)
  std::sort(matching_files.begin(), matching_files.end());

  // 3. Delete oldest files if we have more than maxMessagesToKeep_
  while (matching_files.size() > maxMessagesToKeep_) {
    std::filesystem::remove(matching_files.front());
    matching_files.erase(matching_files.begin());
  }

  return matching_files.size();
}

}  // namespace gestures

