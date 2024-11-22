#pragma once

#include <filesystem>
#include <string_view>

namespace gestures {

class ProtoWriter {
 public:
  ProtoWriter() = default;

  void Initialize(std::filesystem::path path, int maxMessagesToKeep);

  void Write(std::string_view contents);

 private:
  std::string generateFilename();
  size_t clearOldFiles();

  std::filesystem::path path_;
  int maxMessagesToKeep_;
  int counter_ = 0;
};

}  // namespace gestures
