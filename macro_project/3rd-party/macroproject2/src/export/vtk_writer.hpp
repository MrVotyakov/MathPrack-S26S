#pragma once

#include "core/grid.hpp"
#include <string>
#include <vector>

class VtkWriter {
 public:
  VtkWriter(const std::string& prefix, const Grid& grid);
  ~VtkWriter();

  void write_frame(const Grid& grid, double time);
  void finalize();

 private:
  std::string prefix_;
  int frame_index_ = 0;
  bool finalized_ = false;
  std::vector<std::pair<double, std::string>> frames_;
};