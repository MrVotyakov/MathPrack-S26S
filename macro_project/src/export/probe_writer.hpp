#pragma once

#include "core/grid.hpp"
#include <fstream>
#include <string>

class ProbeWriter {
 public:
  explicit ProbeWriter(const std::string& filename);
  ~ProbeWriter();
  void write_point(const Grid& grid, double t, int i, int j, int k);

  void write_front_x(const Grid& grid, double t);

 private:
  std::ofstream f_;
  bool header_written_ = false;
};