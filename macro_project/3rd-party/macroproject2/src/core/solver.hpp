#pragma once

#include "core/grid.hpp"
#include "core/boundary.hpp"
#include <vector>

class Solver {
 private:
  std::vector<double> H_new_;
  BoundaryType bc_;
  Laser laser_;
  double time_  = 0.0;
  int steps_ = 0;

 public:
  explicit Solver(const Grid& grid, BoundaryType bc = BoundaryType::Adiabatic, Laser laser = {});
 
  void step(Grid& grid, double dt);
  double time() const {
    return time_;
  }
  int steps() const {
    return steps_;
  }

  static double stable_dt(const Grid& grid, double safety = 0.4);
};

