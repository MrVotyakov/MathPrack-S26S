#include "export/probe_writer.hpp"
/// ОСТОРОЖНО ВАЙБКОД!
#include <fstream>
#include <string>

ProbeWriter::ProbeWriter(const std::string& filename) : f_(filename) {}

ProbeWriter::~ProbeWriter() {
  f_.close();
}

void ProbeWriter::write_point(const Grid& grid, double t, int i, int j, int k) {
  if (!header_written_) {
    f_ << "t,T,liquid_fraction,H\n";
    header_written_ = true;
  }
  f_ << t << ","
     << grid.T(i, j, k) << ","
     << grid.liquid_fraction(i, j, k) << ","
     << grid.H(i, j, k) << "\n";
}

void ProbeWriter::write_front_x(const Grid& grid, double t) {
  if (!header_written_) {
    f_ << "t,x_front\n";
    header_written_ = true;
  }
  int nx = grid.nx();
  int j = grid.ny() / 2;
  int k = grid.nz() / 2;
  double x_front = -1.0;
  // Идём слева направо: ищем первый узел, который НЕ Solid (с x=0 жидкость).
  // Затем продолжаем — фронт это последний узел, где ещё нет Solid'а.
  for (int i = 0; i < nx; ++i) {
    if (grid.phase(i, j, k) == Phase::Solid) {
      x_front = grid.x(i);
      break;
    }
  }
  if (x_front >= 0.0) {
    f_ << t << "," << x_front << "\n";
  }
}