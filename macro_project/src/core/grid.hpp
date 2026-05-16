#pragma once

#include <vector>
#include "core/materials.hpp"

enum class Phase { Solid, Mushy, Liquid };

class Grid {
 private:
  int nx_, ny_, nz_;
  double Lx_, Ly_, Lz_;
  double dx_, dy_, dz_;
  sim::SimMaterial mat_;
  std::vector<double> data_;

  inline int idx(int i, int j, int k) const {
    return i + nx_ * (j + ny_ * k);
  }

 public:
  Grid(int nx, int ny, int nz,
       double Lx, double Ly, double Lz,
       sim::SimMaterial mat);

  double& H(int i, int j, int k);
  double  H(int i, int j, int k) const;

  double T(int i, int j, int k) const;
  Phase phase(int i, int j, int k) const;
  double conductivity(int i, int j, int k) const;
  double liquid_fraction(int i, int j, int k) const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  double dx() const { return dx_; }
  double dy() const { return dy_; }
  double dz() const { return dz_; }
  double Lx() const { return Lx_; }
  double Ly() const { return Ly_; }
  double Lz() const { return Lz_; }

  double x(int i) const { return (i + 0.5) * dx_; }
  double y(int j) const { return (j + 0.5) * dy_; }
  double z(int k) const { return (k + 0.5) * dz_; }

  const sim::SimMaterial& material() const { return mat_; }

  double H_from_T_solid(double T_kelvin) const;
  double H_from_T_liquid(double T_kelvin) const;
};