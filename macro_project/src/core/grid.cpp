#include "core/grid.hpp"

Grid::Grid(int nx, int ny, int nz,
           double Lx, double Ly, double Lz,
           sim::SimMaterial mat)
  : nx_(nx), ny_(ny), nz_(nz),
    Lx_(Lx), Ly_(Ly), Lz_(Lz),
    dx_(Lx / nx), dy_(Ly / ny), dz_(Lz / nz),
    mat_(mat),
    data_(static_cast<size_t>(nx) * ny * nz, 0.0) {
  }

double& Grid::H(int i, int j, int k) {
  return data_[i + nx_ * (j + ny_ * k)];
}

double Grid::H(int i, int j, int k) const {
  return data_[i + nx_ * (j + ny_ * k)];
}

double Grid::T(int i, int j, int k) const {
  double h = H(i, j, k);
  double Hs = mat_.rho * mat_.c_solid * (mat_.T_melt - mat_.T_ref);
  double Hl = Hs + mat_.rho * mat_.L;
  if (h < Hs) {
    return mat_.T_ref + h / (mat_.rho * mat_.c_solid);
  } else if (h > Hl) {
    return mat_.T_melt + (h - Hl) / (mat_.rho * mat_.c_liquid);
  } else {
    return mat_.T_melt;
  }
}


Phase Grid::phase(int i, int j, int k) const {
  double h = H(i, j, k);
  double Hs = mat_.rho * mat_.c_solid * (mat_.T_melt - mat_.T_ref);
  double Hl = Hs + mat_.rho * mat_.L;
  if (h < Hs){ 
    return Phase::Solid;
  }
  if (h > Hl){ 
    return Phase::Liquid;
  }
  return Phase::Mushy;
}

double Grid::conductivity(int i, int j, int k) const {
  double h = H(i, j, k);
  double Hs = mat_.rho * mat_.c_solid * (mat_.T_melt - mat_.T_ref);
  double Hl = Hs + mat_.rho * mat_.L;

  if (h < Hs) {
    return mat_.k_solid;
  }

  if (h > Hl) {
    return mat_.k_liquid;
  }

  double frac = (h - Hs) / (Hl - Hs);
  
  return (1.0 - frac) * mat_.k_solid + frac * mat_.k_liquid;
}

double Grid::H_from_T_solid(double T_kelvin) const {
  return mat_.rho * mat_.c_solid * (T_kelvin - mat_.T_ref);
}
 
double Grid::H_from_T_liquid(double T_kelvin) const {
  double Hs = mat_.rho * mat_.c_solid * (mat_.T_melt - mat_.T_ref);
  double Hl = Hs + mat_.rho * mat_.L;
  return Hl + mat_.rho * mat_.c_liquid * (T_kelvin - mat_.T_melt);
}

double Grid::liquid_fraction(int i, int j, int k) const {
  double h = H(i, j, k);
  double Hs = mat_.rho * mat_.c_solid * (mat_.T_melt - mat_.T_ref);
  double Hl = Hs + mat_.rho * mat_.L;
  if (h < Hs) return 0.0;
  if (h > Hl) return 1.0;
  return (h - Hs) / (Hl - Hs);
}