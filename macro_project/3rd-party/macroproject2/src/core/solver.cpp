#include "core/solver.hpp"
#include <algorithm>
#include <cmath>

Solver::Solver(const Grid& grid, BoundaryType bc_lateral, Laser laser)
  : H_new_(static_cast<size_t>(grid.nx()) * grid.ny() * grid.nz(), 0.0),
    bc_(bc_lateral),
    laser_(std::move(laser)) {}

double Solver::stable_dt(const Grid& grid, double safety) {
  const auto& m = grid.material();
  double chi_s = m.k_solid  / (m.rho * m.c_solid); //
  double chi_l = m.k_liquid / (m.rho * m.c_liquid);
  double chi   = std::max(chi_s, chi_l);
  double h_min = std::min({grid.dx(), grid.dy(), grid.dz()});
  return safety * h_min * h_min / (6.0 * chi);
}

static inline double k_face(double k1, double k2) {
  if (k1 + k2 <= 0.0) {
    return 0.0;
  }
  return 2.0 * k1 * k2 / (k1 + k2);
}

void Solver::step(Grid& grid, double dt) {
  const int nx = grid.nx();
  const int ny = grid.ny();
  const int nz = grid.nz();
  const double dx = grid.dx();
  const double dy = grid.dy();
  const double dz = grid.dz();

  auto idx = [nx, ny](int i, int j, int k) {
    return i + nx * (j + ny * k);
  };

  for (int k = 1; k < nz - 1; ++k) {
    for (int j = 1; j < ny - 1; ++j) {
      for (int i = 1; i < nx - 1; ++i) {
        double T0 = grid.T(i, j, k);
        double k0 = grid.conductivity(i, j, k);


        double Tx_p = grid.T(i + 1, j, k);
        double kx_p = k_face(k0, grid.conductivity(i + 1, j, k));
        double Tx_m = grid.T(i - 1, j, k);
        double kx_m = k_face(k0, grid.conductivity(i - 1, j, k));
        double divx = (kx_p * (Tx_p - T0) - kx_m * (T0 - Tx_m)) / (dx * dx);


        double Ty_p = grid.T(i, j + 1, k);
        double ky_p = k_face(k0, grid.conductivity(i, j + 1, k));
        double Ty_m = grid.T(i, j - 1, k);
        double ky_m = k_face(k0, grid.conductivity(i, j - 1, k));
        double divy = (ky_p * (Ty_p - T0) - ky_m * (T0 - Ty_m)) / (dy * dy);

        double Tz_p = grid.T(i, j, k + 1);
        double kz_p = k_face(k0, grid.conductivity(i, j, k + 1));
        double Tz_m = grid.T(i, j, k - 1);
        double kz_m = k_face(k0, grid.conductivity(i, j, k - 1));
        double divz = (kz_p * (Tz_p - T0) - kz_m * (T0 - Tz_m)) / (dz * dz);

        H_new_[idx(i, j, k)] = grid.H(i, j, k) + dt * (divx + divy + divz);
      }
    }
  }


  auto apply_lateral = [&](int i, int j, int k, int i_in, int j_in, int k_in) {
    if (bc_ == BoundaryType::Fixed) {
      H_new_[idx(i, j, k)] = grid.H(i, j, k);
    } else {
      H_new_[idx(i, j, k)] = H_new_[idx(i_in, j_in, k_in)];
    }
  };

  // Грани x=0 и x=nx-1
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j) {
      apply_lateral(0, j, k, 1, j, k);
      apply_lateral(nx - 1, j, k, nx - 2, j, k);
    }
  // Грани y=0 и y=ny-1
  for (int k = 0; k < nz; ++k)
    for (int i = 0; i < nx; ++i) {
      apply_lateral(i, 0, k, i, 1, k);
      apply_lateral(i, ny - 1, k, i, ny - 2, k);
    }
  // Нижняя грань z=nz-1=
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      apply_lateral(i, j, nz - 1, i, j, nz - 2);

  // ----- Верхняя грань z=0: либо адиабат, либо лазер -----
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const int k = 0;
      double T0 = grid.T(i, j, k);
      double k0 = grid.conductivity(i, j, k);

      double Tx_p = grid.T(i + 1, j, k);
      double kx_p = k_face(k0, grid.conductivity(i + 1, j, k));
      double Tx_m = grid.T(i - 1, j, k);
      double kx_m = k_face(k0, grid.conductivity(i - 1, j, k));
      double divx = (kx_p * (Tx_p - T0) - kx_m * (T0 - Tx_m)) / (dx * dx);

      double Ty_p = grid.T(i, j + 1, k);
      double ky_p = k_face(k0, grid.conductivity(i, j + 1, k));
      double Ty_m = grid.T(i, j - 1, k);
      double ky_m = k_face(k0, grid.conductivity(i, j - 1, k));
      double divy = (ky_p * (Ty_p - T0) - ky_m * (T0 - Ty_m)) / (dy * dy);

      // Поток вниз
      double Tz_p = grid.T(i, j, k + 1);
      double kz_p = k_face(k0, grid.conductivity(i, j, k + 1));
      double flux_down = -kz_p * (Tz_p - T0) / dz;

      double q_abs_top = 0.0;
      if (laser_.active()) {
        q_abs_top = laser_.q_abs(grid.x(i), grid.y(j), time_);
      }
      double flux_top = +q_abs_top;

      double divz = -(flux_down - flux_top) / dz;

      H_new_[idx(i, j, k)] = grid.H(i, j, k) + dt * (divx + divy + divz);
    }
  }
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        grid.H(i, j, k) = H_new_[idx(i, j, k)];
      }
    }
  }
  time_  += dt;
  steps_ += 1;
}