#include "scenes/laser_pulse.hpp"
#include <cmath>

LaserPulse::LaserPulse(Params p) : p_(p) {}

void LaserPulse::init(Grid& grid) {
  const int nx = grid.nx();
  const int ny = grid.ny();
  const int nz = grid.nz();

  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        grid.H(i, j, k) = grid.H_from_T_solid(p_.T_init);

  xc_ = 0.5 * grid.Lx();
  yc_ = 0.5 * grid.Ly();
}

Laser LaserPulse::laser_bc() const {
  const double I0       = p_.I0;
  const double absorp   = absorption_;
  const double tau      = p_.tau_fwhm;
  const double sigma_t  = tau / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  const double r_spot   = p_.r_spot;
  const double t_center = p_.t_center;
  const double xc       = xc_;
  const double yc       = yc_;

  Laser bc;
  bc.q_abs = [I0, absorp, sigma_t, r_spot, t_center, xc, yc]
             (double x, double y, double t) -> double {
    double dt = t - t_center;
    double time_factor = std::exp(-(dt * dt) / (2.0 * sigma_t * sigma_t));
    double rr = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    double space_factor = std::exp(-2.0 * rr / (r_spot * r_spot));
    return absorp * I0 * time_factor * space_factor;  // (1-R) * I(x,y,t)
  };
  return bc;
}