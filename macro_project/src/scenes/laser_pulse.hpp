#pragma once

#include "scenes/scene.hpp"

class LaserPulse : public Scene {
 public:
  struct Params {
    double I0      = 3e13;     // пиковая интенсивность, Вт/м^2 (= 3 ГВт/см^2)
    double tau_fwhm = 10e-9;   // длительность по FWHM, с
    double r_spot   = 5e-4;    // радиус пятна (1/e^2 в интенсивности), м
    double T_init   = 300.0;   // начальная температура образца, К
    double t_center = 3e-8;    // момент центра импульса, с (чтобы старт ~ 0)
  };

  explicit LaserPulse(Params p);

  void bind_material(const sim::SimMaterial& mat) { absorption_ = 1.0 - mat.R; }

  void init(Grid& grid) override;
  BoundaryType lateral_bc() const override { return BoundaryType::Adiabatic; }
  Laser laser_bc() const override;
  std::string name() const override { return "Laser pulse (Neumann BC)"; }

  const Params& params() const { return p_; }

 private:
  Params p_;
  double xc_ = 0.0;
  double yc_ = 0.0;
  double absorption_ = 1.0;
};