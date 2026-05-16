#pragma once
namespace sim {
struct SimMaterial {
  double k_solid   = 237.0;       // теплопроводность твёрдого Al, Вт/(м·К)
  double k_liquid  = 91.0;        // теплопроводность жидкого Al, Вт/(м·К)
  double c_solid   = 900.0;       // теплоёмкость твёрдого Al, Дж/(кг·К)
  double c_liquid  = 1180.0;      // теплоёмкость жидкого Al, Дж/(кг·К)
  double L         = 397000.0;    // удельная теплота плавления, Дж/кг
  double T_melt    = 933.47;      // температура плавления, К
  double rho       = 2700.0;      // плотность, кг/м^3
  double T_ref     = 300.0;       // опорная температура, К

  double R         = 0.3;       // albedo
  double delta_opt  = 1e-8;      // length of optical penetration

  double E_young   = 70e9;        // модуль Юнга, Па
  double nu        = 0.33;        // коэффициент Пуассона
  double alpha_T   = 23.1e-6;     // коэффициент линейного теплового расширения, 1/К
  double dV_over_V_melt = 0.065;  // скачок объёма при плавлении (Al)
};

}