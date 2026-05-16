#include <algorithm>
#include <cstdio>
#include <string>
#include "core/grid.hpp"
#include "core/solver.hpp"
#include "core/materials.hpp"
#include "scenes/scene.hpp"
#include "scenes/laser_pulse.hpp"
#include "export/vtk_writer.hpp"
#include "export/probe_writer.hpp"

int main() {
  // MESH
  // Размер пятна 0.5 мм; коробка 2x2 мм, 50 мкм глубина
  // 64x64x32 узла
  const int Nx = 64;
  const int Ny = 64;
  const int Nz = 32;
  const double Lx = 2e-3;
  const double Ly = 2e-3;
  const double Lz = 5e-5;
 
  sim::SimMaterial mat;

  Grid grid(Nx, Ny, Nz, Lx, Ly, Lz, mat);
 
  // SCENE
  LaserPulse::Params lp;
  lp.I0       = 1e12;
  lp.tau_fwhm = 10e-9;
  lp.r_spot   = 5e-4;
  lp.T_init   = 300.0;
  lp.t_center = 3e-8;
 
  LaserPulse scene(lp);
  scene.init(grid);
  scene.bind_material(mat);
 
  // SOLVER
  double dt = Solver::stable_dt(grid, 0.4);
  std::printf("dt = %.3e s\n", dt);
  std::fflush(stdout);
 
  Solver solver(grid, scene.lateral_bc(), scene.laser_bc());
 
  // SIMULATION
  const double T_total    = 1e-6;
  const int    n_steps    = static_cast<int>(T_total / dt) + 1;
  const double dump_dt    = 2e-9;
  const int    dump_every = std::max(1, static_cast<int>(dump_dt / dt));
  std::printf("n_steps = %d, dump_every = %d\n", n_steps, dump_every);
  std::fflush(stdout);
 
  // WRITERS
  VtkWriter   vtk("results/laser", grid);
  ProbeWriter probe_center("results/laser_center.csv");
 
  vtk.write_frame(grid, solver.time());
  probe_center.write_point(grid, solver.time(), Nx / 2, Ny / 2, 0);
 
  // SOLVER CYCLE
  for (int s = 0; s < n_steps; ++s) {
    solver.step(grid, dt);
    if ((s + 1) % dump_every == 0) {
      vtk.write_frame(grid, solver.time());
      probe_center.write_point(grid, solver.time(), Nx / 2, Ny / 2, 0);
      std::printf("step %d / %d, t = %.3e s, T_center = %.1f K\n",
                  s + 1, n_steps, solver.time(),
                  grid.T(Nx / 2, Ny / 2, 0));
      std::fflush(stdout);
    }
  }
  vtk.finalize();
  std::printf("Done. ParaView: open results/laser.pvd\n");
  return 0;
}
