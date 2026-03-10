#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkStructuredGrid.h>
#include <vtkSmartPointer.h>

#include "node.hpp"
#include "mesh.hpp"

#include <gmsh.h>



int main() {
  gmsh::initialize();

  Mesh mesh;
  mesh.LoadFromMsh("sword.msh");
  
  double sim_step = 1e-3;
  double sim_duration = 0.5;

  mesh.SetSim(sim_step, sim_duration);
  
  double c = 500;
  double v0 = 100;
  double sigma = 2;
  Node center(80, -74, 2);

  mesh.SetHit(c,v0, sigma, center);
  gmsh::finalize();

  mesh.RunSimulation();
  return 0;
}
