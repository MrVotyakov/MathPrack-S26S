#include <cmath>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include <gmsh.h>

int main(int argc, char **argv) {
  gmsh::initialize();

  gmsh::model::add("sword");

  try {
    gmsh::merge("./sword.stl");
  } catch (...) {
    gmsh::logger::write("Could not load STL mesh: bye!");
    gmsh::finalize();
    return 0;
  }

  double angle = 40;
  bool forceParametrizablePatches = false;
  bool includeBoundary = true;
  double curveAngle = 180;

  gmsh::model::mesh::classifySurfaces(angle * M_PI / 180., includeBoundary,
                                      forceParametrizablePatches,
                                      curveAngle * M_PI / 180.);

  gmsh::model::mesh::createGeometry();

  std::vector<std::pair<int, int>> s;
  gmsh::model::getEntities(s, 2);
  std::vector<int> sl;
  for (auto surf : s) {
    sl.push_back(surf.second);
  }
  int l = gmsh::model::geo::addSurfaceLoop(sl);
  gmsh::model::geo::addVolume({l});

  gmsh::model::geo::synchronize();

  const double h = 2;
  std::vector<std::pair<int, int>> points;
  gmsh::model::getEntities(points, 0);
  gmsh::model::mesh::setSize(points, h);
  gmsh::model::mesh::generate(3);

  gmsh::write("sword.msh");

  std::set<std::string> args(argv, argv + argc);
  if (!args.count("-nopopup"))
    gmsh::fltk::run();

  gmsh::finalize();

  return 0;
}
