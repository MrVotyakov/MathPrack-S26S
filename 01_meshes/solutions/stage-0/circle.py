import gmsh
import sys

gmsh.initialize()
gmsh.model.add("circle2d")

r = 0.5
lc = 0.03

gmsh.model.geo.addPoint(0, 0, 0, lc, 1)

gmsh.model.geo.addPoint( r, 0, 0, lc, 2)
gmsh.model.geo.addPoint( 0, r, 0, lc, 3)
gmsh.model.geo.addPoint(-r, 0, 0, lc, 4)
gmsh.model.geo.addPoint( 0,-r, 0, lc, 5)

gmsh.model.geo.addCircleArc(2, 1, 3, 1)
gmsh.model.geo.addCircleArc(3, 1, 4, 2)
gmsh.model.geo.addCircleArc(4, 1, 5, 3)
gmsh.model.geo.addCircleArc(5, 1, 2, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("circle.msh")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()