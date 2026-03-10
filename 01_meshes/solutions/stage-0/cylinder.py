import gmsh
import sys

gmsh.initialize()


gmsh.model.add("cylinder")

lc = 1e-3

gmsh.model.occ.add_cylinder(0,0,0, 1, 2, 3, 0.5)

gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("cylinder.msh")
gmsh.write("cylinder.geo_unrolled")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()