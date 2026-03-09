import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("torus")

R = 1.0
r_out = 0.2
lc = 0.01
r_in = r_out - 3 * lc

c = gmsh.model.geo.addPoint(R, 0, 0, lc)

p1 = gmsh.model.geo.addPoint(R + r_out, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(R, 0,  r_out, lc)
p3 = gmsh.model.geo.addPoint(R - r_out, 0, 0, lc)
p4 = gmsh.model.geo.addPoint(R, 0, -r_out, lc)

i1 = gmsh.model.geo.addPoint(R + r_in, 0, 0, lc)
i2 = gmsh.model.geo.addPoint(R, 0,  r_in, lc)
i3 = gmsh.model.geo.addPoint(R - r_in, 0, 0, lc)
i4 = gmsh.model.geo.addPoint(R, 0, -r_in, lc)

a1 = gmsh.model.geo.addCircleArc(p1, c, p2, nx=0, ny=1, nz=0)
a2 = gmsh.model.geo.addCircleArc(p2, c, p3, nx=0, ny=1, nz=0)
a3 = gmsh.model.geo.addCircleArc(p3, c, p4, nx=0, ny=1, nz=0)
a4 = gmsh.model.geo.addCircleArc(p4, c, p1, nx=0, ny=1, nz=0)

b1 = gmsh.model.geo.addCircleArc(i1, c, i2, nx=0, ny=1, nz=0)
b2 = gmsh.model.geo.addCircleArc(i2, c, i3, nx=0, ny=1, nz=0)
b3 = gmsh.model.geo.addCircleArc(i3, c, i4, nx=0, ny=1, nz=0)
b4 = gmsh.model.geo.addCircleArc(i4, c, i1, nx=0, ny=1, nz=0)

loop_out = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
loop_in = gmsh.model.geo.addCurveLoop([b1, b2, b3, b4])
disk = gmsh.model.geo.addPlaneSurface([loop_out, loop_in])

volumes = []
section = [(2, disk)]

for _ in range(4):
    out = gmsh.model.geo.revolve(section, 0, 0, 0, 0, 0, 1, math.pi / 2)
    section = [out[0]]
    volumes.append(out[1][1])

gmsh.model.geo.synchronize()

pg = gmsh.model.addPhysicalGroup(3, volumes)
gmsh.model.setPhysicalName(3, pg, "Torus")

gmsh.model.mesh.generate(3)
gmsh.write("torus.msh")
gmsh.write("torus.geo_unrolled")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()