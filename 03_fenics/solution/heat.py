from fenics import *
import math

T = 5.0
num_steps = 200
dt = T / num_steps

mesh = UnitSquareMesh(40, 40)
V = FunctionSpace(mesh, "P", 1)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

def down(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

bc_right = DirichletBC(V, Constant(0.0), right)
bc_down = DirichletBC(V, Constant(0.0), down)
bcs = [bc_right, bc_down]

u_n = interpolate(Constant(0.0), V)

u = TrialFunction(V)
v = TestFunction(V)

# Источник тепла с периодом
class HeartSource(UserExpression):
    def __init__(self, t=0.0, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        X = 3.0 * (x[0] - 0.5)
        Y = 3.0 * (x[1] - 0.55)

        heart = (X * X + Y * Y - 1.0)**3 - X * X * Y * Y * Y

        if heart <= 0.0:
            values[0] = 10.0 + 3.0 * math.sin(10.0 * self.t)
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()

f = HeartSource(t=0.0, degree=2)

a = u * v * dx + dt * dot(grad(u), grad(v)) * dx
L = (u_n + dt * f) * v * dx

u = Function(V)
vtkfile = File("heart_heat/solution.pvd")

t = 0.0
for n in range(num_steps):
    t += dt
    f.t = t

    solve(a == L, u, bcs)
    vtkfile << (u, t)
    u_n.assign(u)