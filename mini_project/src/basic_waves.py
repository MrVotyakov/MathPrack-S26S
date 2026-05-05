from pathlib import Path

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


from mini_project.src.tools.preprocess import load_head_model_from_vtu


BASE_DIR = Path(__file__).resolve().parent

VTU_PATH = BASE_DIR.parent / "3d-models" / "head_with_materials.vtu"

OUT_PATH = BASE_DIR / "wave_animation.pvd"

# Скорость волны в условных единицах.
C0 = 1.0 

# Шаг по времени.
DT = 0.2 

# Число кадров
NUM_STEPS = 80

# Записывать каждый N-й шаг
WRITE_EVERY = 1

# Ширина начального гауссова импульса
SIGMA_FRACTION = 0.08


# ------------------------------------------------------------
# Простая волновая задача
# ------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD


    print("Loading model...", flush=True)

    model = load_head_model_from_vtu(VTU_PATH)

    domain = model.domain

    print("domain:", domain, flush=True)


    # Скалярное пространство:
    # в каждой вершине сетки храним одно число u(x,t)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Неизвестная и тестовая функция
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    dx = ufl.dx(domain)

    # Берём rho из твоей модели.
    # rho — DG0-функция, одно значение на ячейку.
    rho = model.rho

    # --------------------------------------------------------
    # Формы для уравнения:
    #
    #     rho * u_tt = div(rho * C0^2 * grad u)
    #
    # Слабая форма:
    #
    #     M * u_tt + K * u = 0
    #
    # где
    #     M_ij = ∫ rho * phi_i * phi_j dx
    #     K_ij = ∫ rho * C0^2 * grad(phi_i)·grad(phi_j) dx
    # --------------------------------------------------------

    mass_lumped_form = rho * v_test * dx

    stiffness_form = (
        rho
        * (C0 ** 2)
        * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test))
        * dx
    )

    print("Assembling lumped mass...", flush=True)

    # Lumped mass:
    # M_lumped_i = ∫ rho * phi_i dx
    # Это удобно для явной схемы.
    mass_vec = assemble_vector(fem.form(mass_lumped_form))
    mass_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    mass_array = mass_vec.getArray(readonly=True).copy()

    if np.any(mass_array <= 0.0):
        raise RuntimeError("В mass_array есть неположительные значения.")

    print("Assembling stiffness matrix...", flush=True)

    K = assemble_matrix(fem.form(stiffness_form))
    K.assemble()

    # --------------------------------------------------------
    # Функции для временной схемы
    # --------------------------------------------------------

    u_nm1 = fem.Function(V)
    u_n = fem.Function(V)
    u_np1 = fem.Function(V)

    u_nm1.name = "wave"
    u_n.name = "wave"
    u_np1.name = "wave"

    # --------------------------------------------------------
    # Начальное условие: гауссов импульс около одной стороны модели
    # --------------------------------------------------------

    x = domain.geometry.x

    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])
    z_min = np.min(x[:, 2])
    z_max = np.max(x[:, 2])

    bbox_size = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    sigma = SIGMA_FRACTION * bbox_size

    # Источник ставим ближе к левой стороне модели
    source_center = np.array(
        [
            0.5 * (x_max + x_min),
            0.5 * (y_min + y_max),
            z_max,
        ],
        dtype=np.float64,
    )

    print("source center:", source_center, flush=True)
    print("sigma:", sigma, flush=True)

    def initial_gaussian(x):
        """
        x имеет форму (3, N).
        Возвращаем массив значений размера N.
        """
        r2 = (
            (x[0] - source_center[0]) ** 2
            + (x[1] - source_center[1]) ** 2
            + (x[2] - source_center[2]) ** 2
        )

        return np.exp(-r2 / (2.0 * sigma ** 2))

    u_n.interpolate(initial_gaussian)

    # Нулевая начальная скорость:
    # u^{-1} = u^0
    u_nm1.x.array[:] = u_n.x.array[:]

    u_n.x.scatter_forward()
    u_nm1.x.scatter_forward()

    # Временный вектор для K * u_n
    Ku = K.createVecRight()

    print("Writing animation:", OUT_PATH, flush=True)

    # VTKFile пишет:
    # wave_animation.pvd
    # wave_animation/*.vtu или похожие piece-файлы
    with io.VTKFile(comm, OUT_PATH.as_posix(), "w") as vtk:
        vtk.write_function(u_n, 0.0)

        for step in range(1, NUM_STEPS + 1):
            t = step * DT

            # Ku = K * u_n
            K.mult(u_n.x.petsc_vec, Ku)

            ku_array = Ku.getArray(readonly=True)

            # Явная схема:
            #
            # M * (u^{n+1} - 2u^n + u^{n-1}) / dt^2 + K u^n = 0
            #
            # значит:
            #
            # u^{n+1} = 2u^n - u^{n-1} - dt^2 * M^{-1} K u^n

            u_np1.x.array[:] = (
                2.0 * u_n.x.array[:]
                - u_nm1.x.array[:]
                - (DT ** 2) * ku_array / mass_array
            )

            u_np1.x.scatter_forward()

            # Сдвигаем временные слои
            u_nm1.x.array[:] = u_n.x.array[:]
            u_n.x.array[:] = u_np1.x.array[:]

            u_nm1.x.scatter_forward()
            u_n.x.scatter_forward()

            if step % WRITE_EVERY == 0:
                vtk.write_function(u_n, t)

            max_u = np.max(np.abs(u_n.x.array))

            print(
                f"step={step:04d}, time={t:.3f}, max|u|={max_u:.6e}",
                flush=True,
            )

    print("\nDone.", flush=True)
    print("Open in ParaView:", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
