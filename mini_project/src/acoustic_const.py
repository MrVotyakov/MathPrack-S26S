from pathlib import Path
import shutil

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

from mini_project.src.tools.preprocess import load_head_model_from_vtu


BASE_DIR = Path(__file__).resolve().parent

VTU_PATH = BASE_DIR.parent / "3d-models" / "head_with_materials.vtu"

# Папка с отдельными .vtu кадрами
OUT_DIR = BASE_DIR / "acoustic_vtu_frames_1"

# Скорость звука в условных единицах.
C0 = 1.0

# Шаг по времени.
DT = 0.1

# Число шагов
NUM_STEPS = 100

# Записывать каждый N-й шаг
WRITE_EVERY = 5

# Ширина гауссова профиля источника
SIGMA_FRACTION = 0.1

# Амплитуда постоянного во времени источника
SOURCE_AMPLITUDE = 1.0

# Порог аварийной остановки, если решение явно взорвалось
MAX_ALLOWED_PRESSURE = 1.0e6


# ------------------------------------------------------------
# Вспомогательные проверки
# ------------------------------------------------------------

def check_array_global(comm, array, name):
    """
    Проверяет массив на NaN/inf и печатает глобальные min/max.
    Работает корректнее при MPI-запуске.
    """
    local_finite = np.all(np.isfinite(array))

    if array.size > 0:
        local_min = np.min(array)
        local_max = np.max(array)
        local_max_abs = np.max(np.abs(array))
    else:
        local_min = np.inf
        local_max = -np.inf
        local_max_abs = 0.0

    global_finite = comm.allreduce(local_finite, op=MPI.LAND)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_max_abs = comm.allreduce(local_max_abs, op=MPI.MAX)

    print(
        f"{name}: min={global_min:.6e}, "
        f"max={global_max:.6e}, "
        f"max_abs={global_max_abs:.6e}, "
        f"finite={global_finite}",
        flush=True,
    )

    if not global_finite:
        raise RuntimeError(f"{name} contains NaN or inf")

    return global_min, global_max, global_max_abs


def write_vtu_frame(comm, function, out_dir, step, time):
    """
    Пишет один .vtu кадр.

    Важно:
    - если запускать в 1 MPI процесс, будет обычный .vtu;
    - если запускать через mpirun с несколькими процессами,
      Dolfinx/VTK может создать parallel VTK-структуру.
    """
    filename = out_dir / f"pressure_{step:04d}.vtu"

    with io.VTKFile(comm, filename.as_posix(), "w") as vtk:
        vtk.write_function(function, time)

    print(f"wrote: {filename}", flush=True)


# ------------------------------------------------------------
# Акустическая волновая задача для давления p(x, t)
# ------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print("Loading model...", flush=True)

    model = load_head_model_from_vtu(VTU_PATH)
    domain = model.domain

    print("domain:", domain, flush=True)

    # Чистим старые VTU-кадры
    if rank == 0:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    comm.barrier()

    # Скалярное пространство:
    # в каждой вершине сетки храним одно число p(x,t) — давление.
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Неизвестная и тестовая функция
    p_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    dx = ufl.dx(domain)

    # rho — DG0-функция, одно значение на ячейку.
    rho = model.rho

    # --------------------------------------------------------
    # Акустическое уравнение с постоянным источником:
    #
    #     1 / (rho * c^2) * p_tt
    #       - div(1 / rho * grad(p))
    #       = s(x)
    #
    # Слабая форма:
    #
    #     ∫ 1/(rho*c^2) * p_tt * v dx
    #   + ∫ 1/rho * grad(p)·grad(v) dx
    #   = ∫ s * v dx
    #
    # После FEM:
    #
    #     M * p_tt + K * p = F
    #
    # Явная схема:
    #
    #     p^{n+1}
    #       = 2p^n - p^{n-1}
    #         + dt^2 * M^{-1} * (F - Kp^n)
    # --------------------------------------------------------

    inv_bulk_modulus = 1.0 / (rho * (C0 ** 2))
    inv_density = 1.0 / rho

    mass_lumped_form = inv_bulk_modulus * v_test * dx

    stiffness_form = (
        inv_density
        * ufl.inner(ufl.grad(p_trial), ufl.grad(v_test))
        * dx
    )

    print("Assembling lumped acoustic mass...", flush=True)

    mass_vec = assemble_vector(fem.form(mass_lumped_form))
    mass_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    mass_array = mass_vec.getArray(readonly=True).copy()

    check_array_global(comm, mass_array, "mass_array")

    if np.any(mass_array <= 0.0):
        raise RuntimeError("В mass_array есть неположительные значения.")

    print("Assembling acoustic stiffness matrix...", flush=True)

    K = assemble_matrix(fem.form(stiffness_form))
    K.assemble()

    # --------------------------------------------------------
    # Функции для временной схемы
    # --------------------------------------------------------

    p_nm1 = fem.Function(V)
    p_n = fem.Function(V)
    p_np1 = fem.Function(V)

    p_nm1.name = "pressure"
    p_n.name = "pressure"
    p_np1.name = "pressure"

    # --------------------------------------------------------
    # Геометрия модели и положение источника
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

    source_center = np.array(
        [
            0.5 * (x_max + x_min),
            y_min + 0.4 * (-y_min + y_max),
            z_min,
        ],
        dtype=np.float64,
    )

    print("source center:", source_center, flush=True)
    print("sigma:", sigma, flush=True)

    def gaussian_profile(x):
        """
        Пространственный профиль источника.
        x имеет форму (3, N).
        Возвращаем массив размера N.
        """
        r2 = (
            (x[0] - source_center[0]) ** 2
            + (x[1] - source_center[1]) ** 2
            + (x[2] - source_center[2]) ** 2
        )

        return np.exp(-r2 / (2.0 * sigma ** 2))

    # --------------------------------------------------------
    # Постоянный во времени источник s(x)
    # --------------------------------------------------------

    source = fem.Function(V)
    source.name = "source"

    def source_gaussian(x):
        return SOURCE_AMPLITUDE * gaussian_profile(x)

    source.interpolate(source_gaussian)
    source.x.scatter_forward()

    check_array_global(comm, source.x.array, "source")

    source_nonzero_local = np.count_nonzero(np.abs(source.x.array) > 1.0e-12)
    source_nonzero = comm.allreduce(source_nonzero_local, op=MPI.SUM)

    print("source nonzero count:", source_nonzero, flush=True)

    if source_nonzero == 0:
        raise RuntimeError(
            "Источник не попал в сетку: все значения source равны 0."
        )

    # Правая часть F_i = ∫ s * phi_i dx
    source_form = source * v_test * dx

    print("Assembling source vector F...", flush=True)

    F = assemble_vector(fem.form(source_form))
    F.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    F_array = F.getArray(readonly=True).copy()

    check_array_global(comm, F_array, "F_array")

    # --------------------------------------------------------
    # Начальное условие:
    # p(x, 0) = 0
    # p_t(x, 0) = 0
    # --------------------------------------------------------

    p_n.x.array[:] = 0.0
    p_n.x.scatter_forward()

    check_array_global(comm, p_n.x.array, "initial pressure")

    p_nm1.x.array[:] = p_n.x.array[:]
    p_nm1.x.scatter_forward()

    # Временный вектор для K * p_n
    Kp = K.createVecRight()

    print("Writing VTU frames to:", OUT_DIR, flush=True)

    # Пишем начальный кадр
    write_vtu_frame(comm, p_n, OUT_DIR, 0, 0.0)

    # --------------------------------------------------------
    # Основной временной цикл
    # --------------------------------------------------------

    for step in range(1, NUM_STEPS + 1):
        t = step * DT

        # Kp = K * p_n
        K.mult(p_n.x.petsc_vec, Kp)

        kp_array = Kp.getArray(readonly=True)

        check_array_global(comm, kp_array, f"Kp step={step:04d}")

        # M * p_tt + Kp = F
        #
        # p^{n+1}
        #   = 2p^n - p^{n-1}
        #     + dt^2 * M^{-1} * (F - Kp^n)

        p_np1.x.array[:] = (
            2.0 * p_n.x.array[:]
            - p_nm1.x.array[:]
            + (DT ** 2) * (F_array - kp_array) / mass_array
        )

        p_np1.x.scatter_forward()

        _, _, max_p_np1 = check_array_global(
            comm,
            p_np1.x.array,
            f"p_np1 step={step:04d}",
        )

        if max_p_np1 > MAX_ALLOWED_PRESSURE:
            raise RuntimeError(
                f"Pressure exploded: max|p|={max_p_np1:.6e} "
                f"at step={step}, time={t:.6f}. "
                f"Попробуй уменьшить DT или SOURCE_AMPLITUDE."
            )

        # Сдвигаем временные слои
        p_nm1.x.array[:] = p_n.x.array[:]
        p_n.x.array[:] = p_np1.x.array[:]

        p_nm1.x.scatter_forward()
        p_n.x.scatter_forward()

        _, _, max_p = check_array_global(
            comm,
            p_n.x.array,
            f"pressure step={step:04d}",
        )

        if step % WRITE_EVERY == 0:
            check_array_global(
                comm,
                p_n.x.array,
                f"VTU write pressure step={step:04d}",
            )

            write_vtu_frame(comm, p_n, OUT_DIR, step, t)

        print(
            f"step={step:04d}, time={t:.3f}, max|p|={max_p:.6e}",
            flush=True,
        )

    print("\nDone.", flush=True)
    print("Open VTU frames from:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()