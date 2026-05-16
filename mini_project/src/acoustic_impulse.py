from pathlib import Path
import shutil

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

from mini_project.src.tools.preprocess import load_head_model_from_vtu


BASE_DIR = Path(__file__).resolve().parent

VTU_PATH = BASE_DIR.parent / "3d-models" / "head_with_materials.vtu"

# Папка с отдельными .vtu кадрами
OUT_DIR = BASE_DIR / "acoustic_vtu_frames_other_source"

# Геометрия задана в миллиметрах, время - в секундах,
# поэтому скорости звука задаются в mm/s в таблице материалов.
DT = None
AUTO_TIME_STEP_SAFETY = 0.25

# Число шагов
NUM_STEPS = 2000

# Записывать каждый N-й шаг
WRITE_EVERY = 5

# Ширина гауссова профиля источника в пространстве
SIGMA_FRACTION = 0.1

# Амплитуда источника
SOURCE_AMPLITUDE = 1.0

# Частота импульсного источника в Hz.
SOURCE_FREQUENCY = 1.0e6

# Центр оконного burst по времени.
PULSE_T0 = 5.0 / SOURCE_FREQUENCY

# Число периодов несущей внутри burst.
PULSE_NUM_CYCLES = 5.0

# Длительность burst.
PULSE_DURATION = PULSE_NUM_CYCLES / SOURCE_FREQUENCY

MAX_ALLOWED_PRESSURE = 1.0e6


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


def source_time_factor(t):
    """
    Оконный ультразвуковой burst:

        q(t) = sin(2*pi*f*tau) * sin(pi*tau/T)^2

    где tau = t - t_start, а T - длительность burst.
    До t_start и после t_start + T источник строго равен нулю.
    """
    pulse_start = PULSE_T0 - 0.5 * PULSE_DURATION
    local_t = t - pulse_start

    if local_t < 0.0 or local_t > PULSE_DURATION:
        return 0.0

    carrier = np.sin(2.0 * np.pi * SOURCE_FREQUENCY * local_t)
    window = np.sin(np.pi * local_t / PULSE_DURATION) ** 2

    return carrier * window


def estimate_time_step(domain, sound_speed):
    """
    Оценивает устойчивый шаг для явной акустической схемы.

    Сетка в mm, sound_speed в mm/s, поэтому dt получается в секундах.
    """
    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local
    local_cells = np.arange(num_local_cells, dtype=np.int32)

    h_local = cpp.mesh.h(domain._cpp_object, tdim, local_cells)
    h_min_local = np.min(h_local) if h_local.size else np.inf
    h_min = domain.comm.allreduce(h_min_local, op=MPI.MIN)

    c_local = sound_speed.x.array
    c_max_local = np.max(c_local) if c_local.size else 0.0
    c_max = domain.comm.allreduce(c_max_local, op=MPI.MAX)

    if not np.isfinite(h_min) or h_min <= 0.0:
        raise RuntimeError("Failed to estimate positive cell size")
    if not np.isfinite(c_max) or c_max <= 0.0:
        raise RuntimeError("Failed to estimate positive sound speed")

    dt = AUTO_TIME_STEP_SAFETY * h_min / c_max

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError("Failed to estimate stable time step")

    return dt, h_min, c_max


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print("Loading model...", flush=True)

    model = load_head_model_from_vtu(VTU_PATH)
    domain = model.domain

    print("domain:", domain, flush=True)

    if rank == 0:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    comm.barrier()


    V = fem.functionspace(domain, ("Lagrange", 1))

    p_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    dx = ufl.dx(domain)

    rho = model.rho
    sound_speed = model.sound_speed

    check_array_global(comm, sound_speed.x.array, "sound_speed")

    if DT is None:
        dt, h_min, c_max = estimate_time_step(domain, sound_speed)
    else:
        dt = DT
        h_min = None
        c_max = None

    if rank == 0:
        if h_min is not None:
            print(f"estimated h_min={h_min:.6e} mm", flush=True)
        if c_max is not None:
            print(f"estimated c_max={c_max:.6e} mm/s", flush=True)
        print(f"using dt={dt:.6e} s", flush=True)
        print(f"source frequency={SOURCE_FREQUENCY:.6e} Hz", flush=True)
        print(
            f"pulse center={PULSE_T0:.6e} s, "
            f"duration={PULSE_DURATION:.6e} s, "
            f"cycles={PULSE_NUM_CYCLES:.1f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Акустическое уравнение с импульсным источником:
    #
    #     1 / (rho * c^2) * p_tt
    #       - div(1 / rho * grad(p))
    #       = s(x, t)
    #
    # Пусть:
    #
    #     s(x, t) = g(x) * q(t)
    #
    # где:
    #
    #     g(x) — гауссов профиль в пространстве,
    #     q(t) — импульс во времени.
    #
    # После FEM:
    #
    #     M * p_tt + K * p = q(t) * F
    #
    # Явная схема:
    #
    #     p^{n+1}
    #       = 2p^n - p^{n-1}
    #         + dt^2 * M^{-1} * (q(t)F - Kp^n)
    # --------------------------------------------------------

    inv_bulk_modulus = 1.0 / (rho * (sound_speed ** 2))
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


    p_nm1 = fem.Function(V)
    p_n = fem.Function(V)
    p_np1 = fem.Function(V)

    p_nm1.name = "pressure"
    p_n.name = "pressure"
    p_np1.name = "pressure"


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


    source_space = fem.Function(V)
    source_space.name = "source_space"

    def source_gaussian(x):
        return SOURCE_AMPLITUDE * gaussian_profile(x)

    source_space.interpolate(source_gaussian)
    source_space.x.scatter_forward()

    check_array_global(comm, source_space.x.array, "source_space")

    source_nonzero_local = np.count_nonzero(np.abs(source_space.x.array) > 1.0e-12)
    source_nonzero = comm.allreduce(source_nonzero_local, op=MPI.SUM)

    print("source nonzero count:", source_nonzero, flush=True)

    if source_nonzero == 0:
        raise RuntimeError(
            "Источник не попал в сетку: все значения source_space равны 0."
        )

    # Вектор пространственной части источника:
    #     F_i = ∫ g(x) * phi_i dx

    source_form = source_space * v_test * dx

    print("Assembling source vector F...", flush=True)

    F = assemble_vector(fem.form(source_form))
    F.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    F_array = F.getArray(readonly=True).copy()

    check_array_global(comm, F_array, "F_array")

    # Начальное условие:
    # p(x, 0) = 0
    # p_t(x, 0) = 0

    p_n.x.array[:] = 0.0
    p_n.x.scatter_forward()

    check_array_global(comm, p_n.x.array, "initial pressure")

    p_nm1.x.array[:] = p_n.x.array[:]
    p_nm1.x.scatter_forward()

    Kp = K.createVecRight()

    print("Writing VTU frames to:", OUT_DIR, flush=True)

    write_vtu_frame(comm, p_n, OUT_DIR, 0, 0.0)


    for step in range(1, NUM_STEPS + 1):
        t = step * dt

        # Kp = K * p_n
        K.mult(p_n.x.petsc_vec, Kp)

        kp_array = Kp.getArray(readonly=True)

        check_array_global(comm, kp_array, f"Kp step={step:04d}")

        # Импульсный множитель q(t)
        q_t = source_time_factor(t)

        # M * p_tt + Kp = q(t)F
        #
        # p^{n+1}
        #   = 2p^n - p^{n-1}
        #     + dt^2 * M^{-1} * (q(t)F - Kp^n)

        p_np1.x.array[:] = (
            2.0 * p_n.x.array[:]
            - p_nm1.x.array[:]
            + (dt ** 2) * (q_t * F_array - kp_array) / mass_array
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
            f"step={step:04d}, time={t:.6e}, "
            f"q(t)={q_t:.6e}, max|p|={max_p:.6e}",
            flush=True,
        )

    print("\nDone.", flush=True)
    print("Open VTU frames from:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
