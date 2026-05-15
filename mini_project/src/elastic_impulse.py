from pathlib import Path
import shutil

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp, fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

from mini_project.src.tools.preprocess import (
    load_head_model_from_vtu,
    make_elasticity_forms,
)


BASE_DIR = Path(__file__).resolve().parent

VTU_PATH = BASE_DIR.parent / "3d-models" / "head_with_materials.vtu"
OUT_DIR = BASE_DIR / "elastic_vtu_frames_2"

NUM_STEPS = 5000
WRITE_EVERY = 20

SOURCE_AMPLITUDE = 5.0e2
SOURCE_FREQUENCY = 1.0
PULSE_T0 = 0.8
PULSE_TAU = 0.25
SIGMA_FRACTION = 0.08

SOURCE_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float64)
AUTO_TIME_STEP_SAFETY = 0.12
MAX_ALLOWED_DISPLACEMENT = 1.0e6
BASE_CLAMP_FRACTION = 0.06
RAYLEIGH_ALPHA = 0.20
RAYLEIGH_BETA = 0.0


def check_array_global(comm, array, name):
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


def source_time_factor(t):
    carrier = np.sin(2.0 * np.pi * SOURCE_FREQUENCY * t)
    envelope = np.exp(-((t - PULSE_T0) ** 2) / (2.0 * PULSE_TAU ** 2))
    return carrier * envelope


def get_domain_bounds(domain):
    x = domain.geometry.x
    return {
        "x_min": np.min(x[:, 0]),
        "x_max": np.max(x[:, 0]),
        "y_min": np.min(x[:, 1]),
        "y_max": np.max(x[:, 1]),
        "z_min": np.min(x[:, 2]),
        "z_max": np.max(x[:, 2]),
    }


def estimate_time_step(domain, rho, lambda_, mu):
    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local
    local_cells = np.arange(num_local_cells, dtype=np.int32)

    h_local = cpp.mesh.h(domain._cpp_object, tdim, local_cells)
    h_min_local = np.min(h_local) if h_local.size else np.inf
    h_min = domain.comm.allreduce(h_min_local, op=MPI.MIN)

    cp_local = np.sqrt((lambda_.x.array + 2.0 * mu.x.array) / rho.x.array)
    cp_max_local = np.max(cp_local) if cp_local.size else 0.0
    cp_max = domain.comm.allreduce(cp_max_local, op=MPI.MAX)

    if cp_max <= 0.0 or not np.isfinite(cp_max):
        raise RuntimeError("Failed to estimate compressional wave speed")

    dt = AUTO_TIME_STEP_SAFETY * h_min / cp_max

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError("Failed to estimate stable time step")

    return dt, h_min, cp_max


def build_source_profile(domain):
    bounds = get_domain_bounds(domain)
    bbox_size = max(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
    )
    sigma = SIGMA_FRACTION * bbox_size

    source_center = np.array(
        [
            0.5 * (bounds["x_max"] + bounds["x_min"]),
            bounds["y_min"] + 0.4 * (bounds["y_max"] - bounds["y_min"]),
            bounds["z_min"],
        ],
        dtype=np.float64,
    )

    def gaussian_vector(x_eval):
        r2 = (
            (x_eval[0] - source_center[0]) ** 2
            + (x_eval[1] - source_center[1]) ** 2
            + (x_eval[2] - source_center[2]) ** 2
        )
        profile = np.exp(-r2 / (2.0 * sigma ** 2))
        return SOURCE_DIRECTION[:, None] * profile[None, :]

    return gaussian_vector, source_center, sigma


def build_base_dirichlet_bc(domain, V):
    bounds = get_domain_bounds(domain)
    bbox_size = max(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
    )
    clamp_limit = bounds["z_min"] + BASE_CLAMP_FRACTION * bbox_size

    fdim = domain.topology.dim - 1
    base_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda x: x[2] <= clamp_limit,
    )
    base_dofs = fem.locate_dofs_topological(V, fdim, base_facets)

    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    zero.x.scatter_forward()

    bc = fem.dirichletbc(zero, base_dofs)
    return bc, np.asarray(base_dofs, dtype=np.int32), clamp_limit


def apply_zero_on_dofs(function, dofs):
    if dofs.size > 0:
        function.x.array[dofs] = 0.0
    function.x.scatter_forward()


def build_cellwise_scalar_tools(domain):
    Q = fem.functionspace(domain, ("DG", 0))
    q = ufl.TestFunction(Q)

    cell_volume = fem.Function(Q)
    cell_volume.name = "cell_volume"

    cell_volume_vec = assemble_vector(fem.form(q * ufl.dx(domain)))
    cell_volume_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    cell_volume.x.array[:] = cell_volume_vec.getArray(readonly=True)
    cell_volume.x.scatter_forward()

    return Q, q, cell_volume


def update_cellwise_average(out_function, test_function, expression, cell_volume, domain):
    numerator = assemble_vector(
        fem.form(expression * test_function * ufl.dx(domain))
    )
    numerator.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    out_function.x.array[:] = numerator.getArray(readonly=True) / cell_volume.x.array
    out_function.x.scatter_forward()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print("Loading model...", flush=True)
    model = load_head_model_from_vtu(VTU_PATH)
    domain = model.domain

    if rank == 0:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    comm.barrier()

    V, mass_form, stiffness_form = make_elasticity_forms(model)
    bc, fixed_dofs, clamp_limit = build_base_dirichlet_bc(domain, V)
    bcs = [bc]

    v_test = ufl.TestFunction(V)
    dx = ufl.dx(domain)

    print("Assembling lumped vector mass...", flush=True)
    one = fem.Function(V)
    one.x.array[:] = 1.0
    one.x.scatter_forward()

    mass_vec = assemble_vector(fem.form(ufl.action(mass_form, one)))
    mass_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    mass_array = mass_vec.getArray(readonly=True).copy()
    if fixed_dofs.size > 0:
        mass_array[fixed_dofs] = 1.0
    check_array_global(comm, mass_array, "mass_array")

    if np.any(mass_array <= 0.0):
        raise RuntimeError("mass_array contains non-positive entries")

    print("Assembling elastic stiffness matrix...", flush=True)
    K = assemble_matrix(fem.form(stiffness_form), bcs=bcs)
    K.assemble()

    source_profile = fem.Function(V)
    source_profile.name = "source_profile"
    gaussian_vector, source_center, sigma = build_source_profile(domain)
    source_profile.interpolate(gaussian_vector)
    source_profile.x.scatter_forward()

    source_vector = assemble_vector(fem.form(ufl.inner(source_profile, v_test) * dx))
    source_vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    source_array = source_vector.getArray(readonly=True).copy()
    if fixed_dofs.size > 0:
        source_array[fixed_dofs] = 0.0
    check_array_global(comm, source_array, "source_array")

    dt, h_min, cp_max = estimate_time_step(
        domain,
        model.rho,
        model.lambda_,
        model.mu,
    )

    if rank == 0:
        print(f"estimated h_min={h_min:.6e}", flush=True)
        print(f"estimated c_p,max={cp_max:.6e}", flush=True)
        print(f"using dt={dt:.6e}", flush=True)
        print(f"source center={source_center}", flush=True)
        print(f"source sigma={sigma:.6e}", flush=True)
        print(f"clamp limit z<={clamp_limit:.6e}", flush=True)
        print(f"fixed dofs={fixed_dofs.size}", flush=True)
        print(
            f"Rayleigh damping: alpha={RAYLEIGH_ALPHA:.6e}, beta={RAYLEIGH_BETA:.6e}",
            flush=True,
        )

    u_nm1 = fem.Function(V)
    u_n = fem.Function(V)
    u_np1 = fem.Function(V)
    v_n = fem.Function(V)

    u_nm1.name = "displacement"
    u_n.name = "displacement"
    u_np1.name = "displacement"
    v_n.name = "velocity"

    apply_zero_on_dofs(u_nm1, fixed_dofs)
    apply_zero_on_dofs(u_n, fixed_dofs)
    apply_zero_on_dofs(u_np1, fixed_dofs)
    apply_zero_on_dofs(v_n, fixed_dofs)

    Ku = K.createVecRight()
    Kv = K.createVecRight()

    Q, q0, cell_volume = build_cellwise_scalar_tools(domain)
    velocity_magnitude = fem.Function(Q)
    velocity_magnitude.name = "velocity_magnitude"
    strain_energy_density = fem.Function(Q)
    strain_energy_density.name = "strain_energy_density"

    gdim = domain.geometry.dim

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return model.lambda_ * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * model.mu * eps(w)

    velocity_magnitude_expr = ufl.sqrt(ufl.dot(v_n, v_n))
    strain_energy_expr = 0.5 * ufl.inner(sigma(u_n), eps(u_n))

    disp_path = OUT_DIR / "displacement.pvd"
    vel_path = OUT_DIR / "velocity_magnitude.pvd"
    energy_path = OUT_DIR / "strain_energy_density.pvd"
    print(f"Writing displacement: {disp_path}", flush=True)
    print(f"Writing velocity magnitude: {vel_path}", flush=True)
    print(f"Writing strain energy density: {energy_path}", flush=True)

    update_cellwise_average(
        velocity_magnitude,
        q0,
        velocity_magnitude_expr,
        cell_volume,
        domain,
    )
    update_cellwise_average(
        strain_energy_density,
        q0,
        strain_energy_expr,
        cell_volume,
        domain,
    )

    with (
        io.VTKFile(comm, disp_path.as_posix(), "w") as disp_vtk,
        io.VTKFile(comm, vel_path.as_posix(), "w") as vel_vtk,
        io.VTKFile(comm, energy_path.as_posix(), "w") as energy_vtk,
    ):
        disp_vtk.write_function(u_n, 0.0)
        vel_vtk.write_function(velocity_magnitude, 0.0)
        energy_vtk.write_function(strain_energy_density, 0.0)

        for step in range(1, NUM_STEPS + 1):
            t = step * dt

            v_n.x.array[:] = (u_n.x.array[:] - u_nm1.x.array[:]) / dt
            apply_zero_on_dofs(v_n, fixed_dofs)

            K.mult(u_n.x.petsc_vec, Ku)
            ku_array = Ku.getArray(readonly=True)

            if RAYLEIGH_BETA != 0.0:
                K.mult(v_n.x.petsc_vec, Kv)
                kv_array = Kv.getArray(readonly=True)
            else:
                kv_array = 0.0

            force_scale = SOURCE_AMPLITUDE * source_time_factor(t)
            damping_array = RAYLEIGH_ALPHA * mass_array * v_n.x.array[:]
            rhs_array = (
                force_scale * source_array
                - ku_array
                - damping_array
                - RAYLEIGH_BETA * kv_array
            )

            u_np1.x.array[:] = (
                2.0 * u_n.x.array[:]
                - u_nm1.x.array[:]
                + (dt ** 2) * rhs_array / mass_array
            )
            apply_zero_on_dofs(u_np1, fixed_dofs)

            disp_max = np.max(np.abs(u_np1.x.array)) if u_np1.x.array.size else 0.0
            disp_max = comm.allreduce(disp_max, op=MPI.MAX)

            if not np.isfinite(disp_max):
                raise RuntimeError("displacement contains NaN or inf")
            if disp_max > MAX_ALLOWED_DISPLACEMENT:
                raise RuntimeError(
                    f"displacement exceeded limit: {disp_max:.6e} > {MAX_ALLOWED_DISPLACEMENT:.6e}"
                )

            v_n.x.array[:] = (u_np1.x.array[:] - u_nm1.x.array[:]) / (2.0 * dt)
            apply_zero_on_dofs(v_n, fixed_dofs)

            u_nm1.x.array[:] = u_n.x.array[:]
            u_n.x.array[:] = u_np1.x.array[:]
            apply_zero_on_dofs(u_nm1, fixed_dofs)
            apply_zero_on_dofs(u_n, fixed_dofs)

            if step % WRITE_EVERY == 0:
                update_cellwise_average(
                    velocity_magnitude,
                    q0,
                    velocity_magnitude_expr,
                    cell_volume,
                    domain,
                )
                update_cellwise_average(
                    strain_energy_density,
                    q0,
                    strain_energy_expr,
                    cell_volume,
                    domain,
                )
                disp_vtk.write_function(u_n, t)
                vel_vtk.write_function(velocity_magnitude, t)
                energy_vtk.write_function(strain_energy_density, t)

            if rank == 0:
                print(
                    f"step={step:04d}, time={t:.6e}, "
                    f"force_scale={force_scale:.6e}, max|u|={disp_max:.6e}",
                    flush=True,
                )

    if rank == 0:
        print("Done.", flush=True)
        print(f"Open in ParaView: {disp_path}", flush=True)
        print(f"Open in ParaView: {vel_path}", flush=True)
        print(f"Open in ParaView: {energy_path}", flush=True)


if __name__ == "__main__":
    main()
