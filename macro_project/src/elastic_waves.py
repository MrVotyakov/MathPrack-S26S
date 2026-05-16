from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import re
import shutil
import xml.etree.ElementTree as ET

import basix.ufl
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp, fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    mesh_path: Path = BASE_DIR.parent / "3d-models" / "test.vtu"
    output_dir: Path = BASE_DIR / "elastic_wave_output"
    clear_output: bool = True

    num_steps: int = 120
    write_every: int = 2
    dt: float | None = None
    auto_dt_safety: float = 0.25

    rho: float = 7800.0
    young_modulus: float = 210.0e9
    poisson_ratio: float = 0.3

    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5

    pressure_amplitude: float = 1.0e7
    pressure_t0: float = 1.0e-6
    pressure_tau: float = 2.0e-7

    source_center: tuple[float, float, float] | None = None
    source_radius: float | None = None
    source_radius_fraction: float = 0.18
    laser_marker: int = 1
    fallback_to_full_boundary: bool = True

    solver_rtol: float = 1.0e-9
    solver_atol: float = 1.0e-12
    solver_max_it: int = 2000


@dataclass
class MeshData:
    domain: mesh.Mesh
    source_points: int
    source_cells: int


@dataclass
class LaserBoundary:
    facet_tags: mesh.MeshTags
    marker: int
    center: np.ndarray
    radius: float
    num_facets: int
    used_full_boundary_fallback: bool


@dataclass
class FunctionSpaces:
    V: fem.FunctionSpace
    Q: fem.FunctionSpace
    Q0: fem.FunctionSpace


class ResultWriter:
    def __init__(
        self,
        comm: MPI.Comm,
        output_dir: Path,
        functions: list[fem.Function],
    ):
        self.comm = comm
        self.output_dir = output_dir
        self.functions = functions
        self.kind = "VTU"
        self.path = output_dir

    def write(self, step: int, time: float) -> None:
        for function in self.functions:
            filename = self.output_dir / f"{function.name}_{step:04d}.vtu"
            with io.VTKFile(self.comm, filename.as_posix(), "w") as vtk:
                vtk.write_function(function, time)
            del vtk
            self._keep_only_serial_vtu(filename)

    def close(self) -> None:
        pass

    def _keep_only_serial_vtu(self, filename: Path) -> None:
        if self.comm.size != 1 or self.comm.rank != 0:
            return

        # DOLFINx writes a small collection file plus .pvtu metadata even for a
        # serial .vtu request. For this project we keep the real UnstructuredGrid
        # piece as the requested .vtu file.
        pvtu = filename.with_name(f"{filename.stem}000000.pvtu")
        piece = filename.with_name(f"{filename.stem}_p0_000000.vtu")

        if filename.exists():
            collection_text = filename.read_text(errors="ignore")
            pvtu_match = re.search(r'file="([^"]+\.pvtu)"', collection_text)
            if pvtu_match is not None:
                pvtu = filename.with_name(pvtu_match.group(1))

        if pvtu.exists():
            pvtu_text = pvtu.read_text(errors="ignore")
            piece_match = re.search(r'Source="([^"]+\.vtu)"', pvtu_text)
            if piece_match is not None:
                piece = filename.with_name(piece_match.group(1))

        if piece.exists():
            filename.unlink(missing_ok=True)
            piece.replace(filename)
        pvtu.unlink(missing_ok=True)


_VTK_DTYPES = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
}


def _extract_appended_data(payload: bytes) -> tuple[bytes, bytes | None, str, str]:
    match = re.search(br"<AppendedData\b[^>]*>", payload)
    if match is None:
        return payload, None, "", ""

    tag = match.group(0)
    encoding_match = re.search(br'encoding="([^"]+)"', tag)
    encoding = encoding_match.group(1).decode("ascii") if encoding_match else "raw"

    underscore = payload.find(b"_", match.end())
    if underscore < 0:
        raise ValueError("VTU AppendedData section has no '_' sentinel.")

    close = payload.find(b"</AppendedData>", underscore)
    if close < 0:
        raise ValueError("VTU AppendedData section is not closed.")

    xml_without_appended = payload[: match.start()] + b"</VTKFile>"
    appended = payload[underscore + 1 : close]

    if encoding == "base64":
        compact = re.sub(br"\s+", b"", appended)
        appended = base64.b64decode(compact)
    elif encoding != "raw":
        raise ValueError(f"Unsupported VTU appended data encoding: {encoding}")

    return xml_without_appended, appended, encoding, tag.decode("ascii", errors="ignore")


def _read_data_array(
    array_node: ET.Element,
    appended: bytes | None,
    header_dtype: np.dtype,
) -> np.ndarray:
    vtk_type = array_node.attrib.get("type")
    if vtk_type not in _VTK_DTYPES:
        raise ValueError(f"Unsupported VTU DataArray type: {vtk_type}")

    dtype = np.dtype(_VTK_DTYPES[vtk_type]).newbyteorder("<")
    fmt = array_node.attrib.get("format", "ascii")

    if fmt == "ascii":
        text = array_node.text or ""
        array = np.fromstring(text, sep=" ", dtype=dtype)
    elif fmt == "appended":
        if appended is None:
            raise ValueError("DataArray uses appended format, but VTU has no AppendedData.")
        offset = int(array_node.attrib["offset"])
        if offset + header_dtype.itemsize > len(appended):
            raise ValueError("VTU appended offset is outside the file.")
        byte_count = int(np.frombuffer(appended, dtype=header_dtype, count=1, offset=offset)[0])
        start = offset + header_dtype.itemsize
        end = start + byte_count
        if end > len(appended):
            raise ValueError("VTU appended block is truncated.")
        array = np.frombuffer(appended[start:end], dtype=dtype).copy()
    elif fmt == "binary":
        raw = base64.b64decode(re.sub(r"\s+", "", array_node.text or ""))
        byte_count = int(np.frombuffer(raw, dtype=header_dtype, count=1, offset=0)[0])
        start = header_dtype.itemsize
        array = np.frombuffer(raw[start : start + byte_count], dtype=dtype).copy()
    else:
        raise ValueError(f"Unsupported VTU DataArray format: {fmt}")

    components = int(array_node.attrib.get("NumberOfComponents", "1"))
    if components > 1:
        array = array.reshape((-1, components))

    return array


def _read_vtu_tetra_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    xml_payload, appended, _, _ = _extract_appended_data(payload)
    root = ET.fromstring(xml_payload)

    if root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("Only VTU UnstructuredGrid files are supported.")

    header_type = root.attrib.get("header_type", "UInt32")
    if header_type not in _VTK_DTYPES:
        raise ValueError(f"Unsupported VTU header_type: {header_type}")
    header_dtype = np.dtype(_VTK_DTYPES[header_type]).newbyteorder("<")

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("VTU file has no Piece element.")

    points_node = piece.find("./Points/DataArray")
    if points_node is None:
        raise ValueError("VTU file has no Points DataArray.")
    points = _read_data_array(points_node, appended, header_dtype).astype(np.float64)

    cells_node = piece.find("./Cells")
    if cells_node is None:
        raise ValueError("VTU file has no Cells section.")

    connectivity_node = cells_node.find("./DataArray[@Name='connectivity']")
    offsets_node = cells_node.find("./DataArray[@Name='offsets']")
    types_node = cells_node.find("./DataArray[@Name='types']")
    if connectivity_node is None or offsets_node is None or types_node is None:
        raise ValueError("VTU Cells section must contain connectivity, offsets, and types.")

    connectivity = _read_data_array(connectivity_node, appended, header_dtype).astype(np.int64)
    offsets = _read_data_array(offsets_node, appended, header_dtype).astype(np.int64)
    cell_types = _read_data_array(types_node, appended, header_dtype).astype(np.uint8)

    if offsets.size != cell_types.size:
        raise ValueError("VTU offsets and cell types have different lengths.")

    vtk_tetra = 10
    cells: list[np.ndarray] = []
    start = 0
    unsupported: set[int] = set()

    for end, cell_type in zip(offsets, cell_types):
        ids = connectivity[start:end]
        start = int(end)
        if int(cell_type) == vtk_tetra:
            if ids.size != 4:
                raise ValueError("VTK_TETRA cell has a connectivity length different from 4.")
            cells.append(ids)
        else:
            unsupported.add(int(cell_type))

    if unsupported:
        raise ValueError(
            "This minimal solver currently supports only tetrahedral VTU cells "
            f"(VTK type 10). Found unsupported cell types: {sorted(unsupported)}."
        )
    if not cells:
        raise ValueError("VTU file contains no tetrahedral cells.")

    return points, np.asarray(cells, dtype=np.int64)


def load_mesh(config: Config, comm: MPI.Comm = MPI.COMM_WORLD) -> MeshData:
    if not config.mesh_path.exists():
        raise FileNotFoundError(
            "VTU mesh file was not found: "
            f"{config.mesh_path}\n"
            "Set Config.mesh_path to an existing .vtu file."
        )

    points, cells = _read_vtu_tetra_mesh(config.mesh_path)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3D VTU points, got shape {points.shape}.")

    coordinate_element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    domain = mesh.create_mesh(comm, cells, ufl.Mesh(coordinate_element), points)

    return MeshData(domain=domain, source_points=points.shape[0], source_cells=cells.shape[0])


def _global_bounds(domain: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = domain.geometry.x
    if coords.size:
        local_min = np.min(coords, axis=0)
        local_max = np.max(coords, axis=0)
    else:
        local_min = np.full(domain.geometry.dim, np.inf)
        local_max = np.full(domain.geometry.dim, -np.inf)

    global_min = np.array(
        [domain.comm.allreduce(value, op=MPI.MIN) for value in local_min],
        dtype=np.float64,
    )
    global_max = np.array(
        [domain.comm.allreduce(value, op=MPI.MAX) for value in local_max],
        dtype=np.float64,
    )

    return global_min, global_max


def mark_laser_facets(domain: mesh.Mesh, config: Config) -> LaserBoundary:
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    bounds_min, bounds_max = _global_bounds(domain)
    bbox_size = float(np.max(bounds_max - bounds_min))

    if config.source_center is None:
        center = np.array(
            [
                0.5 * (bounds_min[0] + bounds_max[0]),
                0.5 * (bounds_min[1] + bounds_max[1]),
                bounds_max[2],
            ],
            dtype=np.float64,
        )
    else:
        center = np.asarray(config.source_center, dtype=np.float64)

    radius = (
        float(config.source_radius)
        if config.source_radius is not None
        else config.source_radius_fraction * bbox_size
    )
    if radius <= 0.0:
        raise ValueError("Config.source_radius must be positive.")

    def near_source(x: np.ndarray) -> np.ndarray:
        diff = x - center[:, None]
        return np.sum(diff * diff, axis=0) <= radius * radius

    facets = mesh.locate_entities_boundary(domain, fdim, near_source)
    local_count = int(facets.size)
    global_count = domain.comm.allreduce(local_count, op=MPI.SUM)
    used_fallback = False

    if global_count == 0:
        if not config.fallback_to_full_boundary:
            raise RuntimeError(
                "No boundary facets were found near source_center. "
                "Increase source_radius or enable fallback_to_full_boundary."
            )

        used_fallback = True

        def all_boundary(x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[1], True, dtype=bool)

        facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
        local_count = int(facets.size)
        global_count = domain.comm.allreduce(local_count, op=MPI.SUM)

        if global_count == 0:
            raise RuntimeError("Could not locate any exterior boundary facets.")

    facets = np.asarray(facets, dtype=np.int32)
    order = np.argsort(facets)
    facets = facets[order]
    values = np.full(facets.size, config.laser_marker, dtype=np.int32)
    facet_tags = mesh.meshtags(domain, fdim, facets, values)

    return LaserBoundary(
        facet_tags=facet_tags,
        marker=config.laser_marker,
        center=center,
        radius=radius,
        num_facets=int(global_count),
        used_full_boundary_fallback=used_fallback,
    )


def build_function_spaces(domain: mesh.Mesh) -> FunctionSpaces:
    cell = domain.basix_cell()
    gdim = domain.geometry.dim

    vector_element = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    scalar_element = basix.ufl.element("Lagrange", cell, 1)
    dg0_element = basix.ufl.element("DG", cell, 0)

    return FunctionSpaces(
        V=fem.functionspace(domain, vector_element),
        Q=fem.functionspace(domain, scalar_element),
        Q0=fem.functionspace(domain, dg0_element),
    )


def make_function(function_space: fem.FunctionSpace, name: str) -> fem.Function:
    function = fem.Function(function_space)
    function.name = name
    return function


def interpolation_points(function_space: fem.FunctionSpace) -> np.ndarray:
    points = function_space.element.interpolation_points
    return points() if callable(points) else points


def lame_parameters(config: Config) -> tuple[float, float]:
    e = config.young_modulus
    nu = config.poisson_ratio
    mu = e / (2.0 * (1.0 + nu))
    lambda_ = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lambda_, mu


def laser_pressure(time: float, config: Config) -> float:
    return config.pressure_amplitude * np.exp(
        -((time - config.pressure_t0) ** 2) / (2.0 * config.pressure_tau ** 2)
    )


def estimate_time_step(domain: mesh.Mesh, config: Config, lambda_: float, mu: float) -> tuple[float, float, float]:
    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local
    local_cells = np.arange(num_local_cells, dtype=np.int32)

    h_local = cpp.mesh.h(domain._cpp_object, tdim, local_cells)
    h_min_local = np.min(h_local) if h_local.size else np.inf
    h_min = domain.comm.allreduce(float(h_min_local), op=MPI.MIN)

    cp = np.sqrt((lambda_ + 2.0 * mu) / config.rho)
    dt = config.auto_dt_safety * h_min / cp

    if not np.isfinite(h_min) or h_min <= 0.0:
        raise RuntimeError("Failed to estimate a positive mesh size.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError("Failed to estimate a positive time step.")

    return dt, h_min, cp


def prepare_output_dir(config: Config, comm: MPI.Comm) -> None:
    if comm.rank == 0:
        if config.output_dir.exists() and config.clear_output:
            if config.output_dir.is_dir():
                shutil.rmtree(config.output_dir)
            else:
                raise RuntimeError(f"Output path exists and is not a directory: {config.output_dir}")
        config.output_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()


def write_output(
    writer: ResultWriter,
    step: int,
    time: float,
    displacement_magnitude: fem.Function,
    stress_trace: fem.Function,
    displacement_magnitude_expr: fem.Expression,
    stress_trace_expr: fem.Expression,
) -> float:
    displacement_magnitude.interpolate(displacement_magnitude_expr)
    displacement_magnitude.x.scatter_forward()

    stress_trace.interpolate(stress_trace_expr)
    stress_trace.x.scatter_forward()

    local_max = (
        float(np.max(displacement_magnitude.x.array))
        if displacement_magnitude.x.array.size
        else 0.0
    )
    max_u = displacement_magnitude.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)

    writer.write(step, time)
    return max_u


def solve_elastic_waves(config: Config) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    mesh_data = load_mesh(config, comm)
    domain = mesh_data.domain
    laser = mark_laser_facets(domain, config)
    spaces = build_function_spaces(domain)

    lambda_, mu = lame_parameters(config)
    dt = config.dt
    if dt is None:
        dt, h_min, cp = estimate_time_step(domain, config, lambda_, mu)
    else:
        _, h_min, cp = estimate_time_step(domain, config, lambda_, mu)

    prepare_output_dir(config, comm)

    V = spaces.V
    Q = spaces.Q
    Q0 = spaces.Q0

    beta = config.newmark_beta
    gamma = config.newmark_gamma
    if beta <= 0.0:
        raise ValueError("Config.newmark_beta must be positive.")
    if gamma < 0.0:
        raise ValueError("Config.newmark_gamma must be non-negative.")

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    dx = ufl.dx(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=laser.facet_tags)
    n = ufl.FacetNormal(domain)
    gdim = domain.geometry.dim

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lambda_ * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

    u_n = make_function(V, "displacement")
    v_n = make_function(V, "velocity")
    a_n = make_function(V, "acceleration")
    u_np1 = make_function(V, "displacement")
    u_pred = make_function(V, "newmark_displacement_predictor")

    pressure = fem.Constant(domain, PETSc.ScalarType(0.0))
    traction = -pressure * n

    lhs = (
        config.rho * ufl.inner(u_trial, v_test) * dx
        + beta * dt * dt * ufl.inner(sigma(u_trial), eps(v_test)) * dx
    )
    rhs = (
        config.rho * ufl.inner(u_pred, v_test) * dx
        + beta * dt * dt * ufl.inner(traction, v_test) * ds(laser.marker)
    )

    A = assemble_matrix(fem.form(lhs))
    A.assemble()

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    solver.setTolerances(
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        max_it=config.solver_max_it,
    )
    solver.setFromOptions()

    displacement_magnitude = make_function(Q, "displacement_magnitude")
    stress_trace = make_function(Q0, "stress_trace")
    displacement_magnitude_expr = fem.Expression(
        ufl.sqrt(ufl.dot(u_n, u_n)),
        interpolation_points(Q),
    )
    stress_trace_expr = fem.Expression(
        ufl.tr(sigma(u_n)),
        interpolation_points(Q0),
    )

    writer = ResultWriter(
        comm,
        config.output_dir,
        [u_n, displacement_magnitude, stress_trace],
    )

    if rank == 0:
        print("Loaded VTU mesh:", config.mesh_path, flush=True)
        print(f"points={mesh_data.source_points}, tetrahedra={mesh_data.source_cells}", flush=True)
        print(f"rho={config.rho:.6e}", flush=True)
        print(f"E={config.young_modulus:.6e}, nu={config.poisson_ratio:.6e}", flush=True)
        print(f"lambda={lambda_:.6e}, mu={mu:.6e}", flush=True)
        print(f"estimated h_min={h_min:.6e}", flush=True)
        print(f"estimated c_p={cp:.6e}", flush=True)
        print(f"using dt={dt:.6e}", flush=True)
        print(f"laser center={laser.center}", flush=True)
        print(f"laser radius={laser.radius:.6e}", flush=True)
        print(f"laser facets={laser.num_facets}", flush=True)
        if laser.used_full_boundary_fallback:
            print(
                "TODO: source patch was not found; using the whole exterior boundary "
                "as Gamma_laser. Increase source_radius or set source_center.",
                flush=True,
            )
        print(f"output kind={writer.kind}", flush=True)
        print(f"output path={writer.path}", flush=True)

    max_u = write_output(
        writer,
        0,
        0.0,
        displacement_magnitude,
        stress_trace,
        displacement_magnitude_expr,
        stress_trace_expr,
    )

    if rank == 0:
        print(
            f"step=0000, time={0.0:.6e}, "
            f"p(t)={laser_pressure(0.0, config):.6e}, max|u|={max_u:.6e}",
            flush=True,
        )

    rhs_form = fem.form(rhs)

    try:
        for step in range(1, config.num_steps + 1):
            time = step * dt
            p_value = laser_pressure(time, config)
            pressure.value = PETSc.ScalarType(p_value)

            u_pred.x.array[:] = (
                u_n.x.array
                + dt * v_n.x.array
                + dt * dt * (0.5 - beta) * a_n.x.array
            )
            u_pred.x.scatter_forward()

            b = assemble_vector(rhs_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            u_np1.x.array[:] = 0.0
            solver.solve(b, u_np1.x.petsc_vec)
            u_np1.x.scatter_forward()
            b.destroy()

            a_np1_array = (u_np1.x.array - u_pred.x.array) / (beta * dt * dt)
            v_n.x.array[:] = (
                v_n.x.array
                + dt * (1.0 - gamma) * a_n.x.array
                + gamma * dt * a_np1_array
            )
            a_n.x.array[:] = a_np1_array
            u_n.x.array[:] = u_np1.x.array

            u_n.x.scatter_forward()
            v_n.x.scatter_forward()
            a_n.x.scatter_forward()

            should_write = step % config.write_every == 0 or step == config.num_steps
            if should_write:
                max_u = write_output(
                    writer,
                    step,
                    time,
                    displacement_magnitude,
                    stress_trace,
                    displacement_magnitude_expr,
                    stress_trace_expr,
                )
            else:
                local_max = float(np.max(np.abs(u_n.x.array))) if u_n.x.array.size else 0.0
                max_u = comm.allreduce(local_max, op=MPI.MAX)

            its = solver.getIterationNumber()
            reason = solver.getConvergedReason()
            if reason < 0:
                raise RuntimeError(f"PETSc linear solver did not converge at step {step}: {reason}")

            if rank == 0:
                print(
                    f"step={step:04d}, time={time:.6e}, "
                    f"p(t)={p_value:.6e}, max|u|={max_u:.6e}, "
                    f"ksp_it={its}, output={config.output_dir}",
                    flush=True,
                )
    finally:
        writer.close()
        solver.destroy()
        A.destroy()

    if rank == 0:
        print("Done.", flush=True)
        print(f"Open in ParaView: {writer.path}", flush=True)


def main() -> None:
    config = Config()
    solve_elastic_waves(config)


if __name__ == "__main__":
    main()
