from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import base64
import re
import shutil
import time
import xml.etree.ElementTree as ET

import basix.ufl
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)


# Minimal Docker run example:
#   docker run -ti --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable
#   python3 macro_project/src/thermal_stress_from_vtu.py
#
# The input is a sequence of VTU files with a nodal temperature field T.
# For every temperature frame the script solves a static thermoelastic problem
# and writes a new VTU file with displacement and stress-derived fields.


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    input_dir: Path = BASE_DIR.parent / "3rd-party" / "macroproject2" / "results"
    input_pattern: str = "laser_*.vtu"
    input_pvd: Path | None = None
    temperature_field: str = "T"

    output_dir: Path = BASE_DIR / "thermal_stress_output"
    output_prefix: str = "thermal_stress"
    clear_output: bool = False

    max_frames: int | None = None
    coordinate_round_decimals: int = 12

    # Defaults are used when the VTU FieldData does not contain these values.
    use_vtu_field_data_materials: bool = True
    young_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    thermal_expansion: float = 1.2e-5
    reference_temperature: float = 300.0

    # A static thermoelastic solve needs constraints to remove rigid body modes.
    # By default the bottom z-face is fixed.
    clamp_axis: int = 2
    clamp_at_min: bool = True
    clamp_tolerance_fraction: float = 1.0e-6

    solver_rtol: float = 1.0e-9
    solver_atol: float = 1.0e-12
    solver_max_it: int = 2000


@dataclass
class VtuData:
    points: np.ndarray
    cells: np.ndarray
    temperature: np.ndarray
    field_data: dict[str, np.ndarray]


@dataclass
class InputFrame:
    path: Path
    time: float


@dataclass
class Material:
    young_modulus: float
    poisson_ratio: float
    thermal_expansion: float
    reference_temperature: float

    @property
    def mu(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lambda_(self) -> float:
        e = self.young_modulus
        nu = self.poisson_ratio
        return e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


@dataclass
class FunctionSpaces:
    V: fem.FunctionSpace
    Q: fem.FunctionSpace
    Q0: fem.FunctionSpace


_VTK_DTYPES = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int8": np.int8,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
}


def _extract_appended_data(payload: bytes) -> tuple[bytes, bytes | None]:
    match = re.search(br"<AppendedData\b[^>]*>", payload)
    if match is None:
        return payload, None

    tag = match.group(0)
    encoding_match = re.search(br'encoding="([^"]+)"', tag)
    encoding = encoding_match.group(1).decode("ascii") if encoding_match else "raw"

    underscore = payload.find(b"_", match.end())
    close = payload.find(b"</AppendedData>", underscore)
    if underscore < 0 or close < 0:
        raise ValueError("VTU AppendedData section is malformed.")

    xml_without_appended = payload[: match.start()] + b"</VTKFile>"
    appended = payload[underscore + 1 : close]

    if encoding == "base64":
        appended = base64.b64decode(re.sub(br"\s+", b"", appended))
    elif encoding != "raw":
        raise ValueError(f"Unsupported VTU appended data encoding: {encoding}")

    return xml_without_appended, appended


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
        array = np.fromstring(array_node.text or "", sep=" ", dtype=dtype)
    elif fmt == "appended":
        if appended is None:
            raise ValueError("DataArray uses appended format, but VTU has no AppendedData.")
        offset = int(array_node.attrib["offset"])
        byte_count = int(np.frombuffer(appended, dtype=header_dtype, count=1, offset=offset)[0])
        start = offset + header_dtype.itemsize
        end = start + byte_count
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


def _load_vtu_xml(path: Path) -> tuple[ET.Element, bytes | None, np.dtype]:
    payload = path.read_bytes()
    xml_payload, appended = _extract_appended_data(payload)
    root = ET.fromstring(xml_payload)

    if root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError(f"{path} is not a VTU UnstructuredGrid file.")

    header_type = root.attrib.get("header_type", "UInt32")
    if header_type not in _VTK_DTYPES:
        raise ValueError(f"Unsupported VTU header_type: {header_type}")

    return root, appended, np.dtype(_VTK_DTYPES[header_type]).newbyteorder("<")


def _read_field_data(
    root: ET.Element,
    appended: bytes | None,
    header_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    field_data = root.find(".//FieldData")
    if field_data is None:
        return out

    for node in field_data.findall("./DataArray"):
        name = node.attrib.get("Name")
        if name:
            out[name] = _read_data_array(node, appended, header_dtype)

    return out


def _read_point_data_array(
    piece: ET.Element,
    field_name: str,
    appended: bytes | None,
    header_dtype: np.dtype,
) -> np.ndarray:
    point_data = piece.find("./PointData")
    if point_data is None:
        raise ValueError("VTU file has no PointData section.")

    node = point_data.find(f"./DataArray[@Name='{field_name}']")
    if node is None:
        available = [
            item.attrib.get("Name", "<unnamed>")
            for item in point_data.findall("./DataArray")
        ]
        raise ValueError(
            f"PointData field '{field_name}' was not found. "
            f"Available fields: {available}"
        )

    values = _read_data_array(node, appended, header_dtype)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"Temperature field '{field_name}' must be scalar.")

    return values.astype(np.float64)


def read_vtu(path: Path, temperature_field: str) -> VtuData:
    root, appended, header_dtype = _load_vtu_xml(path)
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path} has no Piece element.")

    points_node = piece.find("./Points/DataArray")
    if points_node is None:
        raise ValueError(f"{path} has no Points DataArray.")
    points = _read_data_array(points_node, appended, header_dtype).astype(np.float64)

    cells_node = piece.find("./Cells")
    if cells_node is None:
        raise ValueError(f"{path} has no Cells section.")

    connectivity_node = cells_node.find("./DataArray[@Name='connectivity']")
    offsets_node = cells_node.find("./DataArray[@Name='offsets']")
    types_node = cells_node.find("./DataArray[@Name='types']")
    if connectivity_node is None or offsets_node is None or types_node is None:
        raise ValueError("VTU Cells section must contain connectivity, offsets, and types.")

    connectivity = _read_data_array(connectivity_node, appended, header_dtype).astype(np.int64)
    offsets = _read_data_array(offsets_node, appended, header_dtype).astype(np.int64)
    cell_types = _read_data_array(types_node, appended, header_dtype).astype(np.int64)

    cells: list[np.ndarray] = []
    unsupported: set[int] = set()
    start = 0

    # 10 = VTK_TETRA, 71 = VTK_LAGRANGE_TETRAHEDRON.
    # For linear files written by DOLFINx the Lagrange tetra also has 4 nodes.
    supported_tetra_types = {10, 71}
    for end, cell_type in zip(offsets, cell_types):
        ids = connectivity[start:end]
        start = int(end)
        if int(cell_type) in supported_tetra_types and ids.size == 4:
            cells.append(ids)
        else:
            unsupported.add(int(cell_type))

    if unsupported:
        raise ValueError(
            "Only linear tetrahedral VTU cells are supported. "
            f"Unsupported cell types: {sorted(unsupported)}."
        )
    if not cells:
        raise ValueError(f"{path} contains no tetrahedral cells.")

    temperature = _read_point_data_array(piece, temperature_field, appended, header_dtype)
    if temperature.size != points.shape[0]:
        raise ValueError(
            f"Temperature field size {temperature.size} does not match "
            f"number of points {points.shape[0]}."
        )

    return VtuData(
        points=points,
        cells=np.asarray(cells, dtype=np.int64),
        temperature=temperature,
        field_data=_read_field_data(root, appended, header_dtype),
    )


def read_temperature_only(path: Path, temperature_field: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    root, appended, header_dtype = _load_vtu_xml(path)
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path} has no Piece element.")
    return (
        _read_point_data_array(piece, temperature_field, appended, header_dtype),
        _read_field_data(root, appended, header_dtype),
    )


def material_from_config_and_vtu(config: Config, field_data: dict[str, np.ndarray]) -> Material:
    def scalar(name: str, default: float) -> float:
        if not config.use_vtu_field_data_materials or name not in field_data:
            return default
        value = np.asarray(field_data[name]).reshape(-1)
        return float(value[0]) if value.size else default

    return Material(
        young_modulus=scalar("E_young", config.young_modulus),
        poisson_ratio=scalar("nu", config.poisson_ratio),
        thermal_expansion=scalar("alpha_T", config.thermal_expansion),
        reference_temperature=scalar("T_ref", config.reference_temperature),
    )


def make_domain(points: np.ndarray, cells: np.ndarray, comm: MPI.Comm) -> mesh.Mesh:
    coordinate_element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    return mesh.create_mesh(
        comm,
        np.ascontiguousarray(cells, dtype=np.int64),
        ufl.Mesh(coordinate_element),
        np.ascontiguousarray(points, dtype=np.float64),
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


def global_bounds(domain: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = domain.geometry.x
    if coords.size:
        local_min = np.min(coords, axis=0)
        local_max = np.max(coords, axis=0)
    else:
        local_min = np.full(domain.geometry.dim, np.inf)
        local_max = np.full(domain.geometry.dim, -np.inf)

    bounds_min = np.array(
        [domain.comm.allreduce(value, op=MPI.MIN) for value in local_min],
        dtype=np.float64,
    )
    bounds_max = np.array(
        [domain.comm.allreduce(value, op=MPI.MAX) for value in local_max],
        dtype=np.float64,
    )
    return bounds_min, bounds_max


def build_clamp_bc(domain: mesh.Mesh, V: fem.FunctionSpace, config: Config):
    bounds_min, bounds_max = global_bounds(domain)
    axis = config.clamp_axis
    bbox_size = float(np.max(bounds_max - bounds_min))
    tolerance = max(config.clamp_tolerance_fraction * bbox_size, 1.0e-15)
    target = bounds_min[axis] if config.clamp_at_min else bounds_max[axis]

    fdim = domain.topology.dim - 1

    if config.clamp_at_min:
        facets = mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x: x[axis] <= target + tolerance,
        )
    else:
        facets = mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x: x[axis] >= target - tolerance,
        )

    dofs = fem.locate_dofs_topological(V, fdim, facets)
    fixed_dofs = domain.comm.allreduce(int(dofs.size), op=MPI.SUM)
    if fixed_dofs == 0:
        raise RuntimeError("No boundary dofs were found for the clamp condition.")

    zero = make_function(V, "zero_displacement")
    zero.x.array[:] = 0.0
    zero.x.scatter_forward()

    return fem.dirichletbc(zero, dofs), fixed_dofs, target, tolerance


def build_dof_to_point_map(
    Q: fem.FunctionSpace,
    points: np.ndarray,
    decimals: int,
) -> np.ndarray:
    dof_coords = Q.tabulate_dof_coordinates()
    point_map = {
        tuple(np.round(point, decimals)): index
        for index, point in enumerate(points)
    }

    indices = np.empty(dof_coords.shape[0], dtype=np.int64)
    missing = 0
    for i, coord in enumerate(dof_coords):
        key = tuple(np.round(coord[: points.shape[1]], decimals))
        point_index = point_map.get(key)
        if point_index is None:
            missing += 1
            point_index = -1
        indices[i] = point_index

    if missing:
        raise RuntimeError(
            f"Could not match {missing} temperature dofs to VTU points. "
            "Try lowering Config.coordinate_round_decimals."
        )

    return indices


def assign_temperature(
    temperature_function: fem.Function,
    values_at_points: np.ndarray,
    dof_to_point: np.ndarray,
) -> None:
    if temperature_function.x.array.size != dof_to_point.size:
        raise RuntimeError(
            "Temperature function dof count does not match the precomputed "
            "VTU point map."
        )
    temperature_function.x.array[:] = values_at_points[dof_to_point]
    temperature_function.x.scatter_forward()


def prepare_output_dir(config: Config, comm: MPI.Comm) -> None:
    if comm.rank == 0:
        if config.output_dir.exists() and config.clear_output:
            if config.output_dir.is_dir():
                shutil.rmtree(config.output_dir)
            else:
                raise RuntimeError(f"Output path exists and is not a directory: {config.output_dir}")
        config.output_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()


def keep_only_serial_vtu(comm: MPI.Comm, filename: Path) -> None:
    if comm.size != 1 or comm.rank != 0:
        return

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


def write_vtu_frame(
    comm: MPI.Comm,
    output_path: Path,
    functions: list[fem.Function],
    time: float,
) -> None:
    with io.VTKFile(comm, output_path.as_posix(), "w") as vtk:
        vtk.write_function(functions, time)
    del vtk
    keep_only_serial_vtu(comm, output_path)


def find_input_pvd(config: Config) -> Path | None:
    if config.input_pvd is not None:
        return config.input_pvd if config.input_pvd.exists() else None

    laser_pvd = config.input_dir / "laser.pvd"
    if laser_pvd.exists():
        return laser_pvd

    pvd_files = sorted(config.input_dir.glob("*.pvd"))
    if len(pvd_files) == 1:
        return pvd_files[0]

    return None


def read_pvd_times(pvd_path: Path) -> dict[Path, float]:
    root = ET.parse(pvd_path).getroot()
    times: dict[Path, float] = {}

    for dataset in root.findall(".//DataSet"):
        file_name = dataset.attrib.get("file")
        timestep = dataset.attrib.get("timestep")
        if file_name is None or timestep is None:
            continue
        times[(pvd_path.parent / file_name).resolve()] = float(timestep)

    return times


def input_frames(config: Config) -> list[InputFrame]:
    paths = sorted(config.input_dir.glob(config.input_pattern))
    if config.max_frames is not None:
        paths = paths[: config.max_frames]
    if not paths:
        raise FileNotFoundError(
            f"No VTU files found: {config.input_dir / config.input_pattern}"
        )

    pvd_path = find_input_pvd(config)
    pvd_times = read_pvd_times(pvd_path) if pvd_path is not None else {}

    frames: list[InputFrame] = []
    for frame_id, path in enumerate(paths):
        frames.append(
            InputFrame(
                path=path,
                time=pvd_times.get(path.resolve(), float(frame_id)),
            )
        )

    return frames


def write_output_pvd(
    comm: MPI.Comm,
    output_dir: Path,
    output_prefix: str,
    frames: list[tuple[float, Path]],
) -> None:
    if comm.rank != 0:
        return

    pvd_path = output_dir / f"{output_prefix}.pvd"
    with pvd_path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">\n')
        stream.write("  <Collection>\n")
        for time_value, path in frames:
            stream.write(
                f'    <DataSet timestep="{time_value:.12e}" group="" part="0" '
                f'file="{path.name}"/>\n'
            )
        stream.write("  </Collection>\n")
        stream.write("</VTKFile>\n")


def solve_thermal_stress(config: Config) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    start_total = time.perf_counter()
    frames = input_frames(config)
    if rank == 0:
        print(f"input frames: {len(frames)}", flush=True)
        print(f"input dir: {config.input_dir}", flush=True)
        print(f"temperature field: {config.temperature_field}", flush=True)
        print(f"reading first VTU: {frames[0].path}", flush=True)

    start = time.perf_counter()
    first = read_vtu(frames[0].path, config.temperature_field)
    material = material_from_config_and_vtu(config, first.field_data)
    if rank == 0:
        print(
            f"first VTU loaded in {time.perf_counter() - start:.2f} s: "
            f"points={first.points.shape[0]}, tetrahedra={first.cells.shape[0]}",
            flush=True,
        )
        print("creating DOLFINx mesh; this can take a while for large VTU files...", flush=True)

    start = time.perf_counter()
    domain = make_domain(first.points, first.cells, comm)
    if rank == 0:
        print(f"DOLFINx mesh created in {time.perf_counter() - start:.2f} s", flush=True)
        print("building function spaces and boundary conditions...", flush=True)

    start = time.perf_counter()
    spaces = build_function_spaces(domain)
    bc, fixed_dofs, clamp_target, clamp_tolerance = build_clamp_bc(domain, spaces.V, config)
    bcs = [bc]
    if rank == 0:
        print(f"function spaces ready in {time.perf_counter() - start:.2f} s", flush=True)

    temperature = make_function(spaces.Q, config.temperature_field)
    start = time.perf_counter()
    dof_to_point = build_dof_to_point_map(
        spaces.Q,
        first.points,
        config.coordinate_round_decimals,
    )
    if rank == 0:
        print(f"temperature dofs mapped to VTU points in {time.perf_counter() - start:.2f} s", flush=True)

    lambda_ = material.lambda_
    mu = material.mu
    alpha = material.thermal_expansion
    t_ref = material.reference_temperature

    u_trial = ufl.TrialFunction(spaces.V)
    v_test = ufl.TestFunction(spaces.V)
    dx = ufl.dx(domain)
    gdim = domain.geometry.dim
    identity = ufl.Identity(gdim)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def elastic_stress(w):
        e = eps(w)
        return lambda_ * ufl.tr(e) * identity + 2.0 * mu * e

    def thermal_strain():
        return alpha * (temperature - t_ref) * identity

    def thermal_stress():
        e_th = thermal_strain()
        return lambda_ * ufl.tr(e_th) * identity + 2.0 * mu * e_th

    lhs = ufl.inner(elastic_stress(u_trial), eps(v_test)) * dx
    rhs = ufl.inner(thermal_stress(), eps(v_test)) * dx

    lhs_form = fem.form(lhs)
    rhs_form = fem.form(rhs)

    if rank == 0:
        print("assembling thermoelastic stiffness matrix...", flush=True)
    start = time.perf_counter()
    A = assemble_matrix(lhs_form, bcs=bcs)
    A.assemble()
    if rank == 0:
        print(f"matrix assembled in {time.perf_counter() - start:.2f} s", flush=True)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.GAMG)
    solver.setTolerances(
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        max_it=config.solver_max_it,
    )
    solver.setFromOptions()

    displacement = make_function(spaces.V, "displacement")
    displacement_magnitude = make_function(spaces.Q, "displacement_magnitude")

    # Keep stress-derived quantities on the same nodal P1 space as temperature.
    # ParaView then stores T, displacement and stresses together as PointData.
    stress_trace = make_function(spaces.Q, "stress_trace")
    von_mises = make_function(spaces.Q, "von_mises")
    sigma_xx = make_function(spaces.Q, "sigma_xx")
    sigma_yy = make_function(spaces.Q, "sigma_yy")
    sigma_zz = make_function(spaces.Q, "sigma_zz")
    sigma_xy = make_function(spaces.Q, "sigma_xy")
    sigma_xz = make_function(spaces.Q, "sigma_xz")
    sigma_yz = make_function(spaces.Q, "sigma_yz")

    sigma_total = elastic_stress(displacement) - thermal_stress()
    sigma_dev = sigma_total - (ufl.tr(sigma_total) / 3.0) * identity

    q_points = interpolation_points(spaces.Q)
    displacement_magnitude_expr = fem.Expression(
        ufl.sqrt(ufl.dot(displacement, displacement)),
        q_points,
    )
    stress_trace_expr = fem.Expression(ufl.tr(sigma_total), q_points)
    von_mises_expr = fem.Expression(
        ufl.sqrt(1.5 * ufl.inner(sigma_dev, sigma_dev)),
        q_points,
    )
    component_exprs = [
        fem.Expression(sigma_total[0, 0], q_points),
        fem.Expression(sigma_total[1, 1], q_points),
        fem.Expression(sigma_total[2, 2], q_points),
        fem.Expression(sigma_total[0, 1], q_points),
        fem.Expression(sigma_total[0, 2], q_points),
        fem.Expression(sigma_total[1, 2], q_points),
    ]
    component_functions = [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz]

    output_functions = [
        temperature,
        displacement,
        displacement_magnitude,
        stress_trace,
        von_mises,
        sigma_xx,
        sigma_yy,
        sigma_zz,
        sigma_xy,
        sigma_xz,
        sigma_yz,
    ]

    prepare_output_dir(config, comm)

    if rank == 0:
        print(f"E={material.young_modulus:.6e}", flush=True)
        print(f"nu={material.poisson_ratio:.6e}", flush=True)
        print(f"lambda={lambda_:.6e}, mu={mu:.6e}", flush=True)
        print(f"alpha={alpha:.6e}, T_ref={t_ref:.6e}", flush=True)
        print(
            f"clamp axis={config.clamp_axis}, target={clamp_target:.6e}, "
            f"tolerance={clamp_tolerance:.6e}, fixed dofs={fixed_dofs}",
            flush=True,
        )
        print(f"output dir: {config.output_dir}", flush=True)

    try:
        written_frames: list[tuple[float, Path]] = []
        for frame_id, frame in enumerate(frames):
            path = frame.path
            if frame_id == 0:
                temperature_values = first.temperature
            else:
                temperature_values, _ = read_temperature_only(path, config.temperature_field)

            assign_temperature(temperature, temperature_values, dof_to_point)

            b = assemble_vector(rhs_form)
            apply_lifting(b, [lhs_form], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            displacement.x.array[:] = 0.0
            solver.solve(b, displacement.x.petsc_vec)
            displacement.x.scatter_forward()
            b.destroy()

            reason = solver.getConvergedReason()
            if reason < 0:
                raise RuntimeError(
                    f"PETSc solver did not converge for frame {path.name}: {reason}"
                )

            displacement_magnitude.interpolate(displacement_magnitude_expr)
            displacement_magnitude.x.scatter_forward()
            stress_trace.interpolate(stress_trace_expr)
            stress_trace.x.scatter_forward()
            von_mises.interpolate(von_mises_expr)
            von_mises.x.scatter_forward()

            for function, expression in zip(component_functions, component_exprs):
                function.interpolate(expression)
                function.x.scatter_forward()

            t_min = float(np.min(temperature_values))
            t_max = float(np.max(temperature_values))
            local_max_u = (
                float(np.max(displacement_magnitude.x.array))
                if displacement_magnitude.x.array.size
                else 0.0
            )
            local_max_vm = (
                float(np.max(von_mises.x.array))
                if von_mises.x.array.size
                else 0.0
            )
            max_u = comm.allreduce(local_max_u, op=MPI.MAX)
            max_vm = comm.allreduce(local_max_vm, op=MPI.MAX)

            output_path = config.output_dir / f"{config.output_prefix}_{frame_id:05d}.vtu"
            write_vtu_frame(comm, output_path, output_functions, frame.time)
            written_frames.append((frame.time, output_path))

            if rank == 0:
                print(
                    f"frame={frame_id:05d}, time={frame.time:.6e}, input={path.name}, "
                    f"T=[{t_min:.6e}, {t_max:.6e}], "
                    f"max|u|={max_u:.6e}, max von_mises={max_vm:.6e}, "
                    f"ksp_it={solver.getIterationNumber()}, output={output_path}",
                    flush=True,
                )
        write_output_pvd(comm, config.output_dir, config.output_prefix, written_frames)
    finally:
        solver.destroy()
        A.destroy()

    if rank == 0:
        print("Done.", flush=True)
        print(f"Open VTU files in ParaView from: {config.output_dir}", flush=True)
        print(f"total runtime: {time.perf_counter() - start_total:.2f} s", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Solve static thermoelastic stress fields from VTU temperature frames.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Config.input_dir,
        help="Directory with input VTU temperature frames.",
    )
    parser.add_argument(
        "--pattern",
        default=Config.input_pattern,
        help="Glob pattern for input VTU frames.",
    )
    parser.add_argument(
        "--input-pvd",
        type=Path,
        default=Config.input_pvd,
        help="Optional PVD file with physical timesteps for input VTU frames.",
    )
    parser.add_argument(
        "--temperature-field",
        default=Config.temperature_field,
        help="Name of the scalar PointData temperature field.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Config.output_dir,
        help="Directory for output VTU files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=Config.max_frames,
        help="Limit the number of processed frames, useful for a first test run.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove the output directory before writing new files.",
    )
    args = parser.parse_args()

    return Config(
        input_dir=args.input_dir,
        input_pattern=args.pattern,
        input_pvd=args.input_pvd,
        temperature_field=args.temperature_field,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        clear_output=args.clear_output,
    )


def main() -> None:
    solve_thermal_stress(parse_args())


if __name__ == "__main__":
    main()
