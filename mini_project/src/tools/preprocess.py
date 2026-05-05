from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import meshio
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, io, mesh


# ---------------------------------------------------------------------
# 1. Таблица материалов
# ---------------------------------------------------------------------

MATERIAL_ID_TO_PARAMS = {
    3: {
        "name": "fat",
        "rho": 0.001158,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    4: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    11: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    12: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    21: {
        "name": "bones",
        "rho": 0.001,
        "lambda": 786000000.0,
        "mu": 1180000000.0,
    },
    22: {
        "name": "blood",
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    24: {
        "name": "trachea",
        "rho": 0.002,
        "lambda": 14300000.0,
        "mu": 3570000.0,
    },
    25: {
        "name": "aorta",
        "rho": 0.001,
        "lambda": 9210000.0,
        "mu": 190000.0,
    },
    26: {
        "name": "veins",
        "rho": 0.001,
        "lambda": 32890000.0,
        "mu": 670000.0,
    },
    27: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    28: {
        "name": "blood",
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    29: {
        "name": "blood",
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    30: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    31: {
        "name": "fat",
        "rho": 0.001158,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    32: {
        "name": "muscle",
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
}


@dataclass
class HeadModel:
    """
    Контейнер для всего, что нужно физике.

    domain:
        DOLFINx-сетка.

    cell_tags:
        MeshTags на тетраэдрах.
        Это аналог Physical Volume из Gmsh.

    rho, lambda_, mu:
        DG0-поля. Одно значение на одну ячейку.
        Их можно напрямую использовать в ufl-формах.

    material:
        DG0-функция с численным material_id.
        Удобна для отладки и вывода в ParaView.
    """

    domain: mesh.Mesh
    cell_tags: mesh.MeshTags
    rho: fem.Function
    lambda_: fem.Function
    mu: fem.Function
    material: fem.Function


# ---------------------------------------------------------------------
# 2. Чтение VTU
# ---------------------------------------------------------------------

def read_vtu_tetra_mesh(vtu_path: Path):
    """
    Читает .vtu через meshio.

    Возвращает:
    - points: координаты вершин;
    - cells: tetra-ячейки;
    - material_ids: Material ID для каждой tetra-ячейки.
    """
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    msh = meshio.read(vtu_path)

    if "Material" not in msh.cell_data:
        raise RuntimeError("В VTU нет CellData['Material'].")

    for block_index, cell_block in enumerate(msh.cells):
        if cell_block.type == "tetra":
            points = msh.points.astype(np.float64)
            cells = cell_block.data.astype(np.int64)
            material_ids = msh.cell_data["Material"][block_index].astype(np.int32)
            return points, cells, material_ids

    raise RuntimeError("В VTU не найден блок tetra.")


def check_material_ids(material_ids: np.ndarray) -> None:
    """
    Проверяет, что для всех Material ID есть физические параметры.
    """
    unique_ids = sorted(map(int, np.unique(material_ids)))

    missing_ids = [
        material_id
        for material_id in unique_ids
        if material_id not in MATERIAL_ID_TO_PARAMS
    ]

    if missing_ids:
        raise RuntimeError(
            "Для этих material_id нет параметров в MATERIAL_ID_TO_PARAMS: "
            f"{missing_ids}"
        )


# ---------------------------------------------------------------------
# 3. Создание DOLFINx mesh
# ---------------------------------------------------------------------

def create_dolfinx_mesh(points: np.ndarray, cells: np.ndarray) -> mesh.Mesh:
    """
    Создаёт DOLFINx mesh из массивов points/cells.

    Это аналог того, что gmshio.model_to_mesh делает для Gmsh.
    Только тут источник не Gmsh, а VTU.
    """
    coordinate_element = basix.ufl.element(
        "Lagrange",
        "tetrahedron",
        1,
        shape=(3,),
    )

    ufl_domain = ufl.Mesh(coordinate_element)

    domain = mesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        ufl_domain,
        points,
    )

    return domain


def create_cell_tags(
    domain: mesh.Mesh,
    material_ids_from_vtu: np.ndarray,
) -> mesh.MeshTags:
    """
    Создаёт MeshTags для ячеек.

    Важно:
    DOLFINx может переупорядочить ячейки.
    Поэтому используем domain.topology.original_cell_index.

    cell_tags:
        local cell index -> Material ID
    """
    tdim = domain.topology.dim

    num_local_cells = domain.topology.index_map(tdim).size_local

    local_cell_indices = np.arange(num_local_cells, dtype=np.int32)

    original_cell_indices = np.asarray(
        domain.topology.original_cell_index[:num_local_cells],
        dtype=np.int64,
    )

    local_material_values = material_ids_from_vtu[original_cell_indices].astype(np.int32)

    # meshtags обычно требуют отсортированные индексы
    order = np.argsort(local_cell_indices)

    cell_tags = mesh.meshtags(
        domain,
        tdim,
        local_cell_indices[order],
        local_material_values[order],
    )

    cell_tags.name = "material_id"

    return cell_tags


# ---------------------------------------------------------------------
# 4. DG0-поля rho, lambda, mu
# ---------------------------------------------------------------------

def create_dg0_material_functions(
    domain: mesh.Mesh,
    cell_tags: mesh.MeshTags,
):
    """
    Создаёт DG0-функции:
    - rho
    - lambda_
    - mu
    - material

    DG0 = Discontinuous Galerkin degree 0.
    То есть одно значение на одну ячейку.
    """
    Q = fem.functionspace(domain, ("DG", 0))

    rho = fem.Function(Q)
    lambda_ = fem.Function(Q)
    mu = fem.Function(Q)
    material = fem.Function(Q)

    rho.name = "rho"
    lambda_.name = "lambda"
    mu.name = "mu"
    material.name = "material_id"

    dofmap = Q.dofmap

    # cell_tags.indices: локальные номера ячеек
    # cell_tags.values: material_id на этих ячейках
    for local_cell_id, material_id in zip(cell_tags.indices, cell_tags.values):
        local_cell_id = int(local_cell_id)
        material_id = int(material_id)

        params = MATERIAL_ID_TO_PARAMS[material_id]

        dof = dofmap.cell_dofs(local_cell_id)[0]

        rho.x.array[dof] = params["rho"]
        lambda_.x.array[dof] = params["lambda"]
        mu.x.array[dof] = params["mu"]
        material.x.array[dof] = float(material_id)

    rho.x.scatter_forward()
    lambda_.x.scatter_forward()
    mu.x.scatter_forward()
    material.x.scatter_forward()

    return rho, lambda_, mu, material


# ---------------------------------------------------------------------
# 5. Главная функция: загрузить модель из VTU
# ---------------------------------------------------------------------

def load_head_model_from_vtu(vtu_path: Path | str = Path("body/mesh.vtu")) -> HeadModel:
    """
    Основной способ для твоего проекта.

    Использование:
        model = load_head_model_from_vtu("body/mesh.vtu")

        domain = model.domain
        rho = model.rho
        lambda_ = model.lambda_
        mu = model.mu
    """
    vtu_path = Path(vtu_path)

    points, cells, material_ids = read_vtu_tetra_mesh(vtu_path)

    check_material_ids(material_ids)

    domain = create_dolfinx_mesh(points, cells)

    cell_tags = create_cell_tags(domain, material_ids)

    rho, lambda_, mu, material = create_dg0_material_functions(domain, cell_tags)

    return HeadModel(
        domain=domain,
        cell_tags=cell_tags,
        rho=rho,
        lambda_=lambda_,
        mu=mu,
        material=material,
    )


# ---------------------------------------------------------------------
# 6. Сохранение checkpoint-файла XDMF
# ---------------------------------------------------------------------

def write_head_checkpoint_xdmf(
    model: HeadModel,
    xdmf_path: Path | str = Path("head_checkpoint.xdmf"),
) -> None:
    """
    Сохраняет сетку и material_id как MeshTags.

    Это лучше, чем хранить rho/lambda/mu как единственный источник правды.
    Физические поля потом можно восстановить из material_id.
    """
    xdmf_path = Path(xdmf_path)

    with io.XDMFFile(MPI.COMM_WORLD, xdmf_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(model.domain)
        xdmf.write_meshtags(model.cell_tags, model.domain.geometry)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Saved checkpoint: {xdmf_path}")


def load_head_model_from_xdmf(
    xdmf_path: Path | str = Path("head_checkpoint.xdmf"),
) -> HeadModel:
    """
    Загружает сетку и cell_tags из XDMF,
    затем восстанавливает rho/lambda/mu.

    Важно:
    этот способ работает с checkpoint-файлом,
    который был записан через write_head_checkpoint_xdmf().
    """
    xdmf_path = Path(xdmf_path)

    with io.XDMFFile(MPI.COMM_WORLD, xdmf_path.as_posix(), "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")

        # Нужно создать connectivity, чтобы MeshTags корректно привязались к ячейкам
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, 0)

        cell_tags = xdmf.read_meshtags(domain, name="material_id")

    rho, lambda_, mu, material = create_dg0_material_functions(domain, cell_tags)

    return HeadModel(
        domain=domain,
        cell_tags=cell_tags,
        rho=rho,
        lambda_=lambda_,
        mu=mu,
        material=material,
    )


# ---------------------------------------------------------------------
# 7. Минимальная физика: формы для линейной упругости
# ---------------------------------------------------------------------

def make_elasticity_forms(model: HeadModel):
    """
    Каркас для будущей физики.

    Уравнение линейной упругости:

        rho * u_tt = div(sigma(u)) + f

    где:

        sigma(u) = lambda * tr(eps(u)) * I + 2 * mu * eps(u)
        eps(u) = sym(grad(u))

    Эта функция пока не решает задачу,
    а только создаёт UFL-формы массы и жёсткости.
    """
    domain = model.domain
    rho = model.rho
    lambda_ = model.lambda_
    mu = model.mu

    gdim = domain.geometry.dim

    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        I = ufl.Identity(gdim)
        return lambda_ * ufl.tr(eps(w)) * I + 2.0 * mu * eps(w)

    dx = ufl.dx(domain)

    mass_form = rho * ufl.inner(u, v) * dx
    stiffness_form = ufl.inner(sigma(u), eps(v)) * dx

    return V, mass_form, stiffness_form


# ---------------------------------------------------------------------
# 8. Отладочный запуск
# ---------------------------------------------------------------------

def print_material_summary(model: HeadModel) -> None:
    """
    Печатает сводку по material_id.
    """
    values = np.asarray(model.cell_tags.values, dtype=np.int32)

    unique_ids, counts = np.unique(values, return_counts=True)

    if MPI.COMM_WORLD.rank == 0:
        print("\nMaterial summary:")
        print("ID | local cells | name")
        print("---|-------------|------")

        for material_id, count in zip(unique_ids, counts):
            material_id = int(material_id)
            name = MATERIAL_ID_TO_PARAMS[material_id]["name"]
            print(f"{material_id:2d} | {int(count):11d} | {name}")


def main() -> None:
    model = load_head_model_from_vtu("body/mesh.vtu")

    print_material_summary(model)

    write_head_checkpoint_xdmf(model, "head_checkpoint.xdmf")

    V, mass_form, stiffness_form = make_elasticity_forms(model)

    if MPI.COMM_WORLD.rank == 0:
        print("\nModel is ready for physics.")
        print("Vector function space:", V)


if __name__ == "__main__":
    main()