from pathlib import Path

import numpy as np
import meshio
import basix.ufl
import ufl

from mpi4py import MPI
from dolfinx import fem, io, mesh


# Путь к твоей VTU-модели
VTU_PATH = Path("body/mesh.vtu")

# Файл результата
OUT_XDMF_PATH = Path("head_with_materials.xdmf")


MATERIAL_ID_TO_PARAMS = {
    3: {
        "name": "fat",  # большой внешний/мягкий слой
        "rho": 0.001158,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    4: {
        "name": "muscle",  # основной мягкотканный объем
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    11: {
        "name": "muscle",  # вероятно внутренняя мягкая ткань головы; в XML нет brain
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    12: {
        "name": "muscle",  # вероятно вторая внутренняя мягкая зона головы
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    21: {
        "name": "bones",  # череп / кости / плечевой каркас
        "rho": 0.001,
        "lambda": 786000000.0,
        "mu": 1180000000.0,
    },
    22: {
        "name": "blood",  # маленькая внутренняя зона; heart для головы не подходит
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    24: {
        "name": "trachea",  # вытянутая центральная трубка в шее
        "rho": 0.002,
        "lambda": 14300000.0,
        "mu": 3570000.0,
    },
    25: {
        "name": "aorta",  # крупный сосуд / магистральный канал
        "rho": 0.001,
        "lambda": 9210000.0,
        "mu": 190000.0,
    },
    26: {
        "name": "veins",  # ветвящиеся сосудистые структуры
        "rho": 0.001,
        "lambda": 32890000.0,
        "mu": 670000.0,
    },
    27: {
        "name": "muscle",  # крупный объем головы/шеи; digestive для головы не подходит
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    28: {
        "name": "blood",  # похоже на глаза/жидкую мягкую область; в XML нет eye/vitreous
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    29: {
        "name": "blood",  # очень маленькие структуры, вероятно сосуды/жидкость
        "rho": 0.001,
        "lambda": 239000.0,
        "mu": 500.0,
    },
    30: {
        "name": "muscle",  # компактная мягкая область головы/шеи
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    31: {
        "name": "fat",  # небольшая мягкая область; точность низкая
        "rho": 0.001158,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
    32: {
        "name": "muscle",  # мягкая/шейная структура; не похоже на bones
        "rho": 0.001,
        "lambda": 3070000.0,
        "mu": 2050000.0,
    },
}


def read_vtu_tetra_mesh(vtu_path: Path):
    """
    Читает mesh.vtu через meshio.

    Возвращает:
    - points: координаты вершин
    - cells: tetra-ячейки
    - material_ids: Material ID для каждой tetra-ячейки
    """
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    msh = meshio.read(vtu_path)

    if "Material" not in msh.cell_data:
        raise RuntimeError("В VTU нет CellData['Material'].")

    for block_index, cell_block in enumerate(msh.cells):
        if cell_block.type == "tetra":
            cells = cell_block.data.astype(np.int64)
            material_ids = msh.cell_data["Material"][block_index].astype(np.int32)
            points = msh.points.astype(np.float64)

            return points, cells, material_ids

    raise RuntimeError("В VTU не найден блок tetra.")


def create_dolfinx_mesh(points: np.ndarray, cells: np.ndarray):
    """
    Создаёт DOLFINx mesh из VTU points/cells.
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


def check_material_ids(material_ids: np.ndarray):
    """
    Проверяет, что для каждого material_id есть параметры.
    """
    unique_ids = sorted(map(int, np.unique(material_ids)))

    missing_ids = [
        material_id for material_id in unique_ids
        if material_id not in MATERIAL_ID_TO_PARAMS
    ]

    if missing_ids:
        raise RuntimeError(
            "Для этих material_id нет параметров в MATERIAL_ID_TO_PARAMS: "
            f"{missing_ids}"
        )


def create_dg0_material_functions(domain, material_ids: np.ndarray):
    """
    Создаёт DG0-функции rho, lambda, mu.

    DG0 значит: одно значение на одну ячейку.
    Это правильно, потому что Material в VTU лежит в CellData.
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

    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local

    # DOLFINx может переупорядочить ячейки.
    # Поэтому берём соответствие:
    # local_cell_id в DOLFINx -> original_cell_id из исходного VTU.
    original_cell_indices = domain.topology.original_cell_index

    dofmap = Q.dofmap

    for local_cell_id in range(num_local_cells):
        original_cell_id = int(original_cell_indices[local_cell_id])
        material_id = int(material_ids[original_cell_id])

        params = MATERIAL_ID_TO_PARAMS[material_id]

        # В DG0 на каждой ячейке ровно один degree of freedom.
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


def print_material_summary(material_ids: np.ndarray):
    """
    Печатает статистику по Material ID.
    """
    unique_ids, counts = np.unique(material_ids, return_counts=True)

    print("\nMaterial summary:")
    print("ID | cells | name")
    print("---|-------|------")

    for material_id, count in zip(unique_ids, counts):
        material_id = int(material_id)
        name = MATERIAL_ID_TO_PARAMS.get(material_id, {}).get("name", "UNKNOWN")
        print(f"{material_id:2d} | {int(count):6d} | {name}")


def main():
    print(f"Reading VTU: {VTU_PATH}", flush=True)

    points, cells, material_ids = read_vtu_tetra_mesh(VTU_PATH)

    print("points:", points.shape, flush=True)
    print("cells:", cells.shape, flush=True)
    print("material ids:", np.unique(material_ids), flush=True)

    check_material_ids(material_ids)
    print_material_summary(material_ids)

    print("\nCreating DOLFINx mesh...", flush=True)
    domain = create_dolfinx_mesh(points, cells)

    print("Creating DG0 material functions...", flush=True)
    rho, lambda_, mu, material = create_dg0_material_functions(domain, material_ids)

    print(f"Writing XDMF: {OUT_XDMF_PATH}", flush=True)

    with io.XDMFFile(MPI.COMM_WORLD, OUT_XDMF_PATH.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(material)
        xdmf.write_function(rho)
        xdmf.write_function(lambda_)
        xdmf.write_function(mu)

    print("\nSaved:", OUT_XDMF_PATH, flush=True)
    print("Open this file in ParaView:", OUT_XDMF_PATH, flush=True)


if __name__ == "__main__":
    main()