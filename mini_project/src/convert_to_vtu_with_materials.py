from pathlib import Path

import meshio
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

VTU_PATH = BASE_DIR.parent / "3d-models" / "mesh.vtu"
OUT_PATH = BASE_DIR.parent / "3d-models" / "head_with_materials.vtu"


MATERIAL_ID_TO_PARAMS = {
    3: {"name": "fat", "rho": 0.001158, "lambda": 3070000.0, "mu": 2050000.0},
    4: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
    11: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
    12: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
    21: {"name": "bones", "rho": 0.001, "lambda": 786000000.0, "mu": 1180000000.0},
    22: {"name": "blood", "rho": 0.001, "lambda": 239000.0, "mu": 500.0},
    24: {"name": "trachea", "rho": 0.002, "lambda": 14300000.0, "mu": 3570000.0},
    25: {"name": "aorta", "rho": 0.001, "lambda": 9210000.0, "mu": 190000.0},
    26: {"name": "veins", "rho": 0.001, "lambda": 32890000.0, "mu": 670000.0},
    27: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
    28: {"name": "blood", "rho": 0.001, "lambda": 239000.0, "mu": 500.0},
    29: {"name": "blood", "rho": 0.001, "lambda": 239000.0, "mu": 500.0},
    30: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
    31: {"name": "fat", "rho": 0.001158, "lambda": 3070000.0, "mu": 2050000.0},
    32: {"name": "muscle", "rho": 0.001, "lambda": 3070000.0, "mu": 2050000.0},
}


msh = meshio.read(VTU_PATH)

if "Material" not in msh.cell_data:
    raise RuntimeError("В исходном VTU нет CellData['Material'].")

new_cell_data = {}

for name, data_blocks in msh.cell_data.items():
    new_cell_data[name] = data_blocks

rho_blocks = []
lambda_blocks = []
mu_blocks = []
material_id_blocks = []

for block_index, cell_block in enumerate(msh.cells):
    material_ids = np.asarray(msh.cell_data["Material"][block_index], dtype=np.int32)

    rho = np.zeros_like(material_ids, dtype=np.float64)
    lambda_ = np.zeros_like(material_ids, dtype=np.float64)
    mu = np.zeros_like(material_ids, dtype=np.float64)

    for material_id in np.unique(material_ids):
        material_id = int(material_id)

        if material_id not in MATERIAL_ID_TO_PARAMS:
            raise RuntimeError(f"Нет параметров для material_id = {material_id}")

        mask = material_ids == material_id
        params = MATERIAL_ID_TO_PARAMS[material_id]

        rho[mask] = params["rho"]
        lambda_[mask] = params["lambda"]
        mu[mask] = params["mu"]

    material_id_blocks.append(material_ids)
    rho_blocks.append(rho)
    lambda_blocks.append(lambda_)
    mu_blocks.append(mu)

new_cell_data["material_id"] = material_id_blocks
new_cell_data["rho"] = rho_blocks
new_cell_data["lambda"] = lambda_blocks
new_cell_data["mu"] = mu_blocks

out_mesh = meshio.Mesh(
    points=msh.points,
    cells=msh.cells,
    cell_data=new_cell_data,
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
meshio.write(str(OUT_PATH), out_mesh)

print(f"Saved: {OUT_PATH}")
