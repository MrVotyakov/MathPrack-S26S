import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VTU_PATH = SCRIPT_DIR.parent / "body" / "mesh.vtu"
DEFAULT_XML_PATH = SCRIPT_DIR.parent / "source_code" / "tasks" / "objects" / "head.xml"

DEFAULT_OUT_CELLS_CSV = Path("head_cells_materials.csv")
DEFAULT_OUT_POINTS_CSV = Path("head_points_materials_approx.csv")


# ВАЖНО:
# Это нужно заполнить после восстановления соответствия ID -> tissue.
# Пока здесь пример-заглушка.
MATERIAL_ID_TO_NAME = {
    # 3: "bones",
    # 4: "fat",
    # 11: "muscle",
    # ...
}


def read_material_definitions(xml_path: Path) -> dict[str, dict[str, float]]:
    """
    Читает из head.xml параметры материалов:
    name -> {lambda, mu, rho}
    """
    root = ET.parse(xml_path).getroot()

    materials = {}

    materials_node = root.find("materials")
    if materials_node is None:
        raise RuntimeError(f"В {xml_path} не найден блок <materials>.")

    for elem in materials_node.findall("material"):
        name = elem.attrib.get("name")
        rheology = elem.attrib.get("rheology")

        if name is None or rheology is None:
            continue

        la = elem.findtext("la")
        mu = elem.findtext("mu")
        rho = elem.findtext("rho")

        if la is None or mu is None or rho is None:
            continue

        materials[name] = {
            "lambda": float(la.strip()),
            "mu": float(mu.strip()),
            "rho": float(rho.strip()),
        }

    return materials


def get_cells_and_materials(mesh: meshio.Mesh):
    """
    Достаёт тетраэдры и CellData['Material'] из VTU.
    """
    tetra_cells = None
    tetra_materials = None

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in ("tetra", "tetra10"):
            tetra_cells = cell_block.data

            # В meshio cell_data хранится по блокам ячеек.
            if "Material" not in mesh.cell_data:
                raise RuntimeError("В mesh.vtu нет CellData с именем 'Material'.")

            tetra_materials = mesh.cell_data["Material"][i]
            break

    if tetra_cells is None:
        raise RuntimeError("В mesh.vtu не найдены tetra / tetra10 ячейки.")

    if tetra_materials is None:
        raise RuntimeError("Не найдены Material-метки для tetra-ячеек.")

    return tetra_cells, tetra_materials


def export_cell_materials(vtu_path: Path, xml_path: Path, out_cells_csv: Path):
    mesh = meshio.read(vtu_path)

    points = mesh.points
    cells, material_ids = get_cells_and_materials(mesh)

    material_defs = read_material_definitions(xml_path)

    rows = []

    for cell_id, point_ids in enumerate(cells):
        material_id = int(material_ids[cell_id])
        material_name = MATERIAL_ID_TO_NAME.get(material_id)

        if material_name is None:
            rho = None
            lam = None
            mu = None
        else:
            props = material_defs[material_name]
            rho = props["rho"]
            lam = props["lambda"]
            mu = props["mu"]

        row = {
            "cell_id": cell_id,
            "material_id": material_id,
            "material_name": material_name,
            "rho": rho,
            "lambda": lam,
            "mu": mu,
        }

        # point_id внутри ячейки
        for local_i, point_id in enumerate(point_ids):
            row[f"point_id_{local_i}"] = int(point_id)

            x, y, z = points[point_id]
            row[f"point_{local_i}_x"] = x
            row[f"point_{local_i}_y"] = y
            row[f"point_{local_i}_z"] = z

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_cells_csv, index=False)

    print(f"Saved cell material table to: {out_cells_csv}")
    print(df.head())


def export_point_materials_approx(vtu_path: Path, xml_path: Path, out_points_csv: Path):
    """
    Приближённый экспорт point_id -> material.

    ВАЖНО:
    В исходном VTU материал задан на ячейках, а не на точках.
    Поэтому для точки берём самый частый material_id среди соседних ячеек.
    На границах материалов это может быть неоднозначно.
    """
    mesh = meshio.read(vtu_path)

    points = mesh.points
    cells, material_ids = get_cells_and_materials(mesh)
    material_defs = read_material_definitions(xml_path)

    point_to_materials = {}

    for cell_id, point_ids in enumerate(cells):
        material_id = int(material_ids[cell_id])

        for point_id in point_ids:
            point_id = int(point_id)
            point_to_materials.setdefault(point_id, []).append(material_id)

    rows = []

    for point_id, adjacent_materials in point_to_materials.items():
        # Самый частый материал среди соседних ячеек
        material_id = max(
            set(adjacent_materials),
            key=adjacent_materials.count,
        )

        material_name = MATERIAL_ID_TO_NAME.get(material_id)

        if material_name is None:
            rho = None
            lam = None
            mu = None
        else:
            props = material_defs[material_name]
            rho = props["rho"]
            lam = props["lambda"]
            mu = props["mu"]

        x, y, z = points[point_id]

        rows.append(
            {
                "point_id": point_id,
                "x": x,
                "y": y,
                "z": z,
                "material_id_approx": material_id,
                "material_name_approx": material_name,
                "rho": rho,
                "lambda": lam,
                "mu": mu,
                "adjacent_material_ids": sorted(set(adjacent_materials)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_points_csv, index=False)

    print(f"Saved approximate point material table to: {out_points_csv}")
    print(df.head())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtu", type=Path, default=DEFAULT_VTU_PATH)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--out-cells", type=Path, default=DEFAULT_OUT_CELLS_CSV)
    parser.add_argument("--out-points", type=Path, default=DEFAULT_OUT_POINTS_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vtu_path = args.vtu.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    out_cells_csv = args.out_cells.expanduser()
    out_points_csv = args.out_points.expanduser()

    export_cell_materials(vtu_path, xml_path, out_cells_csv)
    export_point_materials_approx(vtu_path, xml_path, out_points_csv)
