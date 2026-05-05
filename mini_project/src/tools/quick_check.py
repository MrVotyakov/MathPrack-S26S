from pathlib import Path
import numpy as np
import meshio

path = Path("body/mesh.vtu")

print("exists:", path.exists(), path, flush=True)

msh = meshio.read(path)

print("points:", msh.points.shape, flush=True)
print("cell blocks:", [(c.type, c.data.shape) for c in msh.cells], flush=True)
print("cell data keys:", msh.cell_data.keys(), flush=True)

for i, block in enumerate(msh.cells):
    if block.type == "tetra":
        cells = block.data
        mat = msh.cell_data["Material"][i].astype(int)

        print("tetra cells:", cells.shape, flush=True)
        print("material ids:", np.unique(mat), flush=True)

        ids, counts = np.unique(mat, return_counts=True)

        print("\nMaterial | cells | unique points")
        print("---------|-------|--------------")

        for mid, count in zip(ids, counts):
            pts = np.unique(cells[mat == mid].reshape(-1))
            print(f"{int(mid):8d} | {int(count):5d} | {len(pts):13d}")

        break
