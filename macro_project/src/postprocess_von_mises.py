from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import shutil
import xml.etree.ElementTree as ET


# Minimal run example:
#   python3 macro_project/src/postprocess_von_mises.py --yield-strength 2.5e8
#
# The script reads VTU files that already contain a von_mises field, adds
# threshold-based plasticity diagnostics, and writes a new VTU/PVD series.


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    input_dir: Path = BASE_DIR / "thermal_stress_output"
    input_pattern: str = "thermal_stress_*.vtu"
    input_pvd: Path | None = None

    output_dir: Path = BASE_DIR / "plastic_zone_output"
    output_prefix: str = "plastic_zone"
    clear_output: bool = False
    max_frames: int | None = None

    von_mises_field: str = "von_mises"
    yield_strength: float = 250.0e6


@dataclass
class Frame:
    path: Path
    time: float


@dataclass
class FrameStats:
    frame_id: int
    time: float
    input_file: str
    output_file: str
    max_von_mises: float
    max_yield_ratio: float
    max_overstress: float
    plastic_points: int
    total_points: int


def find_input_pvd(config: Config) -> Path | None:
    if config.input_pvd is not None:
        return config.input_pvd if config.input_pvd.exists() else None

    pattern_prefix = config.input_pattern.split("*", 1)[0].rstrip("_-.")
    expected = config.input_dir / f"{pattern_prefix}.pvd"
    if expected.exists():
        return expected

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


def input_frames(config: Config) -> list[Frame]:
    paths = sorted(config.input_dir.glob(config.input_pattern))
    if config.max_frames is not None:
        paths = paths[: config.max_frames]
    if not paths:
        raise FileNotFoundError(
            f"No VTU files found: {config.input_dir / config.input_pattern}"
        )

    pvd_path = find_input_pvd(config)
    pvd_times = read_pvd_times(pvd_path) if pvd_path is not None else {}

    return [
        Frame(path=path, time=pvd_times.get(path.resolve(), float(frame_id)))
        for frame_id, path in enumerate(paths)
    ]


def prepare_output_dir(config: Config) -> None:
    if config.output_dir.exists() and config.clear_output:
        if config.output_dir.is_dir():
            shutil.rmtree(config.output_dir)
        else:
            raise RuntimeError(f"Output path exists and is not a directory: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)


def find_piece(root: ET.Element, path: Path) -> ET.Element:
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path} has no Piece element.")
    return piece


def load_ascii_vtu_tree(path: Path) -> ET.ElementTree:
    payload = path.read_bytes()
    if b"<AppendedData" in payload:
        raise ValueError(
            f"{path} uses VTU AppendedData. This postprocessor expects the "
            "ASCII VTU files produced by thermal_stress_from_vtu.py."
        )
    return ET.ElementTree(ET.fromstring(payload))


def find_field_container(piece: ET.Element, field_name: str) -> tuple[ET.Element, str]:
    point_data = piece.find("./PointData")
    if point_data is not None and point_data.find(f"./DataArray[@Name='{field_name}']") is not None:
        return point_data, "PointData"

    cell_data = piece.find("./CellData")
    if cell_data is not None and cell_data.find(f"./DataArray[@Name='{field_name}']") is not None:
        return cell_data, "CellData"

    available: list[str] = []
    for container in (point_data, cell_data):
        if container is None:
            continue
        for node in container.findall("./DataArray"):
            available.append(node.attrib.get("Name", "<unnamed>"))

    raise ValueError(f"Field '{field_name}' was not found. Available fields: {available}")


def read_ascii_scalar_field(container: ET.Element, field_name: str) -> list[float]:
    node = container.find(f"./DataArray[@Name='{field_name}']")
    if node is None:
        raise ValueError(f"Field '{field_name}' was not found.")

    if node.attrib.get("format", "ascii") != "ascii":
        raise ValueError(
            f"Field '{field_name}' is not ascii. Run this postprocessor on "
            "the VTU files produced by thermal_stress_from_vtu.py."
        )

    components = int(node.attrib.get("NumberOfComponents", "1"))
    if components != 1:
        raise ValueError(f"Field '{field_name}' must be scalar.")

    values = [float(value) for value in (node.text or "").split()]
    if not values:
        raise ValueError(f"Field '{field_name}' is empty.")

    return values


def remove_existing_fields(container: ET.Element, field_names: list[str]) -> None:
    for field_name in field_names:
        for node in list(container.findall(f"./DataArray[@Name='{field_name}']")):
            container.remove(node)


def float_array_text(values: list[float]) -> str:
    return " ".join(f"{value:.9e}" for value in values)


def int_array_text(values: list[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def append_scalar_field(
    container: ET.Element,
    name: str,
    values: list[float] | list[int],
    vtk_type: str,
) -> None:
    node = ET.SubElement(
        container,
        "DataArray",
        {
            "type": vtk_type,
            "Name": name,
            "format": "ascii",
        },
    )
    if vtk_type.startswith("Float"):
        node.text = float_array_text([float(value) for value in values])
    else:
        node.text = int_array_text([int(value) for value in values])


def write_output_pvd(output_dir: Path, output_prefix: str, frames: list[tuple[float, Path]]) -> None:
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


def write_summary(output_dir: Path, stats: list[FrameStats]) -> None:
    summary_path = output_dir / "plastic_zone_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "time",
                "input_file",
                "output_file",
                "max_von_mises",
                "max_yield_ratio",
                "max_overstress",
                "plastic_points",
                "total_points",
                "plastic_fraction",
            ]
        )
        for item in stats:
            writer.writerow(
                [
                    item.frame_id,
                    f"{item.time:.12e}",
                    item.input_file,
                    item.output_file,
                    f"{item.max_von_mises:.12e}",
                    f"{item.max_yield_ratio:.12e}",
                    f"{item.max_overstress:.12e}",
                    item.plastic_points,
                    item.total_points,
                    f"{item.plastic_points / max(item.total_points, 1):.12e}",
                ]
            )


def postprocess_von_mises(config: Config) -> None:
    if config.yield_strength <= 0.0:
        raise ValueError("yield_strength must be positive.")

    frames = input_frames(config)
    prepare_output_dir(config)

    output_frames: list[tuple[float, Path]] = []
    summary: list[FrameStats] = []
    plastic_ever: list[int] | None = None
    max_yield_ratio_so_far: list[float] | None = None

    print(f"input frames: {len(frames)}", flush=True)
    print(f"input dir: {config.input_dir}", flush=True)
    print(f"von Mises field: {config.von_mises_field}", flush=True)
    print(f"yield strength: {config.yield_strength:.6e} Pa", flush=True)
    print(f"output dir: {config.output_dir}", flush=True)

    diagnostic_fields = [
        "yield_ratio",
        "overstress",
        "plastic_zone",
        "max_yield_ratio_so_far",
        "plastic_zone_ever",
    ]

    for frame_id, frame in enumerate(frames):
        tree = load_ascii_vtu_tree(frame.path)
        root = tree.getroot()
        piece = find_piece(root, frame.path)
        container, container_name = find_field_container(piece, config.von_mises_field)
        von_mises = read_ascii_scalar_field(container, config.von_mises_field)

        yield_ratio = [value / config.yield_strength for value in von_mises]
        overstress = [max(value - config.yield_strength, 0.0) for value in von_mises]
        plastic_zone = [1 if ratio >= 1.0 else 0 for ratio in yield_ratio]

        if plastic_ever is None:
            plastic_ever = list(plastic_zone)
            max_yield_ratio_so_far = list(yield_ratio)
        else:
            if len(plastic_ever) != len(plastic_zone) or max_yield_ratio_so_far is None:
                raise RuntimeError("All VTU frames must have the same number of stress values.")
            plastic_ever = [
                max(old_value, new_value)
                for old_value, new_value in zip(plastic_ever, plastic_zone)
            ]
            max_yield_ratio_so_far = [
                max(old_value, new_value)
                for old_value, new_value in zip(max_yield_ratio_so_far, yield_ratio)
            ]

        remove_existing_fields(container, diagnostic_fields)
        append_scalar_field(container, "yield_ratio", yield_ratio, "Float64")
        append_scalar_field(container, "overstress", overstress, "Float64")
        append_scalar_field(container, "plastic_zone", plastic_zone, "Int32")
        append_scalar_field(container, "max_yield_ratio_so_far", max_yield_ratio_so_far, "Float64")
        append_scalar_field(container, "plastic_zone_ever", plastic_ever, "Int32")

        if "Scalars" in container.attrib:
            container.attrib["Scalars"] = "yield_ratio"

        output_path = config.output_dir / f"{config.output_prefix}_{frame_id:05d}.vtu"
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        output_frames.append((frame.time, output_path))

        plastic_points = sum(plastic_zone)
        max_vm = max(von_mises)
        max_ratio = max(yield_ratio)
        max_overstress = max(overstress)
        summary.append(
            FrameStats(
                frame_id=frame_id,
                time=frame.time,
                input_file=frame.path.name,
                output_file=output_path.name,
                max_von_mises=max_vm,
                max_yield_ratio=max_ratio,
                max_overstress=max_overstress,
                plastic_points=plastic_points,
                total_points=len(von_mises),
            )
        )

        print(
            f"frame={frame_id:05d}, time={frame.time:.6e}, source={container_name}, "
            f"max_vm={max_vm:.6e}, max_ratio={max_ratio:.6e}, "
            f"plastic_points={plastic_points}/{len(von_mises)}, output={output_path}",
            flush=True,
        )

    write_output_pvd(config.output_dir, config.output_prefix, output_frames)
    write_summary(config.output_dir, summary)
    print(f"Done. Open: {config.output_dir / f'{config.output_prefix}.pvd'}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Postprocess von Mises stress into possible plasticity zones.",
    )
    parser.add_argument("--input-dir", type=Path, default=Config.input_dir)
    parser.add_argument("--pattern", default=Config.input_pattern)
    parser.add_argument("--input-pvd", type=Path, default=Config.input_pvd)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--output-prefix", default=Config.output_prefix)
    parser.add_argument("--max-frames", type=int, default=Config.max_frames)
    parser.add_argument("--von-mises-field", default=Config.von_mises_field)
    parser.add_argument(
        "--yield-strength",
        type=float,
        default=Config.yield_strength,
        help="Yield strength in Pa. Choose this for your aluminium alloy.",
    )
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()

    return Config(
        input_dir=args.input_dir,
        input_pattern=args.pattern,
        input_pvd=args.input_pvd,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        max_frames=args.max_frames,
        von_mises_field=args.von_mises_field,
        yield_strength=args.yield_strength,
        clear_output=args.clear_output,
    )


def main() -> None:
    postprocess_von_mises(parse_args())


if __name__ == "__main__":
    main()
