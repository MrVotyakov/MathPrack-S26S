#include "export/vtk_writer.hpp"
/// ОСТОРОЖНО ВАЙБКОД!
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

// Имя файла кадра: "<prefix>_NNNNN.vtu"
std::string frame_filename(const std::string& prefix, int idx) {
  std::ostringstream ss;
  ss << prefix << "_" << std::setw(5) << std::setfill('0') << idx << ".vtu";
  return ss.str();
}

// basename для .pvd (он хранит относительные пути к кадрам)
std::string basename(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// Декомпозиция одного кубика (8 вершин) на 5 тетраэдров.
// Локальная нумерация вершин:
//   v0 = (i,   j,   k  )
//   v1 = (i+1, j,   k  )
//   v2 = (i+1, j+1, k  )
//   v3 = (i,   j+1, k  )
//   v4 = (i,   j,   k+1)
//   v5 = (i+1, j,   k+1)
//   v6 = (i+1, j+1, k+1)
//   v7 = (i,   j+1, k+1)
// 5 тетраэдров (классическая декомпозиция, без вспомогательных точек):
//   (v0, v1, v3, v4)
//   (v1, v2, v3, v6)
//   (v1, v3, v4, v6)
//   (v1, v4, v5, v6)
//   (v3, v4, v6, v7)
constexpr int TETS_PER_CUBE = 5;
constexpr int TET_LOCAL[5][4] = {
  {0, 1, 3, 4},
  {1, 2, 3, 6},
  {1, 3, 4, 6},
  {1, 4, 5, 6},
  {3, 4, 6, 7}
};

}  // namespace

VtkWriter::VtkWriter(const std::string& prefix, const Grid& /*grid*/)
  : prefix_(prefix) {}

VtkWriter::~VtkWriter() {
  if (!finalized_) finalize();
}

void VtkWriter::write_frame(const Grid& grid, double time) {
  const int nx = grid.nx();
  const int ny = grid.ny();
  const int nz = grid.nz();
  const auto& mat = grid.material();

  const size_t N = static_cast<size_t>(nx) * ny * nz;

  // ---------- 1. Координаты узлов ----------
  // Узлы лежат в центрах ячеек (x(i)=(i+0.5)*dx и т.д.).
  // Линейный индекс узла: idx = i + nx*(j + ny*k).
  std::vector<float> points(3 * N);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        size_t p = static_cast<size_t>(i) + nx * (j + ny * k);
        points[3 * p + 0] = static_cast<float>(grid.x(i));
        points[3 * p + 1] = static_cast<float>(grid.y(j));
        points[3 * p + 2] = static_cast<float>(grid.z(k));
      }

  // ---------- 2. Тетраэдризация ----------
  const int n_cubes = (nx - 1) * (ny - 1) * (nz - 1);
  const int n_tets  = TETS_PER_CUBE * n_cubes;
  std::vector<int32_t> connectivity;
  std::vector<int32_t> offsets;
  std::vector<uint8_t> cell_types;
  connectivity.reserve(4 * n_tets);
  offsets.reserve(n_tets);
  cell_types.reserve(n_tets);

  auto pidx = [nx, ny](int i, int j, int k) {
    return i + nx * (j + ny * k);
  };

  for (int k = 0; k < nz - 1; ++k) {
    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        int v[8] = {
          pidx(i,     j,     k    ),
          pidx(i + 1, j,     k    ),
          pidx(i + 1, j + 1, k    ),
          pidx(i,     j + 1, k    ),
          pidx(i,     j,     k + 1),
          pidx(i + 1, j,     k + 1),
          pidx(i + 1, j + 1, k + 1),
          pidx(i,     j + 1, k + 1),
        };
        for (int t = 0; t < TETS_PER_CUBE; ++t) {
          for (int a = 0; a < 4; ++a)
            connectivity.push_back(v[TET_LOCAL[t][a]]);
          offsets.push_back(static_cast<int32_t>(connectivity.size()));
          cell_types.push_back(10);  // VTK_TETRA
        }
      }
    }
  }

  // ---------- 3. Поля на узлах ----------
  std::vector<float>   T_field(N);
  std::vector<float>   H_field(N);
  std::vector<float>   fl_field(N);
  std::vector<int32_t> phase_field(N);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        size_t p = static_cast<size_t>(i) + nx * (j + ny * k);
        T_field[p]  = static_cast<float>(grid.T(i, j, k));
        H_field[p]  = static_cast<float>(grid.H(i, j, k));
        fl_field[p] = static_cast<float>(grid.liquid_fraction(i, j, k));
        switch (grid.phase(i, j, k)) {
          case Phase::Solid:  phase_field[p] = 0; break;
          case Phase::Mushy:  phase_field[p] = 1; break;
          case Phase::Liquid: phase_field[p] = 2; break;
        }
      }
    }
  }

  // ---------- 4. FieldData (материальные константы) ----------
  // Каждая константа — массив из одного double. Имена соответствуют SimMaterial.
  struct FieldEntry { const char* name; double value; };
  const FieldEntry field_entries[] = {
    {"rho",            mat.rho},
    {"T_melt",         mat.T_melt},
    {"L_fusion",       mat.L},
    {"c_solid",        mat.c_solid},
    {"c_liquid",       mat.c_liquid},
    {"k_solid",        mat.k_solid},
    {"k_liquid",       mat.k_liquid},
    {"T_ref",          mat.T_ref},
    {"R",              mat.R},
    {"alpha_T",        mat.alpha_T},
    {"dV_over_V_melt", mat.dV_over_V_melt},
    {"E_young",        mat.E_young},
    {"nu",             mat.nu},
  };
  const int n_fields = sizeof(field_entries) / sizeof(field_entries[0]);

  // ---------- 5. Пишем XML ----------
  std::string filename = frame_filename(prefix_, frame_index_);
  std::ofstream f(filename, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "VtkWriter: cannot open %s\n", filename.c_str());
    return;
  }

  // Размеры блоков (без 8-байтного заголовка размера)
  const uint64_t bytes_points       = points.size()       * sizeof(float);
  const uint64_t bytes_connectivity = connectivity.size() * sizeof(int32_t);
  const uint64_t bytes_offsets      = offsets.size()      * sizeof(int32_t);
  const uint64_t bytes_types        = cell_types.size()   * sizeof(uint8_t);
  const uint64_t bytes_T            = T_field.size()      * sizeof(float);
  const uint64_t bytes_H            = H_field.size()      * sizeof(float);
  const uint64_t bytes_fl           = fl_field.size()     * sizeof(float);
  const uint64_t bytes_phase        = phase_field.size()  * sizeof(int32_t);
  const uint64_t bytes_field_one    = sizeof(double);  // на одну материальную константу

  // Накопительные смещения в AppendedData (каждый блок начинается с UInt64 size)
  uint64_t off = 0;
  uint64_t off_points       = off; off += sizeof(uint64_t) + bytes_points;
  uint64_t off_connectivity = off; off += sizeof(uint64_t) + bytes_connectivity;
  uint64_t off_offsets      = off; off += sizeof(uint64_t) + bytes_offsets;
  uint64_t off_types        = off; off += sizeof(uint64_t) + bytes_types;
  uint64_t off_T            = off; off += sizeof(uint64_t) + bytes_T;
  uint64_t off_H            = off; off += sizeof(uint64_t) + bytes_H;
  uint64_t off_fl           = off; off += sizeof(uint64_t) + bytes_fl;
  uint64_t off_phase        = off; off += sizeof(uint64_t) + bytes_phase;
  std::vector<uint64_t> off_fields(n_fields);
  for (int e = 0; e < n_fields; ++e) {
    off_fields[e] = off;
    off += sizeof(uint64_t) + bytes_field_one;
  }

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
       "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
  f << "  <UnstructuredGrid>\n";

  // FieldData — материальные константы
  f << "    <FieldData>\n";
  for (int e = 0; e < n_fields; ++e) {
    f << "      <DataArray type=\"Float64\" Name=\"" << field_entries[e].name
      << "\" NumberOfTuples=\"1\" format=\"appended\" offset=\""
      << off_fields[e] << "\"/>\n";
  }
  f << "    </FieldData>\n";

  f << "    <Piece NumberOfPoints=\"" << N
    << "\" NumberOfCells=\"" << n_tets << "\">\n";

  f << "      <Points>\n";
  f << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
       "format=\"appended\" offset=\"" << off_points << "\"/>\n";
  f << "      </Points>\n";

  f << "      <Cells>\n";
  f << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
       "format=\"appended\" offset=\"" << off_connectivity << "\"/>\n";
  f << "        <DataArray type=\"Int32\" Name=\"offsets\" "
       "format=\"appended\" offset=\"" << off_offsets << "\"/>\n";
  f << "        <DataArray type=\"UInt8\" Name=\"types\" "
       "format=\"appended\" offset=\"" << off_types << "\"/>\n";
  f << "      </Cells>\n";

  f << "      <PointData Scalars=\"T\">\n";
  f << "        <DataArray type=\"Float32\" Name=\"T\" "
       "format=\"appended\" offset=\"" << off_T << "\"/>\n";
  f << "        <DataArray type=\"Float32\" Name=\"H\" "
       "format=\"appended\" offset=\"" << off_H << "\"/>\n";
  f << "        <DataArray type=\"Float32\" Name=\"liquid_fraction\" "
       "format=\"appended\" offset=\"" << off_fl << "\"/>\n";
  f << "        <DataArray type=\"Int32\" Name=\"phase\" "
       "format=\"appended\" offset=\"" << off_phase << "\"/>\n";
  f << "      </PointData>\n";

  f << "    </Piece>\n";
  f << "  </UnstructuredGrid>\n";
  f << "  <AppendedData encoding=\"raw\">\n";
  f << "_";  // обязательный маркер начала raw-блока

  auto write_block = [&](const void* data, uint64_t bytes) {
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(uint64_t));
    f.write(reinterpret_cast<const char*>(data), bytes);
  };

  // Порядок ДОЛЖЕН совпадать с порядком offset'ов выше.
  write_block(points.data(),       bytes_points);
  write_block(connectivity.data(), bytes_connectivity);
  write_block(offsets.data(),      bytes_offsets);
  write_block(cell_types.data(),   bytes_types);
  write_block(T_field.data(),      bytes_T);
  write_block(H_field.data(),      bytes_H);
  write_block(fl_field.data(),     bytes_fl);
  write_block(phase_field.data(),  bytes_phase);
  for (int e = 0; e < n_fields; ++e) {
    write_block(&field_entries[e].value, bytes_field_one);
  }

  f << "\n  </AppendedData>\n";
  f << "</VTKFile>\n";
  f.close();

  frames_.emplace_back(time, basename(filename));
  ++frame_index_;
}

void VtkWriter::finalize() {
  if (finalized_) return;
  finalized_ = true;

  std::string pvd_name = prefix_ + ".pvd";
  std::ofstream f(pvd_name);
  if (!f) {
    std::fprintf(stderr, "VtkWriter: cannot open %s\n", pvd_name.c_str());
    return;
  }
  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"Collection\" version=\"1.0\" "
       "byte_order=\"LittleEndian\">\n";
  f << "  <Collection>\n";
  for (const auto& [t, name] : frames_) {
    f << "    <DataSet timestep=\"" << t
      << "\" group=\"\" part=\"0\" file=\"" << name << "\"/>\n";
  }
  f << "  </Collection>\n";
  f << "</VTKFile>\n";
}