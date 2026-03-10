#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "node.hpp"

#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkTetra.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "gmsh.h"

struct Cell {
  int gmsh_type;
  std::vector<size_t> node_ids;
};

class Mesh {
protected:
  std::vector<Node> nodes_;
  std::vector<Cell> cells_;
  double c_, v0_, sigma_ {};
  Node hit_center_ {};
  double sim_step_ = 1e-6;
  double sim_duration_ = 1e-3;

public:
  void SetHit(double c, double v0, double sigma, Node center) {
    c_ = c;
    v0_ = v0;
    sigma_ = sigma;
    hit_center_ = center;
  }

  void SetSim(double sim_step, double sim_duration) {
    sim_step_ = sim_step;
    sim_duration_ = sim_duration;
  }


  void LoadFromMsh(const std::string &filename) {
    gmsh::open(filename);

    std::vector<std::size_t> nodeTags;
    std::vector<double> coords, parametricCoords;

    gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

    std::unordered_map<std::size_t, size_t> tag_to_index;
    nodes_.resize(nodeTags.size());

    for (size_t i = 0; i < nodeTags.size(); ++i) {
      nodes_[i].x = coords[3 * i + 0];
      nodes_[i].y = coords[3 * i + 1];
      nodes_[i].z = coords[3 * i + 2];
      tag_to_index[nodeTags[i]] = i;
    }

    std::vector<int> elementTypes;
    std::vector<std::vector<std::size_t>> elementTags, elementNodeTags;
    gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags);

    cells_.clear();

    for (size_t b = 0; b < elementTypes.size(); ++b) {
      int type = elementTypes[b];

      std::string name;
      int dim, order, numNodes, numPrimaryNodes;
      std::vector<double> localCoords;
      gmsh::model::mesh::getElementProperties(type, name, dim, order, numNodes,
                                              localCoords, numPrimaryNodes);

      if (dim != 3) {
        continue;
      }

      const auto &conn = elementNodeTags[b];
      size_t numElems = conn.size() / numNodes;

      for (size_t e = 0; e < numElems; ++e) {
        Cell cell;
        cell.gmsh_type = type;
        cell.node_ids.resize(numNodes);

        for (int j = 0; j < numNodes; ++j) {
          std::size_t gmshTag = conn[e * numNodes + j];
          cell.node_ids[j] = tag_to_index.at(gmshTag);
        }

        cells_.push_back(std::move(cell));
      }
    }
  }

  void PointHit(const Node& center, Node& node, double t, double c, double v0, double sigma) {
      double dx = node.x - center.x;
      double dy = node.y - center.y;
      double dz = node.z - center.z;
      double r = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

      if(sigma <= 0.0) {
        node.vx = 0.0;
        node.vy = 0.0;
        node.vz = 0.0;
        return;
      }

      const double wave_offset = r - c * t;
      const double exponent_abs = (wave_offset * wave_offset) / (sigma * sigma);
      const double exp_cutoff = -std::log(0.001);

      if(exponent_abs > exp_cutoff) {
        node.vx = 0.0;
        node.vy = 0.0;
        node.vz = 0.0;
        return;
      }

      double nx = dx / r;
      double ny = dy / r;
      double nz = dz / r;

      double v = v0 * std::exp(-exponent_abs);

      node.vx = nx *  v;
      node.vy = ny *  v;
      node.vz = nz *  v;
  }

  void doTimeStep(double time, double tau) {
    for (auto &node : nodes_) {
      PointHit(hit_center_, node, time,  c_,  v0_,  sigma_);
      node.move(tau);
    }
  }

  void Snapshot(unsigned int snap_number) {
    vtkSmartPointer<vtkUnstructuredGrid> grid =
        vtkSmartPointer<vtkUnstructuredGrid>::New();

    vtkSmartPointer<vtkPoints> dumpPoints = vtkSmartPointer<vtkPoints>::New();

    auto smth = vtkSmartPointer<vtkDoubleArray>::New();
    smth->SetName("smth");

    auto vel = vtkSmartPointer<vtkDoubleArray>::New();
    vel->SetName("velocity");
    vel->SetNumberOfComponents(3);

    for (unsigned int i = 0; i < nodes_.size(); i++) {
      dumpPoints->InsertNextPoint(nodes_[i].x, nodes_[i].y, nodes_[i].z);

      double v[3] = {nodes_[i].vx, nodes_[i].vy, nodes_[i].vz};
      vel->InsertNextTuple(v);

      smth->InsertNextValue(nodes_[i].smth);
    }

    grid->SetPoints(dumpPoints);

    for (unsigned int i = 0; i < cells_.size(); i++) {
      vtkSmartPointer<vtkTetra> tetra = vtkSmartPointer<vtkTetra>::New();

      tetra->GetPointIds()->SetId(0, cells_[i].node_ids[0]);
      tetra->GetPointIds()->SetId(1, cells_[i].node_ids[1]);
      tetra->GetPointIds()->SetId(2, cells_[i].node_ids[2]);
      tetra->GetPointIds()->SetId(3, cells_[i].node_ids[3]);

      grid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
    }

    grid->GetPointData()->AddArray(vel);
    grid->GetPointData()->AddArray(smth);

    std::string fileName = "../sim/sword-step-" + std::to_string(snap_number) + ".vtu";
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
        vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

    writer->SetFileName(fileName.c_str());
    writer->SetInputData(grid);
    writer->Write();
  }

  void RunSimulation() {
    for (double time = 0.0; time < sim_duration_; time += sim_step_) {
      doTimeStep(time, sim_step_);
      std::cout << "Step: " << static_cast<int>(time / sim_step_) << " / "
                << static_cast<int>(sim_duration_ / sim_step_) << " ("
                << (time / sim_duration_ * 100.0) << "%)" << std::endl;
      Snapshot(static_cast<unsigned int>(time / sim_step_));
    }
  }
};
