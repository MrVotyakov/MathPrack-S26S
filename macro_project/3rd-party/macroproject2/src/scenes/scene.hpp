#pragma once

#include <string>
#include "core/grid.hpp"
#include "core/boundary.hpp"

class Scene {
public:
    virtual ~Scene() = default;
    virtual void init(Grid& grid) = 0;

     virtual BoundaryType lateral_bc() const { return BoundaryType::Adiabatic; }
     virtual Laser laser_bc() const { return {}; }
 
    virtual std::string name() const = 0;
};