#pragma once

#include <functional>

enum class BoundaryType {
    Fixed,
    Adiabatic
};

struct Laser {
    std::function<double(double x, double y, double t)> q_abs = nullptr;
    bool active() const {
        return static_cast<bool>(q_abs);
    };
};
