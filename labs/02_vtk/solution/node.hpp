#pragma once

class Mesh;

// Класс расчётной точки
class Node {
  // Класс сетки будет friend-ом точки
  friend class Mesh;

protected:
  // Координаты
  double x;
  double y;
  double z;
  // Некая величина, в попугаях
  double smth;
  // Скорость
  double vx;
  double vy;
  double vz;

public:
  // Конструктор по умолчанию
  Node() : x(0.0), y(0.0), z(0.0), smth(0.0), vx(0.0), vy(0.0), vz(0.0) {}

  Node(double x, double y, double z)
      : x(x), y(y), z(z), smth(0), vx(0), vy(0), vz(0) {}
      
  // Конструктор с указанием всех параметров
  Node(double x, double y, double z, double smth, double vx, double vy,
       double vz)
      : x(x), y(y), z(z), smth(smth), vx(vx), vy(vy), vz(vz) {}

  // Движемся время tau из текущего положения с текущей скоростью
  void move(double tau) {
    x += vx * tau;
    y += vy * tau;
    z += vz * tau;
  }
};
