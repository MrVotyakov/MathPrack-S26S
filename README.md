# Моделирование акустического ультразвукового импульса

Проект моделирует распространение акустического давления в 3D-модели головы на тетраэдральной сетке DOLFINx.
Основной расчетный файл:

```text
mini_project/src/acoustic_impulse.py
```

Сетка и материалы:

```text
mini_project/3d-models/head_with_materials.vtu
mini_project/src/tools/preprocess.py
mini_project/src/tools/convert_to_vtu_with_materials.py
```

`preprocess.py` читает VTU-сетку, восстанавливает `Material ID` и создает DG0-поля `rho`, `lambda`, `mu`, `sound_speed`, `material_id`: одно значение на тетраэдр. В акустической задаче используются `rho(x)` и `c(x) = sound_speed(x)`. Геометрия задана в миллиметрах, время - в секундах, скорости звука - в `mm/s`.

Модель взята из репозитория [gcm-3d](https://github.com/avasyukov/gcm-3d/tree/markered-corpse/models/ani3d).
Расчетный код написан с опорой на готовое решение из [fenics-fus](https://github.com/adeebkor/fenicsx-fus).

## Физическая модель

Искомая величина - акустическое давление $p(x, t)$.

Уравнение:

$$
\frac{1}{\rho(x)c^2(x)}
\frac{\partial^2 p}{\partial t^2}

\nabla \cdot
\left(
  \frac{1}{\rho(x)} \nabla p
\right)
= s(x,t)
$$

Источник разделен на пространственную и временную части:

$$
s(x,t)=g(x)q(t)
$$

Пространственная часть - гауссов профиль:

$$
g(x)=A\exp\left(
  -\frac{\|x-x_s\|^2}{2\sigma^2}
\right)
$$

В текущем коде центр источника $x_s$ ставится около нижней границы расчетной области, а $\sigma$ задается как доля размера bounding box.

Временная часть - оконный ультразвуковой burst:

$$
q(t)=
\begin{cases}
\sin(2\pi f\tau)\sin^2\left(\dfrac{\pi\tau}{T}\right),
& 0 \le \tau \le T,\\
0, & \text{иначе}.
\end{cases}
$$

где:

$$
\tau=t-t_{\mathrm{start}},
\qquad
t_{\mathrm{start}}=t_0-\frac{T}{2},
\qquad
T=\frac{N}{f}
$$

Вне интервала $[t_{\mathrm{start}},\,t_{\mathrm{start}}+T]$ источник строго равен нулю. Сейчас в коде:

$$
f=1\,\mathrm{MHz},
\qquad
N=5,
\qquad
t_0=\frac{5}{f},
\qquad
T=\frac{5}{f}
$$

Такой источник плавно начинается с нуля, делает 5 колебаний и плавно заканчивается в ноль.

## Численный метод
Используется пространство конечных элементов `Lagrange` степени 1 для скалярного поля давления.

После FEM-дискретизации:

$$
\mathbf{M}_L\ddot{\mathbf{p}}+\mathbf{K}\mathbf{p}
=q(t)\mathbf{F}
$$

где:

$$
\begin{aligned}
(\mathbf{M}_L)_i
&=\int_{\Omega}\frac{1}{\rho(x)c^2(x)}\phi_i(x)\,dx,\\
K_{ij}
&=\int_{\Omega}\frac{1}{\rho(x)}
\nabla\phi_j(x)\cdot\nabla\phi_i(x)\,dx,\\
F_i
&=\int_{\Omega}g(x)\phi_i(x)\,dx.
\end{aligned}
$$

Матрица масс используется в lumped-виде, то есть как диагональный вектор. Временная схема - явная центральная разность:

$$
\mathbf{p}^{n+1}
=2\mathbf{p}^{n}-\mathbf{p}^{n-1}
+\Delta t^2\mathbf{M}_L^{-1}
\left(q(t_n)\mathbf{F}-\mathbf{K}\mathbf{p}^{n}\right)
$$

Начальные условия:

$$
p(x,0)=0,
\qquad
\frac{\partial p}{\partial t}(x,0)=0
$$

Шаг времени оценивается автоматически:

$$
\Delta t=\mathrm{safety}\,\frac{h_{\min}}{c_{\max}}
$$

где `h_min` - минимальный размер ячейки, `c_max` - максимальная скорость звука по материалам, `safety = 0.25`.

## Структура

```text
README.md
mini_project/
  requirements.txt
  3d-models/
    mesh.vtu
    head_with_materials.vtu
    *.out
  src/
    acoustic_impulse.py
    tools/
      preprocess.py
      convert_to_vtu_with_materials.py
```

Результаты расчета пишутся в:

```text
mini_project/src/acoustic_vtu_frames_other_source/
```

При новом запуске эта папка удаляется и создается заново.

## Результаты

В результате моделирования получаем `.vtu` файлы. Открыть их можно в `Paraview`, примеры видео моделирования можно найти на [Яндекс Диске](https://disk.360.yandex.ru/d/l9_8UlurSN1kMA)

## Запуск

Команды выполняются из корня репозитория. В Docker-контейнере с DOLFINx и смонтированным проектом:

```bash
/dolfinx-env/bin/python -m mini_project.src.acoustic_impulse
```

Если окружение уже активировано:

```bash
python -m mini_project.src.acoustic_impulse
```

## Подготовка VTU с материалами

Если нужно заново собрать `head_with_materials.vtu` из `mesh.vtu`:

```bash
python -m mini_project.src.tools.convert_to_vtu_with_materials
```
