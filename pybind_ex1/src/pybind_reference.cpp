#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

struct Point {
    double x;
    double y;
};

// MODIFIED: Function now accepts parameters for radius and velocity
std::vector<Point> run_simulation(int num_steps, double radius, double angular_velocity) {
    std::vector<Point> trajectory;
    for (int i = 0; i < num_steps; ++i) {
        double time = static_cast<double>(i);
        double angle = angular_velocity * time;
        Point p;
        p.x = radius * std::cos(angle);
        p.y = radius * std::sin(angle);
        trajectory.push_back(p);
    }
    return trajectory;
}

// m: This is a variable of type py::module_ that represents the module object. You use this variable to add your C++ functions and classes to the module.

PYBIND11_MODULE(cpp_solver_module, m) {
    m.doc() = "A C++ particle simulation module for Python";

    py::class_<Point>(m, "Point")
        .def(py::init<>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);

    // MODIFIED: Expose the new function signature with named arguments and defaults
    m.def("run_simulation", &run_simulation, "Runs the particle simulation",
          py::arg("num_steps"),
          py::arg("radius") = 10.0,
          py::arg("angular_velocity") = 0.1);
}




