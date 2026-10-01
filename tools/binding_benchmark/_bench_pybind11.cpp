#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#define MODTYPE py::module_
#define MODTYPE_ARG py::arg
#define CLASS py::class_
#include "bench_cxx_body.inc"
PYBIND11_MODULE(_bench_pybind11, m) { define_all(m); }
