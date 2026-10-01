#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
#define MODTYPE nb::module_
#define MODTYPE_ARG nb::arg
#define CLASS nb::class_
#include "bench_cxx_body.inc"
NB_MODULE(_bench_nanobind, m) { define_all(m); }
