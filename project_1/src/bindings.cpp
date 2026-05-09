#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>

#include "vector_add.h"

namespace nb = nanobind;

using FArr1D = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static nb::ndarray<nb::numpy, float, nb::ndim<1>> vector_add_py(FArr1D a, FArr1D b) {
    if (a.shape(0) != b.shape(0)) {
        throw std::runtime_error("a and b must have the same length");
    }
    int n = static_cast<int>(a.shape(0));

    float* out = new float[static_cast<size_t>(n)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });

    p1::vector_add(a.data(), b.data(), out, n);

    size_t shape[1] = {static_cast<size_t>(n)};
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, 1, shape, owner);
}

NB_MODULE(_project_1_py, m) {
    m.doc() = "project_1: CUDA elementwise vector addition";
    m.def("vector_add", &vector_add_py,
          nb::arg("a").noconvert(), nb::arg("b").noconvert(),
          "Element-wise add two 1-D float32 arrays on the GPU. Returns a new array.");
}
