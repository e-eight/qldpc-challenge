#include "ris/ris.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>

namespace py = pybind11;

ris::Matrix from_numpy(py::array array) {
    if (array.ndim() != 2 || array.shape(0) > std::numeric_limits<int>::max() ||
        array.shape(1) >= std::numeric_limits<int>::max())
        throw std::invalid_argument("expected a two-dimensional binary matrix");
    if (!array.dtype().is(py::dtype::of<int8_t>()) && !array.dtype().is(py::dtype::of<uint8_t>()) &&
        !array.dtype().is(py::dtype::of<bool>()))
        throw std::invalid_argument("matrix dtype must be int8, uint8, or bool");
    ris::Matrix matrix(int(array.shape(0)), int(array.shape(1)));
    const auto* base = static_cast<const char*>(array.data());
    for (int r = 0; r < matrix.rows(); ++r) for (int c = 0; c < matrix.columns(); ++c) {
        auto value = *reinterpret_cast<const uint8_t*>(base + r * array.strides(0) + c * array.strides(1));
        if (value > 1) throw std::invalid_argument("matrix entries must be binary");
        if (value) matrix.set(r, c);
    }
    return matrix;
}

py::array_t<uint8_t> to_numpy(const ris::Matrix& matrix) {
    py::array_t<uint8_t> result({matrix.rows(), matrix.columns()});
    auto view = result.mutable_unchecked<2>();
    for (int r = 0; r < matrix.rows(); ++r) for (int c = 0; c < matrix.columns(); ++c)
        view(r, c) = matrix.bit(r, c);
    return result;
}

PYBIND11_MODULE(ris_native, module) {
    module.doc() = "Packed CPU random-information-set search; returned distances are upper bounds.";
    module.attr("compiler") = __VERSION__;
    module.attr("host_tuned") = bool(RIS_HOST_TUNED);
    py::class_<ris::Prepared, std::shared_ptr<ris::Prepared>>(module, "Prepared")
        .def(py::init([](py::array own, py::array opposite) {
            auto a = from_numpy(own), b = from_numpy(opposite);
            py::gil_scoped_release release;
            return std::make_shared<ris::Prepared>(a, b);
        }), py::arg("own"), py::arg("opposite"))
        .def_property_readonly("applicable", &ris::Prepared::applicable)
        .def_property_readonly("kernel", [](const ris::Prepared& p) { return to_numpy(p.kernel); })
        .def_property_readonly("logicals", [](const ris::Prepared& p) { return to_numpy(p.logicals); })
        .def_property_readonly("basis_bytes", [](const ris::Prepared& p) { return p.kernel.bytes() + p.logicals.bytes(); });
    py::class_<ris::Improvement>(module, "Improvement")
        .def_readonly("worker", &ris::Improvement::worker)
        .def_readonly("weight", &ris::Improvement::weight)
        .def_readonly("trial", &ris::Improvement::trial)
        .def_readonly("support", &ris::Improvement::support);
    py::class_<ris::Batch>(module, "Batch")
        .def_readonly("trials", &ris::Batch::trials)
        .def_readonly("reductions", &ris::Batch::reductions)
        .def_readonly("proposals", &ris::Batch::proposals)
        .def_readonly("exchanges", &ris::Batch::exchanges)
        .def_readonly("best_weight", &ris::Batch::best_weight)
        .def_readonly("improvements", &ris::Batch::improvements);
    py::class_<ris::Session>(module, "Session")
        .def(py::init<std::shared_ptr<const ris::Prepared>, int, uint64_t, int, bool, int, uint64_t, int>(),
             py::arg("prepared"), py::arg("threads") = 1, py::arg("seed") = 0,
             py::arg("pair_depth") = 8, py::arg("masked") = false, py::arg("block_size") = 1,
             py::arg("restart_interval") = 0, py::arg("exchange_proposals") = 8,
             py::call_guard<py::gil_scoped_release>())
        .def("advance", &ris::Session::advance, py::arg("trials"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("workspace_bytes", &ris::Session::workspace_bytes);
}
