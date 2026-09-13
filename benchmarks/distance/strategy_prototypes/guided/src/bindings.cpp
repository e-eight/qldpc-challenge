#include "ris/ris.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>

namespace py = pybind11;

ris_guided::Matrix from_numpy(py::array array) {
    if (array.ndim() != 2 || array.shape(0) > std::numeric_limits<int>::max() ||
        array.shape(1) >= std::numeric_limits<int>::max())
        throw std::invalid_argument("expected a two-dimensional binary matrix");
    if (!array.dtype().is(py::dtype::of<int8_t>()) && !array.dtype().is(py::dtype::of<uint8_t>()) &&
        !array.dtype().is(py::dtype::of<bool>()))
        throw std::invalid_argument("matrix dtype must be int8, uint8, or bool");
    ris_guided::Matrix matrix(int(array.shape(0)), int(array.shape(1)));
    const auto* base = static_cast<const char*>(array.data());
    for (int r = 0; r < matrix.rows(); ++r) for (int c = 0; c < matrix.columns(); ++c) {
        auto value = *reinterpret_cast<const uint8_t*>(base + r * array.strides(0) + c * array.strides(1));
        if (value > 1) throw std::invalid_argument("matrix entries must be binary");
        if (value) matrix.set(r, c);
    }
    return matrix;
}

py::array_t<uint8_t> to_numpy(const ris_guided::Matrix& matrix) {
    py::array_t<uint8_t> result({matrix.rows(), matrix.columns()});
    auto view = result.mutable_unchecked<2>();
    for (int r = 0; r < matrix.rows(); ++r) for (int c = 0; c < matrix.columns(); ++c)
        view(r, c) = matrix.bit(r, c);
    return result;
}

PYBIND11_MODULE(ris_guided_native, module) {
    module.doc() = "Four-parent guided-basis prototype; witnessed distances are upper bounds.";
    module.attr("compiler") = __VERSION__;
    module.attr("host_tuned") = bool(RIS_HOST_TUNED);
    py::class_<ris_guided::Prepared, std::shared_ptr<ris_guided::Prepared>>(module, "Prepared")
        .def(py::init([](py::array own, py::array opposite) {
            auto a = from_numpy(own), b = from_numpy(opposite);
            py::gil_scoped_release release;
            return std::make_shared<ris_guided::Prepared>(a, b);
        }), py::arg("own"), py::arg("opposite"))
        .def_property_readonly("applicable", &ris_guided::Prepared::applicable)
        .def_property_readonly("kernel", [](const ris_guided::Prepared& p) { return to_numpy(p.kernel); })
        .def_property_readonly("logicals", [](const ris_guided::Prepared& p) { return to_numpy(p.logicals); })
        .def_property_readonly("basis_bytes", [](const ris_guided::Prepared& p) { return p.kernel.bytes() + p.logicals.bytes(); });
    py::class_<ris_guided::Improvement>(module, "Improvement")
        .def_readonly("worker", &ris_guided::Improvement::worker)
        .def_readonly("weight", &ris_guided::Improvement::weight)
        .def_readonly("trial", &ris_guided::Improvement::trial)
        .def_readonly("support", &ris_guided::Improvement::support);
    py::class_<ris_guided::Batch>(module, "Batch")
        .def_readonly("trials", &ris_guided::Batch::trials)
        .def_readonly("reductions", &ris_guided::Batch::reductions)
        .def_readonly("proposals", &ris_guided::Batch::proposals)
        .def_readonly("exchanges", &ris_guided::Batch::exchanges)
        .def_readonly("accepted_children", &ris_guided::Batch::accepted_children)
        .def_readonly("immigrants", &ris_guided::Batch::immigrants)
        .def_readonly("best_weight", &ris_guided::Batch::best_weight)
        .def_readonly("improvements", &ris_guided::Batch::improvements);
    py::class_<ris_guided::Session>(module, "Session")
        .def(py::init<std::shared_ptr<const ris_guided::Prepared>, int, uint64_t, int, bool, int, uint64_t, int>(),
             py::arg("prepared"), py::arg("threads") = 1, py::arg("seed") = 0,
             py::arg("pair_depth") = 8, py::arg("masked") = false, py::arg("block_size") = 1,
             py::arg("restart_interval") = 0, py::arg("exchange_proposals") = 8,
             py::call_guard<py::gil_scoped_release>())
        .def("advance", &ris_guided::Session::advance, py::arg("trials"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("workspace_bytes", &ris_guided::Session::workspace_bytes);
}
