// Restricted-coordinate and orbit-constant kernels, scored by the existing RIS core.
#include "../../../../verify/gf2_fast.cpp"

class Space {
    GF2Matrix basis_, duals_;
    int logical_rank_ = 0;
public:
    Space(py::array_t<int8_t> checks, py::array_t<int8_t> duals,
          const std::vector<std::vector<int>>& groups) : duals_(GF2Matrix::from_numpy(duals)) {
        auto h = GF2Matrix::from_numpy(checks);
        if (h.cols_ != duals_.cols_ || groups.size() > size_t(h.cols_))
            throw std::invalid_argument("Incompatible dimensions");
        std::vector<bool> seen(h.cols_, false);
        for (const auto& group : groups) {
            if (group.empty()) throw std::invalid_argument("Empty group");
            for (int q : group) {
                if (q < 0 || q >= h.cols_ || seen[q]) throw std::invalid_argument("Groups must be disjoint valid coordinates");
                seen[q] = true;
            }
        }
        py::gil_scoped_release release;
        GF2Matrix quotient(h.rows_, groups.size());
        for (int r=0; r<h.rows_; ++r)
            for (int j=0; j<int(groups.size()); ++j) {
                bool parity=false;
                for (int q : groups[j]) parity ^= h.get(r,q);
                quotient.set(r,j,parity);
            }
        auto kernel = kernel_basis(quotient);
        basis_ = GF2Matrix(kernel.rows_, h.cols_);
        for (int r=0; r<kernel.rows_; ++r)
            for (int j=0; j<kernel.cols_; ++j)
                if (kernel.get(r,j)) for (int q : groups[j]) basis_.set(r,q,true);
        GF2Matrix tags(basis_.rows_, duals_.rows_);
        for (int r=0; r<basis_.rows_; ++r)
            for (int d=0; d<duals_.rows_; ++d) {
                int parity=0;
                for (int w=0; w<basis_.wpr_; ++w) parity ^= __builtin_parityll(basis_.row_ptr(r)[w]&duals_.row_ptr(d)[w]);
                tags.set(r,d,parity);
            }
        logical_rank_ = gf2_rref(std::move(tags)).pivots.size();
    }
    int dimension() const { return basis_.rows_; }
    int logical_rank() const { return logical_rank_; }
    size_t workspace_bytes() const { return sizeof(uint64_t)*(basis_.data_.capacity()+duals_.data_.capacity()); }
    py::array_t<int8_t> basis() const { return basis_.to_numpy(); }
    py::tuple batch(int trials, uint64_t seed, int pairs=8) const {
        if (trials < 1 || pairs < 0) throw std::invalid_argument("Invalid work request");
        std::vector<uint64_t> witness;
        int weight=basis_.cols_+1;
        {
            py::gil_scoped_release release;
            if (logical_rank_)
                weight = min_logical_weight_rand_core(basis_.cols_, basis_, duals_, trials, seed, pairs, &witness);
        }
        std::vector<int> support;
        if (weight <= basis_.cols_)
            for (int q=0; q<basis_.cols_; ++q)
                if ((witness[q/64]>>(q%64))&1) support.push_back(q);
        return py::make_tuple(weight,support);
    }
};
PYBIND11_MODULE(large_structure_native,m) {
    py::class_<Space>(m,"Space")
        .def(py::init<py::array_t<int8_t>,py::array_t<int8_t>,const std::vector<std::vector<int>>&>())
        .def_property_readonly("dimension",&Space::dimension)
        .def_property_readonly("logical_rank",&Space::logical_rank)
        .def_property_readonly("workspace_bytes",&Space::workspace_bytes)
        .def_property_readonly("basis",&Space::basis)
        .def("batch",&Space::batch,py::arg("trials"),py::arg("seed"),py::arg("pairs")=8);
}
