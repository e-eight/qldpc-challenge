#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;
using Event = std::pair<int, std::vector<int>>;
namespace {

// All physical words and ALL logical pairing words share each packed row.
// Tables are contiguous; scoring and state updates allocate nothing.
class Session {
    struct Group { size_t offset; int count; };
    int n_, nw_, tw_, stride_, best_, weight_ = 0, position_ = 0;
    int passes_ = 0, stalled_ = 0, pass_weight_ = 0;
    uint64_t scored_ = 0, rejected_ = 0, updates_ = 0, restarts_ = 0, exports_ = 0;
    std::mt19937_64 rng_;
    std::vector<uint64_t> tables_, seeds_, state_, best_word_;
    std::vector<Group> groups_;
    std::vector<int> order_;

    bool logical(const uint64_t* row) const {
        uint64_t tag = 0;
        for (int j = nw_; j < stride_; ++j) tag |= row[j];
        return tag != 0;
    }
    int weight(const uint64_t* row) const {
        int w = 0;
        for (int j = 0; j < nw_; ++j) w += __builtin_popcountll(row[j]);
        return w;
    }
    void xor_row(const uint64_t* row) {
        for (int j = 0; j < stride_; ++j) state_[j] ^= row[j];
    }
    void restart() {
        ++restarts_;
        // One full uniform restart in four; otherwise perturb a common seed
        // or the best search-produced word. All choices stay in the kernel.
        if (restarts_ % 4 == 0) {
            std::fill(state_.begin(), state_.end(), 0);
            for (const auto& g : groups_)
                xor_row(&tables_[g.offset + (rng_() % g.count) * stride_]);
        } else {
            if (restarts_ % 2 == 0 && best_ <= n_)
                state_ = best_word_;
            else {
                size_t which = rng_() % (seeds_.size() / stride_);
                std::copy_n(&seeds_[which * stride_], stride_, state_.begin());
            }
            int kicks = restarts_ <= seeds_.size() / stride_ ? 0 : (2 << (restarts_ % 3));
            for (int k = 0; k < kicks; ++k) {
                const auto& g = groups_[rng_() % groups_.size()];
                xor_row(&tables_[g.offset + (rng_() % g.count) * stride_]);
            }
        }
        // Rare trivial perturbations restart from a known nontrivial seed.
        if (!logical(state_.data())) std::copy_n(seeds_.data(), stride_, state_.begin());
        weight_ = weight(state_.data());
        pass_weight_ = weight_;
        passes_ = stalled_ = position_ = 0;
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
    void consider(std::vector<Event>& out) {
        if (weight_ >= best_ || !logical(state_.data())) return;
        best_ = weight_;
        best_word_ = state_;
        std::vector<int> support;
        support.reserve(best_);
        for (int j = 0; j < nw_; ++j) {
            uint64_t w = state_[j];
            while (w) {
                support.push_back(64 * j + __builtin_ctzll(w));
                w &= w - 1;
            }
        }
        out.emplace_back(best_, std::move(support));
        ++exports_;
    }
    bool update(int index) {
        const auto& g = groups_.at(index);
        int minimum = weight_, selected = 0;
        uint64_t ties = 1;
        for (int i = 1; i < g.count; ++i) {
            const uint64_t* row = &tables_[g.offset + size_t(i) * stride_];
            ++scored_;
            uint64_t tag = 0;
            for (int j = nw_; j < stride_; ++j) tag |= state_[j] ^ row[j];
            if (!tag) { ++rejected_; continue; }
            int w = 0;
            for (int j = 0; j < nw_; ++j) w += __builtin_popcountll(state_[j] ^ row[j]);
            if (w < minimum) { minimum = w; selected = i; ties = 1; }
            else if (w == minimum && rng_() % ++ties == 0) selected = i;
        }
        bool improved = minimum < weight_;
        if (selected) xor_row(&tables_[g.offset + size_t(selected) * stride_]);
        weight_ = minimum;
        ++updates_;
        return improved;
    }
public:
    Session(Array basis, std::vector<int> widths, Array seeds, int n, int tags, uint64_t seed)
        : n_(n), nw_((n+63)/64), tw_((tags+63)/64), stride_(nw_+tw_), best_(n+1), rng_(seed) {
        if (n < 1 || n > 2048 || tags < 1 || tags > 2048 || basis.ndim()!=2 || seeds.ndim()!=2 ||
            basis.shape(1)!=stride_ || seeds.shape(1)!=stride_ || seeds.shape(0)<1 || widths.empty())
            throw std::invalid_argument("Invalid packed dimensions");
        int total = 0;
        for (int w : widths) {
            if (w < 1 || w > 10) throw std::invalid_argument("Group width must be in 1..10");
            total += w;
        }
        if (total != basis.shape(0) || total > 2048) throw std::invalid_argument("Invalid basis grouping");
        auto validate_padding = [&](const Array& a) {
            for (py::ssize_t i=0; i<a.shape(0); ++i) {
                const uint64_t* r=a.data()+i*stride_;
                if ((n%64 && (r[nw_-1] >> (n%64))) || (tags%64 && (r[stride_-1] >> (tags%64))))
                    throw std::invalid_argument("Nonzero padding");
            }
        };
        validate_padding(basis); validate_padding(seeds);
        seeds_.assign(seeds.data(), seeds.data()+seeds.size());
        for (py::ssize_t i=0; i<seeds.shape(0); ++i)
            if (!logical(&seeds_[i*stride_])) throw std::invalid_argument("Trivial seed");
        size_t size=0;
        for (int w : widths) { groups_.push_back({size, 1<<w}); size += size_t(1<<w)*stride_; }
        tables_.resize(size,0);
        int first=0;
        for (size_t k=0; k<groups_.size(); ++k) {
            const auto& g=groups_[k];
            for (int i=1; i<g.count; ++i) {
                int bit=__builtin_ctz(i), prev=i&(i-1);
                for (int j=0; j<stride_; ++j)
                    tables_[g.offset+size_t(i)*stride_+j] = tables_[g.offset+size_t(prev)*stride_+j] ^
                        basis.data()[(first+bit)*stride_+j];
            }
            first += widths[k];
        }
        state_.resize(stride_); best_word_.resize(stride_);
        order_.resize(groups_.size()); std::iota(order_.begin(),order_.end(),0);
        restart();
    }
    std::vector<Event> advance(int steps=1) {
        if (steps<1 || steps>10000) throw std::invalid_argument("Invalid batch");
        std::vector<Event> out;
        for (int i=0; i<steps; ++i) {
            consider(out);
            update(order_[position_++]);
            consider(out);
            if (position_ == int(order_.size())) {
                ++passes_; stalled_ = weight_ < pass_weight_ ? 0 : stalled_+1;
                if (stalled_>=2 || passes_>=8) restart();
                else { position_=0; pass_weight_=weight_; std::shuffle(order_.begin(),order_.end(),rng_); }
            }
        }
        return out;
    }
    // Diagnostic hook: exact update on an explicitly supplied packed state.
    // Caller is responsible for kernel membership, as for packed input basis.
    std::vector<uint64_t> probe(std::vector<uint64_t> row, int group) {
        if (row.size()!=size_t(stride_) || group<0 || group>=int(groups_.size()) || !logical(row.data()))
            throw std::invalid_argument("Invalid probe");
        state_=row; weight_=weight(state_.data()); update(group); return state_;
    }
    py::dict stats() const {
        py::dict d;
        d["scored"]=scored_; d["trivial_rejections"]=rejected_; d["updates"]=updates_;
        d["restarts"]=restarts_; d["exports"]=exports_; d["best"]=best_<=n_ ? py::cast(best_) : py::none();
        d["groups"]=groups_.size(); d["logical_words"]=tw_;
        d["table_bytes"]=tables_.size()*8;
        d["workspace_bytes"]=(tables_.capacity()+seeds_.capacity()+state_.capacity()+best_word_.capacity())*8
                             +groups_.capacity()*sizeof(Group)+order_.capacity()*sizeof(int);
        return d;
    }
};
}
PYBIND11_MODULE(polynomial_weight_native, m) {
    py::class_<Session>(m,"Session")
        .def(py::init<Array,std::vector<int>,Array,int,int,uint64_t>())
        .def("advance",&Session::advance,py::arg("steps")=1)
        .def("probe",&Session::probe)
        .def_property_readonly("stats",&Session::stats);
}
