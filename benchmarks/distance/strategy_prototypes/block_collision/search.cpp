// Packed systematic-generator Stern p=4 search and exact small-space enumeration.
#include "../../../../verify/gf2_fast.cpp"
#include <numeric>
#include <random>

struct Event {
    int weight;
    std::vector<int> support;
    const char* stage;
};

static GF2Matrix checked_matrix(py::array_t<int8_t> a) {
    if (a.ndim() != 2 || a.shape(0) > 2048 || a.shape(1) > 2048)
        throw std::invalid_argument("Expected a matrix with dimensions <=2048");
    auto view = a.unchecked<2>();
    for (ssize_t r=0; r<a.shape(0); ++r)
        for (ssize_t c=0; c<a.shape(1); ++c)
            if (view(r,c) != 0 && view(r,c) != 1)
                throw std::invalid_argument("Matrix entries must be binary");
    return GF2Matrix::from_numpy(a);
}

class Session {
    GF2Matrix basis_, work_;
    int original_n_, k_, nw_, best_, ell_=0;
    bool stern_;
    std::mt19937_64 rng_;
    std::vector<int> coordinates_, columns_, row_order_, projected_, pivots_;
    std::vector<uint8_t> is_pivot_;
    std::vector<uint64_t> basis_tags_, tags_, candidate_, enum_word_;
    std::vector<uint32_t> keys_;
    struct Pair { uint16_t a,b; int next; };
    std::vector<Pair> pairs_;
    std::vector<int> heads_;
    uint64_t reductions_=0, singles_=0, pair_scores_=0, collisions_=0, rejected_=0;
    uint64_t exported_=0, enum_next_=1, enum_nontrivial_=0, enum_tag_=0;

    void save_word(int weight, const char* stage, std::vector<Event>& out) {
        best_ = weight;
        Event event{weight, {}, stage};
        event.support.reserve(weight);
        for (int j=0; j<basis_.cols_; ++j)
            if ((candidate_[j/64] >> (j%64)) & 1) event.support.push_back(coordinates_[j]);
        out.push_back(std::move(event));
        ++exported_;
    }
    void score(int a, int b, int c, int d, const char* stage, std::vector<Event>& out) {
        uint64_t tag = tags_[a];
        if (b>=0) tag ^= tags_[b];
        if (c>=0) tag ^= tags_[c];
        if (d>=0) tag ^= tags_[d];
        if (!tag) { ++rejected_; return; }
        int weight=0;
        for (int w=0; w<nw_; ++w) {
            uint64_t value = work_.row_ptr(a)[w];
            if (b>=0) value ^= work_.row_ptr(b)[w];
            if (c>=0) value ^= work_.row_ptr(c)[w];
            if (d>=0) value ^= work_.row_ptr(d)[w];
            candidate_[w] = value;
            weight += __builtin_popcountll(value);
            if (weight >= best_) return;
        }
        save_word(weight, stage, out);
    }
    void reduce() {
        std::copy(basis_.data_.begin(), basis_.data_.end(), work_.data_.begin());
        std::copy(basis_tags_.begin(), basis_tags_.end(), tags_.begin());
        std::shuffle(columns_.begin(), columns_.end(), rng_);
        std::fill(is_pivot_.begin(), is_pivot_.end(), 0);
        int rank=0;
        for (int q : columns_) {
            int pivot=rank;
            while (pivot<k_ && !work_.get(pivot,q)) ++pivot;
            if (pivot==k_) continue;
            work_.swap_rows(rank,pivot);
            std::swap(tags_[rank],tags_[pivot]);
            for (int r=0; r<k_; ++r) if (r!=rank && work_.get(r,q)) {
                work_.xor_rows(r,rank);
                tags_[r] ^= tags_[rank];
            }
            pivots_[rank] = q;
            is_pivot_[q] = 1;
            if (++rank==k_) break;
        }
        if (rank!=k_) throw std::runtime_error("Dependent input basis");
        ++reductions_;
    }
    void trial(std::vector<Event>& out) {
        reduce();
        for (int a=0; a<k_; ++a) {
            ++singles_; score(a,-1,-1,-1,"single",out);
            for (int b=0; b<a; ++b) {
                ++pair_scores_; score(a,b,-1,-1,"pair",out);
            }
        }
        if (!stern_ || k_<4) return;
        std::shuffle(row_order_.begin(), row_order_.end(), rng_);
        const int split=k_/2;
        const int count=split*(split-1)/2;
        ell_=std::min({basis_.cols_-k_,16,31-__builtin_clz(unsigned(count))});
        projected_.clear();
        for (int q : columns_) if (!is_pivot_[q]) projected_.push_back(q);
        std::shuffle(projected_.begin(),projected_.end(),rng_);
        projected_.resize(ell_);
        for (int r=0; r<k_; ++r) {
            uint32_t key=0;
            for (int j=0; j<ell_; ++j) key |= uint32_t(work_.get(r,projected_[j])) << j;
            keys_[r]=key;
        }
        std::fill(heads_.begin(),heads_.begin()+(1<<ell_),-1);
        int index=0;
        for (int i=0; i<split; ++i) for (int j=0; j<i; ++j) {
            const int a=row_order_[i],b=row_order_[j];
            const uint32_t key=keys_[a]^keys_[b];
            pairs_[index]={uint16_t(a),uint16_t(b),heads_[key]};
            heads_[key]=index++;
        }
        for (int i=split; i<k_; ++i) for (int j=split; j<i; ++j) {
            const int c=row_order_[i],d=row_order_[j];
            for (int p=heads_[keys_[c]^keys_[d]]; p>=0; p=pairs_[p].next) {
                ++collisions_; score(pairs_[p].a,pairs_[p].b,c,d,"collision4",out);
            }
        }
    }
    static py::list export_events(const std::vector<Event>& events) {
        py::list out;
        for (const auto& e : events) out.append(py::make_tuple(e.weight,e.support,e.stage));
        return out;
    }
public:
    Session(py::array_t<int8_t> basis, py::array_t<int8_t> duals, uint64_t seed, bool stern)
        : stern_(stern), rng_(seed) {
        auto input=checked_matrix(basis), logical=checked_matrix(duals);
        original_n_=input.cols_; k_=input.rows_; best_=original_n_+1;
        if (original_n_!=logical.cols_ || k_>512 || logical.rows_>64)
            throw std::invalid_argument("Incompatible dimensions or rank cap exceeded");
        if (int(gf2_rref(input).pivots.size())!=k_)
            throw std::invalid_argument("Independent basis required");
        for (int q=0; q<original_n_; ++q) {
            bool active=false;
            for (int r=0; r<k_ && !active; ++r) active=input.get(r,q);
            if (active) coordinates_.push_back(q);
        }
        basis_=GF2Matrix(k_,coordinates_.size());
        for (int r=0; r<k_; ++r) for (int j=0; j<basis_.cols_; ++j)
            basis_.set(r,j,input.get(r,coordinates_[j]));
        basis_tags_.resize(k_,0);
        for (int r=0; r<k_; ++r) for (int d=0; d<logical.rows_; ++d) {
            int parity=0;
            for (int w=0; w<input.wpr_; ++w)
                parity ^= __builtin_parityll(input.row_ptr(r)[w]&logical.row_ptr(d)[w]);
            basis_tags_[r] |= uint64_t(parity) << d;
        }
        nw_=basis_.wpr_; work_=GF2Matrix(k_,basis_.cols_);
        tags_.resize(k_); candidate_.resize(nw_); enum_word_.resize(nw_,0);
        columns_.resize(basis_.cols_); std::iota(columns_.begin(),columns_.end(),0);
        row_order_.resize(k_); std::iota(row_order_.begin(),row_order_.end(),0);
        pivots_.resize(k_); is_pivot_.resize(basis_.cols_); keys_.resize(k_);
        projected_.reserve(basis_.cols_);
        if (stern_ && k_>=4) {
            int count=(k_/2)*(k_/2-1)/2;
            int max_l=std::min({basis_.cols_-k_,16,31-__builtin_clz(unsigned(count))});
            pairs_.resize(count); heads_.resize(1<<max_l);
        }
    }
    py::list advance(int trials) {
        if (trials<1) throw std::invalid_argument("Positive trial count required");
        std::vector<Event> events;
        { py::gil_scoped_release release;
          if (k_) for (int t=0; t<trials; ++t) trial(events); }
        return export_events(events);
    }
    py::list enumerate_chunk(uint64_t words) {
        if (!words || k_>24) throw std::invalid_argument("Enumeration needs positive work and dimension <=24");
        std::vector<Event> events;
        { py::gil_scoped_release release;
          uint64_t end=std::min(uint64_t(1)<<k_,enum_next_+std::min(words,uint64_t(1)<<24));
          for (; enum_next_<end; ++enum_next_) {
              int bit=__builtin_ctzll(enum_next_);
              enum_tag_ ^= basis_tags_[bit];
              for (int w=0; w<nw_; ++w) enum_word_[w] ^= basis_.row_ptr(bit)[w];
              if (!enum_tag_) continue;
              ++enum_nontrivial_;
              int weight=0;
              for (uint64_t word : enum_word_) weight += __builtin_popcountll(word);
              if (weight<best_) {
                  std::copy(enum_word_.begin(),enum_word_.end(),candidate_.begin());
                  save_word(weight,"enumeration",events);
              }
          }
        }
        return export_events(events);
    }
    py::dict stats() const {
        py::dict out;
        out["dimension"]=k_; out["active_columns"]=basis_.cols_; out["ell"]=ell_;
        out["reductions"]=reductions_; out["single_scores"]=singles_; out["pair_scores"]=pair_scores_;
        out["collisions"]=collisions_; out["trivial_rejections"]=rejected_; out["exports"]=exported_;
        out["best"]=best_<=original_n_ ? py::cast(best_) : py::none();
        out["enumerated"]=enum_next_-1; out["enumerated_nontrivial"]=enum_nontrivial_;
        out["enumeration_complete"]= k_<=24 && enum_next_==(uint64_t(1)<<k_);
        size_t bytes=8*(basis_.data_.capacity()+work_.data_.capacity()+basis_tags_.capacity()+tags_.capacity()
                         +candidate_.capacity()+enum_word_.capacity());
        bytes+=sizeof(Pair)*pairs_.capacity()+sizeof(int)*(heads_.capacity()+coordinates_.capacity()+columns_.capacity()
                 +row_order_.capacity()+projected_.capacity()+pivots_.capacity());
        bytes+=is_pivot_.capacity()+sizeof(uint32_t)*keys_.capacity();
        out["workspace_bytes"]=bytes;
        return out;
    }
    py::dict snapshot() const {
        GF2Matrix expanded(k_,original_n_);
        for (int r=0; r<k_; ++r) for (int j=0; j<basis_.cols_; ++j)
            expanded.set(r,coordinates_[j],work_.get(r,j));
        std::vector<int> projection,pivots;
        for (int q : projected_) projection.push_back(coordinates_[q]);
        for (int q : pivots_) pivots.push_back(coordinates_[q]);
        py::dict out;
        out["basis"]=expanded.to_numpy(); out["tags"]=tags_; out["keys"]=keys_;
        out["projection"]=projection; out["row_order"]=row_order_; out["pivots"]=pivots;
        return out;
    }
};

PYBIND11_MODULE(block_collision_native,m) {
    py::class_<Session>(m,"Session")
        .def(py::init<py::array_t<int8_t>,py::array_t<int8_t>,uint64_t,bool>(),
             py::arg("basis"),py::arg("duals"),py::arg("seed"),py::arg("stern")=true)
        .def("advance",&Session::advance)
        .def("enumerate_chunk",&Session::enumerate_chunk)
        .def_property_readonly("stats",&Session::stats)
        .def("snapshot",&Session::snapshot);
}
