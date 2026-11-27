// ops.h on a Tenstorrent card through tt-metal's C++ API (ttnn ops, metal traces, raw buffer writes).
// Built only on the box: make TUFF_TTNN=1 TT_METAL_HOME=... ; CI does not have tt-metal.
//
// ponytail: a draft written from the ttnn Python docs (device/ops.py's TTNN class is the same backend in Python) without
// a card or tt-metal's headers at hand. Names to check against the installed tt-metal first: the ttnn:: op signatures
// (slice, sum, mean, sparse_matmul), the gated-delta-rule op's C++ name, the trace calls, and EnqueueWriteSubBuffer for
// bank_write_tiles. Each op here mirrors one in ops.py's TTNN class; run tuff_engine --ttnn on engine/export.py's tiny
// model and compare with the CPU reference before the real model.
#include <ttnn/core.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/kv_cache/kv_cache.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/trace.hpp>
#include <tt-metalium/host_api.hpp>

#include <map>

#include "ops.h"

namespace tuff {
namespace {

using TT = ttnn::Tensor;

TT& tt(const Tensor& t) { return *std::static_pointer_cast<TT>(t.impl); }
Tensor wrap(TT x, Shape s) { return Tensor{std::move(s), std::make_shared<TT>(std::move(x))}; }
ttnn::Shape shp(const Shape& s) { return ttnn::Shape(ttnn::SmallVector<uint32_t>(s.begin(), s.end())); }

ttnn::DataType dtype(const std::string& f) {
    if (f == "bfp8") return ttnn::DataType::BFLOAT8_B;
    if (f == "bfp4") return ttnn::DataType::BFLOAT4_B;
    if (f == "bf16") return ttnn::DataType::BFLOAT16;
    return ttnn::DataType::FLOAT32;
}

struct Bank { TT w; int n, k, N; };                    // [n, 1, k, N] tiles: slot s = a contiguous run of pages

struct TTNN : Ops {
    std::shared_ptr<ttnn::MeshDevice> dev;
    std::map<std::string, std::tuple<ttnn::MeshTraceId, std::vector<TT>, std::vector<Tensor>>> traces;

    TTNN(int id, size_t trace_bytes) {
        dev = ttnn::MeshDevice::create_unit_mesh(id, /*l1_small_size=*/0, trace_bytes, /*num_command_queues=*/2);
    }
    ~TTNN() override { dev->close(); }

    TT upload(const std::vector<float>& a, const Shape& s, ttnn::DataType dt, ttnn::Layout lay = ttnn::TILE_LAYOUT) {
        auto spec = ttnn::TensorSpec(shp(s), ttnn::TensorLayout(dt, ttnn::PageConfig(lay), ttnn::DRAM_MEMORY_CONFIG));
        return TT::from_vector(a, spec, dev.get());
    }

    Tensor put(const std::vector<float>& a, Shape s) override { return wrap(upload(a, s, ttnn::DataType::FLOAT32), s); }
    Tensor put_weight(const float* w, int out, int in, const std::string& fmt) override {
        std::vector<float> t((size_t)in * out);
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++) t[(size_t)i * out + o] = w[(size_t)o * in + i];
        std::string f = (fmt == "bfp8" || fmt == "bfp4") && (out % 16 || in % 8) ? "bf16" : fmt;
        return wrap(upload(t, {in, out}, dtype(f)), {in, out});      // ttnn packs BFP on the host here: fine once
    }
    std::vector<float> get(const Tensor& x) override { return ttnn::from_device(tt(x)).to_vector<float>(); }
    Tensor zeros(Shape s) override { return wrap(ttnn::zeros(shp(s), ttnn::DataType::FLOAT32, ttnn::TILE_LAYOUT, *dev), s); }
    Tensor put_ids(const std::vector<int>& ids) override {
        std::vector<uint32_t> v(ids.begin(), ids.end());
        auto spec = ttnn::TensorSpec(shp({1, (int)v.size()}), ttnn::TensorLayout(ttnn::DataType::UINT32,
                                     ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT), ttnn::DRAM_MEMORY_CONFIG));
        return wrap(TT::from_vector(v, spec, dev.get()), {(int)v.size()});
    }

    static Shape bshape(const Shape& a, const Shape& b) {         // numpy broadcast of the logical shapes
        Shape s(std::max(a.size(), b.size()), 1);
        for (size_t i = 0; i < s.size(); i++) {
            int x = i < s.size() - a.size() ? 1 : a[i - (s.size() - a.size())];
            int y = i < s.size() - b.size() ? 1 : b[i - (s.size() - b.size())];
            s[i] = std::max(x, y);
        }
        return s;
    }
    Tensor linear(const Tensor& x, const Tensor& w, Act a) override {
        std::optional<std::string> act;
        if (a == Act::silu) act = "silu";
        if (a == Act::sigmoid) act = "sigmoid";
        Shape s = x.shape;
        s.back() = w.dim(1);
        return wrap(ttnn::linear(tt(x), tt(w), std::nullopt, false, false, std::nullopt, std::nullopt, std::nullopt, act), s);
    }
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        Shape s = bshape(Shape(a.shape.begin(), a.shape.end() - 2), Shape(b.shape.begin(), b.shape.end() - 2));
        s.push_back(a.dim(-2));
        s.push_back(b.dim(-1));
        return wrap(ttnn::matmul(tt(a), tt(b)), s);
    }
    Tensor add(const Tensor& a, const Tensor& b) override { return wrap(ttnn::add(tt(a), tt(b)), bshape(a.shape, b.shape)); }
    Tensor mul(const Tensor& a, const Tensor& b) override { return wrap(ttnn::multiply(tt(a), tt(b)), bshape(a.shape, b.shape)); }
    Tensor scale(const Tensor& a, float s) override { return wrap(ttnn::multiply(tt(a), s), a.shape); }
    Tensor unary(const Tensor& a, Unary u) override {
        const TT& x = tt(a);
        switch (u) {
            case Unary::silu: return wrap(ttnn::silu(x), a.shape);
            case Unary::sigmoid: return wrap(ttnn::sigmoid_accurate(x), a.shape);
            case Unary::softplus: return wrap(ttnn::softplus(x, 1.0f, 20.0f), a.shape);
            case Unary::relu: return wrap(ttnn::relu(x), a.shape);
            case Unary::exp: return wrap(ttnn::exp(x), a.shape);
            case Unary::sqrt: return wrap(ttnn::sqrt(x), a.shape);
            case Unary::rsqrt: return wrap(ttnn::rsqrt(x), a.shape);
            case Unary::abs: return wrap(ttnn::abs(x), a.shape);
            case Unary::sign: return wrap(ttnn::sign(x), a.shape);
        }
        return a;
    }
    Tensor clamp_min(const Tensor& a, float m) override { return wrap(ttnn::clamp(tt(a), m, std::nullopt), a.shape); }
    Tensor sum(const Tensor& a, int dim) override {
        Shape s = a.shape;
        s[dim < 0 ? dim + s.size() : dim] = 1;
        return wrap(ttnn::sum(tt(a), dim, /*keepdim=*/true), s);
    }
    Tensor mean(const Tensor& a, int dim) override {
        Shape s = a.shape;
        s[dim < 0 ? dim + s.size() : dim] = 1;
        return wrap(ttnn::mean(tt(a), dim, /*keepdim=*/true), s);
    }
    Tensor softmax(const Tensor& a) override { return wrap(ttnn::softmax(tt(a), -1), a.shape); }
    Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) override {
        return mul(wrap(ttnn::rms_norm(tt(x), eps), x.shape), w);
    }

    Tensor reshape(const Tensor& a, Shape s) override { return wrap(ttnn::reshape(tt(a), shp(s)), s); }
    Tensor slice(const Tensor& a, int dim, int start, int end) override {
        int d = dim < 0 ? dim + a.rank() : dim;
        ttnn::SmallVector<int> lo(a.rank(), 0), hi(a.shape.begin(), a.shape.end()), st(a.rank(), 1);
        lo[d] = start;
        hi[d] = end;
        Shape s = a.shape;
        s[d] = end - start;
        return wrap(ttnn::slice(tt(a), lo, hi, st), s);
    }
    Tensor concat(const std::vector<Tensor>& xs, int dim) override {
        std::vector<TT> v;
        Shape s = xs[0].shape;
        int d = dim < 0 ? dim + (int)s.size() : dim;
        s[d] = 0;
        for (auto& x : xs) { v.push_back(tt(x)); s[d] += x.shape[d]; }
        return wrap(ttnn::concat(v, d), s);
    }
    Tensor transpose(const Tensor& a) override {
        Shape s = a.shape;
        std::swap(s[s.size() - 1], s[s.size() - 2]);
        return wrap(ttnn::transpose(tt(a), -2, -1), s);
    }
    Tensor permute(const Tensor& a, const std::vector<int>& dims) override {
        Shape s(dims.size());
        for (size_t i = 0; i < dims.size(); i++) s[i] = a.shape[dims[i]];
        return wrap(ttnn::permute(tt(a), ttnn::SmallVector<int64_t>(dims.begin(), dims.end())), s);
    }

    Tensor gather_rows(const Tensor& table, const Tensor& ids) override {
        int n = ids.dim(0), D = table.dim(-1);
        return wrap(ttnn::reshape(ttnn::embedding(tt(ids), tt(table), false, std::nullopt, ttnn::TILE_LAYOUT), shp({n, D})), {n, D});
    }
    void write_rows(Tensor& table, int row, const Tensor& rows) override {
        int S = table.dim(0), D = table.dim(1);
        TT cache = ttnn::reshape(tt(table), shp({1, 1, S, D}));
        for (int r = 0; r < rows.dim(0); r++)
            ttnn::update_cache(cache, ttnn::reshape(tt(slice(rows, 0, r, r + 1)), shp({1, 1, 1, D})), row + r);
    }

    std::pair<Tensor, Tensor> gdr(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
                                  const Tensor& state, float scale, bool per_token) override {
        int m = q.dim(0), H = q.dim(1), K = q.dim(2), HV = v.dim(1), V = v.dim(2);
        auto [o, s] = ttnn::transformer::fused_recurrent_gated_delta_rule(
            ttnn::reshape(tt(q), shp({1, m, H, K})), ttnn::reshape(tt(k), shp({1, m, H, K})),
            ttnn::reshape(tt(v), shp({1, m, HV, V})), ttnn::reshape(tt(g), shp({1, m, HV})),
            ttnn::reshape(tt(beta), shp({1, m, HV})), scale, ttnn::reshape(tt(state), shp({1, HV, K, V})), per_token);
        int n = per_token ? m : 1;
        return {wrap(ttnn::reshape(o, shp({m, HV, V})), {m, HV, V}), wrap(ttnn::reshape(s, shp({n, HV, K, V})), {n, HV, K, V})};
    }

    Tensor bank(int n, int k, int N, const std::string& fmt) override {
        auto b = std::make_shared<Bank>(Bank{ttnn::zeros(shp({n, 1, k, N}), dtype(fmt), ttnn::TILE_LAYOUT, *dev), n, k, N});
        return Tensor{{n, k, N}, b};
    }
    void bank_write(Tensor& bank, int slot, const float* w, int N, int k) override {
        Bank& b = *std::static_pointer_cast<Bank>(bank.impl);
        Tensor t = put_weight(w, N, k, b.w.dtype() == ttnn::DataType::BFLOAT4_B ? "bfp4" : "bfp8");
        ttnn::fill_cache(b.w, ttnn::reshape(tt(t), shp({1, 1, k, N})), slot);
    }
    bool takes_tiles() const override { return true; }
    void bank_write_tiles(Tensor& bank, int slot, const void* tiles, size_t bytes) override {
        // the slot's pages as they are: tstore_expert_tt already wrote TT's bfp tiles, so no packing on the host,
        // only the PCIe copy (on command queue 1, next to the decode on queue 0)
        Bank& b = *std::static_pointer_cast<Bank>(bank.impl);
        auto buffer = b.w.mesh_buffer()->get_reference_buffer();
        tt::tt_metal::BufferRegion region((size_t)slot * bytes, bytes);
        tt::tt_metal::EnqueueWriteSubBuffer(dev->mesh_command_queue(1), *buffer, tiles, region, /*blocking=*/false);
    }
    Tensor gather_matmul(const Tensor& x, const Tensor& bank, const Tensor& ids, Act a) override {
        Bank& b = *std::static_pointer_cast<Bank>(bank.impl);
        int g = x.dim(0), m = x.dim(1), n = ids.dim(0);
        TT sp = ttnn::zeros(shp({1, 1, 1, b.n}), ttnn::DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT, *dev);  // unread
        TT y = ttnn::sparse_matmul(ttnn::reshape(tt(x), shp({1, g, m, b.k})), ttnn::reshape(b.w, shp({1, b.n, b.k, b.N})),
                                   sp, /*is_input_a_sparse=*/g > 1, /*is_input_b_sparse=*/true, tt(ids));
        Tensor out = wrap(ttnn::reshape(y, shp({n, m, b.N})), {n, m, b.N});
        return a == Act::none ? out : unary(out, a == Act::silu ? Unary::silu : Unary::sigmoid);
    }

    std::vector<Tensor> trace(const std::string& key, const Fn& fn, const std::vector<Tensor>& args) override {
        auto it = traces.find(key);
        if (it == traces.end()) {
            fn(args);                                          // compile, upload constants
            std::vector<TT> buf;
            std::vector<Tensor> bargs;
            for (auto& a : args) { buf.push_back(ttnn::clone(tt(a))); bargs.push_back(wrap(buf.back(), a.shape)); }
            auto id = ttnn::operations::trace::begin_trace_capture(dev.get(), ttnn::QueueId(0));
            std::vector<Tensor> outs = fn(bargs);
            ttnn::operations::trace::end_trace_capture(dev.get(), id, ttnn::QueueId(0));
            it = traces.emplace(key, std::make_tuple(id, buf, outs)).first;
        }
        auto& [id, buf, outs] = it->second;
        for (size_t i = 0; i < args.size(); i++) ttnn::copy(tt(args[i]), buf[i]);
        ttnn::operations::trace::execute_trace(dev.get(), id, ttnn::QueueId(0), /*blocking=*/false);
        return outs;
    }
};

}  // namespace

std::unique_ptr<Ops> ttnn_ops(int device_id, size_t trace_region_bytes) {
    return std::make_unique<TTNN>(device_id, trace_region_bytes);
}

}  // namespace tuff
