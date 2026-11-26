// The operations the C++ engine (qfn.cpp) is written in: device/ops.py's interface, in C++.
//
//   cpu_ops()    float32 reference on the host (cpu_ops.cpp). CI runs the engine on it against the Python card code.
//   ttnn_ops()   a Tenstorrent card through tt-metal's C++ API (ttnn_ops.cpp, built with TUFF_TTNN=1 on the box).
//
// A weight W [out, in] (torch layout) is uploaded as W^T [in, out]; linear(x, w) = x @ W^T. Tensors are handles: the
// backend owns what `impl` points to.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tuff {

using Shape = std::vector<int>;

struct Tensor {
    Shape shape;
    std::shared_ptr<void> impl;
    int dim(int i) const { return shape[i < 0 ? (int)shape.size() + i : i]; }
    int rank() const { return (int)shape.size(); }
};

enum class Act { none, silu, sigmoid };
enum class Unary { silu, sigmoid, softplus, relu, exp, sqrt, rsqrt, abs, sign };

struct Ops {
    virtual ~Ops() = default;

    // host <-> device
    virtual Tensor put(const std::vector<float>& a, Shape s) = 0;
    virtual Tensor put_weight(const float* w, int out, int in, const std::string& fmt) = 0;  // -> W^T [in, out]
    virtual std::vector<float> get(const Tensor& x) = 0;
    virtual Tensor zeros(Shape s) = 0;
    virtual Tensor put_ids(const std::vector<int>& ids) = 0;

    // math (binary ops broadcast like numpy)
    virtual Tensor linear(const Tensor& x, const Tensor& w, Act act = Act::none) = 0;
    virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;                 // batched, batch dims broadcast
    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor mul(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor scale(const Tensor& a, float s) = 0;
    virtual Tensor unary(const Tensor& a, Unary u) = 0;
    virtual Tensor clamp_min(const Tensor& a, float m) = 0;
    virtual Tensor sum(const Tensor& a, int dim) = 0;                              // keeps the dim
    virtual Tensor mean(const Tensor& a, int dim) = 0;
    virtual Tensor softmax(const Tensor& a) = 0;                                   // over the last dim
    virtual Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) = 0;      // last dim, times w (broadcast)

    // shape
    virtual Tensor reshape(const Tensor& a, Shape s) = 0;
    virtual Tensor slice(const Tensor& a, int dim, int start, int end) = 0;
    virtual Tensor concat(const std::vector<Tensor>& xs, int dim) = 0;
    virtual Tensor transpose(const Tensor& a) = 0;                                 // the last two dims
    virtual Tensor permute(const Tensor& a, const std::vector<int>& dims) = 0;

    // tables kept on the device (KV cache, indexer keys): rows by id, rows written in place
    virtual Tensor gather_rows(const Tensor& table, const Tensor& ids) = 0;
    virtual void write_rows(Tensor& table, int row, const Tensor& rows) = 0;

    // gated delta rule over m tokens: q, k [m, H, K] (L2-normed), v [m, HV, V], g, beta [m, HV], state [HV, K, V]
    // -> (o [m, HV, V], the state after each token [m, HV, K, V], or only the last one [1, HV, K, V])
    virtual std::pair<Tensor, Tensor> gdr(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                          const Tensor& beta, const Tensor& state, float scale, bool per_token) = 0;

    // expert banks: n slots of W^T [k, N]
    virtual Tensor bank(int n, int k, int N, const std::string& fmt) = 0;
    virtual void bank_write(Tensor& bank, int slot, const float* w, int N, int k) = 0;     // W [N][k], packed here
    // a slot from TT tiles as tstore_expert_tt gives them (no repacking): the reason the engine is C++
    virtual void bank_write_tiles(Tensor&, int, const void*, size_t) {
        throw std::runtime_error("bank_write_tiles: not on this backend");
    }
    // x [1 or n, m, k] @ bank[ids[i]] for each of the n ids -> [n, m, N]
    virtual Tensor gather_matmul(const Tensor& x, const Tensor& bank, const Tensor& ids, Act act = Act::none) = 0;

    // a traced segment: captured once per key, replayed after; fn may use device ops only
    using Fn = std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
    virtual std::vector<Tensor> trace(const std::string&, const Fn& fn, const std::vector<Tensor>& args) {
        return fn(args);
    }
};

std::unique_ptr<Ops> cpu_ops();
#ifdef TUFF_TTNN
std::unique_ptr<Ops> ttnn_ops(int device_id, size_t trace_region_bytes);
#endif

}  // namespace tuff
