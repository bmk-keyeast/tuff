// tuff_engine <dir> [--store <experts dir> bfp4|bfp8] [--ttnn]
//
// <dir> holds what engine/export.py writes: config.txt, weights.bin, case.txt. Runs two cases on fresh cards:
//   steps    the tokens of `steps` one at a time; each step's mixer output -> <dir>/out_steps.bin (float32 [T][d])
//   prompt   greedy decoding of `n` tokens after `prompt` -> <dir>/out_tokens.txt
// engine/test_engine.py compares both with device/qfn.py on the same weights.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "qfn.h"

using namespace tuff;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: tuff_engine <dir> [--store <dir> bfp4|bfp8] [--ttnn]\n");
        return 2;
    }
    std::string dir = argv[1], store, sfmt = "bfp4";
    bool ttnn = false;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--store") && i + 2 < argc) { store = argv[++i]; sfmt = argv[++i]; }
        else if (!std::strcmp(argv[i], "--ttnn")) ttnn = true;
    }
    Config c = Config::load(dir + "/config.txt");
    Weights W = Weights::load(dir + "/weights.bin");
    std::map<std::string, std::vector<int>> cs;
    std::ifstream f(dir + "/case.txt");
    for (std::string line; std::getline(f, line);) {
        std::istringstream ss(line);
        std::string k;
        ss >> k;
        for (int v; ss >> v;) cs[k].push_back(v);
    }
    std::unique_ptr<Ops> ops;
#ifdef TUFF_TTNN
    if (ttnn) ops = ttnn_ops(0, (size_t)256 << 20);
#endif
    if (!ops) {
        if (ttnn) std::fprintf(stderr, "built without TUFF_TTNN; using the CPU reference\n");
        ops = cpu_ops();
    }
    std::unique_ptr<Host> host = store.empty() ? std::make_unique<Host>(c, W) : std::make_unique<StoreHost>(c, W, store, sfmt);
    std::string dense = cs.count("dense_bfp") ? "bfp8,lm_head=bfp4,lin_z=bfp4" : "f32";
    int slots = cs.at("slots")[0], budget = cs.at("budget")[0];

    Card a(*ops, c, W, *host, dense, "f32", slots, budget, 256);
    std::ofstream out(dir + "/out_steps.bin", std::ios::binary);
    for (int tok : cs.at("steps")) {
        std::vector<float> y = ops->get(a.step(tok));
        out.write((const char*)y.data(), 4 * y.size());
    }
    std::printf("steps: %zu tokens, card expert hits %ld/%ld\n", cs.at("steps").size(), a.hits, a.lookups);

    Card b(*ops, c, W, *host, dense, "f32", slots, budget, 256);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> toks = b.generate(cs.at("prompt"), cs.at("n")[0]);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream ot(dir + "/out_tokens.txt");
    for (int tk : toks) ot << tk << " ";
    std::printf("greedy: %zu tokens in %.2f s, card expert hits %ld/%ld\n", toks.size(), s, b.hits, b.lookups);
    return 0;
}
