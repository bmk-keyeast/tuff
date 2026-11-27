// tuff_engine: Qwen3.8-Flash-Next decode in C++.
//
//   tuff_engine run --model <snapshot> --store <experts dir> --ids "<prompt token ids>" [--n 128] [--mtp 2]
//                   [--ttnn] [--gpu] [--slots N] [--budget 25] [--ctx 4096] [--vocab <draft vocab ids file>]
//       the real model: a Hugging Face snapshot of Qwen/Qwen3.8-Flash-Next, the experts converted by loader.py
//       convert; prints the generated ids (engine/tok.py turns text into ids and back) and tokens/s.
//   tuff_engine test <dir> [--store <dir> bfp4|bfp8]
//       engine/test_engine.py's cases on the tiny random QFN it exported: steps, greedy, MTP speculative, the split.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "qfn.h"

using namespace tuff;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> ints(const std::string& s) {
    std::istringstream ss(s);
    std::vector<int> v;
    for (int x; ss >> x;) v.push_back(x);
    return v;
}

static std::unique_ptr<Ops> card_ops(bool ttnn, size_t trace_bytes) {
#ifdef TUFF_TTNN
    if (ttnn) return ttnn_ops(0, trace_bytes);
#endif
    if (ttnn) std::fprintf(stderr, "built without TUFF_TTNN=1: the CPU reference stands in for the card\n");
    (void)trace_bytes;
    return cpu_ops();
}

static std::unique_ptr<Ops> gpu_ops() {
#ifdef TUFF_CUDA
    return cuda_ops();
#else
    std::fprintf(stderr, "built without TUFF_CUDA=1: the CPU reference stands in for the GPU\n");
    return cpu_ops();
#endif
}

static int run(std::map<std::string, std::string>& a) {
    Config c = Config::load_hf(a.at("model") + "/config.json");
    Weights W = Weights::open_hf(a.at("model"), "model.language_model.", c);
    StoreHost host(c, W, a.at("store"), "bfp4");
    std::string dense = "bfp8,lm_head=bfp4,lin_z=bfp4";             // tuff's default formats
    int ctx = std::stoi(a.count("ctx") ? a["ctx"] : "4096"), budget = std::stoi(a.count("budget") ? a["budget"] : "25");
    int slots = std::stoi(a.count("slots") ? a["slots"] : "10019");   // plan.memory_map's room at 4k context
    auto ops = card_ops(a.count("ttnn"), (size_t)256 << 20);
    std::unique_ptr<Ops> g;
    std::unique_ptr<Decoder> dec;
    if (a.count("gpu")) {                              // plan.split_multi's split for a 4060 Ti 16 GB
        g = gpu_ops();
        dec = std::make_unique<TP>(*ops, *g, c, W, host, 0.41f, 10581, 4791, budget, dense, "bfp4", ctx);
    } else {
        dec = std::make_unique<Card>(*ops, c, W, host, dense, "bfp4", slots, budget, ctx);
    }
    int k = std::stoi(a.count("mtp") ? a["mtp"] : "2");
    std::unique_ptr<Weights> MW;
    std::unique_ptr<MTPDrafter> drafter;
    Config mcfg = c.mtp();
    if (k) {
        MW = std::make_unique<Weights>(Weights::open_hf(a.at("model"), "mtp.", mcfg));
        std::vector<int> vocab;
        if (a.count("vocab")) {
            std::ifstream f(a["vocab"]);
            for (int v; f >> v;) vocab.push_back(v);
        }
        // ponytail: the MTP layer's experts from the checkpoint into its own 256 slots (store them too if it matters)
        drafter = std::make_unique<MTPDrafter>(*ops, c, *MW, host, dense, "bfp4", 256, 10, ctx, vocab);
    }
    int n = std::stoi(a.count("n") ? a["n"] : "128");
    std::vector<int> prompt = ints(a.at("ids"));
    double t0 = now();
    std::vector<int> out = dec->generate(prompt, n, drafter.get(), k);
    double t1 = now();
    for (int x : out) std::printf("%d ", x);
    std::printf("\nprompt %zu tokens: %.1f s (includes weight uploads); %d tokens: %.1f s = %.1f t/s, %d steps, "
                "card expert hits %.1f%%\n", prompt.size(), dec->t_prompt - t0, n, t1 - dec->t_prompt,
                n / std::max(t1 - dec->t_prompt, 1e-9), dec->steps, 100.0 * dec->hits / std::max(dec->lookups, 1L));
    return 0;
}

static int test(const std::string& dir, const std::string& store, const std::string& sfmt) {
    Config c = Config::load(dir + "/config.txt");
    Weights W = Weights::load(dir + "/weights.bin");
    std::map<std::string, std::vector<int>> cs;
    std::ifstream f(dir + "/case.txt");
    for (std::string line; std::getline(f, line);) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        for (int v; ss >> v;) cs[key].push_back(v);
    }
    auto ops = cpu_ops();
    std::unique_ptr<Host> host = store.empty() ? std::make_unique<Host>(c, W) : std::make_unique<StoreHost>(c, W, store, sfmt);
    int slots = cs.at("slots")[0], budget = cs.at("budget")[0], n = cs.at("n")[0], max_t = 256;
    auto write = [&](const std::string& name, const std::vector<int>& v, int steps) {
        std::ofstream o(dir + "/" + name);
        for (int x : v) o << x << " ";
        o << "\n" << steps << "\n";
    };

    Card a(*ops, c, W, *host, "f32", "f32", slots, budget, max_t);   // 1. steps
    std::ofstream out(dir + "/out_steps.bin", std::ios::binary);
    for (int tok : cs.at("steps")) {
        std::vector<float> y = ops->get(a.step(tok));
        out.write((const char*)y.data(), 4 * y.size());
    }
    std::printf("steps: %zu tokens, card expert hits %ld/%ld\n", cs.at("steps").size(), a.hits, a.lookups);

    Card b(*ops, c, W, *host, "f32", "f32", slots, budget, max_t);   // 2. greedy
    write("out_tokens.txt", b.generate(cs.at("prompt"), n), b.steps);

    Card l(*ops, c, W, *host, "f32", "f32", slots, budget, max_t);   // 3. one long pass (prefill), then steps
    std::ofstream ol(dir + "/out_prefill.bin", std::ios::binary);
    std::vector<int> st = cs.at("steps");
    int p = cs.at("prefill")[0];
    std::vector<float> y = ops->get(l.verify(std::vector<int>(st.begin(), st.begin() + p)));
    l.commit(p);
    ol.write((const char*)y.data(), 4 * y.size());
    for (size_t i = p; i < st.size(); i++) {
        y = ops->get(l.step(st[i]));
        ol.write((const char*)y.data(), 4 * y.size());
    }

    if (cs.count("mtp")) {                             // 4. speculative with QFN's MTP drafter (and a draft vocabulary)
        Weights MW = Weights::load(dir + "/mtp.bin");
        for (int pass = 0; pass < 2; pass++) {
            std::vector<int> vocab = pass ? cs.at("vocab") : std::vector<int>{};
            Card s(*ops, c, W, *host, "f32", "f32", slots, budget, max_t);
            MTPDrafter d(*ops, c, MW, *host, "f32", "f32", slots, budget, max_t, vocab);
            std::vector<int> toks = s.generate(cs.at("prompt"), n, &d, cs.at("mtp")[0]);
            write(pass ? "out_mtp_vocab.txt" : "out_mtp.txt", toks, s.steps);
            std::printf("MTP%s: %zu tokens in %d steps\n", pass ? " (draft vocabulary)" : "", toks.size(), s.steps);
        }
    }
    if (cs.count("hf")) {                              // 6. the same model from a Hugging Face snapshot (safetensors)
        Config hc = Config::load_hf(dir + "/hf/config.json");
        Weights hw = Weights::open_hf(dir + "/hf", "model.language_model.", hc);
        Host hh(hc, hw);
        Card s(*ops, hc, hw, hh, "f32", "f32", slots, budget, max_t);
        std::ofstream oh(dir + "/out_hf_steps.bin", std::ios::binary);
        for (int tok : cs.at("steps")) {
            std::vector<float> v = ops->get(s.step(tok));
            oh.write((const char*)v.data(), 4 * v.size());
        }
        Config hm = hc.mtp();
        Weights mw = Weights::open_hf(dir + "/hf", "mtp.", hm);
        Card g(*ops, hc, hw, hh, "f32", "f32", slots, budget, max_t);
        MTPDrafter d(*ops, hc, mw, hh, "f32", "f32", slots, budget, max_t);
        write("out_hf_mtp.txt", g.generate(cs.at("prompt"), n, &d, cs.at("mtp")[0]), g.steps);
        std::printf("snapshot: %zu steps, MTP %d steps\n", cs.at("steps").size(), g.steps);
    }
    if (cs.count("tp")) {                              // 5. the card + GPU split (two CPU backends here)
        auto gops = cpu_ops();
        float fs[2] = {0.5f, 0.3f};
        for (int i = 0; i < 2; i++) {
            TP tp(*ops, *gops, c, W, *host, fs[i], cs.at("tp")[0], cs.at("tp")[1], budget, "f32", "f32", max_t);
            std::ofstream ot(dir + "/out_tp" + std::to_string(i) + ".bin", std::ios::binary);
            for (int tok : cs.at("steps")) {
                std::vector<float> v = ops->get(tp.step(tok));
                ot.write((const char*)v.data(), 4 * v.size());
            }
            std::printf("split %.0f%%: card A key heads %d, GPU %d; slot hits %ld/%ld\n", 100 * fs[i], tp.ca.lin_k_heads,
                        tp.cb.lin_k_heads, tp.hits, tp.lookups);
        }
        TP tp(*ops, *gops, c, W, *host, 0.3f, cs.at("tp")[0], cs.at("tp")[1], budget, "f32", "f32", max_t);
        if (cs.count("mtp")) {
            Weights MW = Weights::load(dir + "/mtp.bin");
            MTPDrafter d(*ops, c, MW, *host, "f32", "f32", slots, budget, max_t);
            write("out_tp_mtp.txt", tp.generate(cs.at("prompt"), n, &d, cs.at("mtp")[0]), tp.steps);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && !std::strcmp(argv[1], "test")) {
            std::string store, fmt = "bfp4";
            for (int i = 3; i + 2 < argc + 1; i++)
                if (!std::strcmp(argv[i], "--store") && i + 2 < argc) { store = argv[i + 1]; fmt = argv[i + 2]; }
            return test(argv[2], store, fmt);
        }
        if (argc >= 2 && !std::strcmp(argv[1], "run")) {
            std::map<std::string, std::string> a;
            for (int i = 2; i < argc; i++) {
                std::string k = argv[i];
                if (k.rfind("--", 0)) continue;
                bool flag = i + 1 >= argc || !std::strncmp(argv[i + 1], "--", 2);
                a[k.substr(2)] = flag ? "1" : argv[++i];
            }
            return run(a);
        }
        std::fprintf(stderr, "usage: tuff_engine run --model <snapshot> --store <dir> --ids \"...\" [--n N] [--mtp K]"
                             " [--ttnn] [--gpu]\n       tuff_engine test <dir> [--store <dir> bfp4]\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
