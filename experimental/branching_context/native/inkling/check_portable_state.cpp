// GPU regression for the native payload. PCC owns checkpoint identity and integrity.
#include "ink_backend.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace ink2;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static void same_logits(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "full-vocabulary logits differ");
}

int main(int argc, char** argv) try {
    require(argc == 4, "usage: check_portable_state MODEL write|read FILE");
    const std::string mode = argv[2];
    require(mode == "write" || mode == "read", "invalid mode");
    constexpr int prefix = 4097;
    BackendOpts opts;
    opts.max_ctx = 8192;
    opts.chunk = 512;
    opts.chunk_auto = false;
    opts.enable_dspark = false;
    opts.kv_dtype_auto = false;
    opts.kv_offload = "off";

    InkModelFile model;
    std::string error;
    require(model.open(argv[1], error, false, ""), error);
    auto backend = make_cuda_backend();
    require(backend->init(model, opts, error), error);
    require(backend->alloc_slot(0, opts.max_ctx, opts.kv_dtype, error), error);
    std::vector<float> logits(model.m.c.vocab_unpadded);
    std::vector<float> expected(logits.size());
    auto forward = [&](int start, int count, int offset) {
        StepBatch batch;
        for (int i = 0; i < count; ++i) {
            const int position = start + i;
            batch.push(0, position, 100 + (position + offset) % 1000, i == count - 1);
        }
        require(backend->forward_batch(batch, logits.data(), nullptr, error), error);
    };

    std::vector<char> state(backend->portable_state_size(0, prefix));
    require(!state.empty(), "portable state unavailable");
    if (mode == "write") {
        for (int start = 0; start < prefix; start += opts.chunk)
            forward(start, std::min(opts.chunk, prefix - start), 0);
        require(backend->export_portable_state(0, prefix, state.data(), state.size()), "export failed");
        forward(prefix, 1, 0);
        expected = logits;
        require(backend->import_portable_state(0, prefix, state.data(), state.size()), "RAM restore failed");
        forward(prefix, 2, 191);   // write to a sibling continuation
        require(backend->import_portable_state(0, prefix, state.data(), state.size()), "parent restore failed");
        forward(prefix, 1, 0);
        same_logits(logits, expected);

        std::ofstream out(argv[3], std::ios::binary);
        out.write(state.data(), state.size());
        out.write(reinterpret_cast<const char*>(expected.data()), expected.size() * sizeof(float));
        out.close();
        require(bool(out), "fixture write failed");
    } else {
        std::ifstream in(argv[3], std::ios::binary);
        in.read(state.data(), state.size());
        in.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float));
        require(bool(in) && in.peek() == EOF, "fixture size mismatch");
        require(!backend->import_portable_state(0, prefix, state.data(), state.size() - 1),
                "truncated state accepted");
        require(backend->import_portable_state(0, prefix, state.data(), state.size()), "fresh restore failed");
        forward(prefix, 1, 0);
        same_logits(logits, expected);
    }
    std::cout << mode << ": " << state.size() << " state bytes, " << logits.size()
              << " exact continuation logits\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
