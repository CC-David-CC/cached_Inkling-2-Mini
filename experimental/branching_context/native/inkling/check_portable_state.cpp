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
    const bool draft = argc == 5 && std::string(argv[4]) == "--dspark";
    const bool layer_split = argc == 5 && std::string(argv[4]) == "--three-gpu";
    require(argc == 4 || draft || layer_split,
            "usage: check_portable_state MODEL write|read FILE [--dspark|--three-gpu]");
    const std::string mode = argv[2];
    require(mode == "write" || mode == "read", "invalid mode");
    constexpr int prefix = 4097;
    BackendOpts opts;
    if (layer_split) {
        opts.gpus = 3;
        opts.gpu_ids = {0, 1, 2};
        opts.layer_split = "auto";
    }
    opts.max_ctx = 8192;
    opts.chunk = 512;
    opts.chunk_auto = false;
    opts.enable_dspark = draft;
    opts.kv_dtype_auto = false;
    opts.kv_offload = "off";

    InkModelFile model;
    std::string error;
    require(model.open(argv[1], error, false, ""), error);
    auto backend = make_cuda_backend();
    require(backend->init(model, opts, error), error);
    if (layer_split)
        require(std::string(backend->backend_name()) == "cuda-ls", "three-GPU layer split unavailable");
    require(backend->alloc_slot(0, opts.max_ctx, opts.kv_dtype, error), error);
    std::vector<float> logits(model.m.c.vocab_unpadded);
    std::vector<float> expected(logits.size());
    auto forward = [&](int start, int count, int offset) {
        StepBatch batch;
        batch.capture_taps = draft;
        for (int i = 0; i < count; ++i) {
            const int position = start + i;
            batch.push(0, position, 100 + (position + offset) % 1000, i == count - 1);
        }
        require(backend->forward_batch(batch, logits.data(), nullptr, error), error);
    };

    if (draft) {
        // The saved state must be sufficient to draft immediately in a new process.
        int position = prefix, anchor = 0;
        std::fstream fixture(argv[3], std::ios::binary |
                             (mode == "write" ? std::ios::out | std::ios::trunc : std::ios::in));
        require(bool(fixture), "cannot open fixture");
        auto transfer = [&](void* data, size_t bytes) {
            if (mode == "write") fixture.write(static_cast<const char*>(data), bytes);
            else fixture.read(static_cast<char*>(data), bytes);
            require(bool(fixture), "fixture transfer failed");
        };
        int32_t emitted[8];
        int drafted = 0;
        auto step = [&] {
            int count = backend->dspark_spec_step_greedy(0, anchor, position, 8, emitted, &drafted, error);
            require(count >= 1 && count <= 8, "fused drafting unavailable");
            return count;
        };
        if (mode == "write") {
            for (int start = 0; start < prefix; start += opts.chunk)
                forward(start, std::min(opts.chunk, prefix - start), 0);
            anchor = std::max_element(logits.begin(), logits.end()) - logits.begin();
            bool rewound = false;
            for (int attempt = 0; attempt < 64 && !rewound; ++attempt) {
                int count = step();
                rewound = count >= 2 && drafted + 1 > count;
                if (rewound) {
                    count /= 2;
                    backend->rollback_slot(0, position + count);
                }
                position += count;
                anchor = emitted[count - 1];
            }
            require(rewound, "fixture did not exercise partial tree rewind");
        }
        transfer(&position, sizeof(position));
        transfer(&anchor, sizeof(anchor));
        require(position >= prefix && position <= prefix + 512 && anchor >= 0 &&
                size_t(anchor) < logits.size(), "invalid fixture boundary");
        std::vector<char> image(backend->portable_state_size(0, position));
        require(!image.empty(), "draft checkpoint unavailable");
        if (mode == "write")
            require(backend->export_portable_state(0, position, image.data(), image.size()), "draft export failed");
        transfer(image.data(), image.size());
        if (mode == "read")
            require(backend->import_portable_state(0, position, image.data(), image.size()), "draft import failed");
        auto compare = [&](const void* value, size_t bytes) {
            std::vector<char> stored(bytes);
            std::memcpy(stored.data(), value, bytes);
            transfer(stored.data(), bytes);
            require(std::memcmp(value, stored.data(), bytes) == 0, "draft continuation differs");
        };
        for (int i = 0; i < 16; ++i) {
            int count = step();
            compare(&count, sizeof(count));
            compare(&drafted, sizeof(drafted));
            compare(emitted, count * sizeof(int32_t));
            position += count;
            StepBatch batch;
            batch.capture_taps = true;
            batch.push(0, position++, emitted[count - 1], true);
            require(backend->forward_batch(batch, logits.data(), nullptr, error), error);
            compare(logits.data(), logits.size() * sizeof(float));
            anchor = std::max_element(logits.begin(), logits.end()) - logits.begin();
        }
        if (mode == "read") require(fixture.peek() == EOF, "trailing fixture bytes");
        fixture.close();
        require(!fixture.bad(), "fixture close failed");
        std::cout << mode << ": " << image.size() << " bytes, 16 fused rounds after rewind" << std::endl;
        return 0;
    }
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
