// Exercise the optional FP16 parent-suffix exporter against full native images.
#include "ink_backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    using namespace ink2;
    check(argc >= 4, "usage: check_parent_suffix MODEL write|read FIXTURE [--wrapped] [--gpu-experts] [--multi-step]");
    bool wrapped = false, gpu_experts = false;
    int steps = 1;
    for (int i = 4; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--wrapped") wrapped = true;
        else if (flag == "--gpu-experts") gpu_experts = true;
        else if (flag == "--multi-step") steps = 32;
        else throw std::runtime_error("unknown argument: " + flag);
    }
    std::string mode = argv[2];
    check(mode == "write" || mode == "read", "invalid fixture mode");
    BackendOpts opts;
    opts.max_ctx = 8192;
    opts.chunk = 512;
    opts.chunk_auto = false;
    opts.enable_dspark = false;
    opts.kv_dtype_auto = false;
    opts.kv_offload = "off";
    opts.vram_experts_gb = gpu_experts ? 20 : 3;
    InkModelFile model;
    std::string error;
    check(model.open(argv[1], error, false, ""), error);
    auto backend = make_cuda_backend();
    check(backend->init(model, opts, error), error);
    if (gpu_experts) {
        const auto stats = backend->stats();
        check(stats.total_experts > 0 && stats.vram_experts_loaded == stats.total_experts &&
              stats.ram_experts_loaded == 0, "GPU expert residency requirement failed");
        std::cout << "GPU-resident experts: " << stats.vram_experts_loaded << '\n';
    }
    check(backend->alloc_slot(0, opts.max_ctx, opts.kv_dtype, error), error);
    std::vector<float> logits(model.m.c.vocab_unpadded);
    check(logits.size() == 200058, "unexpected vocabulary");
    auto forward = [&](int start, int end) {
        while (start < end) {
            StepBatch batch;
            const int stop = std::min(start + opts.chunk, end);
            for (; start < stop; ++start)
                batch.push(0, start, 100 + start % 1000, start + 1 == stop);
            check(backend->forward_batch(batch, logits.data(), nullptr, error), error);
        }
    };
    auto full = [&](int length) {
        std::vector<char> image(backend->portable_state_size(0, length));
        check(!image.empty() && backend->export_portable_state(0, length, image.data(), image.size()), "full export failed");
        return image;
    };
    const int parent = wrapped ? 1537 : 1017;
    const int child = parent + (wrapped ? 1027 : 18);
    std::vector<char> base(backend->portable_state_size(0, parent));
    std::vector<char> suffix(backend->portable_suffix_size(0, parent, child));
    std::vector<float> reference(steps * logits.size());
    auto continuation = [&](bool capture) {
        for (int step = 0; step < steps; ++step) {
            forward(child + step, child + step + 1);
            float* row = reference.data() + step * logits.size();
            if (capture) std::copy(logits.begin(), logits.end(), row);
            else if (std::memcmp(row, logits.data(), logits.size() * sizeof(float)) != 0)
                throw std::runtime_error("continuation mismatch at step " + std::to_string(step + 1));
        }
    };
    std::vector<char> expected;
    check(!base.empty() && !suffix.empty() && suffix.size() < backend->portable_state_size(0, child), "invalid fixture sizes");
    auto transfer = [&](std::fstream& file, auto& values) {
        auto* data = reinterpret_cast<char*>(values.data());
        const auto bytes = values.size() * sizeof(values.front());
        if (mode == "write") file.write(data, bytes);
        else file.read(data, bytes);
        check(bool(file), "fixture transfer failed");
    };
    if (mode == "write") {
        forward(0, parent);
        base = full(parent);
        forward(parent, child);
        expected = full(child);
        check(!backend->export_portable_suffix(0, parent, child, suffix.data(), suffix.size()-1), "accepted truncated buffer");
        check(backend->export_portable_suffix(0, parent, child, suffix.data(), suffix.size()), "suffix export failed");
        continuation(true);
    }
    {
        std::fstream file(argv[3], std::ios::binary | (mode == "write" ? std::ios::out | std::ios::trunc : std::ios::in));
        transfer(file, base);
        transfer(file, suffix);
        transfer(file, reference);
        if (mode == "read") check(file.peek() == std::char_traits<char>::eof(), "trailing fixture bytes");
        else { file.close(); check(bool(file), "fixture flush failed"); }
    }
    constexpr size_t conv = 48 * 3 * 6656 * sizeof(float), row = 2048;
    auto assembled = std::vector<char>(suffix.begin(), suffix.begin() + conv);
    size_t parent_pos = conv, suffix_pos = conv;
    for (int visit = 0; visit < 48; ++visit) {
        const int layer = visit < 6 ? visit : visit < 42 ? 6 + (visit - 6) % 12 : visit - 24;
        const bool local = model.m.c.is_swa[layer];
        const int child_rows = local ? std::min(child, 1024) : child;
        const int parent_rows = local ? std::min(parent, 1024) : parent;
        const int added = std::min(child_rows, child - parent);
        const int retained = child_rows - added;
        for (int kv = 0; kv < 2; ++kv) {
            size_t parent_end = parent_pos + parent_rows * row;
            size_t suffix_end = suffix_pos + added * row;
            check(parent_end <= base.size() && suffix_end <= suffix.size(), "range outside payload");
            assembled.insert(assembled.end(), base.begin()+parent_end-retained*row, base.begin()+parent_end);
            assembled.insert(assembled.end(), suffix.begin()+suffix_pos, suffix.begin()+suffix_end);
            parent_pos = parent_end;
            suffix_pos = suffix_end;
        }
    }
    check(parent_pos == base.size() && suffix_pos == suffix.size(), "fixture size mismatch");
    if (mode == "write") check(assembled == expected, "reconstructed bytes differ");
    check(std::all_of(reference.begin(), reference.end(), [](float v){ return std::isfinite(v); }), "nonfinite logits");
    check(backend->import_portable_state(0, parent, base.data(), base.size()), "parent import failed");
    check(backend->import_portable_suffix(0, parent, child, suffix.data(), suffix.size()), "direct suffix import failed");
    continuation(false);
    check(backend->import_portable_state(0, child, assembled.data(), assembled.size()), "import failed");
    continuation(false);
    std::cout << "Parent " << parent << ", child " << child << ", suffix " << suffix.size()
              << " bytes, full " << assembled.size() << " bytes, exact continuation " << reference.size() << " logits\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
