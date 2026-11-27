// PCC bridge for the pinned CUDA FP16 Inkling profile.
#include "ink_backend.h"
#define INKLING_DEFINE_STBI
#include "ink_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace {
constexpr int default_capacity = 8192;
constexpr int chunk = 512;
constexpr int max_image_rows = 1024;
thread_local std::string error;

struct Model {
    ink2::InkModelFile file;
    std::unique_ptr<ink2::InkBackend> backend;
    ink2::CpuVisionEncoder vision;
    std::vector<float> lora_a, lora_b;
    int capacity = default_capacity;
    bool dspark = false;
    int lora_rank = 0;
    uint64_t work = 0;
    bool leased = false;
    int pinned_length = 0;
    int exported_length = 0;
};

struct ImageEmbedding {
    std::vector<float> values;
    int width;
    int height;
};

struct Lease {
    Model& model;
    std::vector<float> logits;
    int length = 0;
    bool frontier = false;
};
}

extern "C" {
const char* ctt_error() { return error.c_str(); }

void* ctt_load_profile(const char* path, int capacity, bool dspark) try {
    if (capacity < default_capacity || capacity > 133120) { error = "Invalid Inkling context capacity"; return nullptr; }
    auto model = std::make_unique<Model>();
    model->capacity = capacity;
    if (!model->file.open(path, error, false, "")) return nullptr;
    ink2::BackendOpts opts;
    opts.max_ctx = capacity;
    opts.chunk = chunk;
    opts.chunk_auto = false;
    opts.enable_dspark = dspark;
    model->dspark = dspark;
    opts.kv_dtype_auto = false;
    opts.kv_offload = "off";
    model->backend = ink2::make_cuda_backend();
    if (!model->backend->init(model->file, opts, error)) return nullptr;
    if (model->backend->effective_max_ctx() != capacity ||
        model->backend->effective_chunk() != chunk) {
        error = "Inkling CUDA runtime did not honor the pinned capacity or chunk";
        return nullptr;
    }
    return model.release();
} catch (const std::exception& e) {
    error = e.what();
    return nullptr;
}

void* ctt_load_capacity(const char* path, int capacity) { return ctt_load_profile(path, capacity, false); }

void* ctt_load(const char* path) { return ctt_load_capacity(path, default_capacity); }

void ctt_unload(void* model) { delete static_cast<Model*>(model); }
int ctt_vocab(void* model) { return static_cast<Model*>(model)->file.m.c.vocab_unpadded; }
uint64_t ctt_model_work(void* model) { return static_cast<Model*>(model)->work; }

int ctt_meta(void*, const char* key, char* value, size_t size) {
    const std::string_view name(key);
    const std::string_view result = name == "general.architecture" ? "inkling2_loop" : "";
    if (result.empty() || size <= result.size()) return -1;
    std::memcpy(value, result.data(), result.size());
    value[result.size()] = 0;
    return int(result.size());
}

// This bridge consumes canonical token IDs. Text rendering is owned by the frontend.
int ctt_piece(void*, int, char*, int) { return -1; }

bool ctt_eog(void* model, int token) {
    const auto& config = static_cast<Model*>(model)->file.m.c;
    return token == config.eos_token_id || token == config.eot_token_id;
}

void* ctt_new(void* raw) try {
    auto& model = *static_cast<Model*>(raw);
    if (model.leased) {
        error = "Inkling CUDA slot already leased";
        return nullptr;
    }
    auto lease = std::make_unique<Lease>(Lease{model, std::vector<float>(ctt_vocab(raw))});
    if (!model.pinned_length && !model.backend->alloc_slot(0, model.capacity, ink2::KvDtype::FP16, error)) return nullptr;
    model.leased = true;
    return lease.release();
} catch (const std::exception& e) {
    error = e.what();
    return nullptr;
}

void ctt_free(void* raw) {
    std::unique_ptr<Lease> lease(static_cast<Lease*>(raw));
    if (!lease->model.pinned_length) lease->model.backend->free_slot(0);
    lease->model.leased = false;
}

int ctt_position(void* raw) { return static_cast<Lease*>(raw)->length - 1; }
uint64_t ctt_work(void* raw) { return static_cast<Lease*>(raw)->model.work; }

static int consume(void* raw, const int32_t* ids, const int32_t* image_rows,
                      const float* embeddings, int rows, int count,
                      float* training_hidden = nullptr) try {
    auto& lease = *static_cast<Lease*>(raw);
    if (!ids || count < 1 || count > lease.model.capacity - lease.length ||
        rows < 0 || rows > count || (rows && (!embeddings || !image_rows)) ||
        std::any_of(ids, ids + count, [&](int token) {
            return token < 0 || size_t(token) >= lease.logits.size();
        })) {
        error = "Invalid Inkling tokens, image rows, or context length";
        return 1;
    }
    const int pad = lease.model.file.m.c.image_token_id;
    int seen = 0;
    if (image_rows) {
        for (int i = 0; i < count; ++i) {
            if (image_rows[i] == -1) continue;
            if (image_rows[i] != seen || seen >= rows || ids[i] != pad) {
                error = "Invalid ordered Inkling image embedding placement";
                return 1;
            }
            ++seen;
        }
        if (seen != rows || image_rows[count - 1] != -1) {
            error = "Image embeddings require a trailing text token";
            return 1;
        }
    }
    lease.model.exported_length = 0;
    if (!lease.length && lease.model.pinned_length) {
        lease.model.backend->reset_slot(0);
        lease.model.pinned_length = 0;
    }
    lease.frontier = false;
    for (int offset = 0; offset < count;) {
        ink2::StepBatch batch;
        batch.capture_taps = lease.model.dspark;
        batch.img_embeds = embeddings;
        const int take = std::min(chunk, count - offset);
        for (int i = 0; i < take; ++i) {
            const int index = offset + i;
            batch.push(0, lease.length + i, ids[index], index + 1 == count,
                       image_rows ? image_rows[index] : -1);
        }
        float* frontier = offset + take == count ? lease.logits.data() : nullptr;
        const auto& model = lease.model;
        if (frontier && (training_hidden || model.lora_rank)) {
            const int width = model.file.m.c.hidden;
            std::vector<float> scratch(training_hidden ? 0 : width);
            float* hidden = training_hidden ? training_hidden : scratch.data();
            if (!model.backend->forward_head_features(batch, frontier, hidden, error)) return 1;
            if (!training_hidden) {
                std::vector<float> projected(model.lora_rank, 0.0f);
                for (int r = 0; r < model.lora_rank; ++r)
                    for (int i = 0; i < width; ++i)
                        projected[r] += model.lora_a[size_t(r) * width + i] * hidden[i];
                const float cap = model.file.m.c.logit_softcap;
                for (size_t token = 0; token < lease.logits.size(); ++token) {
                    float delta = 0.0f;
                    for (int r = 0; r < model.lora_rank; ++r)
                        delta += model.lora_b[token * model.lora_rank + r] * projected[r];
                    frontier[token] = cap * std::tanh((frontier[token] + delta) / cap);
                }
            }
        } else if (!model.backend->forward_batch(batch, frontier, nullptr, error)) return 1;
        lease.length += take;
        lease.model.work += take;
        offset += take;
    }
    lease.frontier = training_hidden == nullptr;
    return 0;
} catch (const std::exception& e) {
    error = e.what();
    return 1;
}

int ctt_consume_mixed(void* raw, const int32_t* ids, const int32_t* image_rows,
                      const float* embeddings, int rows, int count) {
    return consume(raw, ids, image_rows, embeddings, rows, count);
}

int ctt_head_features(void* raw, const int32_t* ids, int count,
                      float* hidden, float* logits) {
    auto& lease = *static_cast<Lease*>(raw);
    if (!hidden || !logits || lease.model.lora_rank) return 1;
    if (consume(raw, ids, nullptr, nullptr, 0, count, hidden)) return 1;
    std::copy(lease.logits.begin(), lease.logits.end(), logits);
    return 0;
}

float ctt_logit_softcap(void* raw) {
    return static_cast<Model*>(raw)->file.m.c.logit_softcap;
}

int ctt_attach_head_lora(void* raw, int rank, const float* a, const float* b) try {
    auto& model = *static_cast<Model*>(raw);
    if (model.leased || model.work || model.lora_rank || rank < 1 || rank > 16 || !a || !b) return 1;
    const size_t a_count = size_t(rank) * model.file.m.c.hidden;
    const size_t b_count = size_t(rank) * model.file.m.c.vocab_unpadded;
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(a, a + a_count, finite) || !std::all_of(b, b + b_count, finite)) return 1;
    model.lora_a.assign(a, a + a_count);
    model.lora_b.assign(b, b + b_count);
    model.lora_rank = rank;
    return 0;
} catch (const std::exception& e) {
    error = e.what();
    return 1;
}

int ctt_consume(void* raw, const int32_t* ids, int count) {
    return ctt_consume_mixed(raw, ids, nullptr, nullptr, 0, count);
}

int ctt_image_pad(void* raw) {
    return static_cast<Model*>(raw)->file.m.c.image_token_id;
}

void* ctt_encode_image(void* raw, const char* tower, const char* path) try {
    auto& model = *static_cast<Model*>(raw);
    ink2::VisionShardInfo metadata;
    ink2::ImageRec picture;
    if (!ink2::vision_probe(tower, metadata, error) ||
        !ink2::prepare_picture(path, metadata.c, picture, error)) return nullptr;
    if (picture.num_lm_tokens() < 1 || picture.num_lm_tokens() > max_image_rows) {
        error = "Inkling vision row limit exceeded";
        return nullptr;
    }
    auto image = std::make_unique<ImageEmbedding>();
    image->width = picture.grid_w;
    image->height = picture.grid_h;
    if (!model.backend->encode_image(picture, image->values, error)) {
        if (!model.vision.load(tower, error) ||
            !model.vision.encode(picture, image->values, error)) return nullptr;
    }
    if (image->values.size() != size_t(picture.num_lm_tokens()) * model.file.m.c.hidden) {
        error = "Unexpected Inkling vision embedding dimensions";
        return nullptr;
    }
    return image.release();
} catch (const std::exception& e) {
    error = e.what();
    return nullptr;
}

const float* ctt_image_data(void* raw, int* width, int* height, size_t* size) {
    const auto& image = *static_cast<ImageEmbedding*>(raw);
    *width = image.width;
    *height = image.height;
    *size = image.values.size() * sizeof(float);
    return image.values.data();
}

void ctt_free_image(void* raw) { delete static_cast<ImageEmbedding*>(raw); }

const float* ctt_logits(void* raw) {
    auto& lease = *static_cast<Lease*>(raw);
    return lease.frontier ? lease.logits.data() : nullptr;
}

int ctt_select_greedy(void* raw) {
    const auto& lease = *static_cast<Lease*>(raw);
    if (!lease.frontier || !std::all_of(lease.logits.begin(), lease.logits.end(),
                                      [](float value) { return std::isfinite(value); })) {
        error = "Inkling greedy selection requires finite frontier logits";
        return -1;
    }
    return int(std::max_element(lease.logits.begin(), lease.logits.end()) - lease.logits.begin());
}

// Return consumed tokens with a full verifier frontier at the last token.
int ctt_draft_block(void* raw, int remaining, int32_t* output, int* verified_nodes) try {
    auto& lease = *static_cast<Lease*>(raw);
    if (!lease.model.dspark || !output || !verified_nodes || remaining < 1) return -1;
    *verified_nodes = 0;
    const int anchor = ctt_select_greedy(raw);
    if (anchor < 0) return -1;
    output[0] = anchor;
    int n = 0;
    if (remaining > 1 && !ctt_eog(&lease.model, anchor)) {
        n = lease.model.backend->dspark_spec_step_greedy(0, anchor, lease.length,
            std::min(8, remaining - 1), output + 1, verified_nodes, error);
        if (n < 0) return -1;
    }
    int last = n;
    for (int i = 1; i < n; ++i) {
        if (ctt_eog(&lease.model, output[i])) {
            last = i;
            lease.model.backend->rollback_slot(0, lease.length + last);
            break;
        }
    }
    lease.length += last;
    lease.model.work += last;
    if (ctt_consume(raw, output + last, 1)) return -1;
    return last + 1;
} catch (const std::exception& e) {
    error = e.what();
    return -1;
}

size_t ctt_state_size(void* raw) {
    auto& lease = *static_cast<Lease*>(raw);
    return lease.frontier ? lease.model.backend->portable_state_size(0, lease.length) : 0;
}

size_t ctt_export(void* raw, void* target, size_t size) {
    auto& lease = *static_cast<Lease*>(raw);
    if (!lease.frontier || !lease.model.backend->export_portable_state(0, lease.length, target, size)) return 0;
    lease.model.exported_length = lease.length;
    return size;
}

// A resident parent keeps its append-only global KV on the device. The backend
// saves the ShortConv and sliding-window rows overwritten by later questions.
int ctt_pin_shared(void* raw) try {
    auto& lease = *static_cast<Lease*>(raw);
    if (!lease.frontier || lease.model.pinned_length ||
        !lease.model.backend->supports_shared_checkpoint() ||
        !lease.model.backend->save_shared_checkpoint(0, lease.length)) return 0;
    lease.model.pinned_length = lease.length;
    return 1;
} catch (const std::exception& e) { error = e.what(); return 0; }

// The latest exported child still occupies slot zero across leases while its parent is pinned.
int ctt_reuse_exported(void* raw, int length, const float* frontier, size_t count) {
    auto& lease = *static_cast<Lease*>(raw);
    if (!lease.model.pinned_length || length <= 0 || length != lease.model.exported_length ||
        !frontier || count != lease.logits.size()) return 0;
    std::copy_n(frontier, count, lease.logits.data());
    lease.length = length;
    lease.frontier = true;
    return 1;
}

int ctt_reuse_shared(void* raw, int length, const float* frontier, size_t count) try {
    auto& lease = *static_cast<Lease*>(raw);
    if (length <= 0 || length != lease.model.pinned_length || !frontier ||
        count != lease.logits.size()) return 0;
    lease.model.exported_length = 0;
    if (!lease.model.backend->restore_shared_checkpoint(0, length)) return 0;
    std::copy_n(frontier, count, lease.logits.data());
    lease.length = length;
    lease.frontier = true;
    return 1;
} catch (const std::exception& e) { error = e.what(); return 0; }

// PCC provides the authenticated position and separately stored frontier.
size_t ctt_import_checkpoint(void* raw, const void* data, size_t size, int length,
                             const float* frontier, size_t count) try {
    auto& lease = *static_cast<Lease*>(raw);
    if (!frontier || count != lease.logits.size() || !size ||
        size != lease.model.backend->portable_state_size(0, length)) return 0;
    lease.frontier = false;
    lease.model.pinned_length = 0;
    lease.model.exported_length = 0;
    if (!lease.model.backend->import_portable_state(0, length, data, size)) return 0;
    std::copy_n(frontier, count, lease.logits.data());
    lease.length = length;
    lease.frontier = true;
    return size;
} catch (const std::exception& e) {
    error = e.what();
    return 0;
}

size_t ctt_import(void*, const void*, size_t) { return 0; }
}
