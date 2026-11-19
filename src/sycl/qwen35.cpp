// SPDX-License-Identifier: Apache-2.0
#include "attention.hpp"
#include "format.hpp"
#include "signs.hpp"
#include "yskn_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace yskn::sycl_backend {
namespace {
namespace graph = sycl::ext::oneapi::experimental;

sycl::device cuda_device(int index) {
  if (index < 0) throw std::runtime_error("GPU index must be non-negative");
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_cuda) continue;
    for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
      if (index-- == 0) return device;
    }
  }
  throw std::runtime_error("SYCL CUDA GPU index unavailable; check the DPC++ CUDA adapter and NVIDIA driver");
}

class DeviceMemory {
public:
  explicit DeviceMemory(sycl::queue& queue) : queue_(queue) {}
  ~DeviceMemory() {
    try { queue_.wait(); } catch (...) {}
    for (auto* p : allocations_) {
      try { sycl::free(p, queue_); } catch (...) {}
    }
  }

  template <class T> T* allocate(size_t count) {
    T* p = sycl::malloc_device<T>(count, queue_);
    if (!p) throw std::runtime_error("SYCL device allocation failed");
    try { allocations_.push_back(p); }
    catch (...) { sycl::free(p, queue_); throw; }
    return p;
  }
  template <class T> T* copy(const T* from, size_t count) {
    T* p = allocate<T>(count);
    queue_.copy(from, p, count).wait_and_throw();
    return p;
  }

private:
  sycl::queue& queue_;
  std::vector<void*> allocations_;
};

template <class T>
const T* payload(const Container& pack, uint64_t off, size_t count, const std::string& name) {
  if (off % alignof(T) || off > pack.len || count > (pack.len - off) / sizeof(T))
    throw std::runtime_error("invalid YSKN payload range: " + name);
  return pack.at<T>(off);
}

template <int Bits>
std::vector<uint32_t> pack_lnord(Codes source, size_t count) {
  std::vector<uint32_t> packed(count);
  auto* dst = packed.data();
  // Match the CUDA LNORD stream: sixteen output rows per warp and four
  // consecutive 64-column fragments per row in each 256-column sequence.
  for (int row = 0; row < source.rows; row += 16) {
    for (int seq = 0; seq < source.cols / 256; ++seq) {
      for (int chunk = 0; chunk < Bits / 2; ++chunk) {
        for (int lane = 0; lane < 32; ++lane) {
          for (int w = 0; w < 4; ++w) {
            const int field = 4 * chunk + w;
            const int output = row + lane / 4 + 8 * (field / Bits);
            const int index = (lane % 4) * Bits + field % Bits;
            *dst++ = source.word(output, seq, index);
          }
        }
      }
    }
  }
  return packed;
}

struct Layer {
  bool full = false;
  FullAttention attention;
  LinearAttention linear;
  Projection gate, up, down;
  const uint16_t *norm1 = nullptr, *norm2 = nullptr;
};

class Engine {
public:
  Engine(const char* path, int context, int index)
      : queue_(cuda_device(index), [](sycl::exception_list errors) {
          if (errors.begin() != errors.end()) std::rethrow_exception(*errors.begin());
        }, sycl::property::queue::in_order{}),
        mem_(queue_), context_(context) {
    // Bound attention score scratch space to 32 MiB for large contexts.
    constexpr int score_elements = 8 << 20;
    rows_ = std::min({prefill_rows(), context_, std::max(1, score_elements / query_heads / context_)});
    packed_decode_ = !std::getenv("YSKN_SYCL_REFERENCE_DECODE");
    tensor_decode_ = packed_decode_ && !std::getenv("YSKN_SYCL_DP4A_DECODE");
    graph_decode_ = packed_decode_ && !std::getenv("YSKN_SYCL_EAGER_DECODE") &&
                    queue_.get_device().has(sycl::aspect::ext_oneapi_limited_graph);
    Container pack;
    pack.open(path);
    const auto& cfg = pack.model;
    const auto& model = pack.raw.at("model");
    if (cfg.arch != "qwen3_5" || cfg.hidden != hidden || cfg.ffn != ffn || cfg.layers != layers_count
        || cfg.vocab != vocab || cfg.heads != query_heads || cfg.kv_heads != kv_heads || cfg.head_dim != head_dim
        || model.at("linear_num_key_heads") != linear_keys || model.at("linear_num_value_heads") != linear_values
        || model.at("linear_key_head_dim") != linear_dim || model.at("linear_value_head_dim") != linear_dim
        || model.at("linear_conv_kernel_dim") != 4 || model.at("rotary_dim") != rotary_dim
        || !model.at("attn_output_gate").get<bool>() || model.at("norm_offset") != 1
        || !std::isfinite(cfg.eps) || cfg.eps <= 0 || !std::isfinite(cfg.rope_theta) || cfg.rope_theta <= 0)
      throw std::runtime_error(std::string("unsupported model shape; expected a ") + model_name + " text pack");

    eps_ = cfg.eps;
    theta_ = cfg.rope_theta;
    const auto& types = model.at("layer_types");
    if (!types.is_array() || types.size() != layers_count || std::count(types.begin(), types.end(), "full_attention") != 8)
      throw std::runtime_error(std::string("invalid layer_types in ") + model_name + " pack");
    const size_t kv_count = size_t(context_) * kv_heads * head_dim;
    for (int i = 0; i < layers_count; ++i) {
      auto& layer = layers_[i];
      const auto p = "layers." + std::to_string(i) + ".";
      layer.full = types.at(i) == "full_attention";
      if (layer.full) {
        const auto prefix = p + "self_attn.";
        auto& a = layer.attention;
        a.q = projection(pack, prefix + "q_proj", 2 * query_heads * head_dim, hidden);
        a.k = projection(pack, prefix + "k_proj", kv_heads * head_dim, hidden);
        a.v = projection(pack, prefix + "v_proj", kv_heads * head_dim, hidden);
        a.out = projection(pack, prefix + "o_proj", hidden, query_heads * head_dim);
        a.qn = vector(pack, prefix + "q_norm", head_dim);
        a.kn = vector(pack, prefix + "k_norm", head_dim);
        a.keys = mem_.allocate<uint16_t>(kv_count);
        a.values = mem_.allocate<uint16_t>(kv_count);
      } else {
        if (types.at(i) != "linear_attention") throw std::runtime_error("unsupported layer type");
        const auto prefix = p + "linear_attn.";
        auto& a = layer.linear;
        a.mixed = projection(pack, prefix + "in_proj_qkv", mixed_size, hidden);
        a.z = projection(pack, prefix + "in_proj_z", linear_width, hidden);
        a.out = projection(pack, prefix + "out_proj", hidden, linear_width);
        a.a = vector(pack, prefix + "in_proj_a", linear_values * hidden);
        a.b = vector(pack, prefix + "in_proj_b", linear_values * hidden);
        a.conv = vector(pack, prefix + "conv1d", mixed_size * 4);
        a.a_log = vector(pack, prefix + "A_log", linear_values);
        a.dt = vector(pack, prefix + "dt_bias", linear_values);
        a.norm = vector(pack, prefix + "norm", linear_dim);
        a.state = mem_.allocate<float>(linear_values * linear_dim * linear_dim);
        a.history = mem_.allocate<uint16_t>(mixed_size * 3);
      }
      layer.gate = projection(pack, p + "mlp.gate_proj", ffn, hidden);
      layer.up = projection(pack, p + "mlp.up_proj", ffn, hidden);
      layer.down = projection(pack, p + "mlp.down_proj", hidden, ffn);
      layer.norm1 = vector(pack, p + "input_layernorm", hidden);
      layer.norm2 = vector(pack, p + "post_attention_layernorm", hidden);
    }

    const auto& e = pack.tensor("embed_tokens");
    embed_.codes = codes(pack, e, vocab, hidden, true);
    if (e.has_rot_out) throw std::runtime_error("unexpected embedding output rotation");
    embed_.sign = signs(e.rot_in, hidden);
    embed_.lut = half_values(pack, e.lut_off, 8192, e.name);
    embed_.scale = half_values(pack, e.scale_off, vocab, e.name);
    head_ = projection(pack, "lm_head", vocab, hidden, false);
    final_norm_ = vector(pack, "norm", hidden);

    const int padded = rows_ > 1 ? padded_projection_tokens(rows_) : 1;
    hidden_ = mem_.allocate<float>(rows_ * hidden);
    normalized_ = mem_.allocate<float>(rows_ * hidden);
    projection_out_ = mem_.allocate<float>(rows_ * hidden);
    work_ = mem_.allocate<float>(rows_ * ffn);
    activations_ = mem_.allocate<int8_t>(padded * ffn);
    scales_ = mem_.allocate<float>(rows_ * (ffn / block));
    const int batch_dots = rows_ > 1 && needs_batch_partials_ ? padded * ffn * (hidden / block) : 0;
    const int decode_dots = packed_decode_ ? vocab * (hidden / 256) : 0;
    if (batch_dots || decode_dots)
      partial_ = mem_.allocate<int32_t>(std::max(batch_dots, decode_dots));
    if (rows_ > 1) input_ids_ = mem_.allocate<int32_t>(rows_);
    first_ = mem_.allocate<float>(rows_ * ffn);
    second_ = mem_.allocate<float>(rows_ * ffn);
    third_ = mem_.allocate<float>(rows_ * ffn);
    attention_out_ = mem_.allocate<float>(rows_ * linear_width);
    q16_ = mem_.allocate<uint16_t>(rows_ * query_heads * head_dim);
    scores_ = mem_.allocate<float>(size_t(rows_) * query_heads * context_);
    convolved_ = mem_.allocate<float>(rows_ * mixed_size);
    core_ = mem_.allocate<float>(rows_ * linear_width);
    beta_ = mem_.allocate<float>(rows_ * linear_values);
    decay_ = mem_.allocate<float>(rows_ * linear_values);
    logits_ = mem_.allocate<float>(vocab);
    maxima_ = mem_.allocate<float>(head_groups);
    indices_ = mem_.allocate<int>(head_groups);
    prediction_ = mem_.allocate<int>(1);
    if (graph_decode_) decode_position_ = mem_.allocate<int32_t>(1);
    reset();
  }

  void reset() {
    position_ = 0;
    next_ = -1;
    valid_ = false;
    for (const auto& layer : layers_) {
      if (layer.full) continue; // Each accessible KV row is written before it is read.
      queue_.fill(layer.linear.state, 0.f, linear_values * linear_dim * linear_dim);
      queue_.fill(layer.linear.history, uint16_t(0), mixed_size * 3);
    }
    queue_.wait_and_throw();
    valid_ = true;
  }

  int prefill(const int32_t* ids, int n) {
    check_capacity(n);
    if (!ids) throw std::runtime_error("prefill: null token pointer");
    for (int i = 0; i < n; ++i) {
      if (ids[i] < 0 || ids[i] >= vocab) throw std::runtime_error("prefill: token outside vocabulary");
    }
    valid_ = false;
    for (int done = 0; done < n;) {
      const int rows = std::min(rows_, n - done);
      if (rows > 1) queue_.copy(ids + done, input_ids_, rows);
      forward(ids[done], rows > 1 ? input_ids_ : nullptr, rows, done + rows == n);
      done += rows;
      position_ += rows;
      // Bound the number of queued kernels and report asynchronous failures.
      if (rows_ > 1 || done % 8 == 0) queue_.wait_and_throw();
    }
    if (graph_decode_) queue_.fill(decode_position_, int32_t(position_), 1);
    next_ = read_prediction();
    valid_ = true;
    return next_;
  }

  void decode(int32_t* out, int steps) {
    check_capacity(steps);
    if (!out || next_ < 0) throw std::runtime_error("decode requires a successful prefill and an output buffer");
    valid_ = false;
    if (graph_decode_ && !decode_graph_) prepare_decode_graph();
    for (int i = 0; i < steps; ++i) {
      if (decode_graph_) queue_.ext_oneapi_graph(*decode_graph_);
      else forward(next_, nullptr, 1, true);
      next_ = read_prediction();
      out[i] = next_;
      ++position_;
    }
    valid_ = true;
  }

  void logits(float* out) {
    if (!valid_ || next_ < 0 || !out) throw std::runtime_error("no logits available or missing output buffer");
    queue_.copy(logits_, out, vocab).wait_and_throw();
  }

  int position() const { return valid_ ? position_ : -1; }

private:
  static constexpr int head_groups = 128, max_prefill_rows = 256;
  sycl::queue queue_;
  DeviceMemory mem_;
  const int context_;
  Layer layers_[layers_count];
  std::map<std::tuple<std::string, uint64_t, uint64_t, int>, const int8_t*> rotation_signs_;
  Embedding embed_;
  Projection head_;
  const uint16_t* final_norm_ = nullptr;
  int position_ = 0, next_ = -1, rows_ = 1;
  bool valid_ = false, packed_decode_ = true, tensor_decode_ = true, graph_decode_ = false;
  bool needs_batch_partials_ = false;
  std::optional<graph::command_graph<graph::graph_state::executable>> decode_graph_;
  int32_t* decode_position_ = nullptr;
  float eps_ = 0.f, theta_ = 0.f;
  float *hidden_, *normalized_, *projection_out_, *work_, *scales_;
  int8_t* activations_;
  int32_t *partial_ = nullptr, *input_ids_ = nullptr;
  float *first_, *second_, *third_, *attention_out_, *scores_;
  uint16_t* q16_;
  float *convolved_, *core_, *beta_, *decay_;
  float *logits_, *maxima_;
  int *indices_, *prediction_;

  static int prefill_rows() {
    const char* env = std::getenv("YSKN_SYCL_PREFILL_ROWS");
    if (!env) return max_prefill_rows;
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || *end || value < 1 || value > max_prefill_rows)
      throw std::runtime_error("YSKN_SYCL_PREFILL_ROWS must be an integer from 1 to 256");
    return int(value);
  }

  void check_capacity(int count) const {
    if (!valid_) throw std::runtime_error("inference failed; reset the engine before retrying");
    if (count < 1 || count > context_ - position_) throw std::runtime_error("invalid token count or context overflow");
  }

  const uint16_t* vector(const Container& pack, const std::string& name, int size) {
    const auto& desc = pack.vector(name);
    if (desc.n != size || desc.dtype != "bf16") throw std::runtime_error("unsupported vector: " + name);
    return mem_.copy(payload<uint16_t>(pack, desc.off, size, name), size);
  }

  const float* half_values(const Container& pack, uint64_t off, int size, const std::string& name) {
    const auto* source = payload<uint16_t>(pack, off, size, name);
    std::vector<float> values(size);
    for (int i = 0; i < size; ++i) values[i] = fp16(source[i]);
    return mem_.copy(values.data(), values.size());
  }

  const int8_t* signs(const RotDesc& rot, int size) {
    if (rot.kind != "H512" || rot.m != size / block ||
        rot.sign_offset > 4u * vocab)
      throw std::runtime_error("unsupported YSKN rotation");
    const auto key = std::make_tuple(rot.rule, rot.sign_seed, rot.sign_offset, size);
    if (const auto existing = rotation_signs_.find(key); existing != rotation_signs_.end())
      return existing->second;
    const auto values = sign_stream_rule(rot.rule, rot.sign_seed, rot.sign_offset, size);
    const auto* device_signs = mem_.copy(values.data(), values.size());
    rotation_signs_.emplace(key, device_signs);
    return device_signs;
  }

  Codes codes(const Container& pack, const TensorDesc& d, int rows, int cols, bool row_layout) {
    if (d.rows != rows || d.physical_rows != rows || d.cols != cols || d.L != 12 || d.has_rar
        || d.start != "zero" || d.bias_off != 0 || d.layout != (row_layout ? "row" : "warp"))
      throw std::runtime_error("unsupported coded tensor: " + d.name);
    const size_t count = size_t(rows) * (cols / 256) * (4 * d.bits);
    if (d.stream_bytes != count * sizeof(uint32_t)) throw std::runtime_error("invalid stream size: " + d.name);
    Codes stream{payload<uint32_t>(pack, d.stream_off, count, d.name), rows, cols, d.bits, row_layout};
    if (tensor_decode_ && !row_layout && rows % 16 == 0 && cols % block == 0 && (d.bits == 6 || d.bits == 8)) {
      const auto packed = d.bits == 6 ? pack_lnord<6>(stream, count) : pack_lnord<8>(stream, count);
      stream.words = mem_.copy(packed.data(), count);
      stream.lnord_layout = true;
    } else {
      stream.words = mem_.copy(stream.words, count);
    }
    return stream;
  }

  Projection projection(const Container& pack, const std::string& name, int rows, int cols, bool rotate = true) {
    const auto& d = pack.tensor(name);
    if (d.has_rot_out != rotate) throw std::runtime_error("unexpected output rotation: " + name);
    Projection out;
    out.codes = codes(pack, d, rows, cols, false);
    if (rotate && !out.codes.lnord_layout) needs_batch_partials_ = true;
    out.input_sign = signs(d.rot_in, cols);
    if (rotate) out.output_sign = signs(d.rot_out, rows);
    const auto lut = payload<uint16_t>(pack, d.lut_off, 8192, name);
    const auto scales = payload<uint16_t>(pack, d.scale_off, rows, name);
    float magnitude = 0.f;
    for (int i = 0; i < 8192; ++i) {
      const float v = fp16(lut[i]);
      if (!std::isfinite(v)) throw std::runtime_error("non-finite LUT: " + name);
      magnitude = std::max(magnitude, std::abs(v));
    }
    const float step = magnitude / 127.f;
    uint16_t lut8[4096]{};
    if (step > 0) {
      for (int i = 0; i < 4096; ++i) {
        const auto quant = [&](int j) { return int8_t(std::clamp(std::nearbyint(fp16(lut[j]) / step), -127.f, 127.f)); };
        lut8[i] = uint8_t(quant(2 * i)) | (uint16_t(uint8_t(quant(2 * i + 1))) << 8);
      }
    }
    std::vector<float> tau(rows);
    for (int i = 0; i < rows; ++i) tau[i] = fp16(scales[i]) * step;
    out.lut = mem_.copy(lut8, 4096);
    out.scale = mem_.copy(tau.data(), tau.size());
    return out;
  }

  void run_projection(Projection matrix, const float* in, float* out, int tokens = 1,
                      const Projection* previous = nullptr) {
    if (tokens > 1) {
      // The caller supplies the last projection only when its input rows are identical.
      const bool reuse_input = previous && previous->input_sign == matrix.input_sign &&
          previous->codes.cols == matrix.codes.cols && previous->codes.lnord_layout == matrix.codes.lnord_layout;
      project_batch(queue_, matrix, in, out, tokens, activations_, scales_, work_, partial_, reuse_input);
    } else if (packed_decode_) project_decode(queue_, matrix, in, out, activations_, scales_, work_, partial_);
    else project(queue_, matrix, in, out, activations_, scales_, work_);
  }

  static bool can_group_decode(const DecodeProjections& batch) {
    const auto& first = batch.matrices[0];
    if (!first.codes.lnord_layout) return false;
    for (int i = 1; i < batch.count; ++i) {
      const auto& next = batch.matrices[i];
      if (!next.codes.lnord_layout || next.codes.bits != first.codes.bits ||
          next.codes.cols != first.codes.cols || next.input_sign != first.input_sign) return false;
    }
    return true;
  }

  void run_projections(const float* in, int tokens, DecodeProjections batch) {
    if (tokens == 1 && can_group_decode(batch)) {
      project_decode_group(queue_, batch, in, activations_, scales_, partial_);
      return;
    }
    for (int i = 0; i < batch.count; ++i)
      run_projection(batch.matrices[i], in, batch.outputs[i], tokens, i ? &batch.matrices[i - 1] : nullptr);
  }

  void residual(int tokens) {
    auto* out = hidden_;
    const auto* delta = projection_out_;
    queue_.parallel_for(sycl::range<1>(tokens * hidden), [=](sycl::id<1> i) {
      out[i] = round_bf16(out[i] + round_bf16(delta[i]));
    });
  }

  void residual_norm(const uint16_t* weights) {
    auto* hidden_row = hidden_;
    auto* out = normalized_;
    const auto* delta = projection_out_;
    const float eps = eps_;
    queue_.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      float sum = 0.f;
      for (int i = lane; i < hidden; i += 256) {
        const float x = round_bf16(hidden_row[i] + round_bf16(delta[i]));
        hidden_row[i] = x;
        sum += x * x;
      }
      const float scale = sycl::rsqrt(sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>())
                                     / hidden + eps);
      for (int i = lane; i < hidden; i += 256)
        out[i] = round_bf16(hidden_row[i] * scale * (1.f + bf16(weights[i])));
    });
  }

  void project_residual_norm(Projection matrix, const float* in, const uint16_t* weights,
                             const float* prepared_scales = nullptr) {
    if (!matrix.codes.lnord_layout) {
      run_projection(matrix, in, projection_out_);
      residual_norm(weights);
      return;
    }
    if (!prepared_scales) quantize_rows(queue_, matrix, in, 1, activations_, scales_);
    launch_decode_projection(queue_, matrix, activations_, partial_);
    finish_decode_residual(queue_, matrix, partial_, prepared_scales ? prepared_scales : scales_, hidden_);
    norm(queue_, hidden_, weights, normalized_, eps_);
  }

  void run_mlp(const Layer& layer, int tokens, const uint16_t* next_norm) {
    DecodeProjections gate_up{{layer.gate, layer.up}, {first_, second_}, 2};
    if (next_norm && layer.down.codes.lnord_layout && can_group_decode(gate_up)) {
      launch_decode_group(queue_, gate_up, normalized_, activations_, scales_, partial_);
      // Keep the down-projection scales separate from the gate/up scales,
      // which are still being read by the other work-groups in this kernel.
      finish_decode_swiglu(queue_, layer.gate, layer.up, layer.down, partial_, scales_, activations_, work_);
      project_residual_norm(layer.down, nullptr, next_norm, work_);
      return;
    }
    run_projections(normalized_, tokens, gate_up);
    auto* gate = first_;
    const auto* up = second_;
    queue_.parallel_for(sycl::range<1>(tokens * ffn), [=](sycl::id<1> i) {
      gate[i] = round_bf16(round_bf16(silu(round_bf16(gate[i]))) * round_bf16(up[i]));
    });
    if (next_norm) project_residual_norm(layer.down, gate, next_norm);
    else {
      run_projection(layer.down, gate, projection_out_, tokens);
      residual(tokens);
    }
  }

  void prepare_decode_graph() {
    try {
      graph::command_graph recording(queue_);
      recording.begin_recording(queue_);
      try {
        forward(0, prediction_, 1, true, decode_position_);
        auto* position = decode_position_;
        queue_.single_task([=] { ++*position; });
      } catch (...) {
        recording.end_recording(queue_);
        throw;
      }
      recording.end_recording(queue_);
      decode_graph_.emplace(recording.finalize());
      std::fprintf(stderr, "[yskn-sycl] decode graph ready\n");
    } catch (const sycl::exception& err) {
      if (queue_.ext_oneapi_get_state() != graph::queue_state::executing) throw;
      graph_decode_ = false;
      std::fprintf(stderr, "[yskn-sycl] using eager decode: %s\n", err.what());
    }
  }

  void forward(int id, const int32_t* ids, int tokens, bool output, const int32_t* live_position = nullptr) {
    embed(queue_, embed_, id, ids, tokens, hidden_);
    for (int i = 0; i < layers_count; ++i) {
      const auto& l = layers_[i];
      if (!live_position || i == 0) norm(queue_, hidden_, l.norm1, normalized_, eps_, tokens);
      if (l.full) {
        run_projections(normalized_, tokens, {{l.attention.q, l.attention.k, l.attention.v},
                        {first_, second_, third_}, 3});
        attend(queue_, l.attention, first_, second_, third_, q16_, scores_, attention_out_,
               position_, theta_, eps_, tokens, live_position, context_);
      } else {
        run_projections(normalized_, tokens, {{l.linear.mixed, l.linear.z}, {first_, second_}, 2});
        delta_net(queue_, l.linear, normalized_, first_, second_, convolved_, core_, beta_,
                  decay_, attention_out_, eps_, tokens, live_position != nullptr);
      }
      const Projection out = l.full ? l.attention.out : l.linear.out;
      if (live_position) project_residual_norm(out, attention_out_, l.norm2);
      else {
        run_projection(out, attention_out_, projection_out_, tokens);
        residual(tokens);
        norm(queue_, hidden_, l.norm2, normalized_, eps_, tokens);
      }
      const auto* next_norm = i + 1 < layers_count ? layers_[i + 1].norm1 : final_norm_;
      run_mlp(l, tokens, live_position ? next_norm : nullptr);
    }
    if (!output) return;
    if (!live_position) norm(queue_, hidden_ + (tokens - 1) * hidden, final_norm_, normalized_, eps_);
    run_projection(head_, normalized_, logits_);
    greedy();
  }

  void greedy() {
    const float* logits = logits_;
    auto* best = maxima_;
    auto* indices = indices_;
    auto* prediction = prediction_;
    queue_.parallel_for(sycl::nd_range<1>(head_groups * 256, 256), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0), g = item.get_group(0);
      float value = -INFINITY;
      int id = vocab;
      for (int i = g * 256 + lane; i < vocab; i += head_groups * 256) {
        if (sycl::isfinite(logits[i]) && logits[i] > value) { value = logits[i]; id = i; }
      }
      const float maximum = sycl::reduce_over_group(item.get_group(), value, sycl::maximum<float>());
      const int index = sycl::reduce_over_group(item.get_group(), value == maximum ? id : vocab,
                                              sycl::minimum<int>());
      if (lane == 0) { best[g] = maximum; indices[g] = index; }
    });
    queue_.parallel_for(sycl::nd_range<1>(head_groups, head_groups), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      const float maximum = sycl::reduce_over_group(item.get_group(), best[lane], sycl::maximum<float>());
      const int index = sycl::reduce_over_group(item.get_group(), best[lane] == maximum ? indices[lane] : vocab,
                                              sycl::minimum<int>());
      if (lane == 0) *prediction = index;
    });
  }

  int read_prediction() {
    int id;
    queue_.copy(prediction_, &id, 1).wait_and_throw();
    if (id < 0 || id >= vocab) throw std::runtime_error("no finite SYCL logits");
    return id;
  }
};

} // namespace
} // namespace yskn::sycl_backend

struct yskn_engine { yskn::sycl_backend::Engine impl; };

namespace {
thread_local std::string last_error;

template <class Call>
int invoke(yskn_engine* e, Call call) {
  try {
    if (!e) throw std::runtime_error("missing SYCL engine");
    call(e->impl);
    return 0;
  } catch (const std::exception& err) {
    last_error = err.what();
    return -1;
  } catch (...) {
    last_error = "unknown SYCL error";
    return -1;
  }
}
}

extern "C" yskn_engine* yskn_open(const char* path, int max_ctx, int device, int flags) {
  try {
    if (!path || !*path || max_ctx < 1 || max_ctx > 262144 || flags != 0)
      throw std::runtime_error("SYCL open requires a plain Qwen3.5-4B pack, context 1 to 262144 and zero flags");
    return new yskn_engine{{path, max_ctx, device}};
  } catch (const std::exception& err) {
    last_error = err.what();
  } catch (...) {
    last_error = "unknown SYCL initialization error";
  }
  return nullptr;
}

extern "C" void yskn_close(yskn_engine* e) { delete e; }
extern "C" const char* yskn_last_error() { return last_error.c_str(); }
extern "C" int yskn_position(yskn_engine* e) { return e ? e->impl.position() : -1; }
extern "C" int yskn_reset(yskn_engine* e) { return invoke(e, [](auto& engine) { engine.reset(); }); }
extern "C" int yskn_prefill(yskn_engine* e, const int32_t* ids, int n, int32_t* out) {
  return invoke(e, [&](auto& engine) {
    if (!out) throw std::runtime_error("prefill: missing prediction buffer");
    *out = engine.prefill(ids, n);
  });
}
extern "C" int yskn_decode(yskn_engine* e, int32_t* out, int n) {
  return invoke(e, [&](auto& engine) { engine.decode(out, n); });
}
extern "C" int yskn_logits(yskn_engine* e, float* out) {
  return invoke(e, [&](auto& engine) { engine.logits(out); });
}
