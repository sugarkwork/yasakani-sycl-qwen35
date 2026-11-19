// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdint>

namespace yskn::sycl_backend {

constexpr int hidden = YSKN_Q35_HIDDEN;
static_assert(hidden == 2560 || hidden == 4096, "unsupported SYCL Qwen3.5 hidden size");
constexpr int ffn = hidden == 2560 ? 9216 : 12288;
constexpr const char* model_name = hidden == 2560 ? "Qwen3.5-4B" : "Qwen3.5-9B";
constexpr int vocab = 248320, layers_count = 32;
constexpr int query_heads = 16, kv_heads = 4, head_dim = 256, rotary_dim = 64;
constexpr int linear_keys = 16, linear_values = 32, linear_dim = 128;
constexpr int mixed_size = (2 * linear_keys + linear_values) * linear_dim;
constexpr int linear_width = linear_values * linear_dim;
constexpr int block = 512, projection_token_tile = 8, projection_row_tile = 32;
inline int padded_projection_tokens(int tokens) {
  return (tokens + projection_token_tile - 1) / projection_token_tile * projection_token_tile;
}
constexpr float hadamard_scale = 0.044194173824159216f;

inline float fp16(uint16_t bits) { return static_cast<float>(sycl::bit_cast<sycl::half>(bits)); }
inline float bf16(uint16_t bits) { return sycl::bit_cast<float>(uint32_t(bits) << 16); }
inline uint16_t to_bf16(float value) {
  const auto bits = sycl::bit_cast<uint32_t>(value);
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline float round_bf16(float value) { return bf16(to_bf16(value)); }
inline float sigmoid(float value) { return 1.f / (1.f + sycl::exp(-value)); }
inline float silu(float value) { return value * sigmoid(value); }

struct Codes {
  const uint32_t* words = nullptr;
  int rows = 0, cols = 0, bits = 0;
  bool row_layout = false;
  bool lnord_layout = false;

  uint32_t word(int row, int seq, int index) const {
    const int width = 4 * bits;
    if (row_layout) return words[(size_t(row) * (cols / 256) + seq) * width + index];
    if (lnord_layout) {
      const int lane = (row % 8) * 4 + index / bits;
      const int field = ((row % 16) / 8) * bits + index % bits;
      const size_t tile = size_t(row / 16) * (cols / 256) + seq;
      return words[((tile * (bits / 2) + field / 4) * 32 + lane) * 4 + field % 4];
    }
    const size_t lane = size_t(seq) * rows + row;
    return words[((lane / 32 * bits + index / 4) * 32 + lane % 32) * 4 + index % 4];
  }

  uint32_t state(int row, int seq, int pair) const {
    // The bitstream is little endian. Each sequence starts with a zero state.
    const int end = bits * (pair + 1);
    if (end < 12) return (word(row, seq, 0) & ((1u << end) - 1)) << (12 - end);
    const int start = end - 12, index = start / 32, shift = start % 32;
    uint32_t result = word(row, seq, index) >> shift;
    if (shift > 20) result |= word(row, seq, index + 1) << (32 - shift);
    return result & 4095u;
  }
};

struct Projection {
  Codes codes;
  const int8_t* input_sign = nullptr;
  const int8_t* output_sign = nullptr;
  const uint16_t* lut = nullptr;
  const float* scale = nullptr;
};

struct Embedding {
  Codes codes;
  const int8_t* sign = nullptr;
  const float* lut = nullptr;
  const float* scale = nullptr;
};

template <typename Source, typename Output>
void hadamard(sycl::queue& queue, int n, Source source, Output output) {
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<float, 1> data(block, cmd);
    cmd.parallel_for(sycl::nd_range<1>(n, block / 2), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      // One work-group per 512 values. Each work-item owns one butterfly pair.
      const int offset = item.get_group(0) * block;
      data[lane] = source(offset + lane);
      data[lane + block / 2] = source(offset + lane + block / 2);
      sycl::group_barrier(item.get_group());
      for (int stride = 1; stride < block; stride *= 2) {
        const int left = (lane / stride) * (2 * stride) + lane % stride;
        const float a = data[left], b = data[left + stride];
        data[left] = a + b;
        data[left + stride] = a - b;
        sycl::group_barrier(item.get_group());
      }
      output(item.get_group(), offset, lane, data[lane] * hadamard_scale,
             data[lane + block / 2] * hadamard_scale);
    });
  });
}

inline void embed(sycl::queue& queue, Embedding emb, int id, const int32_t* ids, int tokens, float* out) {
  hadamard(queue, tokens * hidden / 2, [=](int i) {
    const int token = ids ? ids[i / hidden] : id;
    const int col = i % hidden;
    const auto state = emb.codes.state(token, col / 256, (col % 256) / 2);
    return emb.lut[2 * state + col % 2] * emb.scale[token];
  }, [=](sycl::group<1>, int base, int lane, float lo, float hi) {
    const int a = base + lane, b = a + block / 2;
    out[a] = round_bf16(lo * emb.sign[a % hidden]);
    out[b] = round_bf16(hi * emb.sign[b % hidden]);
  });
}

// Eight token rows share a warp's MMA B fragment. Each 32-lane slice holds
// two packed INT8 words per lane and is read as a coalesced 256-byte tile.
inline size_t prefill_activation_index(int token, int group, int col, int groups) {
  const int step = (col / 256) * 8 + (col % 64) / 8;
  const int lane = (token % 8) * 4 + (col % 256) / 64;
  return (((size_t(token / 8) * groups + group) * 16 * 32 + step * 32 + lane) * 8) + col % 8;
}

template <bool Packed = false>
inline void quantize_rows(sycl::queue& queue, Projection matrix, const float* in, int tokens,
                          int8_t* activations, float* scales) {
  const int cols = matrix.codes.cols;
  hadamard(queue, tokens * cols / 2, [=](int i) {
    return in[i] * matrix.input_sign[i % cols];
  }, [=](sycl::group<1> group, int base, int lane, float lo, float hi) {
    lo = static_cast<float>(sycl::half(lo));
    hi = static_cast<float>(sycl::half(hi));
    const float scale = sycl::reduce_over_group(group, sycl::fmax(sycl::fabs(lo), sycl::fabs(hi)),
                                                sycl::maximum<float>()) / 127.f;
    if (lane == 0) scales[base / block] = scale;
    size_t dest = base + lane, high_offset = block / 2;
    if constexpr (Packed) {
      dest = prefill_activation_index(base / cols, (base % cols) / block, lane, cols / block);
      high_offset = 8 * 32 * 8;
    }
    activations[dest] = scale > 0 ? int8_t(sycl::clamp(sycl::rint(lo / scale), -127.f, 127.f)) : 0;
    activations[dest + high_offset] = scale > 0 ? int8_t(sycl::clamp(sycl::rint(hi / scale), -127.f, 127.f)) : 0;
  });

}

inline void unrotate_rows(sycl::queue& queue, Projection matrix, int tokens, const float* work, float* out) {
  if (!matrix.output_sign) return;
  const int rows = matrix.codes.rows;
  hadamard(queue, tokens * rows / 2, [=](int i) { return work[i]; },
           [=](sycl::group<1>, int base, int lane, float lo, float hi) {
    const int a = base + lane, b = a + block / 2;
    out[a] = lo * matrix.output_sign[a % rows];
    out[b] = hi * matrix.output_sign[b % rows];
  });
}

inline void project(sycl::queue& queue, Projection matrix, const float* in, float* out,
                    int8_t* activations, float* scales, float* work) {
  const int rows = matrix.codes.rows, cols = matrix.codes.cols;
  quantize_rows(queue, matrix, in, 1, activations, scales);

  // Thirty-two rows share a work-group. Row lanes decode independent parts of the
  // same sequence; keeping the row lanes adjacent helps warp-layout reads.
  const auto result = matrix.output_sign ? work : out;
  queue.submit([&](sycl::handler& cmd) {
    constexpr int row_tile = 32, columns_per_row = 8;
    sycl::local_accessor<float, 1> partial(row_tile * columns_per_row, cmd);
    cmd.parallel_for(sycl::nd_range<1>(size_t(rows) * columns_per_row, row_tile * columns_per_row),
                     [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0), row_lane = lane % row_tile, column_lane = lane / row_tile;
      const int row = item.get_group(0) * row_tile + row_lane;
      float value = 0.f;
      for (int seq = 0; seq < cols / 256; seq += 2) {
        int sum = 0;
        for (int b = 0; b < 2; ++b) {
          const int8_t* input = activations + (seq + b) * 256;
          for (int j = column_lane; j < 128; j += columns_per_row) {
            const unsigned s = matrix.codes.state(row, seq + b, j);
            const unsigned pair = matrix.lut[s];
            sum += input[2 * j] * int8_t(pair) + input[2 * j + 1] * int8_t(pair >> 8);
          }
        }
        value += float(sum) * scales[seq / 2];
      }
      partial[lane] = value;
      sycl::group_barrier(item.get_group());
      if (column_lane == 0) {
        for (int col = 1; col < columns_per_row; ++col) value += partial[col * row_tile + row_lane];
        result[row] = value * matrix.scale[row];
      }
    });
  });
  unrotate_rows(queue, matrix, 1, work, out);
}

inline int dot4(uint32_t a, uint32_t b, int sum) {
#if defined(__NVPTX__)
  asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(sum) : "r"(a), "r"(b));
#else
  for (int byte = 0; byte < 32; byte += 8) sum += int8_t(a >> byte) * int8_t(b >> byte);
#endif
  return sum;
}

template <int Bits>
inline unsigned trellis_state(const uint32_t (&words)[4 * Bits], int pair) {
  const int end = Bits * (pair + 1);
  if (end < 12) return (words[0] & ((1u << end) - 1)) << (12 - end);
  const int start = end - 12, word = start / 32, shift = start % 32;
  const uint32_t hi = shift > 20 ? words[word + 1] << (32 - shift) : 0;
  return ((words[word] >> shift) | hi) & 4095u;
}

template <int Bits>
void decode_dots(sycl::queue& queue, Projection matrix, const int8_t* activations, int32_t* dots) {
  constexpr size_t threads = 128;
  using Pack = sycl::vec<uint32_t, 4>;
  using Input = sycl::vec<int8_t, 4>;
  const int rows = matrix.codes.rows, seqs = matrix.codes.cols / 256;
  const size_t work_items = size_t(rows) * seqs;
  queue.parallel_for(sycl::nd_range<1>((work_items + threads - 1) / threads * threads, threads),
                     [=](sycl::nd_item<1> item) {
    const size_t i = item.get_global_id(0);
    if (i >= work_items) return;
    const int seq = i / rows;
    const size_t base = (i / 32) * (Bits * 32) + i % 32;
    const auto* codes = reinterpret_cast<const Pack*>(matrix.codes.words);
    uint32_t words[4 * Bits];
#pragma unroll
    for (int p = 0; p < Bits; ++p) {
      const Pack pack = codes[base + p * 32];
      words[4 * p] = pack.x();
      words[4 * p + 1] = pack.y();
      words[4 * p + 2] = pack.z();
      words[4 * p + 3] = pack.w();
    }
    const auto* input = reinterpret_cast<const Input*>(activations + seq * 256);
    int sum = 0;
#pragma unroll
    for (int p = 0; p < 64; ++p) {
      const unsigned a = matrix.lut[trellis_state<Bits>(words, 2 * p)];
      const unsigned b = matrix.lut[trellis_state<Bits>(words, 2 * p + 1)];
      sum = dot4(sycl::bit_cast<uint32_t>(input[p]), a | (b << 16), sum);
    }
    dots[i] = sum;
  });
}

template <int Bits>
inline unsigned lnord_field(const uint32_t* words, int step) {
  constexpr int width = 4 * Bits;
  const int bit = width * step, word = bit / 32, shift = bit % 32;
  unsigned value = words[word] >> shift;
  if constexpr (width < 32) {
    if (shift + width > 32) value |= words[word + 1] << (32 - shift);
    value &= (1u << width) - 1;
  }
  return value;
}

inline void decode_mma(int (&out)[4], const unsigned (&weight)[4], uint32_t lo, uint32_t hi) {
#if defined(__NVPTX__)
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(out[0]), "+r"(out[1]), "+r"(out[2]), "+r"(out[3])
      : "r"(weight[0]), "r"(weight[1]), "r"(weight[2]), "r"(weight[3]), "r"(lo), "r"(hi));
#endif
}

struct DecodeProjections {
  Projection matrices[3]{};
  float* outputs[3]{};
  int count = 0;
  int block_ends[3]{};
  int row_ends[3]{};
};

template <int Bits, bool Grouped = false>
void decode_tensor_dots(sycl::queue& queue, Projection matrix, const int8_t* activations, int32_t* dots,
                        DecodeProjections batch = {}) {
  constexpr int warp = 32, warps = 4, threads = warp * warps;
  constexpr int tail = 12 - Bits, high_pair_shift = 3 * Bits - 12;
  constexpr unsigned pair_mask = (1u << (2 * Bits)) - 1, tail_mask = (1u << tail) - 1;
  using Pack = sycl::vec<uint32_t, 4>;
  const int rows = matrix.codes.rows, groups = matrix.codes.cols / block;
  const int tiles = (rows / 16) * groups;
  const int blocks = Grouped ? batch.block_ends[batch.count - 1] : (tiles + warps - 1) / warps;
  // Fixed captures avoid dynamically indexing the kernel parameter block.
  const auto first = batch.matrices[0], second = batch.matrices[1], third = batch.matrices[2];
  const int block1 = batch.block_ends[0], block2 = batch.block_ends[1];
  const int row1 = batch.row_ends[0], row2 = batch.row_ends[1], count = batch.count;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<sycl::vec<uint16_t, 8>, 1> lut_storage(512, cmd);
    cmd.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
      const int lid = item.get_local_id(0), lane = lid % warp;
      int cta = item.get_group(0);
      const uint32_t* weights = matrix.codes.words;
      const uint16_t* table = matrix.lut;
      int local_rows = rows;
      auto* dst = dots;
      if constexpr (Grouped) {
        if (cta < block1) {
          weights = first.codes.words;
          table = first.lut;
          local_rows = first.codes.rows;
        } else if (count == 2 || cta < block2) {
          weights = second.codes.words;
          table = second.lut;
          local_rows = second.codes.rows;
          cta -= block1;
          dst += row1 * groups;
        } else {
          weights = third.codes.words;
          table = third.lut;
          local_rows = third.codes.rows;
          cta -= block2;
          dst += row2 * groups;
        }
      }
      const int tile = cta * warps + lid / warp;
      const auto* src_lut = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(table);
      for (int i = lid; i < 512; i += threads) lut_storage[i] = src_lut[i];
      sycl::group_barrier(item.get_group());
      if (tile >= (local_rows / 16) * groups) return;

      const int row_group = lane / 4, k_lane = lane % 4, group = tile % groups;
      const auto* lut = reinterpret_cast<const uint16_t*>(&lut_storage[0]);
      const auto* codes = reinterpret_cast<const Pack*>(weights);
      const auto* inputs = reinterpret_cast<const Pack*>(activations + group * block);
      const auto pack4 = [=](unsigned pairs, unsigned prev) {
        const unsigned state = prev | (pairs << tail);
        return unsigned(lut[state & 4095]) | (unsigned(lut[(state >> Bits) & 4095]) << 16);
      };
      int acc[4]{};
#pragma unroll
      for (int seq = 0; seq < 2; ++seq) {
        uint32_t words[2 * Bits];
        const size_t base = (size_t(tile) * 2 + seq) * (Bits / 2) * warp + lane;
#pragma unroll
        for (int p = 0; p < Bits / 2; ++p) {
          const Pack code = codes[base + p * warp];
          words[4 * p] = code.x();
          words[4 * p + 1] = code.y();
          words[4 * p + 2] = code.z();
          words[4 * p + 3] = code.w();
        }
        const unsigned edge = (words[Bits - 1] >> (32 - tail)) |
                              ((words[2 * Bits - 1] >> (32 - tail)) << tail);
        const unsigned prev = sycl::select_from_group(item.get_sub_group(), edge,
                                                      row_group * 4 + (k_lane + 3) % 4);
        unsigned prev0 = k_lane ? prev & tail_mask : 0, prev1 = k_lane ? prev >> tail : 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          // Only token column zero is populated. Rows and K fragments follow
          // the LNORD m16n8k32 lane order used by the CUDA kernel.
          const Pack x = row_group == 0 ? inputs[16 * seq + 4 * k_lane + p] : Pack(0);
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const unsigned f0 = lnord_field<Bits>(words, 2 * p + half);
            const unsigned f1 = lnord_field<Bits>(words + Bits, 2 * p + half);
            const unsigned low0 = f0 & pair_mask, low1 = f1 & pair_mask;
            const unsigned high0 = f0 >> (2 * Bits), high1 = f1 >> (2 * Bits);
            const unsigned a[4] = {pack4(low0, prev0), pack4(low1, prev1),
                                   pack4(high0, low0 >> high_pair_shift), pack4(high1, low1 >> high_pair_shift)};
            decode_mma(acc, a, half ? x.z() : x.x(), half ? x.w() : x.y());
            prev0 = high0 >> high_pair_shift;
            prev1 = high1 >> high_pair_shift;
          }
        }
      }
      if (k_lane == 0) {
        const size_t i = size_t(group) * local_rows + (tile / groups) * 16 + row_group;
        dst[i] = acc[0];
        dst[i + 8] = acc[2];
      }
    });
  });
}

inline void inverse_hadamard_warp(sycl::sub_group sg, int lane, float (&x)[block / 32]) {
  constexpr int warp = 32, per_lane = block / warp;
#pragma unroll
  for (int stride = 1; stride < warp; stride *= 2) {
#pragma unroll
    for (int j = 0; j < per_lane; ++j) {
      const float other = sycl::permute_group_by_xor(sg, x[j], stride);
      x[j] = (lane & stride) ? other - x[j] : x[j] + other;
    }
  }
#pragma unroll
  for (int stride = 1; stride < per_lane; stride *= 2) {
#pragma unroll
    for (int j = 0; j < per_lane; ++j) {
      if (!(j & stride)) {
        const float lo = x[j], hi = x[j + stride];
        x[j] = lo + hi;
        x[j + stride] = lo - hi;
      }
    }
  }
}

inline void launch_decode_group(sycl::queue& queue, DecodeProjections& batch, const float* in,
                                int8_t* activations, float* scales, int32_t* dots) {
  constexpr int projection_warps = 4;
  const Projection first = batch.matrices[0];
  const int groups = first.codes.cols / block;
  int blocks = 0, rows = 0;
  for (int i = 0; i < batch.count; ++i) {
    const int n = batch.matrices[i].codes.rows;
    blocks += ((n / 16) * groups + projection_warps - 1) / projection_warps;
    rows += n;
    batch.block_ends[i] = blocks;
    batch.row_ends[i] = rows;
  }
  quantize_rows(queue, first, in, 1, activations, scales);
  if (first.codes.bits == 6) decode_tensor_dots<6, true>(queue, first, activations, dots, batch);
  else decode_tensor_dots<8, true>(queue, first, activations, dots, batch);
}

inline void load_decode_output(sycl::sub_group sg, int lane, Projection matrix, int row,
                               const int32_t* dots, const float* scales, float (&x)[block / 32]) {
  constexpr int warp = 32, per_lane = block / warp;
  const int groups = matrix.codes.cols / block;
  const auto* tile = dots + row;
  for (int g = 0; g < groups; ++g) {
    const float scale = scales[g];
#pragma unroll
    for (int j = 0; j < per_lane; ++j) x[j] += float(tile[j * warp]) * scale;
    tile += matrix.codes.rows;
  }
#pragma unroll
  for (int j = 0; j < per_lane; ++j) x[j] *= matrix.scale[row + j * warp];
  inverse_hadamard_warp(sg, lane, x);
#pragma unroll
  for (int j = 0; j < per_lane; ++j)
    x[j] = (x[j] * hadamard_scale) * matrix.output_sign[row + j * warp];
}

inline void project_decode_group(sycl::queue& queue, DecodeProjections batch, const float* in,
                                 int8_t* activations, float* scales, int32_t* dots) {
  constexpr int warp = 32, per_lane = block / warp;
  launch_decode_group(queue, batch, in, activations, scales, dots);
  const auto first = batch.matrices[0];
  const int groups = first.codes.cols / block, tiles = batch.row_ends[batch.count - 1] / block;
  const Projection second = batch.matrices[1], third = batch.matrices[2];
  const int row1 = batch.row_ends[0], row2 = batch.row_ends[1], count = batch.count;
  auto* dst1 = batch.outputs[0];
  auto* dst2 = batch.outputs[1];
  auto* dst3 = batch.outputs[2];
  queue.parallel_for(sycl::nd_range<1>(tiles * warp, warp),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
    const int base = item.get_group(0) * block;
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const bool at_first = base < row1, at_second = !at_first && (count == 2 || base < row2);
    const int begin = at_first ? 0 : (at_second ? row1 : row2);
    const auto matrix = at_first ? first : (at_second ? second : third);
    auto* dst = at_first ? dst1 : (at_second ? dst2 : dst3);
    const int row = base - begin + lane;
    float x[per_lane]{};
    load_decode_output(sg, lane, matrix, row, dots + begin * groups, scales, x);
#pragma unroll
    for (int j = 0; j < per_lane; ++j) dst[row + j * warp] = x[j];
  });
}

inline void finish_decode_swiglu(sycl::queue& queue, Projection gate, Projection up, Projection down,
                                 const int32_t* dots, const float* scales, int8_t* act, float* next_scales) {
  constexpr int warp = 32, threads = 4 * warp, per_thread = block / threads;
  const int groups = gate.codes.cols / block, up_offset = gate.codes.rows * groups;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<float, 1> scratch(block, cmd);
    cmd.parallel_for(sycl::nd_range<1>(gate.codes.rows / block * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
      const auto sg = item.get_sub_group();
      const int tid = item.get_local_id(0), lane = tid % warp, w = tid / warp;
      const int tile = item.get_group(0), row = tile * block + tid;
      auto transform = [&](float (&x)[per_thread]) {
#pragma unroll
        for (int stride = 1; stride < warp; stride *= 2) {
#pragma unroll
          for (int j = 0; j < per_thread; ++j) {
            const float other = sycl::permute_group_by_xor(sg, x[j], stride);
            x[j] = (lane & stride) ? other - x[j] : x[j] + other;
          }
        }
#pragma unroll
        for (int j = 0; j < per_thread; ++j) scratch[j * threads + tid] = x[j];
        sycl::group_barrier(item.get_group());
        // Shared exchange for strides 32 and 64. Keep the original
        // butterfly order within each four-warp column.
#pragma unroll
        for (int j = 0; j < per_thread; ++j) {
          const int base = j * threads + lane;
          const float a = scratch[base], b = scratch[base + warp];
          const float c = scratch[base + 2 * warp], d = scratch[base + 3 * warp];
          const float lo = (w & 1) ? a - b : a + b, hi = (w & 1) ? c - d : c + d;
          x[j] = (w & 2) ? lo - hi : lo + hi;
        }
        sycl::group_barrier(item.get_group());
        const float a = x[0] + x[1], b = x[0] - x[1];
        const float c = x[2] + x[3], d = x[2] - x[3];
        x[0] = a + c;
        x[1] = b + d;
        x[2] = a - c;
        x[3] = b - d;
      };
      auto load = [&](Projection matrix, const int32_t* input, float (&x)[per_thread]) {
        const auto* partial = input + row;
        for (int g = 0; g < groups; ++g) {
          const float alpha = scales[g];
#pragma unroll
          for (int j = 0; j < per_thread; ++j) x[j] += float(partial[j * threads]) * alpha;
          partial += matrix.codes.rows;
        }
#pragma unroll
        for (int j = 0; j < per_thread; ++j) x[j] *= matrix.scale[row + j * threads];
        transform(x);
#pragma unroll
        for (int j = 0; j < per_thread; ++j)
          x[j] = round_bf16((x[j] * hadamard_scale) * matrix.output_sign[row + j * threads]);
      };
      float g[per_thread]{}, u[per_thread]{};
      load(gate, dots, g);
      load(up, dots + up_offset, u);
#pragma unroll
      for (int j = 0; j < per_thread; ++j) {
        const float v = round_bf16(round_bf16(silu(g[j])) * u[j]);
        g[j] = v * down.input_sign[row + j * threads];
      }
      transform(g);
      float maximum = 0.f;
#pragma unroll
      for (int j = 0; j < per_thread; ++j) {
        g[j] = static_cast<float>(sycl::half(g[j] * hadamard_scale));
        maximum = sycl::fmax(maximum, sycl::fabs(g[j]));
      }
      const float scale = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>()) / 127.f;
      if (tid == 0) next_scales[tile] = scale;
#pragma unroll
      for (int j = 0; j < per_thread; ++j)
        act[row + j * threads] = scale > 0 ? int8_t(sycl::clamp(sycl::rint(g[j] / scale), -127.f, 127.f)) : 0;
    });
  });
}

inline void finish_decode_residual(sycl::queue& queue, Projection matrix, const int32_t* dots,
                                    const float* scales, float* hidden_row) {
  const int groups = matrix.codes.cols / block, rows = matrix.codes.rows;
  hadamard(queue, rows / 2, [=](int row) {
    float sum = 0.f;
    for (int g = 0; g < groups; ++g) sum += float(dots[size_t(g) * rows + row]) * scales[g];
    return sum * matrix.scale[row];
  }, [=](sycl::group<1>, int base, int lane, float lo, float hi) {
    const int a = base + lane, b = a + block / 2;
    hidden_row[a] = round_bf16(hidden_row[a] + round_bf16(lo * matrix.output_sign[a]));
    hidden_row[b] = round_bf16(hidden_row[b] + round_bf16(hi * matrix.output_sign[b]));
  });
}

inline void launch_decode_projection(sycl::queue& queue, Projection matrix, const int8_t* activations,
                                     int32_t* dots) {
  if (matrix.codes.bits == 6) decode_tensor_dots<6>(queue, matrix, activations, dots);
  else decode_tensor_dots<8>(queue, matrix, activations, dots);
}

inline void project_decode(sycl::queue& queue, Projection matrix, const float* in, float* out,
                           int8_t* activations, float* scales, float* work, int32_t* dots) {
  if (matrix.codes.bits != 6 && matrix.codes.bits != 8) {
    project(queue, matrix, in, out, activations, scales, work);
    return;
  }
  quantize_rows(queue, matrix, in, 1, activations, scales);
  if (matrix.codes.lnord_layout) {
    launch_decode_projection(queue, matrix, activations, dots);
  } else if (matrix.codes.bits == 6) decode_dots<6>(queue, matrix, activations, dots);
  else decode_dots<8>(queue, matrix, activations, dots);

  const int rows = matrix.codes.rows, groups = matrix.codes.cols / block;
  auto value = [=](int row) {
    float sum = 0.f;
    const int stride = matrix.codes.lnord_layout ? 1 : 2;
    for (int g = 0; g < groups; ++g) {
      const size_t i = size_t(stride * g) * rows + row;
      const int dot = dots[i] + (stride == 2 ? dots[i + rows] : 0);
      sum += float(dot) * scales[g];
    }
    return sum * matrix.scale[row];
  };
  if (matrix.output_sign) {
    hadamard(queue, rows / 2, value, [=](sycl::group<1>, int base, int lane, float lo, float hi) {
      const int a = base + lane, b = a + block / 2;
      out[a] = lo * matrix.output_sign[a];
      out[b] = hi * matrix.output_sign[b];
    });
  } else {
    queue.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> i) { out[i] = value(i[0]); });
  }
}

// This backend is built for NVIDIA. The DPC++ joint_matrix runtime check
// rejects sm_120 on the bundled toolchain, so use its NVPTX primitives.
// Keep the call in both compiler passes so the kernel layout is identical.
inline void int8_tensor_tile(const int8_t* input, int cols, const int8_t* weights,
                            int32_t* dots, int rows) {
#if defined(__NVPTX__)
  namespace jm = sycl::ext::oneapi::experimental::matrix;
  namespace nv = sycl::ext::oneapi::detail;
  constexpr int token_tile = projection_token_tile, row_tile = projection_row_tile, k_tile = 16;
  nv::joint_matrix_cuda<int8_t, jm::use::a, token_tile, k_tile, jm::layout::row_major> a;
  nv::joint_matrix_cuda<int8_t, jm::use::b, k_tile, row_tile, jm::layout::row_major> b;
  nv::joint_matrix_cuda<int32_t, jm::use::accumulator, token_tile, row_tile> acc{};
  using sycl::access::address_space;
  using sycl::access::decorated;
  for (int k = 0; k < block; k += k_tile) {
    nv::load_multiplicand_cuda(a, sycl::address_space_cast<address_space::global_space, decorated::no>(
        input + k), cols);
    nv::load_multiplicand_cuda(b, sycl::address_space_cast<address_space::local_space, decorated::no>(
        weights + k * row_tile), row_tile);
    nv::joint_matrix_mad_cuda(acc, a, b, acc);
  }
  nv::store_layoutT<jm::layout::row_major>(acc,
      sycl::address_space_cast<address_space::global_space, decorated::no>(dots), rows);
#endif
}

template <int Bits, class Store>
void unpack_prefill_fragment(sycl::sub_group sg, const sycl::vec<uint32_t, 4>* codes,
                             const uint16_t* lut, Store store) {
  constexpr int carry_bits = 12 - Bits, carry_mask = (1u << carry_bits) - 1;
  constexpr unsigned pair_mask = (1u << (2 * Bits)) - 1;
  constexpr int carry_shift = 3 * Bits - 12;
  const int lane = sg.get_local_linear_id(), k_lane = lane % 4;
  uint32_t word[2 * Bits];
#pragma unroll
  for (int p = 0; p < Bits / 2; ++p) {
    const auto code = codes[p * 32 + lane];
    word[4 * p] = code.x();
    word[4 * p + 1] = code.y();
    word[4 * p + 2] = code.z();
    word[4 * p + 3] = code.w();
  }
  const unsigned edge = (word[Bits - 1] >> (32 - carry_bits)) |
                        ((word[2 * Bits - 1] >> (32 - carry_bits)) << carry_bits);
  const unsigned carry = sycl::select_from_group(sg, edge, (lane & ~3) | ((k_lane + 3) & 3));
  const auto pair = [=](unsigned field, unsigned prev) {
    const unsigned state = prev | (field << carry_bits);
    return unsigned(lut[state & 4095]) | (unsigned(lut[(state >> Bits) & 4095]) << 16);
  };
  unsigned prev0 = k_lane ? carry & carry_mask : 0, prev1 = k_lane ? carry >> carry_bits : 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned f0 = lnord_field<Bits>(word, k), f1 = lnord_field<Bits>(word + Bits, k);
    const unsigned lo0 = f0 & pair_mask, lo1 = f1 & pair_mask;
    const unsigned hi0 = f0 >> (2 * Bits), hi1 = f1 >> (2 * Bits);
    store(k, pair(lo0, prev0), pair(lo1, prev1), pair(hi0, lo0 >> carry_shift), pair(hi1, lo1 >> carry_shift));
    prev0 = hi0 >> carry_shift;
    prev1 = hi1 >> carry_shift;
  }
}

template <int Bits>
void prefill_lnord_scaled(sycl::queue& queue, Projection matrix, const int8_t* activations,
                          const float* scales, float* rotated, int tokens, int padded) {
  constexpr int warp = 32, output_tile = 32, token_tile = 8, warp_count = 8;
  constexpr int token_tiles_per_warp = 2, token_block = token_tiles_per_warp * warp_count * token_tile;
  using Pack = sycl::vec<uint32_t, 4>;
  using Pair = sycl::vec<uint32_t, 2>;
  const int rows = matrix.codes.rows, cols = matrix.codes.cols, groups = cols / block;
  const int token_warps = std::min(warp_count, padded / token_tile), threads = token_warps * warp;
  const int token_blocks = (padded + token_block - 1) / token_block;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<sycl::vec<uint16_t, 8>, 1> lut_storage(512, cmd);
    sycl::local_accessor<uint32_t, 1> fragments(2 * 16 * 4 * warp, cmd);
    cmd.parallel_for(sycl::nd_range<1>(size_t(rows / output_tile) * token_blocks * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
      const int tile = item.get_group(0), lid = item.get_local_id(0), w = lid / warp, lane = lid % warp;
      const int row_tile = tile / token_blocks, base_row = row_tile * output_tile;
      const int row_group = lane / 4, k_lane = lane % 4;
      const int token0 = (tile % token_blocks) * token_block + w * token_tile;
      const auto* lut = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(matrix.lut);
      for (int i = lid; i < 512; i += threads) lut_storage[i] = lut[i];
      sycl::group_barrier(item.get_group());
      const auto* codes = reinterpret_cast<const Pack*>(matrix.codes.words) + size_t(2 * row_tile) * groups * Bits * warp;
      float total[token_tiles_per_warp][2][4]{};
      for (int g = 0; g < groups; ++g) {
        for (int seq = w; seq < 4; seq += token_warps) {
          const auto store = [=](int k, unsigned a0, unsigned a1, unsigned a2, unsigned a3) {
            const int base = (seq * 8 + k) * 4 * warp + lane;
            fragments[base] = a0;
            fragments[base + warp] = a1;
            fragments[base + 2 * warp] = a2;
            fragments[base + 3 * warp] = a3;
          };
          const size_t code_offset = ((size_t(seq / 2) * groups + g) * 2 + seq % 2) * (Bits / 2) * warp;
          unpack_prefill_fragment<Bits>(item.get_sub_group(), codes + code_offset,
                                       reinterpret_cast<const uint16_t*>(&lut_storage[0]), store);
        }
        sycl::group_barrier(item.get_group());

        int sums[token_tiles_per_warp][2][4]{};
#pragma unroll 4
        for (int k = 0; k < 16; ++k) {
          const int base = k * 4 * warp + lane;
          unsigned a[2][4];
#pragma unroll
          for (int o = 0; o < 2; ++o) {
            for (int j = 0; j < 4; ++j) a[o][j] = fragments[base + (o * 16 * 4 + j) * warp];
          }
#pragma unroll
          for (int tok = 0; tok < token_tiles_per_warp; ++tok) {
            const int t = token0 + tok * warp_count * token_tile;
            if (t < padded) {
              const auto* x = reinterpret_cast<const Pair*>(activations) +
                              ((size_t(t / token_tile) * groups + g) * 16 + k) * warp + lane;
              const auto b = *x;
              decode_mma(sums[tok][0], a[0], b.x(), b.y());
              decode_mma(sums[tok][1], a[1], b.x(), b.y());
            }
          }
        }
#pragma unroll
        for (int tok = 0; tok < token_tiles_per_warp; ++tok) {
          const int t = token0 + tok * warp_count * token_tile + 2 * k_lane;
          const float alpha0 = t < tokens ? scales[t * groups + g] : 0.f;
          const float alpha1 = t + 1 < tokens ? scales[(t + 1) * groups + g] : 0.f;
#pragma unroll
          for (int o = 0; o < 2; ++o) {
            total[tok][o][0] += float(sums[tok][o][0]) * alpha0;
            total[tok][o][1] += float(sums[tok][o][1]) * alpha1;
            total[tok][o][2] += float(sums[tok][o][2]) * alpha0;
            total[tok][o][3] += float(sums[tok][o][3]) * alpha1;
          }
        }
        sycl::group_barrier(item.get_group());
      }
#pragma unroll
      for (int tok = 0; tok < token_tiles_per_warp; ++tok) {
        const int t = token0 + tok * warp_count * token_tile + 2 * k_lane;
#pragma unroll
        for (int o = 0; o < 2; ++o) {
          const int row = base_row + row_group + 16 * o;
          if (t < tokens) {
            rotated[size_t(t) * rows + row] = total[tok][o][0];
            rotated[size_t(t) * rows + row + 8] = total[tok][o][2];
          }
          if (t + 1 < tokens) {
            rotated[size_t(t + 1) * rows + row] = total[tok][o][1];
            rotated[size_t(t + 1) * rows + row + 8] = total[tok][o][3];
          }
        }
      }
    });
  });
}

template <bool Scaled>
void finish_prefill_projection(sycl::queue& queue, Projection matrix, int tokens, int padded,
                               const int32_t* partial, const float* scales, const float* rotated, float* out) {
  constexpr int warp = 32, warps = 4, threads = warp * warps, per_lane = block / warp;
  const int rows = matrix.codes.rows, groups = matrix.codes.cols / block, row_blocks = rows / block;
  const int tiles = tokens * row_blocks;
  queue.parallel_for(sycl::nd_range<1>((tiles + warps - 1) / warps * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
    const int tile = item.get_global_id(0) / warp;
    if (tile >= tiles) return;
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id(), token = tile / row_blocks;
    const int row = tile % row_blocks * block + lane;
    float x[per_lane]{};
    if constexpr (Scaled) {
#pragma unroll
      for (int j = 0; j < per_lane; ++j) x[j] = rotated[size_t(token) * rows + row + j * warp];
    } else {
      const auto* dots = partial + size_t(token) * rows + row;
      for (int g = 0; g < groups; ++g) {
        const float alpha = scales[token * groups + g];
#pragma unroll
        for (int j = 0; j < per_lane; ++j) x[j] += float(dots[j * warp]) * alpha;
        dots += size_t(padded) * rows;
      }
    }
#pragma unroll
    for (int j = 0; j < per_lane; ++j) x[j] *= matrix.scale[row + j * warp];
    inverse_hadamard_warp(sg, lane, x);
#pragma unroll
    for (int j = 0; j < per_lane; ++j) {
      const int r = row + j * warp;
      out[size_t(token) * rows + r] = (x[j] * hadamard_scale) * matrix.output_sign[r];
    }
  });
}

inline void project_batch(sycl::queue& queue, Projection matrix, const float* in, float* out, int tokens,
                          int8_t* activations, float* scales, float* work, int32_t* partial, bool reuse_input) {
  constexpr int token_tile = projection_token_tile, row_tile = projection_row_tile, warp = 32;
  const int rows = matrix.codes.rows, cols = matrix.codes.cols, groups = cols / block;
  const int padded = padded_projection_tokens(tokens);
  const int token_warps = std::min(padded / token_tile, 8);
  const int group_threads = token_warps * warp;

  if (matrix.codes.lnord_layout && padded <= 256) {
    if (!reuse_input) {
      if (tokens != padded) {
        // Clear the whole final tile before quantization writes its real rows.
        queue.fill(activations + (padded - token_tile) * cols, int8_t(0), token_tile * cols);
      }
      quantize_rows<true>(queue, matrix, in, tokens, activations, scales);
    }
    if (matrix.codes.bits == 6) prefill_lnord_scaled<6>(queue, matrix, activations, scales, work, tokens, padded);
    else prefill_lnord_scaled<8>(queue, matrix, activations, scales, work, tokens, padded);
    if (matrix.output_sign) {
      finish_prefill_projection<true>(queue, matrix, tokens, padded, nullptr, nullptr, work, out);
    } else {
      queue.parallel_for(sycl::range<1>(tokens * rows), [=](sycl::id<1> i) { out[i] = work[i] * matrix.scale[i[0] % rows]; });
    }
    return;
  } else {
    if (!reuse_input) {
      quantize_rows(queue, matrix, in, tokens, activations, scales);
      if (padded != tokens) queue.fill(activations + tokens * cols, int8_t(0), (padded - tokens) * cols);
    }
    queue.submit([&](sycl::handler& cmd) {
      struct alignas(32) WeightBlock { int8_t values[block * row_tile]; };
      sycl::local_accessor<WeightBlock, 1> weight_storage(1, cmd);
      cmd.parallel_for(sycl::nd_range<1>(size_t(rows / row_tile) * groups * group_threads, group_threads),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
        const int lid = item.get_local_id(0), tile = item.get_group(0);
        const int g = tile % groups, row_base = (tile / groups) * row_tile;
        auto* weights = weight_storage[0].values;
        for (int i = lid; i < block * row_tile / 2; i += group_threads) {
          const int r = i % row_tile, pair = i / row_tile;
          const unsigned state = matrix.codes.state(row_base + r, 2 * g + pair / 128, pair % 128);
          const unsigned pair_values = matrix.lut[state];
          weights[(2 * pair) * row_tile + r] = int8_t(pair_values);
          weights[(2 * pair + 1) * row_tile + r] = int8_t(pair_values >> 8);
        }
        sycl::group_barrier(item.get_group());
        for (int token_base = (lid / warp) * token_tile; token_base < padded; token_base += token_warps * token_tile) {
          int8_tensor_tile(activations + token_base * cols + g * block, cols, weights,
                           partial + (size_t(g) * padded + token_base) * rows + row_base, rows);
        }
      });
    });
  }

  if (matrix.output_sign) {
    finish_prefill_projection<false>(queue, matrix, tokens, padded, partial, scales, nullptr, out);
    return;
  }

  const auto value = [=](int i) {
    const int token = i / rows;
    float sum = 0.f;
    for (int g = 0; g < groups; ++g)
      sum += float(partial[size_t(g) * padded * rows + i]) * scales[token * groups + g];
    return sum * matrix.scale[i % rows];
  };
  queue.parallel_for(sycl::range<1>(tokens * rows), [=](sycl::id<1> i) { out[i] = value(i[0]); });
}

inline void norm(sycl::queue& queue, const float* in, const uint16_t* weights, float* out, float eps,
                 int tokens = 1) {
  queue.parallel_for(sycl::nd_range<1>(tokens * 256, 256), [=](sycl::nd_item<1> item) {
    const int lane = item.get_local_id(0), base = item.get_group(0) * hidden;
    float sum = 0.f;
    for (int i = lane; i < hidden; i += 256) sum += in[base + i] * in[base + i];
    const float r = sycl::rsqrt(sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>())
                                 / hidden + eps);
    for (int i = lane; i < hidden; i += 256)
      out[base + i] = round_bf16(in[base + i] * r * (1.f + bf16(weights[i])));
  });
}

} // namespace yskn::sycl_backend
