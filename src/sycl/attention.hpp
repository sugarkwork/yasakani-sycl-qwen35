// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernels.hpp"

namespace yskn::sycl_backend {

struct FullAttention {
  Projection q, k, v, out;
  const uint16_t *qn = nullptr, *kn = nullptr;
  uint16_t *keys = nullptr, *values = nullptr;
};

struct LinearAttention {
  Projection mixed, z, out;
  const uint16_t *a = nullptr, *b = nullptr, *conv = nullptr;
  const uint16_t *a_log = nullptr, *dt = nullptr, *norm = nullptr;
  float* state = nullptr;
  uint16_t* history = nullptr;
};

inline void bf16_mma(float (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if defined(__NVPTX__)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

inline unsigned shared_address(const void* local) {
  unsigned address = 0;
#if defined(__NVPTX__)
  asm("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }"
      : "=r"(address) : "l"(local));
#endif
  return address;
}

template <int Registers, bool Transpose = false>
inline void load_bf16_fragment(unsigned (&values)[Registers], const uint16_t* local) {
#if defined(__NVPTX__)
  const unsigned address = shared_address(local);
  if constexpr (Registers == 4) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]) : "r"(address) : "memory");
  } else if constexpr (Transpose) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(values[0]), "=r"(values[1]) : "r"(address) : "memory");
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(values[0]), "=r"(values[1]) : "r"(address) : "memory");
  }
#endif
}

inline void prefill_attention_scores(sycl::queue& queue, const uint16_t* queries, const uint16_t* keys,
                                     float* scores, int position, int tokens) {
  constexpr int query_tile = 16, key_tile = 32, threads = 128, stride = head_dim + 8;
  constexpr int kv_width = kv_heads * head_dim, vector_width = 8;
  using Vec = sycl::vec<uint16_t, vector_width>;
  const int length = position + tokens, key_tiles = (length + key_tile - 1) / key_tile;
  const int query_tiles = (tokens + query_tile - 1) / query_tile;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<Vec, 1> scratch((query_tile + key_tile) * stride / vector_width, cmd);
    cmd.parallel_for(sycl::nd_range<1>(size_t(query_tiles) * query_heads * key_tiles * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
      const int tile = item.get_group(0), q0 = tile / (key_tiles * query_heads) * query_tile;
      const int head = (tile / key_tiles) % query_heads, k0 = tile % key_tiles * key_tile;
      const int end = position + sycl::min(tokens, q0 + query_tile);
      if (k0 >= end) return;
      const int lid = item.get_local_id(0), warp = lid / 32, lane = lid % 32;
      const int kv_head = head / (query_heads / kv_heads);
      for (int i = lid; i < (query_tile + key_tile) * (head_dim / vector_width); i += threads) {
        const int row = i / (head_dim / vector_width), col = (i % (head_dim / vector_width)) * vector_width;
        Vec data(0);
        if (row < query_tile) {
          if (q0 + row < tokens) data = *reinterpret_cast<const Vec*>(
              queries + (size_t(q0 + row) * query_heads + head) * head_dim + col);
        } else {
          const int time = k0 + row - query_tile;
          if (time < end) data = *reinterpret_cast<const Vec*>(
              keys + size_t(time) * kv_width + kv_head * head_dim + col);
        }
        scratch[(row * stride + col) / vector_width] = data;
      }
      sycl::group_barrier(item.get_group());
      const auto* shared = reinterpret_cast<const uint16_t*>(&scratch[0]);
      const auto* qa = shared + (lane & 15) * stride + (lane / 16) * 8;
      const auto* kb = shared + (query_tile + warp * 8 + (lane & 7)) * stride + ((lane / 8) & 1) * 8;
      float dot[4]{};
#pragma unroll
      for (int d = 0; d < head_dim; d += 16) {
        unsigned a[4], b[2];
        load_bf16_fragment(a, qa + d);
        load_bf16_fragment(b, kb + d);
        bf16_mma(dot, a, b);
      }
      const int key = k0 + warp * 8 + (lane % 4) * 2;
      for (int j = 0; j < 4; ++j) {
        const int row = q0 + lane / 4 + (j / 2) * 8, time = key + (j % 2);
        if (row < tokens && time <= position + row)
          scores[(size_t(row) * query_heads + head) * length + time] = dot[j] * 0.0625f;
      }
    });
  });
}

inline void prefill_attention_values(sycl::queue& queue, const float* scores, const uint16_t* values,
                                     const float* q, float* out, int position, int tokens) {
  constexpr int query_tile = 16, key_tile = 64, value_tile = 32, threads = 128, vector_width = 8;
  constexpr int score_stride = key_tile + 8, value_stride = value_tile + 8;
  constexpr int score_elements = query_tile * score_stride;
  constexpr int value_offset = 2 * score_elements;
  constexpr int kv_width = kv_heads * head_dim, value_tiles = head_dim / value_tile;
  using Vec = sycl::vec<uint16_t, vector_width>;
  const int length = position + tokens, query_tiles = (tokens + query_tile - 1) / query_tile;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<Vec, 1> scratch((value_offset + key_tile * value_stride) / vector_width, cmd);
    cmd.parallel_for(sycl::nd_range<1>(size_t(query_tiles) * query_heads * value_tiles * threads, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
      const int tile = item.get_group(0), head = (tile / value_tiles) % query_heads;
      const int q0 = tile / (value_tiles * query_heads) * query_tile, d0 = tile % value_tiles * value_tile;
      const int kv_head = head / (query_heads / kv_heads), end = position + sycl::min(tokens, q0 + query_tile);
      const int lid = item.get_local_id(0), warp = lid / 32, lane = lid % 32;
      auto* shared = reinterpret_cast<uint16_t*>(&scratch[0]);
      auto* residual = shared + score_elements;
      float sum[4]{};
      for (int k0 = 0; k0 < end; k0 += key_tile) {
        for (int i = lid; i < query_tile * key_tile; i += threads) {
          const int r = i / key_tile, k = i % key_tile, time = k0 + k;
          const float w = q0 + r < tokens && time <= position + q0 + r ?
              scores[(size_t(q0 + r) * query_heads + head) * length + time] : 0.f;
          const uint16_t hi = to_bf16(w);
          shared[r * score_stride + k] = hi;
          // Retain the low part of the FP32 probability as a second BF16 term.
          residual[r * score_stride + k] = to_bf16(w - bf16(hi));
        }
        for (int i = lid; i < key_tile * (value_tile / vector_width); i += threads) {
          const int k = i / (value_tile / vector_width), time = k0 + k;
          const int d = (i % (value_tile / vector_width)) * vector_width;
          Vec v(0);
          if (time < end) v = *reinterpret_cast<const Vec*>(
              values + size_t(time) * kv_width + kv_head * head_dim + d0 + d);
          scratch[(value_offset + k * value_stride + d) / vector_width] = v;
        }
        sycl::group_barrier(item.get_group());
        const auto* wa = shared + (lane & 15) * score_stride + (lane / 16) * 8;
        const auto* vb = shared + value_offset + ((lane & 7) + ((lane / 8) & 1) * 8) * value_stride + warp * 8;
#pragma unroll
        for (int k = 0; k < key_tile; k += 16) {
          unsigned a[4], low[4], b[2];
          load_bf16_fragment(a, wa + k);
          load_bf16_fragment(low, wa + k + score_elements);
          load_bf16_fragment<2, true>(b, vb + k * value_stride);
          bf16_mma(sum, a, b);
          bf16_mma(sum, low, b);
        }
        sycl::group_barrier(item.get_group());
      }
      for (int j = 0; j < 4; ++j) {
        const int row = q0 + lane / 4 + (j / 2) * 8, d = d0 + warp * 8 + (lane % 4) * 2 + j % 2;
        if (row < tokens) {
          const size_t h = size_t(row) * query_heads + head;
          const float gate = round_bf16(sigmoid(round_bf16(q[h * 2 * head_dim + head_dim + d])));
          out[h * head_dim + d] = round_bf16(round_bf16(sum[j]) * gate);
        }
      }
    });
  });
}

inline void rope(sycl::queue& queue, const float* in, const uint16_t* weights,
                 uint16_t* out, int heads, int stride, int position, float theta, float eps, int tokens,
                 const int32_t* live_position, bool cache) {
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<float, 1> normed(rotary_dim, cmd);
    cmd.parallel_for(sycl::nd_range<1>(tokens * heads * head_dim, head_dim), [=](sycl::nd_item<1> item) {
      const int head = item.get_group(0), d = item.get_local_id(0);
      const int base = live_position ? *live_position : position;
      const float x = round_bf16(in[head * stride + d]);
      const float ss = sycl::reduce_over_group(item.get_group(), x * x, sycl::plus<float>());
      const float normalized = round_bf16(x * sycl::rsqrt(ss / head_dim + eps) * (1.f + bf16(weights[d])));
      if (d < rotary_dim) normed[d] = normalized;
      sycl::group_barrier(item.get_group());
      float rotated = normalized;
      if (d < rotary_dim) {
        const int pair = d % (rotary_dim / 2);
        const float phase = (base + head / heads) * sycl::pow(theta, -2.f * pair / rotary_dim);
        const float other = normed[d ^ (rotary_dim / 2)] * (d < rotary_dim / 2 ? -1.f : 1.f);
        rotated = round_bf16(normalized * round_bf16(sycl::cos(phase))) +
                  round_bf16(other * round_bf16(sycl::sin(phase)));
      }
      const size_t out_head = cache ? size_t(base) * heads + head : head;
      out[out_head * head_dim + d] = to_bf16(rotated);
    });
  });
}

inline void attend(sycl::queue& queue, FullAttention layer, const float* q, const float* k,
                   const float* v, uint16_t* q16, float* scores, float* out,
                   int position, float theta, float eps, int tokens,
                   const int32_t* live_position = nullptr, int context = 0) {
  const int max_len = live_position ? context : position + tokens;
  const int kv_width = kv_heads * head_dim;
  const int query_width = query_heads * head_dim;
  rope(queue, q, layer.qn, q16, query_heads, 2 * head_dim, position, theta, eps, tokens,
       live_position, false);
  rope(queue, k, layer.kn, layer.keys, kv_heads, head_dim, position, theta, eps, tokens,
       live_position, true);
  queue.parallel_for(sycl::range<1>(tokens * kv_width), [=](sycl::id<1> index) {
    const int base = live_position ? *live_position : position;
    layer.values[size_t(base) * kv_width + index[0]] = to_bf16(v[index]);
  });

  if (live_position) {
    // The graph keeps its launch dimensions constant. Each warp strides over
    // the live KV length to avoid running work proportional to max context.
    constexpr int warp = 32, threads = 128, blocks_per_head = 32;
    queue.parallel_for(sycl::nd_range<1>(query_heads * blocks_per_head * threads, threads),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
      const int block_id = item.get_group(0), head = block_id / blocks_per_head;
      const int limit = *live_position + 1;
      const int start = (block_id % blocks_per_head) * (threads / warp) + item.get_local_id(0) / warp;
      const auto sg = item.get_sub_group();
      const int lane = sg.get_local_linear_id();
      const int kv_head = head / (query_heads / kv_heads);
      const auto* qr = q16 + head * head_dim;
      for (int t = start; t < limit; t += blocks_per_head * (threads / warp)) {
        const auto* kr = layer.keys + size_t(t) * kv_width + kv_head * head_dim;
        float dot = 0.f;
        for (int d = lane; d < head_dim; d += warp) dot += bf16(qr[d]) * bf16(kr[d]);
        const float total = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        if (lane == 0) scores[size_t(head) * max_len + t] = total * 0.0625f;
      }
    });
  } else if (tokens == 1) {
    queue.parallel_for(sycl::nd_range<1>(size_t(query_heads) * max_len * head_dim, head_dim),
                       [=](sycl::nd_item<1> item) {
      const int head = item.get_group(0) / max_len, time = item.get_group(0) % max_len;
      const int d = item.get_local_id(0), kv_head = head / (query_heads / kv_heads);
      const float dot = bf16(q16[head * head_dim + d]) *
                        bf16(layer.keys[size_t(time) * kv_width + kv_head * head_dim + d]);
      const float total = sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>());
      if (d == 0) scores[head * max_len + time] = total * 0.0625f;
    });
  } else {
    prefill_attention_scores(queue, q16, layer.keys, scores, position, tokens);
  }
  queue.parallel_for(sycl::nd_range<1>(tokens * query_width, head_dim), [=](sycl::nd_item<1> item) {
    const int head = item.get_group(0), d = item.get_local_id(0);
    const int base = live_position ? *live_position : position;
    const int len = base + head / query_heads + 1;
    float* row = scores + size_t(head) * max_len;
    float maximum = -INFINITY;
    for (int t = d; t < len; t += head_dim) maximum = sycl::fmax(maximum, row[t]);
    maximum = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>());
    float sum = 0.f;
    for (int t = d; t < len; t += head_dim) {
      row[t] = sycl::exp(row[t] - maximum);
      sum += row[t];
    }
    const float total = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    for (int t = d; t < len; t += head_dim) row[t] /= total;
  });
  if (tokens > 1) {
    prefill_attention_values(queue, scores, layer.values, q, out, position, tokens);
    return;
  }
  if (live_position) {
    // Parallelize the single query's value reduction over KV time, while each
    // warp reads a contiguous slice of the BF16 value cache.
    constexpr int warp = 32, parts = 8, threads = warp * parts;
    queue.submit([&](sycl::handler& cmd) {
      sycl::local_accessor<float, 1> reduced(threads, cmd);
      cmd.parallel_for(sycl::nd_range<1>(query_heads * (head_dim / warp) * threads, threads),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(warp)]] {
        const int tile = item.get_group(0), lid = item.get_local_id(0);
        const int head = tile / (head_dim / warp), part = lid / warp;
        const int d = (tile % (head_dim / warp)) * warp + lid % warp;
        const int kv_head = head / (query_heads / kv_heads), limit = *live_position + 1;
        const auto* probabilities = scores + size_t(head) * max_len;
        const auto* values = layer.values + kv_head * head_dim + d;
        float sum = 0.f;
        for (int time = part; time < limit; time += parts)
          sum += probabilities[time] * bf16(values[size_t(time) * kv_width]);
        reduced[lid] = sum;
        sycl::group_barrier(item.get_group());
        if (part == 0) {
          for (int p = 1; p < parts; ++p) sum += reduced[p * warp + lid];
          const float gate = round_bf16(sigmoid(round_bf16(q[head * head_dim * 2 + head_dim + d])));
          out[head * head_dim + d] = round_bf16(round_bf16(sum) * gate);
        }
      });
    });
    return;
  }
  queue.parallel_for(sycl::range<1>(tokens * query_width), [=](sycl::id<1> index) {
    const int i = index[0], head = i / head_dim, d = i % head_dim;
    const int kv_head = (head % query_heads) / (query_heads / kv_heads);
    const int base = live_position ? *live_position : position;
    const int len = base + head / query_heads + 1;
    float sum = 0.f;
    for (int t = 0; t < len; ++t) {
      sum += scores[size_t(head) * max_len + t] *
             bf16(layer.values[size_t(t) * kv_width + kv_head * head_dim + d]);
    }
    const float gate = round_bf16(sigmoid(round_bf16(q[head * head_dim * 2 + head_dim + d])));
    out[i] = round_bf16(round_bf16(sum) * gate);
  });
}

inline void stage_float4(float* shared, const float* global) {
#if defined(__NVPTX__)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
      :: "r"(shared_address(shared)), "l"(global) : "memory");
#endif
}

template <bool Single = false>
void tiled_delta_state(sycl::queue& queue, float* state, const float* convolved,
                        const float* beta, const float* decay, float* core, int tokens) {
  constexpr int parts = 4, columns = 32, threads = parts * columns, time_tile = Single ? 1 : 8;
  constexpr int key_count = linear_dim / parts, key_stride = key_count + 4;
  constexpr int row_stride = parts * key_stride, qk_elements = time_tile * row_stride;
  constexpr int values_offset = 2 * qk_elements, decay_offset = values_offset + time_tile * columns;
  constexpr int stage_size = (decay_offset + 2 * time_tile + 3) / 4 * 4;
  using Vec = sycl::vec<float, 4>;
  queue.submit([&](sycl::handler& cmd) {
    sycl::local_accessor<Vec, 1> scratch((Single ? 1 : 2) * stage_size / 4, cmd);
    cmd.parallel_for(sycl::nd_range<1>(linear_width * parts, threads),
                     [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
      const int lid = item.get_local_id(0), tile = item.get_group(0), part = lid % parts;
      const int head = tile / (linear_dim / columns), col0 = tile % (linear_dim / columns) * columns;
      const int column = lid / parts, key_head = head / (linear_values / linear_keys);
      const int value = head * linear_dim + col0 + column;
      auto* smem = reinterpret_cast<float*>(&scratch[0]);
      float* s = state + (head * linear_dim + part * key_count) * linear_dim + col0 + column;
      float local_state[key_count];
#pragma unroll
      for (int j = 0; j < key_count; ++j) local_state[j] = s[j * linear_dim];
      const auto sg = item.get_sub_group();
      auto sum_parts = [=](float term) {
        term += sycl::permute_group_by_xor(sg, term, 1);
        return term + sycl::permute_group_by_xor(sg, term, 2);
      };
      const int token_count = Single ? 1 : tokens;
      auto stage = [=](int t0, float* buffer) {
        const int count = sycl::min(time_tile, token_count - t0);
        for (int i = lid; i < count * (linear_dim / 4); i += threads) {
          const int t = i / (linear_dim / 4), k = (i % (linear_dim / 4)) * 4;
          const auto* from = convolved + size_t(t0 + t) * mixed_size + key_head * linear_dim + k;
          auto* to = buffer + t * row_stride + (k / key_count) * key_stride + k % key_count;
          stage_float4(to, from);
          stage_float4(to + qk_elements, from + linear_keys * linear_dim);
        }
        for (int i = lid; i < count * (columns / 4); i += threads) {
          const int t = i / (columns / 4), d = (i % (columns / 4)) * 4;
          const auto* from = convolved + size_t(t0 + t) * mixed_size + 2 * linear_keys * linear_dim;
          stage_float4(buffer + values_offset + t * columns + d, from + head * linear_dim + col0 + d);
        }
        if (lid < count) {
          buffer[decay_offset + lid] = decay[(t0 + lid) * linear_values + head];
          buffer[decay_offset + time_tile + lid] = beta[(t0 + lid) * linear_values + head];
        }
#if defined(__NVPTX__)
        asm volatile("cp.async.commit_group;" ::: "memory");
#endif
      };
      stage(0, smem);
      for (int t0 = 0, slot = 0; t0 < token_count; t0 += time_tile, slot ^= 1) {
#if defined(__NVPTX__)
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
        sycl::group_barrier(item.get_group());
        if (!Single && t0 + time_tile < token_count) stage(t0 + time_tile, smem + (slot ^ 1) * stage_size);
        const auto* buffer = smem + slot * stage_size;
        const int count = sycl::min(time_tile, token_count - t0);
        for (int t = 0; t < count; ++t) {
          const auto* q = buffer + t * row_stride + part * key_stride;
          const auto* k = q + qk_elements;
          const float decay_now = buffer[decay_offset + t];
          float estimate = 0.f;
#pragma unroll
          for (int j = 0; j < key_count; ++j) {
            local_state[j] *= decay_now;
            estimate += local_state[j] * k[j];
          }
          const float v = buffer[values_offset + t * columns + column];
          const float delta = (v - sum_parts(estimate)) * buffer[decay_offset + time_tile + t];
          float result = 0.f;
#pragma unroll
          for (int j = 0; j < key_count; ++j) {
            local_state[j] += k[j] * delta;
            result += local_state[j] * q[j];
          }
          result = sum_parts(result);
          if (part == 0) core[(t0 + t) * linear_width + value] = round_bf16(result);
        }
      }
#pragma unroll
      for (int j = 0; j < key_count; ++j) s[j * linear_dim] = local_state[j];
    });
  });
}

inline void delta_net(sycl::queue& queue, LinearAttention layer, const float* normalized,
                      const float* mixed, const float* z, float* convolved, float* core,
                      float* beta, float* decay, float* out, float eps, int tokens, bool streaming) {
  queue.parallel_for(sycl::nd_range<1>(tokens * linear_values * 128, 128), [=](sycl::nd_item<1> item) {
    const int row = item.get_group(0), head = row % linear_values;
    const int base = (row / linear_values) * hidden, lane = item.get_local_id(0);
    float a = 0.f, b = 0.f;
    for (int i = lane; i < hidden; i += 128) {
      const float x = round_bf16(normalized[base + i]);
      a += x * bf16(layer.a[head * hidden + i]);
      b += x * bf16(layer.b[head * hidden + i]);
    }
    a = sycl::reduce_over_group(item.get_group(), a, sycl::plus<float>()) + bf16(layer.dt[head]);
    b = sycl::reduce_over_group(item.get_group(), b, sycl::plus<float>());
    if (lane == 0) {
      beta[row] = round_bf16(sigmoid(b));
      const float softplus = a > 20.f ? a : sycl::log1p(sycl::exp(a));
      decay[row] = sycl::exp(-sycl::exp(bf16(layer.a_log[head])) * softplus);
    }
  });

  queue.parallel_for(sycl::nd_range<1>(tokens * mixed_size, linear_dim), [=](sycl::nd_item<1> item) {
    const int index = item.get_global_id(0), t = index / mixed_size, i = index % mixed_size;
    auto h = layer.history + i * 3;
    const auto w = layer.conv + i * 4;
    auto input_at = [=](int pos) {
      return pos < 0 ? h[pos + 3] : to_bf16(mixed[pos * mixed_size + i]);
    };
    const float c = bf16(input_at(t - 3)) * bf16(w[0]) + bf16(input_at(t - 2)) * bf16(w[1]) +
                    bf16(input_at(t - 1)) * bf16(w[2]) + bf16(input_at(t)) * bf16(w[3]);
    if (tokens == 1) {
      h[0] = h[1];
      h[1] = h[2];
      h[2] = input_at(t);
    }
    float act = round_bf16(silu(round_bf16(c)));
    if (i < 2 * linear_keys * linear_dim) {
      const auto ss = sycl::reduce_over_group(item.get_group(), act * act, sycl::plus<float>());
      act *= sycl::rsqrt(ss + 1.e-6f);
      if (i < linear_keys * linear_dim) act *= 0.08838834764831845f;
    }
    convolved[index] = act;
  });

  if (tokens > 1) {
    // Commit the last three inputs only after all convolution rows have read the old history.
    queue.parallel_for(sycl::range<1>(mixed_size), [=](sycl::id<1> index) {
      const int i = index[0];
      auto* h = layer.history + i * 3;
      for (int k = 0; k < 3; ++k) {
        const int pos = tokens - 3 + k;
        h[k] = pos < 0 ? h[pos + 3] : to_bf16(mixed[pos * mixed_size + i]);
      }
    });
    // Preserve the token order and overlap computation with the next input tile.
    tiled_delta_state(queue, layer.state, convolved, beta, decay, core, tokens);
  } else if (streaming) {
    tiled_delta_state<true>(queue, layer.state, convolved, beta, decay, core, 1);
  } else {
    queue.parallel_for(sycl::range<1>(linear_width), [=](sycl::id<1> index) {
      const int i = index[0], h = i / linear_dim, d = i % linear_dim;
      const int key_head = h / (linear_values / linear_keys);
      const float* q = convolved + key_head * linear_dim;
      const float* k = q + linear_keys * linear_dim;
      float* state = layer.state + h * linear_dim * linear_dim + d;
      float estimate = 0.f;
      for (int j = 0; j < linear_dim; ++j) {
        state[j * linear_dim] *= decay[h];
        estimate += state[j * linear_dim] * k[j];
      }
      const float delta = (convolved[2 * linear_keys * linear_dim + i] - estimate) * beta[h];
      float result = 0.f;
      for (int j = 0; j < linear_dim; ++j) {
        state[j * linear_dim] += k[j] * delta;
        result += state[j * linear_dim] * q[j];
      }
      core[i] = round_bf16(result);
    });
  }
  queue.parallel_for(sycl::nd_range<1>(tokens * linear_width, linear_dim), [=](sycl::nd_item<1> item) {
    const int i = item.get_global_id(0), d = item.get_local_id(0);
    const float c = core[i];
    const float sum = sycl::reduce_over_group(item.get_group(), c * c, sycl::plus<float>());
    const float value = round_bf16(bf16(layer.norm[d]) *
                                  round_bf16(c * sycl::rsqrt(sum / linear_dim + eps)));
    out[i] = round_bf16(value * silu(round_bf16(z[i])));
  });
}

} // namespace yskn::sycl_backend
