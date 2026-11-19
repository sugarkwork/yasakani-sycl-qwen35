# YASAKANI SYCL Qwen3.5

[日本語](README.ja.md) · [Windows build and model conversion](docs/WINDOWS-QWEN35.md) · [Benchmarks](docs/QWEN35-BENCHMARKS.md)

An experimental source fork of [YASAKANI v0.1.6](https://github.com/yasakani-dev/yasakani/tree/v0.1.6) for local Qwen3.5 text inference. The native Windows SYCL DLL uses the NVIDIA CUDA backend of SYCL. It supports YASAKANI model packs, batched prefill, and one-token streaming decode.

| Model | Backend | GPU verification |
|---|---|---|
| Qwen3.5-4B | SYCL and CUDA | Windows, RTX 5060 Ti and RTX 3060, both speed and quality |
| Qwen3.5-9B | SYCL and CUDA | Windows, RTX 5060 Ti, both speed and quality |

Build the DLL for the model size and GPU architecture you want to use. The 9B SYCL build also compiles for `sm_86` (RTX 3060). The table lists the hardware used for inference.

On the RTX 5060 Ti, SYCL generated **161.54 / 140.44 tokens/s** with 4B and **102.89 / 85.26 tokens/s** with 9B (speed / quality) in the long reading test. In the RTX 3060 run, 4B reached **114.50 / 104.18 tokens/s**. See the [benchmark report](docs/QWEN35-BENCHMARKS.md) for the CUDA comparison, TTFT, VRAM and answer scores.

This repository contains the source, test inputs and benchmark records. Download the speed and quality packs from Hugging Face. [Qwen3.5-4B](https://huggingface.co/sugarknight/Qwen3.5-4B-YASAKANI) · [Qwen3.5-9B](https://huggingface.co/sugarknight/Qwen3.5-9B-YASAKANI)

The [build guide](docs/WINDOWS-QWEN35.md) covers the toolchain and C API. The [conversion script](tools/convert/convert_qwen35.py) builds speed and quality packs for 4B and 9B. The same pack works with the CUDA and SYCL DLLs built for its size.

The source fork retains the upstream [Apache-2.0 license](LICENSE), [NOTICE](NOTICE), and [third-party notices](THIRD_PARTY_NOTICES.md). Model checkpoints are available separately from the Qwen team.
