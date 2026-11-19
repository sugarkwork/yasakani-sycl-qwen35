# Build and run Qwen3.5 on Windows

Use an x64 Visual Studio 2022 developer shell with the C++ toolchain, CUDA Toolkit 12.9, CMake 3.24 or later, and Ninja. Run the PowerShell examples from the repository root.

## SYCL DLLs for Qwen3.5-4B and Qwen3.5-9B

The tested compiler is the Windows build of [Intel LLVM nightly 2026-11-10](https://github.com/intel/llvm/releases/tag/nightly-2026-11-10) with DPC++ and its CUDA adapter. Set `$sycl` to the `bin` directory from that toolchain.

```powershell
$sycl = 'C:/tools/dpcpp/bin'
$env:Path = "$sycl;$env:Path"
$syclArgs = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DYSKN_ENABLE_SYCL=ON', '-DYSKN_SYCL_CUDA_ARCH=sm_120', "-DCMAKE_CXX_COMPILER=$sycl/clang-cl.exe")
cmake -S . -B build-sycl-4b @syclArgs -DYSKN_Q35_HIDDEN=2560
cmake --build build-sycl-4b
cmake -S . -B build-sycl-9b @syclArgs -DYSKN_Q35_HIDDEN=4096
cmake --build build-sycl-9b
```

For an RTX 3060, use `-DYSKN_SYCL_CUDA_ARCH=sm_86`. Use a separate build directory for each GPU architecture and model size. The DLL is under `src/sycl` in the build directory. Keep the DPC++ runtime and CUDA adapter discoverable when loading it. The target is the NVIDIA CUDA backend of SYCL. See the [results](QWEN35-BENCHMARKS.md) for the hardware tested with each size.

The DLL reads a plain text `.yskn` pack of the selected size and stores K/V in BF16. Its exported C functions are `yskn_open`, `yskn_close`, `yskn_last_error`, `yskn_reset`, `yskn_position`, `yskn_prefill`, `yskn_decode`, and `yskn_logits`. Use `flags=0`, pass token IDs, and load the tokenizer separately. The [benchmark runner](../tools/experiments/qwen35/benchmark_sycl.py) shows the supported calling sequence. Code that requires the complete upstream CUDA ABI should use the CUDA library.

The backend batches prefill, uses NVIDIA Tensor Cores for packed projection and BF16 prefill attention, and replays a SYCL command graph for streaming decode when supported. `YSKN_SYCL_PREFILL_ROWS` sets the maximum prefill chunk size from 1 to 256. `YSKN_SYCL_EAGER_DECODE=1` disables graph replay. `YSKN_SYCL_DP4A_DECODE=1` selects the DP4A projection path. These settings are read when the engine opens.

## CUDA DLLs for Qwen3.5-4B and Qwen3.5-9B

Build each model size into its own directory. Both small-model builds use BF16 KV. Set `$arch` to `120` for RTX 5060 Ti or `86` for RTX 3060. The published measurements for 9B cover the RTX 5060 Ti.

```powershell
$arch = '120'
cmake -S . -B build-cuda-4b -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DYSKN_Q35_HIDDEN=2560
cmake --build build-cuda-4b
cmake -S . -B build-cuda-9b -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DYSKN_Q35_HIDDEN=4096
cmake --build build-cuda-9b
```

Each build produces `yskn.dll` at the root of its build directory. Use the DLL built for the pack's model size with either backend.

## Model packs

This repository distributes source and benchmark records. Obtain or convert compatible `.yskn` packs and keep the source tokenizer files. `speed` uses 3-bit projection weights and `quality` uses 4-bit projection weights. In the converter, these correspond to `--bits 6` and `--bits 8`, since each code represents two weights. The tested text packs use a 3-bit embedding and a 4-bit output head in both modes. The pack hashes used for testing are in the [benchmark records](../bench/qwen35/results/).

The [conversion script](../tools/convert/convert_qwen35.py) downloads the pinned original checkpoint by default, prepares calibration, quantizes the text weights, and writes `speed.yskn` and `quality.yskn` with the tokenizer files. Both DLL backends use these same packs. Run the conversion on Linux or WSL2 with Python 3.11+, CUDA-enabled PyTorch, NumPy, Safetensors, PyArrow, Transformers (5.5.0 for the original conversion), and the Hugging Face CLI. The native Trellis library uses CUDA. Build it for your conversion GPU (`120` for RTX 5060 Ti, `89` for L4, `86` for RTX 3060). The build architecture does not affect which DLL can use the resulting packs.

| Size | Source repository | Pinned revision |
|---|---|---|
| 4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| 9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

```bash
cmake -S tools/convert/native -B build-trellis -DCMAKE_BUILD_TYPE=Release -DYSKN_TRELLIS_CUDA_ARCHS=120
cmake --build build-trellis
native=build-trellis/bin/libyskn_trellis.so
python tools/convert/convert_qwen35.py 4b --native "$native"
python tools/convert/convert_qwen35.py 9b --native "$native"
```

The default output directories are `assets/4b` and `assets/9b`. With `--mode speed` or `--mode quality`, the script builds just that pack. `--checkpoint` uses an existing sharded safetensors checkpoint and `--calibration` supplies existing `.npy` token windows. The default calibration uses the pinned public WikiText dataset. Use `--output` for a separate destination and repeat the same command to resume. Run one conversion at a time per output directory. The `.conversion` subdirectory holds the download, calibration, and intermediate files. Allow disk space for those files as well as the packs.

## Run the benchmark

Install NumPy and Transformers in a Python 3.11 or later environment. The runner uses `nvidia-smi`; use `nvidia-smi -L` to select the GPU UUID. Replace `$gpu` with the desired device, and set `$assets` to the folder containing the input pack and tokenizer files. The benchmark uses the committed, pretokenized [prompt manifest](../bench/qwen35/prompts.json), including scoring criteria and generation limits. The runner refuses to overwrite an existing report.

```powershell
$gpu = 'GPU-REPLACE-WITH-YOUR-DEVICE-UUID'
$bench = 'tools/experiments/qwen35/benchmark_sycl.py'
$prompts = 'bench/qwen35/prompts.json'
$assets = 'assets/4b'
python $bench run sycl build-sycl-4b/src/sycl/yskn_sycl.dll "$assets/speed.yskn" $assets $prompts benchmark-output/sycl-4b-speed.json --gpu $gpu --runtime $sycl --context 2048
python $bench run cuda build-cuda-4b/yskn.dll "$assets/speed.yskn" $assets $prompts benchmark-output/cuda-4b-speed.json --gpu $gpu --context 2048
$assets = 'assets/9b'
python $bench run sycl build-sycl-9b/src/sycl/yskn_sycl.dll "$assets/speed.yskn" $assets $prompts benchmark-output/sycl-9b-speed.json --gpu $gpu --runtime $sycl --context 2048
python $bench run cuda build-cuda-9b/yskn.dll "$assets/speed.yskn" $assets $prompts benchmark-output/cuda-9b-speed.json --gpu $gpu --context 2048
```

Repeat with `quality.yskn` and a different output filename for the quality pack. Add `--smoke` for the first two arithmetic cases. Both tested revisions produce the committed token manifest. To retokenize the fixture with your local checkpoint and write a fresh manifest, use `python $bench prepare $assets bench/qwen35 benchmark-output/prompts.json`.

Use [LICENSE](../LICENSE), [NOTICE](../NOTICE), and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for source attribution. The source checkpoints have their own license files at the pinned revisions above.
