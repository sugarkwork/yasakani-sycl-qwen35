"""Compare single-request text generation with identical Qwen prompts on one GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes as C
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from transformers import AutoTokenizer

ARITHMETIC = (
    ("a01", "728 + 465", 1193), ("a02", "902 - 174", 728),
    ("a03", "63 × 47", 2961), ("a04", "1575 ÷ 35", 45),
    ("a05", "(-18) + 27", 9), ("a06", "56 - 93", -37),
    ("a07", "18 × 24", 432), ("a08", "3456 ÷ 64", 54),
    ("a09", "(36 + 54) ÷ 6", 15), ("a10", "91 - 7 × 8", 35),
    ("a11", "125 × 16 - 450", 1550), ("a12", "(455 - 95) ÷ (9 + 3)", 30),
)
PROSE = (
    ("theme01", "小学生に向けて、なぜ月の満ち欠けが起こるのかを日本語で3文で説明してください。"),
    ("theme02", "都市でヒートアイランドが起こる原因を2つ、対策を2つ、理由とともに日本語で説明してください。"),
    ("theme03", "コンピューターでキャッシュが効く理由と、古いデータが返る問題の対策を日本語で説明してください。"),
    ("theme04", "地球の季節が変わる理由を日本語で説明し、北半球と南半球の季節の関係にも触れてください。"),
    ("theme05", "Write a short explanation of public-key encryption for a beginner. Explain which key encrypts a message and which one decrypts it."),
)

def gpu_used():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

@contextmanager
def memory_sampler(read_used=None):
    if read_used is None:
        read_used = gpu_used
    idle = read_used()
    values = [idle]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.1):
            values.append(read_used())

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    memory = {"idle_mib": idle}
    try:
        yield memory
    finally:
        stop.set()
        thread.join()
        memory["peak_mib"] = max(values)
        memory["peak_delta_mib"] = max(values) - idle

INT = C.c_int32
PTR = C.POINTER(INT)

class Native:
    def __init__(self, binary, model, eos, *, context=2048, device=0):
        cuda_path = os.environ.get("CUDA_PATH")
        if os.name == "nt" and cuda_path and (Path(cuda_path) / "bin").is_dir():
            with os.add_dll_directory(str(Path(cuda_path) / "bin")):
                self.lib = C.CDLL(str(binary))
        else:
            self.lib = C.CDLL(str(binary))
        for name, restype, args in (
            ("yskn_open", C.c_void_p, (C.c_char_p, INT, INT, INT)),
            ("yskn_close", None, (C.c_void_p,)),
            ("yskn_reset", INT, (C.c_void_p,)),
            ("yskn_prefill", INT, (C.c_void_p, PTR, INT, PTR)),
            ("yskn_decode", INT, (C.c_void_p, PTR, INT)),
            ("yskn_logits", INT, (C.c_void_p, C.POINTER(C.c_float))),
            ("yskn_last_error", C.c_char_p, ()),
        ):
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, args
        start = time.perf_counter()
        self.engine = self.lib.yskn_open(os.fsencode(model), context, device, 0)
        if not self.engine:
            self.check(-1)
        self.load_s = time.perf_counter() - start
        self.eos = eos

    def close(self):
        self.lib.yskn_close(self.engine)

    def check(self, code):
        if code != 0:
            raise RuntimeError(self.lib.yskn_last_error().decode(errors="replace"))

    def complete(self, prompt, limit):
        ids = (INT * len(prompt))(*prompt)
        token = INT()
        self.check(self.lib.yskn_reset(self.engine))
        start = time.perf_counter()
        self.check(self.lib.yskn_prefill(self.engine, ids, len(prompt), C.byref(token)))
        first = time.perf_counter()
        output = [token.value]
        while len(output) < limit and output[-1] not in self.eos:
            self.check(self.lib.yskn_decode(self.engine, C.byref(token), 1))
            output.append(token.value)
        decode_s = time.perf_counter() - first
        return output, first - start, decode_s, {}

    def first_logits(self, prompt, vocab):
        ids = (INT * len(prompt))(*prompt)
        token = INT()
        logits = np.empty(vocab, np.float32)
        self.check(self.lib.yskn_reset(self.engine))
        self.check(self.lib.yskn_prefill(self.engine, ids, len(prompt), C.byref(token)))
        self.check(self.lib.yskn_logits(self.engine, logits.ctypes.data_as(C.POINTER(C.c_float))))
        if np.argmax(logits) != token.value or not np.isfinite(logits).all():
            raise RuntimeError("Native logits do not agree with greedy prefill")
        return logits

class Server:
    def __init__(self, backend, binary, model, eos, log):
        del eos
        self.backend = backend
        self.log = log.open("w", encoding="utf-8")
        self.url = "http://127.0.0.1:18081"
        if backend == "llama":
            command = [str(binary), "-m", str(model), "-ngl", "999",
                "--ctx-size", "2048", "--parallel", "1", "-fa", "on", "--host", "127.0.0.1", "--port", "18081",
                "--cache-ram", "0", "--no-context-shift"]
            runtime_dirs = {binary.parent}
            runtime_dirs.update(path.parent for path in binary.parent.parent.rglob("libcudart.so.*"))
            library_path = list(map(str, sorted(runtime_dirs)))
            if os.environ.get("LD_LIBRARY_PATH"):
                library_path.append(os.environ["LD_LIBRARY_PATH"])
            env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(library_path))
            devices = subprocess.run([str(binary), "--list-devices"], env=env, capture_output=True,
                                     text=True, check=True)
            if not re.search(r"CUDA\d+:.*NVIDIA L4", devices.stdout + devices.stderr):
                raise RuntimeError("llama.cpp did not discover the Colab CUDA device")
        else:
            command = [str(binary), "serve", str(model), "--served-model-name", "target",
                "--host", "127.0.0.1", "--port", "18081", "--dtype", "bfloat16", "--language-model-only",
                "--max-model-len", "2048", "--max-num-seqs", "1", "--no-enable-prefix-caching",
                "--kv-cache-memory-bytes", "805306368", "--gpu-memory-utilization", "0.94"]
            env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
        start = time.perf_counter()
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            while time.perf_counter() - start < 300:
                if self.process.poll() is not None:
                    raise RuntimeError(f"{backend} stopped with {self.process.returncode}, see {log}")
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                        if response.status == 200:
                            self.load_s = time.perf_counter() - start
                            return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.2)
            raise TimeoutError(f"{backend} health timeout")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def complete(self, prompt, limit):
        if self.backend == "llama":
            path = "/completion"
            payload = dict(prompt=prompt, n_predict=limit, temperature=0, stream=True,
                           cache_prompt=False, return_tokens=True, seed=1234)
        else:
            path = "/v1/completions"
            payload = dict(model="target", prompt=prompt, max_tokens=limit, temperature=0, stream=True,
                           return_token_ids=True, seed=1234)
        request = urllib.request.Request(self.url + path, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        output, first, stats = [], None, {}
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                event = json.loads(value)
                if self.backend == "llama":
                    chunk = event.get("tokens", [])
                    if event.get("stop_type"):
                        stats["finish_reason"] = event["stop_type"]
                else:
                    choices = event.get("choices") or []
                    chunk = (choices[0].get("token_ids") or []) if choices else []
                    if choices and choices[0].get("finish_reason"):
                        stats["finish_reason"] = choices[0]["finish_reason"]
                if chunk and first is None:
                    first = time.perf_counter()
                output.extend(chunk)
                stats.update(event.get("timings", {}))
        end = time.perf_counter()
        if not output or first is None:
            raise RuntimeError(f"{self.backend} stream returned no token IDs")
        return output, first - start, end - first, stats

class Transformers:
    def __init__(self, binary, model, eos, log):
        del binary, log
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        self.torch = torch
        self.eos = eos
        start = time.perf_counter()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - start

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()

    def complete(self, prompt, limit):
        class Timer:
            first = None
            seen_prompt = False
            def put(self, value):
                if not self.seen_prompt:
                    self.seen_prompt = True
                elif self.first is None:
                    self.first = time.perf_counter()
            def end(self):
                pass

        torch = self.torch
        streamer = Timer()
        tokens = torch.tensor([prompt], device="cuda")
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(input_ids=tokens, max_new_tokens=limit, do_sample=False,
                eos_token_id=sorted(self.eos), pad_token_id=min(self.eos), streamer=streamer)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if streamer.first is None:
            raise RuntimeError("Transformers returned no token")
        return generated[0, len(prompt):].tolist(), streamer.first - start, end - streamer.first, {}

    def first_logits(self, prompt, vocab):
        torch = self.torch
        ids = torch.tensor([prompt], device="cuda")
        with torch.inference_mode():
            logits = self.model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float().cpu().numpy()
        return logits[:vocab]

def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

def stop_ids(tok, model):
    gen = model / "generation_config.json"
    config = gen if gen.exists() else model / "config.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    eos = data.get("eos_token_id", data.get("text_config", {}).get("eos_token_id"))
    ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    for token in ("<|im_end|>", "<|endoftext|>"):
        if token in tok.get_vocab():
            ids.add(tok.convert_tokens_to_ids(token))
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["native", "llama", "vllm", "transformers"])
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--logits", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eos = stop_ids(tok, args.tokenizer)

    def encode(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)

    paths = {"model": str(args.model), "binary": str(args.binary)}
    report = {"engine": args.engine, **paths, "arithmetic": [], "prose": []}
    with memory_sampler() as memory:
        if args.engine == "native":
            runner = Native(args.binary.resolve(), args.model.resolve(), eos)
        elif args.engine == "transformers":
            runner = Transformers(None, args.model.resolve(), eos, args.out.with_suffix(".log"))
        else:
            runner = Server(args.engine, args.binary.resolve(), args.model.resolve(), eos, args.out.with_suffix(".log"))
        report["load_s"] = runner.load_s
        memory["loaded_mib"] = gpu_used()
        if memory["loaded_mib"] - memory["idle_mib"] < 512:
            runner.close()
            raise RuntimeError("GPU VRAM check failed after model load")
        try:
            runner.complete(encode("日本語で挨拶してください。"), 12)
            arith = ARITHMETIC[:4] if args.smoke else ARITHMETIC
            prose = PROSE[:1] if args.smoke else PROSE
            tests = [("arithmetic", name, f"次の式を計算してください。答えの整数だけを出力してください。\n{expr}", expected, 20)
                     for name, expr, expected in arith]
            tests += [("prose", name, prompt, None, 192) for name, prompt in prose]
            for kind, name, text, expected, cap in tests:
                prompt = encode(text)
                ids, ttft, decode, extra = runner.complete(prompt, cap)
                visible = tok.decode(ids, skip_special_tokens=True).strip()
                emitted = len(ids) - int(ids[-1] in eos)
                case = {"name": name, "prompt": text, "prompt_tokens": len(prompt), "max_tokens": cap,
                        "output": visible, "generated_tokens": emitted, "ids": ids,
                        "finish_reason": extra.get("finish_reason", "eos" if ids[-1] in eos else "length"),
                        "ttft_s": ttft, "decode_s": decode, "total_s": ttft + decode,
                        "decode_tps": (emitted - 1) / decode if emitted > 1 and decode else None}
                if expected is not None:
                    clean = visible.replace("−", "-")
                    exact = re.fullmatch(r"[+-]?\d+", clean)
                    case.update(expected=expected, correct=bool(exact and int(exact[0]) == expected))
                if extra:
                    case["server_timings"] = extra
                report[kind].append(case)
                print(name, len(ids), f"{case['decode_tps'] or 0:.2f} tok/s", flush=True)
                write(args.out, report)
            if args.logits and hasattr(runner, "first_logits"):
                sample = tests[:4] + tests[-1:]
                vocab = json.loads((args.tokenizer / "config.json").read_text(encoding="utf-8"))["text_config"]["vocab_size"]
                first = np.stack([runner.first_logits(encode(row[2]), vocab) for row in sample])
                np.save(args.logits, first)
        finally:
            runner.close()
    report["vram"] = memory
    report["arithmetic_correct"] = sum(row["correct"] for row in report["arithmetic"])
    report["arithmetic_total"] = len(report["arithmetic"])
    for field in ("ttft_s", "total_s", "decode_tps"):
        samples = [row[field] for row in report["prose"] if row[field] is not None]
        report[f"prose_median_{field}"] = statistics.median(samples) if samples else None
    write(args.out, report)

if __name__ == "__main__":
    main()
