"""Prepare identical prompts or benchmark one Qwen3.5-4B DLL per process."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import unicodedata


def prepare(args):
    from transformers import AutoTokenizer
    from benchmark import ARITHMETIC, stop_ids, write

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    spec = json.loads((args.fixture / "questions.json").read_text(encoding="utf-8"))
    body = (args.fixture / "body-ja.txt").read_text(encoding="utf-8").strip()
    cases = [
        {"name": name, "kind": "arithmetic", "prompt": f"次の式を計算してください。答えの整数だけを出力してください。\n{expr}",
         "expected": expected, "limit": 20}
        for name, expr, expected in ARITHMETIC
    ]
    for case in spec["sizes"]["4b"]["questions"]:
        prompt = f"{spec['instruction']}\n\n記録\n{body}\n\n質問\n{case['question']}"
        cases.append({"name": case["zone"], "kind": "reading", "prompt": prompt,
                      "expected": case["expected"], "facts": case["facts"], "limit": 64})

    for case in cases:
        case["input_ids"] = tok.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
    write(args.manifest, {"eos": sorted(stop_ids(tok, args.tokenizer)), "cases": cases})


def smi(gpu, fields):
    return subprocess.check_output(
        ["nvidia-smi", "-i", gpu, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def matches(case, answer):
    text = unicodedata.normalize("NFKC", answer).strip().replace("−", "-")
    if case["kind"] == "arithmetic":
        return bool(re.fullmatch(r"[+-]?\d+", text) and int(text) == case["expected"])
    return all(any(word in text for word in alternatives) for alternatives in case["facts"])


def run(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from transformers import AutoTokenizer
    from benchmark import Native, memory_sampler, write

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    eos, cases = set(manifest["eos"]), manifest["cases"]
    if args.smoke:
        cases = cases[:2]
    largest = max(len(case["input_ids"]) + case["limit"] - 1 for case in cases)
    if largest > args.context:
        raise ValueError(f"Context {args.context} is smaller than the required {largest} tokens")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = {
        "backend": args.backend, "model": str(args.model), "model_sha256": sha256(args.model),
        "dll": str(args.dll), "dll_sha256": sha256(args.dll), "manifest_sha256": sha256(args.manifest),
        "gpu": smi(args.gpu, "uuid,name,driver_version,memory.total"),
        "context": args.context, "kv": "BF16", "smoke": args.smoke, "cases": [],
    }
    if args.backend == "cuda":
        report["cuda_engine_environment"] = {key: val for key, val in os.environ.items() if key.startswith("YSKN_")}
    else:
        report["sycl_prefill_rows_override"] = os.environ.get("YSKN_SYCL_PREFILL_ROWS")
    read_used = lambda: int(smi(args.gpu, "memory.used"))
    runtime = os.add_dll_directory(str(args.runtime.resolve())) if args.runtime else nullcontext()
    with runtime, memory_sampler(read_used) as memory:
        engine = Native(args.dll.resolve(), args.model.resolve(), eos, context=args.context, device=0)
        try:
            report["load_s"] = engine.load_s
            memory["loaded_mib"] = read_used()
            engine.complete(cases[0]["input_ids"], 3)
            for case in cases:
                ids, ttft, decode, _ = engine.complete(case["input_ids"], case["limit"])
                answer = tok.decode(ids, skip_special_tokens=True).strip()
                row = {"name": case["name"], "kind": case["kind"], "expected": case["expected"],
                       "answer": answer, "correct": matches(case, answer), "output_ids": ids,
                       "input_tokens": len(case["input_ids"]), "ttft_s": ttft, "decode_s": decode,
                       "decode_tps": (len(ids) - 1) / decode if len(ids) > 1 and decode > 0 else None,
                       "finish_reason": "eos" if ids[-1] in eos else "limit"}
                report["cases"].append(row)
                print(f"{row['name']}: TTFT {ttft:.3f}s, {len(ids)} sampled tokens", flush=True)
                write(args.out, report)
        finally:
            engine.close()
    report["vram"] = memory
    report["accuracy"] = {}
    for kind in ("arithmetic", "reading"):
        rows = [row for row in report["cases"] if row["kind"] == kind]
        if rows:
            rates = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
            report["accuracy"][kind] = {
                "correct": sum(row["correct"] for row in rows), "total": len(rows),
                "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
                "median_decode_tps": statistics.median(rates) if rates else None,
            }
    write(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="create the shared prompt manifest without touching the GPU")
    prep.add_argument("tokenizer", type=Path)
    prep.add_argument("fixture", type=Path)
    prep.add_argument("manifest", type=Path)
    bench = commands.add_parser("run", help="execute one DLL on the selected physical GPU")
    bench.add_argument("backend", choices=("cuda", "sycl"))
    for name in ("dll", "model", "tokenizer", "manifest", "out"):
        bench.add_argument(name, type=Path)
    bench.add_argument("--gpu", required=True, help="physical GPU UUID from nvidia-smi -L")
    bench.add_argument("--runtime", type=Path, help="directory with the SYCL runtime DLLs")
    bench.add_argument("--context", type=int, default=2048)
    bench.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        prepare(args)
    else:
        if not args.gpu.startswith("GPU-"):
            parser.error("--gpu must be a physical GPU UUID")
        if args.backend == "sycl" and not args.runtime:
            parser.error("--runtime is required for the SYCL DLL")
        if args.out.exists():
            parser.error("output file already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run(args)


if __name__ == "__main__":
    main()
