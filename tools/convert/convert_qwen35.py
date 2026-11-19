#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert Qwen3.5-4B or 9B text weights to speed and/or quality .yskn packs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import subprocess
import sys


MODELS = {
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 2560, 9216),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 4096, 12288),
}
BITS = {"speed": "6", "quality": "8"}
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=MODELS)
    parser.add_argument("--mode", choices=("speed", "quality", "both"), default="both")
    parser.add_argument("--output", type=Path, help="pack directory (default: assets/<size>)")
    parser.add_argument("--checkpoint", type=Path, help="local safetensors checkpoint; otherwise download the pinned revision")
    parser.add_argument("--native", type=Path, required=True, help="path to the built YASAKANI trellis library")
    parser.add_argument("--calibration", type=Path, help="existing calibration .npy; otherwise generate the standard windows")
    args = parser.parse_args()

    repository, revision, hidden, ffn = MODELS[args.size]
    output = (args.output or Path("assets") / args.size).resolve()
    state = output / ".conversion"
    checkpoint = args.checkpoint.resolve() if args.checkpoint else state / revision
    library = args.native.resolve()
    if not library.is_file():
        parser.error(f"trellis library does not exist: {library}")
    calibration = args.calibration.resolve() if args.calibration else state / "calibration.npy"
    if args.calibration and not calibration.is_file():
        parser.error(f"calibration file does not exist: {calibration}")

    if args.checkpoint is None:
        hf = shutil.which("hf")
        if hf is None:
            parser.error("install the Hugging Face CLI (`hf`) or provide --checkpoint")
        subprocess.run([hf, "download", repository, "--revision", revision, "--local-dir", checkpoint,
                        "--include", "*.safetensors", "--include", "*.json",
                        "--include", "chat_template.jinja", "--include", "LICENSE"], check=True)

    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    if config.get("model_type") != "qwen3_5" or (text.get("hidden_size"), text.get("intermediate_size")) != (hidden, ffn):
        parser.error(f"expected {repository} text weights in {checkpoint}")
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing = sorted(name for name in set(index.values()) if not (checkpoint / name).is_file())
    if missing:
        parser.error(f"checkpoint shards are missing: {', '.join(missing)}")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    for name in tokenizer_files:
        if not (checkpoint / name).is_file():
            parser.error(f"missing {name} in {checkpoint}")

    modes = tuple(BITS) if args.mode == "both" else (args.mode,)
    source = state / "inputs.json"
    inputs = {"size": args.size, "checkpoint": str(checkpoint), "config_sha256": digest(config_path),
              "index_sha256": digest(index_path), "calibration_sha256": digest(calibration) if args.calibration else None}
    if source.exists():
        if json.loads(source.read_text(encoding="utf-8")) != inputs:
            parser.error("this output directory belongs to a different checkpoint or calibration; use another --output")
    else:
        if any((output / f"{mode}.yskn").exists() for mode in BITS):
            parser.error("this output directory already contains packs; use another --output")
        state.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")

    for name in (*tokenizer_files, "config.json", "chat_template.jinja", "LICENSE"):
        original, destination = checkpoint / name, output / name
        if original.is_file() and original.resolve() != destination.resolve():
            shutil.copy2(original, destination)

    todo = [mode for mode in modes if not (output / f"{mode}.yskn").exists()]
    if not todo:
        print(f"Requested packs already exist in {output}")
        return
    if not calibration.exists():
        pending = state / "calibration.partial.npy"
        subprocess.run([sys.executable, HERE.parent / "experiments/qwen35/calibration.py", checkpoint, pending], check=True)
        pending.replace(calibration)

    converter = [sys.executable, HERE / "yskn_convert.py"]
    work = state / args.mode
    common = ["--checkpoint", checkpoint, "--work", work]
    bits = "both" if args.mode == "both" else BITS[args.mode]
    subprocess.run([*converter, "convert", *common, "--bits", bits, "--native", library,
                    "--calibration", calibration], check=True)
    for mode in todo:
        subprocess.run([*converter, "assemble", *common, "--bits", BITS[mode],
                        "--out", output / f"{mode}.yskn", "--json", output / f"{mode}.json"], check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
