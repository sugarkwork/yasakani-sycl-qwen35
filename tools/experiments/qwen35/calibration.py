"""Create held-out-independent calibration windows for the two native model packs."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--each", type=int, default=16)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                              repo_type="dataset", revision="b08601e04326c79dfdd32d625aee71d232d685c3", token=False)
    text = "".join(pq.read_table(dataset, columns=["text"]).column("text").to_pylist())
    ids = tokenizer.encode(text, add_special_tokens=False)
    starts = np.random.default_rng(20260922).integers(0, len(ids) - 2048, size=args.each)
    wiki = [ids[start:start + 2048] for start in starts]
    source = Path(__file__).resolve().parents[2] / "convert/data/calibration_transcripts.json"
    windows = json.loads(source.read_text(encoding="utf-8"))["windows"]
    source_ids = tokenizer.encode("\n\n".join(item["text"] for item in windows), add_special_tokens=False)
    starts = np.random.default_rng(20261117).integers(0, len(source_ids) - 2048, size=args.each)
    local = [source_ids[start:start + 2048] for start in starts]
    ids = np.asarray(wiki + local, dtype="<i4")
    np.save(args.out, ids)
    print(f"{ids.shape[0]} windows of {ids.shape[1]} tokens", flush=True)

if __name__ == "__main__":
    main()
