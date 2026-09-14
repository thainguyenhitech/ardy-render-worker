#!/usr/bin/env python3
"""Build an offline ARDY text-encoder directory WITHOUT gated meta-llama access.

Why this works
--------------
ARDY loads its text encoder as
    base : McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp            (ungated: config + tokenizer + LoRA adapter, NO base weights)
    peft : McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised (ungated: LoRA adapter)
The mntp repo carries an adapter_config.json whose base_model_name_or_path = meta-llama/Meta-Llama-3-8B-Instruct.
transformers (integrations/peft.py::maybe_load_adapters) follows that pointer and downloads the GATED base
-- unless the path it was given is a LOCAL directory that already contains config.json, in which case it
loads the weights from that directory and never touches the hub pointer.

So we assemble one local directory =
    mntp repo files  +  Llama-3-8B-Instruct safetensors from the ungated NousResearch mirror
(same 4 shards, byte-identical sizes to Meta's release), and point ARDY at it with TEXT_ENCODERS_DIR.
No ARDY source changes are needed.

Resulting layout ($OUT defaults to /runpod-volume/text_encoders):
    $OUT/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/            <- config, tokenizer, adapter, model-0000{1..4}-of-00004.safetensors, index
    $OUT/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/ <- adapter_config.json, adapter_model.safetensors

Runtime env for ARDY workers:
    TEXT_ENCODERS_DIR=$OUT
    HF_HOME=/runpod-volume/hf          (ARDY diffusion checkpoints, downloaded by --ardy-models)
    LOCAL_CACHE=true                   (ARDY tries the local HF cache first)
    HF_HUB_OFFLINE=1                   (optional, forbid any network access at runtime)

Usage (run once on a temporary Pod that has the network volume mounted):
    pip install -U "huggingface_hub>=1.0"
    python prepare_text_encoder.py --out /runpod-volume/text_encoders --hf-home /runpod-volume/hf --ardy-models core8,core40
    python prepare_text_encoder.py --skip-download --verify   # after `pip install -e .` of ARDY; loads encoder + encodes a prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
BASE_MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"  # ungated re-upload of meta-llama/Meta-Llama-3-8B-Instruct
WEIGHT_PATTERNS = ["model-*.safetensors", "model.safetensors.index.json"]

ARDY_MODELS = {
    "core8": "nvidia/ARDY-Core-RP-20FPS-Horizon8",
    "core40": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
    "g18": "nvidia/ARDY-G1-RP-25FPS-Horizon8",
    "g152": "nvidia/ARDY-G1-RP-25FPS-Horizon52",
}

EXPECTED_SHARDS = {
    "model-00001-of-00004.safetensors": 4976698672,
    "model-00002-of-00004.safetensors": 4999802720,
    "model-00003-of-00004.safetensors": 4915916176,
    "model-00004-of-00004.safetensors": 1168138808,
}


def log(msg: str) -> None:
    print(f"[prepare_text_encoder] {msg}", flush=True)


def download(repo_id: str, local_dir: Path, allow_patterns=None) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"downloading {repo_id} -> {local_dir}" + (f" (patterns={allow_patterns})" if allow_patterns else ""))
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)


def patch_adapter_base(adapter_dir: Path, base_dir: Path) -> None:
    """Belt-and-braces: make the adapter's base pointer local too, so nothing can ever resolve to meta-llama."""
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("base_model_name_or_path") != str(base_dir):
        cfg["base_model_name_or_path"] = str(base_dir)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log(f"patched {cfg_path}: base_model_name_or_path -> {base_dir}")


def check_layout(mntp_dir: Path, sup_dir: Path) -> bool:
    ok = True
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors.index.json",
    ]:
        if not (mntp_dir / name).exists():
            log(f"MISSING {mntp_dir / name}")
            ok = False
    for shard, size in EXPECTED_SHARDS.items():
        p = mntp_dir / shard
        if not p.exists():
            log(f"MISSING {p}")
            ok = False
        elif p.stat().st_size != size:
            log(f"SIZE MISMATCH {p}: {p.stat().st_size} != {size}")
            ok = False
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        if not (sup_dir / name).exists():
            log(f"MISSING {sup_dir / name}")
            ok = False
    # ARDY keys its Llama-3 instruct prompt template off this exact string; keep it intact.
    cfg_file = mntp_dir / "config.json"
    if cfg_file.exists():
        name_or_path = json.loads(cfg_file.read_text()).get("_name_or_path")
        if name_or_path != "meta-llama/Meta-Llama-3-8B-Instruct":
            log(f"WARNING config.json _name_or_path is {name_or_path!r}; ARDY expects 'meta-llama/Meta-Llama-3-8B-Instruct'")
            ok = False
    log("layout check: " + ("OK" if ok else "FAILED"))
    return ok


def verify_with_ardy(out: Path) -> None:
    os.environ["TEXT_ENCODERS_DIR"] = str(out)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # prove no network is needed
    try:
        import torch
        from ardy.model.load_model import load_text_encoder
    except ImportError as e:
        log(f"ARDY not importable ({e}); run `pip install -e .` in the ardy repo first")
        sys.exit(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading text encoder in local mode on {device} (offline)...")
    enc = load_text_encoder(mode="local", device=device)
    feat, length = enc(["A person walks forward and waves."])
    log(f"encoded OK: feat.shape={tuple(feat.shape)} dtype={feat.dtype} length={length}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/runpod-volume/text_encoders", help="TEXT_ENCODERS_DIR target")
    ap.add_argument("--hf-home", default=None, help="Set HF_HOME (e.g. /runpod-volume/hf) so ARDY checkpoints land on the volume")
    ap.add_argument("--ardy-models", default="", help=f"Comma list of ARDY checkpoints to pre-download: {','.join(ARDY_MODELS)}")
    ap.add_argument("--skip-download", action="store_true", help="Only run layout check / verify")
    ap.add_argument("--verify", action="store_true", help="Load the encoder through ARDY and encode a test prompt")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    out = Path(args.out).resolve()
    mntp_dir = out / MNTP
    sup_dir = out / SUPERVISED

    if not args.skip_download:
        download(MNTP, mntp_dir)
        download(BASE_MIRROR, mntp_dir, allow_patterns=WEIGHT_PATTERNS)  # weights merged INTO the mntp dir
        download(SUPERVISED, sup_dir)
        patch_adapter_base(mntp_dir, mntp_dir)
        patch_adapter_base(sup_dir, mntp_dir)
        for nick in [s.strip() for s in args.ardy_models.split(",") if s.strip()]:
            repo = ARDY_MODELS.get(nick)
            if repo is None:
                log(f"unknown ARDY model nickname {nick!r}; choose from {list(ARDY_MODELS)}")
                sys.exit(1)
            from huggingface_hub import snapshot_download

            log(f"caching {repo} into HF cache ({os.environ.get('HF_HOME', '~/.cache/huggingface')})")
            snapshot_download(repo_id=repo)

    if not check_layout(mntp_dir, sup_dir):
        sys.exit(1)

    log("done. Runtime env:")
    log(f"  TEXT_ENCODERS_DIR={out}")
    if args.hf_home:
        log(f"  HF_HOME={args.hf_home}  LOCAL_CACHE=true  HF_HUB_OFFLINE=1")

    if args.verify:
        verify_with_ardy(out)


if __name__ == "__main__":
    main()
