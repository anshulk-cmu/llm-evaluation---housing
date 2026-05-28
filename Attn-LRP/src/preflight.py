# -*- coding: utf-8 -*-
"""
Preflight — verify the environment, GPU, data, model, and LXT BEFORE a launch.

Fast checks (default):
    python preflight.py
Full check (also loads Llama and runs ONE real AttnLRP attribution):
    python preflight.py --full

Exit code 0 only if all REQUIRED checks pass. Designed to catch the known
gotchas: missing deps, a non-Blackwell torch wheel ("no kernel image"),
insufficient VRAM, gated-model access, and a broken LXT install — so we never
kick off the multi-hour run on a broken setup.
"""

from __future__ import annotations
import argparse
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log import get_logger, banner
from config import (MODEL_ID, LOCAL_MODEL_DIR, DATA_PATH, N_LAYERS_EXPECTED, DTYPE)

logger = get_logger(__name__)

REQUIRED_ANALYSIS = "analysis"   # needed for stats + plots
REQUIRED_GPU = "gpu-run"         # needed to launch Phase 1/2
INFO = "info"

_results = []   # (severity, name, ok, detail)


def record(sev, name, ok, detail=""):
    _results.append((sev, name, ok, detail))
    mark = "PASS" if ok else ("---- " if sev == INFO else "FAIL")
    logger.info("[%-4s] %-28s %s", mark, name, detail)


def _try_import(mod):
    try:
        m = __import__(mod)
        return True, getattr(m, "__version__", "?")
    except Exception as e:
        return False, repr(e)


def check_python():
    ok = sys.version_info >= (3, 9)
    record(INFO, "python", True, f"{platform.python_version()} on {platform.system()}")
    record(REQUIRED_ANALYSIS, "python>=3.9", ok, platform.python_version())


def check_packages():
    for mod, sev in [("numpy", REQUIRED_ANALYSIS), ("pandas", REQUIRED_ANALYSIS),
                     ("scipy", REQUIRED_ANALYSIS), ("matplotlib", REQUIRED_ANALYSIS),
                     ("torch", REQUIRED_GPU), ("transformers", REQUIRED_GPU),
                     ("lxt", REQUIRED_GPU)]:
        ok, ver = _try_import(mod)
        record(sev, f"import {mod}", ok, ver if ok else ver[:80])


def check_gpu():
    try:
        import torch
    except Exception:
        record(REQUIRED_GPU, "cuda available", False, "torch not importable")
        return
    avail = torch.cuda.is_available()
    record(REQUIRED_GPU, "cuda available", avail, "" if avail else "no CUDA device visible")
    if not avail:
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
    record(INFO, "gpu", True, f"{name} | capability sm_{cap[0]}{cap[1]} | {vram:.1f} GB")
    record(INFO, "torch arch_list", True, ",".join(archs))
    # Blackwell (sm_120): the wheel must include matching kernels.
    sm_tag = f"sm_{cap[0]}{cap[1]}"
    blackwell_ok = any(a.replace("sm_", "") == f"{cap[0]}{cap[1]}" for a in archs) or cap[0] < 12
    record(REQUIRED_GPU, "kernels for this GPU", blackwell_ok,
           f"{sm_tag} {'present' if blackwell_ok else 'MISSING — install cu128 torch'}")
    record(REQUIRED_GPU, "vram >= 11 GB", vram >= 11.0, f"{vram:.1f} GB")
    # Actually launch a kernel — this is what raises "no kernel image" on a bad wheel.
    try:
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        _ = float((x @ x).sum())
        record(REQUIRED_GPU, "cuda kernel exec", True, "bf16 matmul ran")
    except Exception as e:
        record(REQUIRED_GPU, "cuda kernel exec", False, repr(e)[:90])


def check_data():
    ok = os.path.exists(DATA_PATH)
    detail = DATA_PATH
    if ok:
        try:
            import pandas as pd
            n = sum(1 for _ in open(DATA_PATH, encoding="utf-8")) - 1
            detail = f"{n} rows"
        except Exception:
            pass
    record(REQUIRED_ANALYSIS, "data file", ok, detail)


def check_model_access():
    local = os.path.isdir(LOCAL_MODEL_DIR) and bool(os.listdir(LOCAL_MODEL_DIR))
    if local:
        record(REQUIRED_GPU, "model access", True, f"local: {LOCAL_MODEL_DIR}")
        return
    has_tok = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    # Try to fetch the config (needs network + access to the gated repo)
    cfg_ok, detail = False, "no local dir; "
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(MODEL_ID)
        cfg_ok, detail = True, f"hub config OK ({MODEL_ID})"
    except Exception as e:
        detail += ("HF_TOKEN set but config fetch failed: " if has_tok else "HF_TOKEN NOT set; ") + repr(e)[:60]
    record(REQUIRED_GPU, "model access", cfg_ok, detail)


def check_lxt_api():
    try:
        from lxt.efficient import monkey_patch  # noqa: F401
        record(REQUIRED_GPU, "lxt.efficient API", True, "monkey_patch importable")
    except Exception as e:
        record(REQUIRED_GPU, "lxt.efficient API", False, repr(e)[:90])


def check_full_attribution():
    banner("FULL preflight — loading Llama and running ONE AttnLRP attribution")
    try:
        from attribution import load_model_and_tokenizer, prepare_inputs, attribute_target
        from data import load_pairs, sample_pairs
        from prompts import build_attribution_prompt
        from aggregate import map_spans_to_tokens, attribution_fraction
        from config import FEATURES

        model, tok, method = load_model_and_tokenizer()
        record(REQUIRED_GPU, "method == attnlrp", method == "attnlrp", method)
        n_layers = len(model.model.layers)
        record(INFO, "model layers", True, f"{n_layers} (expected {N_LAYERS_EXPECTED})")

        row = sample_pairs(load_pairs(), n=1, oversample=1.0).iloc[0]
        ut, spans, meta = build_attribution_prompt(row)
        prep = prepare_inputs(model, tok, ut)
        tm = map_spans_to_tokens(prep["offsets"], prep["user_off"], spans)
        spans_ok = all(len(tm[(s, f)]) >= 1 for s in (1, 2) for f in FEATURES)
        record(REQUIRED_GPU, "feature spans -> tokens", spans_ok,
               f"chosen={prep['chosen']} (logit {prep['logit_chosen']:.2f} vs {prep['logit_other']:.2f})")

        res = attribute_target(model, prep, "single_logit", capture_layers=True)
        shape_ok = res["R0"].shape[0] == len(prep["offsets"])
        record(REQUIRED_GPU, "R0 shape == seq_len", shape_ok, str(res["R0"].shape))
        cons = res["conservation_rel"]
        record(REQUIRED_GPU, "conservation finite", cons == cons and cons < 1.0,
               f"relative residual = {cons:.4f}")
        layer_ok = res["layer_rel"] is not None and res["layer_rel"].shape[0] == n_layers
        record(INFO, "layer relevance captured", layer_ok,
               str(None if res["layer_rel"] is None else res["layer_rel"].shape))
        frac = attribution_fraction(res["R0"], tm)
        record(INFO, "feature attribution fraction", True, f"{frac:.3f}")
    except Exception as e:
        record(REQUIRED_GPU, "full attribution", False, repr(e)[:120])


def summarize(require_gpu: bool) -> int:
    banner("PREFLIGHT SUMMARY")
    sevs = {REQUIRED_ANALYSIS}
    if require_gpu:
        sevs.add(REQUIRED_GPU)
    failed = [(n, d) for sev, n, ok, d in _results if sev in sevs and not ok]
    for sev, n, ok, d in _results:
        if sev in sevs and not ok:
            logger.error("REQUIRED FAILED: %s — %s", n, d)
    if failed:
        logger.error("PREFLIGHT FAILED: %d required check(s) failed. Fix before launching.", len(failed))
        return 1
    logger.info("PREFLIGHT OK: required checks passed%s.",
                " (GPU path included)" if require_gpu else " (analysis-only; add --full for GPU)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also load the model and run one real AttnLRP attribution")
    ap.add_argument("--no-gpu-required", action="store_true",
                    help="treat GPU checks as informational (analysis-only machine)")
    a = ap.parse_args()

    banner(f"ATTN-LRP PREFLIGHT (dtype={DTYPE}, model={MODEL_ID})")
    check_python()
    check_packages()
    check_gpu()
    check_data()
    check_model_access()
    check_lxt_api()
    if a.full:
        check_full_attribution()
    sys.exit(summarize(require_gpu=not a.no_gpu_required))


if __name__ == "__main__":
    main()
