# -*- coding: utf-8 -*-
"""AttnLRP attribution on Llama-3.2-3B via LXT (PLAN 6). Batched on the GPU."""

from __future__ import annotations
import os
import warnings
import numpy as np
import torch

from config import MODEL_ID, LOCAL_MODEL_DIR, DTYPE, N_LAYERS_EXPECTED, EPS, CPU_THREADS
from prompts import locate_user_text_offset
from log import get_logger

logger = get_logger(__name__)

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]
_DISABLE_LXT = os.environ.get("ATTNLRP_DISABLE_LXT", "0") == "1"   # debug only: plain grad x input


def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(CPU_THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    src = LOCAL_MODEL_DIR if (os.path.isdir(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR)) else MODEL_ID
    logger.info("loading %s dtype=%s cpu_threads=%d", src, DTYPE, CPU_THREADS)
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                              # last column = real final token for all rows

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=_DTYPE).to(device)
    model.eval()

    n = len(model.model.layers)
    if n != N_LAYERS_EXPECTED:
        warnings.warn(f"expected {N_LAYERS_EXPECTED} layers, found {n}")
    method = apply_attnlrp_rules(model)
    logger.info("device=%s layers=%d method=%s", device, n, method)
    return model, tok, method


def apply_attnlrp_rules(model) -> str:
    if _DISABLE_LXT:
        warnings.warn("ATTNLRP_DISABLE_LXT=1: plain grad x input, NOT AttnLRP. Do not report.")
        return "gradxinput_DEBUG"
    try:
        from lxt.efficient import monkey_patch
        import transformers.models.llama.modeling_llama as llama_mod
        monkey_patch(llama_mod, verbose=False)
        return "attnlrp"
    except Exception as e:
        raise RuntimeError(f"LXT AttnLRP rules unavailable (pip install lxt, v2.1). {e!r}")


def choice_token_ids(tokenizer):
    out = {1: set(), 2: set()}
    for d in (1, 2):
        for variant in (f" {d}", f"{d}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[d].add(ids[0])
        if not out[d]:
            out[d].add(tokenizer.encode(f" {d}", add_special_tokens=False)[-1])
    return {d: sorted(v) for d, v in out.items()}


def _decide(last_logits, cand):
    B = last_logits.shape[0]
    chosen, other, cid, oid, lc, lo = [], [], [], [], [], []
    for i in range(B):
        best = {d: max(((t, last_logits[i, t].item()) for t in cand[d]), key=lambda kv: kv[1])
                for d in (1, 2)}
        c = 1 if best[1][1] >= best[2][1] else 2
        o = 2 if c == 1 else 1
        chosen.append(c); other.append(o)
        cid.append(best[c][0]); oid.append(best[o][0])
        lc.append(best[c][1]); lo.append(best[o][1])
    return chosen, other, cid, oid, lc, lo


# --- batched path (default) ------------------------------------------------
def prepare_batch(model, tok, user_texts):
    full = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                    add_generation_prompt=True) for u in user_texts]
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt", padding=True,
              add_special_tokens=False)
    return {
        "input_ids": enc["input_ids"].to(model.device),
        "attention_mask": enc["attention_mask"].to(model.device),
        "offsets": [[tuple(o) for o in row] for row in enc["offset_mapping"].tolist()],
        "user_off": [locate_user_text_offset(f, u) for f, u in zip(full, user_texts)],
        "cand": choice_token_ids(tok),
    }


def attribute_batch(model, prep, targets, capture_layers=True):
    input_ids = prep["input_ids"]
    mask = prep["attention_mask"]
    B, L = input_ids.shape
    embeds = model.get_input_embeddings()(input_ids).detach().clone().requires_grad_(True)

    layer_h, handles = [], []
    if capture_layers:
        for layer in model.model.layers:
            def hook(_m, _i, out, store=layer_h):
                h = out[0] if isinstance(out, tuple) else out
                h.retain_grad(); store.append(h)
            handles.append(layer.register_forward_hook(hook))

    results = {t: [] for t in targets}
    try:
        logits = model(inputs_embeds=embeds, attention_mask=mask).logits[:, -1, :]
        chosen, other, cid, oid, lc, lo = _decide(logits.detach().float(), prep["cand"])

        for ti, target in enumerate(targets):
            if embeds.grad is not None:
                embeds.grad = None
            for h in layer_h:
                h.grad = None
            terms = [logits[i, cid[i]] - logits[i, oid[i]] if target == "logit_diff"
                     else logits[i, cid[i]] for i in range(B)]
            torch.stack(terms).sum().backward(retain_graph=(ti < len(targets) - 1))

            R0 = (embeds * embeds.grad).sum(-1).detach().float().cpu().numpy()      # [B, L]
            layer_rel = None
            if capture_layers and layer_h:
                layer_rel = np.stack([(h * h.grad).sum(-1).detach().float().cpu().numpy()
                                      if h.grad is not None else np.zeros((B, L), np.float32)
                                      for h in layer_h], axis=0)                    # [Lc, B, L]
            for i in range(B):
                real = [j for j, (a, b) in enumerate(prep["offsets"][i]) if b > a]
                tval = (lc[i] - lo[i]) if target == "logit_diff" else lc[i]
                cons = abs(float(R0[i, real].sum()) - tval) / (abs(tval) + EPS)
                results[target].append({
                    "R0": R0[i], "layer_rel": None if layer_rel is None else layer_rel[:, i, :],
                    "offsets": prep["offsets"][i], "user_off": prep["user_off"][i],
                    "chosen": chosen[i], "other": other[i],
                    "target_value": tval, "conservation_rel": cons,
                })
    finally:
        for hd in handles:
            hd.remove()
    return results


# --- single-sample path (preflight, deletion curve) ------------------------
def prepare_inputs(model, tok, user_text):
    prep = prepare_batch(model, tok, [user_text])
    with torch.no_grad():
        last = model(input_ids=prep["input_ids"],
                     attention_mask=prep["attention_mask"]).logits[0, -1].float()
    chosen, other, cid, oid, lc, lo = _decide(last.unsqueeze(0), prep["cand"])
    return {"input_ids": prep["input_ids"], "attention_mask": prep["attention_mask"],
            "offsets": prep["offsets"][0], "user_off": prep["user_off"][0],
            "chosen": chosen[0], "other": other[0], "chosen_id": cid[0], "other_id": oid[0],
            "logit_chosen": lc[0], "logit_other": lo[0]}


def attribute_target(model, prep, target, capture_layers=True):
    out = attribute_batch(model,
                          {"input_ids": prep["input_ids"], "attention_mask": prep["attention_mask"],
                           "offsets": [prep["offsets"]], "user_off": [prep["user_off"]],
                           "cand": choice_token_ids_from_prep(prep)},
                          [target], capture_layers=capture_layers)
    return out[target][0]


def choice_token_ids_from_prep(prep):
    return {1: [prep["chosen_id"] if prep["chosen"] == 1 else prep["other_id"]],
            2: [prep["chosen_id"] if prep["chosen"] == 2 else prep["other_id"]]}


def target_logit_fn_factory(model, prep, target):
    base = model.get_input_embeddings()(prep["input_ids"]).detach()
    mask = prep["attention_mask"]

    @torch.no_grad()
    def f(masked_idx) -> float:
        e = base.clone()
        if len(masked_idx):
            e[0, list(masked_idx), :] = 0.0
        lg = model(inputs_embeds=e, attention_mask=mask).logits[0, -1]
        return float(lg[prep["chosen_id"]] - lg[prep["other_id"]]) if target == "logit_diff" \
            else float(lg[prep["chosen_id"]])
    return f
