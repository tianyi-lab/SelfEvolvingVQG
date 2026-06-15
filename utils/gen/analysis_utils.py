import os
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Iterable
from collections import Counter

def load_qs(json_path):
    with open(json_path, 'r') as f:
        list_data = json.load(f)
    
    list_questions = []
    for item in list_data:
        image_path = item['image_path']
        if isinstance(image_path, list):
            image_path = [item.split('/')[-1] for item in image_path]
            
        eval_item = item['eval']
        for s_item in eval_item:
            qs_ori = s_item['original']
            ver = s_item['question'][-1]['ver']
            qs_simplify = s_item['simplified']
            list_questions.append((image_path, ver, qs_ori, qs_simplify))
    return list_questions

def save_csv(list_qs, save_path):
    df = pd.DataFrame(list_qs, columns=['image', 'ver', 'original', 'simplified'])
    df.to_csv(save_path, sep='\t', index=False)
    return df

# -----------------------------
# Core scoring helper (single source of truth)
# -----------------------------
def _get_model_following_by_constraint(mfc) -> List[dict]:
    """Return model-following constraint entries from either supported judge shape."""
    if isinstance(mfc, dict):
        entries = mfc.get("by_constraint") or []
    elif isinstance(mfc, list):
        entries = mfc
    else:
        entries = []
    return [entry for entry in entries if isinstance(entry, dict)]


def _extract_scores_and_mfc(item: dict) -> Tuple[Dict[str, float], int, int]:
    """
    Returns:
      scores: flat aspect -> score
        - Hard constraints: aspect 'count' if present; 'Validity' uses 0/1 via answerable(False->1, True->0)
        - Soft constraints: aspect 'count' if present; nested ones as aspect_subkey (e.g., perception_difficulty_text)
        - Per-item model following ratio as 'model_following_capability' if there are non-empty constraints
      mfc_true, mfc_total: raw counts across all by_constraint entries with non-empty constraint_text
    """
    scores: Dict[str, float] = {}
    api_res = item.get("api_res") or {}
    qs_info = item.get('question_info') or {}
    mfc_true = 0
    mfc_total = 0

    # ---- Hard constraints ----
    hard = api_res.get("hard_constraints", {}) or {}
    for aspect, info in hard.items():
        if not isinstance(info, dict):
            continue
        if "count" in info:
            scores[aspect] = float(info["count"])
        if "answerable" in info:
            # Validity: 0 if answerable True, 1 if False
            scores[aspect] = 0.0 if info.get("answerable", True) else 1.0

    # ---- Soft constraints ----
    soft = api_res.get("soft_constraints", {}) or {}
    for aspect, info in soft.items():
        if not isinstance(info, dict):
            continue
        if "count" in info:
            scores[aspect] = float(info["count"])
        for subkey, subinfo in info.items():
            if isinstance(subinfo, dict) and "count" in subinfo:
                scores[f"{aspect}_{subkey}"] = float(subinfo["count"])

    # ---- Model-following per-item score + global raw counts ----
    mfc = soft.get("model_following_capability", {}) or {}
    constraint = qs_info.get('constraint', [])
    if len([item for item in constraint if item != '']) > 0: 
        valid = [
            c for c in _get_model_following_by_constraint(mfc)
            if str(c.get("constraint_text", "")).strip()
        ]
        if valid:
            ratio = sum(1 for c in valid if c.get("complies")) / len(valid)
            scores["model_following_capability"] = float(ratio)
            mfc_true += sum(1 for c in valid if c.get("complies"))
            mfc_total += len(valid)

    return scores, mfc_true, mfc_total


# -----------------------------
# Small utilities to remove repetition
# -----------------------------
def build_index_by_q_img(list_res: List[dict]) -> Dict[Tuple[str, str], dict]:
    """(question, image_path) -> result item"""
    return {
        (it.get("question"), it.get("image_path")): it
        for it in list_res
        if it.get("question") and it.get("image_path")
    }


def build_rewrite_seed_mapping(list_ori: List[dict]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Map (rewritten_q, image_path) -> (seed_q, image_path)."""
    mapping = {}
    for meta in list_ori:
        img = meta.get("image_path")
        for it in (meta.get("rewritten") or []):
            seed_q = it.get("seed")
            rewr_q = it.get("Rewritten question")
            if rewr_q is None and isinstance(it.get("rewritten"), dict):
                rewr_q = it["rewritten"].get("Rewritten question")
            if img and seed_q and rewr_q:
                mapping[(rewr_q, img)] = (seed_q, img)
    return mapping


def _get_scores(item: dict) -> Dict[str, float]:
    """Convenience wrapper to only pull scores."""
    scores, _, _ = _extract_scores_and_mfc(item)
    return scores


def _validity_score(item: dict) -> Optional[float]:
    """0 if answerable; 1 if unanswerable; falls back to count if provided; else None."""
    hard = (item.get("api_res") or {}).get("hard_constraints", {}) or {}
    info = hard.get("Validity", {})
    if isinstance(info, dict):
        if "answerable" in info:
            return 0.0 if info.get("answerable") else 1.0
        if "count" in info:
            return float(info["count"])
    return None


def iter_rewrite_pairs(
    list_ori: List[dict],
    list_res: List[dict]
) -> Iterable[Tuple[str, str, str, dict, dict]]:
    """
    Yields tuples:
      (rewritten_question, seed_question, image_path, rewritten_item, seed_item)
    Only yields when both items are found in list_res.
    """
    mapping = build_rewrite_seed_mapping(list_ori)
    index = build_index_by_q_img(list_res)
    for (rewr_q, img), (seed_q, seed_img) in mapping.items():
        rewr_item = index.get((rewr_q, img))
        seed_item = index.get((seed_q, seed_img))
        if rewr_item and seed_item:
            yield rewr_q, seed_q, img, rewr_item, seed_item


# -----------------------------
# Dataset-level averages (with global MFC ratio)
# -----------------------------
def calculate_avg_scores(data: List[dict]) -> Dict[str, float]:
    """
    Average per-aspect scores using the unified extractor.
    Also computes a GLOBAL model_following_compliance_ratio across all valid by_constraint entries.
    """
    totals = defaultdict(float)
    counts = defaultdict(int)
    mfc_true_total = 0
    mfc_total_total = 0

    for item in data:
        api_res = item.get("api_res")
        if not api_res:
            continue
        scores, mfc_true, mfc_total = _extract_scores_and_mfc(item)
        for k, v in scores.items():
            totals[k] += v
            counts[k] += 1
        mfc_true_total += mfc_true
        mfc_total_total += mfc_total

    avg_scores = {k: round(totals[k] / counts[k], 3) for k in totals if counts[k] > 0}
    avg_scores["model_following_compliance_ratio"] = (
        round(mfc_true_total / mfc_total_total, 3) if mfc_total_total > 0 else None
    )
    return avg_scores


# -----------------------------
# Ranking by aspect
# -----------------------------
def rank_questions_by_aspect(
    data: List[dict],
    aspect_name: str,
    top_k: int = 10
) -> Tuple[List[tuple], List[tuple]]:
    """
    Rank items by a flattened aspect key (e.g., 'Factuality', 'Validity',
    'perception_difficulty_text', 'reasoning_difficulty', 'model_following_capability').

    Returns:
      (top_k_list, bottom_k_list) where each list contains (idx, question, score).
    """
    rows = []
    for item in data:
        if not item.get("api_res"):
            continue
        scores = _get_scores(item)
        if aspect_name in scores:
            rows.append((item.get("idx"), item.get("question", ""), scores[aspect_name]))

    rows.sort(key=lambda x: x[2], reverse=True)
    top_k_list = rows[:top_k]
    bottom_k_list = rows[-top_k:] if len(rows) > top_k else rows[::-1]
    return top_k_list, bottom_k_list


# -----------------------------
# Rewrites with negative validity delta
# -----------------------------
def find_rewrites_with_negative_validity(
    list_ori: List[dict],
    list_res: List[dict],
    eps: float = 0.0
) -> List[Dict]:
    """
    Returns pairs where (rewritten Validity - seed Validity) < -eps.
    Each dict: rewritten_question, seed_question, image_path, rewritten_validity, seed_validity, diff
    """
    negatives: List[Dict] = []
    for rewr_q, seed_q, img, rewr_item, seed_item, rew_evol, rew_prompt in iter_rewrite_pairs_with_meta(list_ori, list_res):
        v_rewr = _validity_score(rewr_item)
        v_seed = _validity_score(seed_item)
        if v_rewr is None or v_seed is None:
            continue
        diff = v_rewr - v_seed
        if diff < -abs(eps):
            negatives.append({
                "rewritten_question": rewr_q,
                "seed_question": seed_q,
                "image_path": img,
                "rewritten_validity": v_rewr,
                "seed_validity": v_seed,
                "diff": round(diff, 4)
            })
    negatives.sort(key=lambda x: x["diff"])  # most negative first
    return negatives


# -----------------------------
# Improvements finder
# -----------------------------
def _auto_higher_is_better(aspect: str) -> bool:
    """Lower is better for Validity/Factuality/Ambiguity; otherwise higher is better."""
    return aspect not in {"Validity", "Factuality", "Ambiguity"}


def find_improved_samples(
    list_ori: List[dict],
    list_res: List[dict],
    aspect: str,
    *,
    higher_is_better: Optional[bool] = None,
    eps: float = 0.0,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Return samples where rewritten improved over seed for the given aspect.
      - If higher_is_better=True:   (rewr - seed) >  +eps
      - If higher_is_better=False:  (seed - rewr) >  +eps
    """
    if higher_is_better is None:
        higher_is_better = _auto_higher_is_better(aspect)

    improved: List[Dict] = []
    for rewr_q, seed_q, img, rewr_item, seed_item in iter_rewrite_pairs(list_ori, list_res):
        rewr_scores = _get_scores(rewr_item)
        seed_scores = _get_scores(seed_item)
        if aspect not in rewr_scores or aspect not in seed_scores:
            continue

        r, s = rewr_scores[aspect], seed_scores[aspect]
        diff = r - s
        signed = diff if higher_is_better else (s - r)
        if signed > abs(eps):
            improved.append({
                "rewritten_question": rewr_q,
                "seed_question": seed_q,
                "image_path": img,
                "seed_score": s,
                "rewritten_score": r,
                "diff": round(diff, 6),
                "improved_amount": round(signed, 6),
            })

    improved.sort(key=lambda x: x["improved_amount"], reverse=True)
    return improved[:top_k] if top_k is not None else improved


def union_improved_samples(*lists: List[Dict]) -> List[Dict]:
    """Union of improved samples, deduped by (rewritten_question, seed_question, image_path)."""
    seen = set()
    out: List[Dict] = []
    for lst in lists:
        for it in lst:
            key = (it.get("rewritten_question"), it.get("seed_question"), it.get("image_path"))
            if key not in seen:
                seen.add(key)
                out.append(it)
    out.sort(key=lambda x: x.get("improved_amount", 0), reverse=True)
    return out

# --- Add these helpers ---
def _merge_api_res_from_evaluation(blocks) -> dict:
    """
    Merge evaluation blocks into a single {"hard_constraints": {...}, "soft_constraints": {...}}.
    - Tolerates blocks being a dict or a list.
    - Ignores non-dict entries safely.
    """
    api_res = {"hard_constraints": {}, "soft_constraints": {}}

    # Allow a single dict or a list of dicts
    if isinstance(blocks, dict):
        blocks = [blocks]

    for blk in (blocks or []):
        if not isinstance(blk, dict):
            continue  # <-- robust to strings/None/etc.

        hc = blk.get("hard_constraints")
        if isinstance(hc, dict):
            for k, v in hc.items():
                api_res["hard_constraints"][k] = v

        sc = blk.get("soft_constraints")
        if isinstance(sc, dict):
            for k, v in sc.items():
                if isinstance(v, dict) and isinstance(api_res["soft_constraints"].get(k), dict):
                    # shallow deep-merge one level
                    api_res["soft_constraints"][k] = {**api_res["soft_constraints"][k], **v}
                else:
                    api_res["soft_constraints"][k] = v

    # Clean empty sections so downstream expects the same shape
    if not api_res["hard_constraints"]:
        api_res.pop("hard_constraints")
    if not api_res["soft_constraints"]:
        api_res.pop("soft_constraints")

    return api_res


def _normalize_results_schema(list_res: List[dict]) -> List[dict]:
    """
    Normalize to flat items: {"question", "image_path", "api_res"}.
    - Handles old schema (already has api_res)
    - Explodes new schema under item["eval"][*]["evaluation"]
    - Skips malformed entries safely
    """
    out: List[dict] = []

    for idx, it in enumerate(list_res):
        img = it.get("image_path") or it.get("image")

        # 1) Keep already-flat items (old schema)
        if img and isinstance(it.get("api_res"), dict) and isinstance(it.get("question"), str):
            out.append({"question": it["question"], "image_path": img, "api_res": it["api_res"]})

        # 2) Explode new schema entries
        for ev in (it.get("eval") or []):
            if not isinstance(ev, dict):
                continue
            q = ev.get("question")
            # question may be ["text", "seed"/"rewritten", ..., {"ver": "v1"}] or a plain string
            q_text = q[0] if isinstance(q, list) and q else (q if isinstance(q, str) else "")
            qs_meta = q[-1]
            blocks = ev.get("evaluation") or []
            api_res = _merge_api_res_from_evaluation(blocks)
            if q_text and img and api_res:
                out.append({"question": q_text, "question_info": qs_meta, "image_path": img, "api_res": api_res})

    # If nothing was normalized, fall back to original input
    return out if out else list_res

def avg_diff_rewritten_vs_seed(list_ori: List[dict], list_res: List[dict]) -> Dict[str, Dict]:
    """
    Compute average (rewritten - seed) per aspect.
    Also compute averages grouped by rewrite_evol and rewrite_prompt.

    Returns:
        {
            "overall": {aspect: avg_diff, ...},
            "by_evol": {evol_value: {aspect: avg_diff, ...}},
            "by_prompt": {promptver_value: {aspect: avg_diff, ...}}
        }
    """
    # Normalize to flat schema
    flat_res = _normalize_results_schema(list_res)
    index = build_index_by_q_img(flat_res)
    mapping = build_rewrite_seed_mapping(list_ori)

    # --- global accumulators ---
    totals, counts = defaultdict(float), defaultdict(int)

    # --- group-level accumulators ---
    evol_totals, evol_counts = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    prompt_totals, prompt_counts = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))

    for (rewr_q, img), (seed_q, seed_img) in mapping.items():
        rewr_item = index.get((rewr_q, img))
        seed_item = index.get((seed_q, seed_img))
        if not rewr_item or not seed_item:
            continue

        # optional metadata
        qinfo = rewr_item.get("question_info", {})
        rewrite_evol = qinfo.get("evolution", "unknown")
        rewrite_prompt = qinfo.get("promptver", "unknown")

        rewr_scores, _, _ = _extract_scores_and_mfc(rewr_item)
        seed_scores, _, _ = _extract_scores_and_mfc(seed_item)

        common = set(rewr_scores) & set(seed_scores)
        if not common:
            continue

        for aspect in common:
            diff = rewr_scores[aspect] - seed_scores[aspect]

            # overall
            totals[aspect] += diff
            counts[aspect] += 1

            # by evolution
            evol_totals[rewrite_evol][aspect] += diff
            evol_counts[rewrite_evol][aspect] += 1

            # by prompt version
            prompt_totals[rewrite_prompt][aspect] += diff
            prompt_counts[rewrite_prompt][aspect] += 1

    # --- compute averages ---
    def avg_dict(tot: dict, cnt: dict) -> dict:
        return {a: round(tot[a] / cnt[a], 4) for a in tot if cnt[a] > 0}

    overall_avg = avg_dict(totals, counts)
    by_evol_avg = {e: avg_dict(evol_totals[e], evol_counts[e]) for e in evol_totals}
    by_prompt_avg = {p: avg_dict(prompt_totals[p], prompt_counts[p]) for p in prompt_totals}

    return {
        "overall": overall_avg,
        "by_evol": by_evol_avg,
        "by_prompt": by_prompt_avg
    }

def build_index_by_q_img(flat_res: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {
        (it.get("question"), it.get("image_path")): it
        for it in flat_res
        if it.get("question") and it.get("image_path")
    }

def build_rewrite_seed_mapping(list_ori: List[dict]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    mapping = {}
    for meta in list_ori:
        img = meta.get("image_path")
        for it in (meta.get("rewritten") or []):
            seed_q = it.get("seed")
            rewr_q = it.get("Rewritten question")
            if rewr_q is None and isinstance(it.get("rewritten"), dict):
                rewr_q = it["rewritten"].get("Rewritten question")
            if img and seed_q and rewr_q:
                mapping[(rewr_q, img)] = (seed_q, img)
    return mapping

def _get_scores(item: dict) -> Dict[str, float]:
    scores, _, _ = _extract_scores_and_mfc(item)
    return scores

def iter_rewrite_pairs_with_meta(list_ori: List[dict], list_res: List[dict]):
    """
    Yields: (rewr_q, seed_q, img, rewr_item, seed_item, rewrite_evol, rewrite_prompt)
    - rewrite_evol from rewr_item['question_info'].get('evolution', 'unknown')
    - rewrite_prompt from rewr_item['question_info'].get('promptver', 'unknown')
    """
    flat_res = _normalize_results_schema(list_res)
    index = build_index_by_q_img(flat_res)
    mapping = build_rewrite_seed_mapping(list_ori)

    for (rewr_q, img), (seed_q, seed_img) in mapping.items():
        rewr_item = index.get((rewr_q, img))
        seed_item = index.get((seed_q, seed_img))
        if not rewr_item or not seed_item:
            continue
        qinfo = rewr_item.get("question_info", {}) or {}
        rewrite_evol = qinfo.get("evolution", "unknown")
        rewrite_prompt = qinfo.get("promptver", "unknown")
        yield rewr_q, seed_q, img, rewr_item, seed_item, rewrite_evol, rewrite_prompt

def extract_qualified_rewritten(list_ori: List[dict], list_res: List[dict]):
    """
    Returns:
      qualified: List[dict] with keys:
        rewritten_question, seed_question, image_path, increased_aspect, rewrite_evol, rewrite_prompt
      mis_hard: List[dict] with keys:
        rewritten_question, seed_question, image_path, factuality_ok, ambiguity_ok, validity_ok,
        rewrite_evol, rewrite_prompt
    """
    qualified = []
    mis_hard = []
    mis_soft = []
    all_cnt = 0

    for rewr_q, seed_q, img, rewr_item, seed_item, rew_evol, rew_prompt in iter_rewrite_pairs_with_meta(list_ori, list_res):
        all_cnt += 1
        rewr_scores = _get_scores(rewr_item)
        seed_scores = _get_scores(seed_item)

        # Hard constraints check
        factuality_ok = rewr_scores.get("Factuality", 1) == 0
        ambiguity_ok  = rewr_scores.get("Ambiguity", 1) == 0
        validity_ok   = rewr_scores.get("Validity", 1) == 0
        if not (factuality_ok and ambiguity_ok and validity_ok):
        # if not (facuality_ok and ambiguity_ok):
            mis_hard.append({
                "rewritten_question": rewr_q,
                "seed_question": seed_q,
                "image_path": img,
                "factuality_ok": factuality_ok,
                "ambiguity_ok": ambiguity_ok,
                "validity_ok": validity_ok,
                "rewrite_evol": rew_evol,
                "rewrite_prompt": rew_prompt,
            })
            continue

        # Any difficulty increased?
        if_qualified = []
        for asp in ("reasoning_difficulty", "perception_difficulty_text", "perception_difficulty_image"):
            if rewr_scores.get(asp, 0) > seed_scores.get(asp, 0):
                if_qualified.append(asp)
        
        if len(if_qualified) > 0:
            qualified.append({
                "rewritten_question": rewr_q,
                "seed_question": seed_q,
                "image_path": img,
                "increased_aspect": if_qualified,
                "rewrite_evol": rew_evol,
                "rewrite_prompt": rew_prompt,
            })
        else:
            mis_soft.append({
                    "rewritten_question": rewr_q,
                    "seed_question": seed_q,
                    "image_path": img,
                    "rewrite_evol": rew_evol,
                    "rewrite_prompt": rew_prompt,
            })

    return all_cnt, qualified, mis_hard, mis_soft


def _hard_score_from_info(info: dict) -> float | None:
    """Map a single hard-constraint block to a numeric score (0 pass, >0 fail)."""
    if not isinstance(info, dict):
        return None
    if "answerable" in info:       # Validity semantics: True->0, False->1
        return 0.0 if info.get("answerable", True) else 1.0
    if "count" in info:
        try:
            return float(info["count"])
        except Exception:
            return None
    return None

def _extract_hard_scores(api_res: dict) -> Dict[str, float]:
    """Return numeric hard scores for {Factuality, Ambiguity, Validity} if present."""
    out = {}
    hard = (api_res or {}).get("hard_constraints") or {}
    for k in ("Factuality", "Ambiguity", "Validity"):
        s = _hard_score_from_info(hard.get(k, {}))
        if s is not None:
            out[k] = s
    return out

def find_hard_constraint_violations(list_res: List[dict], threshold: float = 0.0):
    """
    Returns:
      violations: List[dict] each with
        - question, image_path, question_info
        - hard_scores: dict of numeric scores
        - failed_keys: list of aspects that failed (> threshold)
      summary: Counter with counts per failed aspect
    """
    flat = _normalize_results_schema(list_res)  # you already have this robust normalizer
    violations: List[dict] = []
    summary = Counter()

    for it in flat:
        api_res = it.get("api_res") or {}
        hard_scores = _extract_hard_scores(api_res)
        if not hard_scores:
            continue
        failed = [k for k, v in hard_scores.items() if v is not None and v > threshold]
        if failed:
            violations.append({
                "question": it.get("question"),
                "image_path": it.get("image_path"),
                "question_info": it.get("question_info", {}),
                "hard_scores": hard_scores,
                "failed_keys": failed,
            })
            summary.update(failed)

    return violations, summary


def attach_gpt_eval(data_list, gpt_map):
    """
    For each sample in data_list, look up seed_question and rewritten_question
    in gpt_map (keyed by (image_path, question)) and attach scores if found.

    Adds a new field:
      item["gpt_eval"] = {
          "seed_question": {...} or None,
          "rewritten_question": {...} or None
      }
    """
    num_matched = 0
    for item in data_list:
        img = item["image_path"]
        seed_q = item.get("seed_question")
        rew_q = item.get("rewritten_question")

        eval_seed = None
        eval_rew = None

        if seed_q is not None:
            eval_seed = gpt_map.get((img, seed_q))
        if rew_q is not None:
            eval_rew = gpt_map.get((img, rew_q))

        if eval_seed is not None and eval_rew is not None:
            num_matched += 1
            item["gpt_eval"] = {
                "seed_question": eval_seed,
                "rewritten_question": eval_rew,
            }

    print(f"Matched GPT evals for {num_matched} / {len(data_list)} items")
    return data_list


def check_improvement_for_gpt(list_res, gpt_map, check_improve: bool=True):
    """
    For each item in qualified:
      - look up GPT eval for seed_question and rewritten_question
      - compute delta = rewritten - seed for each metric
      - flag whether there is any improvement (any delta > 0)

    Adds a field:
      item["gpt_eval_compare"] = {
          "seed": {...} or None,
          "rewritten": {...} or None,
          "delta": {metric: diff} or None,
          "any_improved": bool,
          "any_worse": bool,
      }
    """
    metrics = ["Fluency", "Clarity", "Conciseness", "Relevance", "Consistency", "Answerability"]

    num_with_both = 0
    num_any_improved = 0
    num_missing = 0

    for item in list_res:
        img = item["image_path"]
        seed_q = item.get("seed_question")
        rew_q = item.get("rewritten_question")

        eval_seed = gpt_map.get((img, seed_q)) if seed_q is not None else None
        eval_rew = gpt_map.get((img, rew_q)) if rew_q is not None else None

        # Default
        delta = None
        any_improved = False
        any_worse = False

        if eval_seed is None or eval_rew is None:
            num_missing += 1
        else:
            num_with_both += 1
            delta = {}
            for m in metrics:
                d = eval_rew[m] - eval_seed[m]
                delta[m] = d
                if d > 0:
                    any_improved = True
                if d < 0:
                    any_worse = True

            if any_improved:
                num_any_improved += 1

        item["gpt_eval_compare"] = {
            "delta": delta,
            "any_improved": any_improved,
            "any_worse": any_worse,
        }

    print(f"Items with both seed & rewritten GPT evals: {num_with_both}")
    if check_improve:
        print(f"Items with ANY improved metric: {num_any_improved}")
    else:
        print(f"Items with ALL worse metric: {num_with_both - num_any_improved}")

    # print(f"Qualified items missing at least one GPT eval: {num_missing}")

    return list_res


def save_qgevol_csvs(qualified, mishard, missoft,
                     out_dir="scripts/data/analysis",):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 1. QUALIFIED
    # -------------------------
    rows = []
    for item in qualified:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if gpt_compare_res.get("any_improved", False):
            continue

        delta = gpt_compare_res.get("delta", {})
        worse_keys = [k for k, v in delta.items() if v < 0]

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
            "increased_aspect": ", ".join(item.get("increased_aspect", [])),
            "gpt_worse": ", ".join(worse_keys),
        })

    df_quality = pd.DataFrame(rows)
    df_quality.to_csv(os.path.join(out_dir, "qgevol_qualified.csv"),
                      sep="\t", index=False)

    print(f"[qualified] Saved {len(df_quality)} rows.")


    # -------------------------
    # 2. MIS-HARD (only improved)
    # -------------------------
    rows = []
    for item in mishard:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if not gpt_compare_res.get("any_improved", False):
            continue

        delta = gpt_compare_res.get("delta", {})
        better_keys = [k for k, v in delta.items() if v > 0]

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
            "gpt_better": ", ".join(better_keys),
        })

    df_mishard = pd.DataFrame(rows)
    df_mishard.to_csv(os.path.join(out_dir, "qgevol_mishard.csv"),
                      sep="\t", index=False)

    print(f"[mis-hard] Saved {len(df_mishard)} rows.")


    # -------------------------
    # 3. MIS-SOFT (only improved)
    # -------------------------
    rows = []
    for item in missoft:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if not gpt_compare_res.get("any_improved", False):
            continue

        delta = gpt_compare_res.get("delta", {})
        better_keys = [k for k, v in delta.items() if v > 0]

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
            "gpt_better": ", ".join(better_keys),
        })

    df_missoft = pd.DataFrame(rows)
    df_missoft.to_csv(os.path.join(out_dir, "qgevol_missoft.csv"),
                      sep="\t", index=False)

    print(f"[mis-soft] Saved {len(df_missoft)} rows.")

    return df_quality, df_mishard, df_missoft



def attach_gpt_eval_pair(data_list, gpt_map):
    """
    For each sample in data_list, look up seed_question and rewritten_question
    in gpt_map (keyed by (image_path, question)) and attach scores if found.

    Adds a new field:
      item["gpt_eval"] = {
          "seed_question": {...} or None,
          "rewritten_question": {...} or None
      }
    """
    num_matched = 0
    for item in data_list:
        img = item["image_path"]
        seed_q = item.get("seed_question")
        rew_q = item.get("rewritten_question")

        eval_res = gpt_map.get((img, seed_q, rew_q), None)
        if eval_res is not None:
            num_matched += 1
            item["gpt_eval"] = eval_res
        else:
            pass

    print(f"Matched GPT evals for {num_matched} / {len(data_list)} items")
    return data_list



def check_improvement_for_gpt_pair(list_res, gpt_map, check_improve: bool=True):
    """
    For each item in qualified:
      - look up GPT eval for seed_question and rewritten_question
      - compute delta = rewritten - seed for each metric
      - flag whether there is any improvement (any delta > 0)

    Adds a field:
      item["gpt_eval_compare"] = {
          "seed": {...} or None,
          "rewritten": {...} or None,
          "delta": {metric: diff} or None,
          "any_improved": bool,
          "any_worse": bool,
      }
    """
    num_with_both = 0
    num_any_improved = 0
    num_missing = 0

    for item in list_res:
        if_improved = None
        score = -1

        img = item["image_path"]
        seed_q = item.get("seed_question")
        rew_q = item.get("rewritten_question")

        eval_res = gpt_map.get((img, seed_q, rew_q), None)

        if eval_res is None:
            num_missing += 1
        else:
            num_with_both += 1
            if_improved = eval_res.get('improved', 'no')
            if if_improved.lower() == 'yes':
                if_improved = True
            else:
                if_improved = False

            score = eval_res.get('score', 0)
            if if_improved:
                num_any_improved += 1

        item["gpt_eval_compare"] = {
            "if_improved": if_improved,
            "score": score,
        }

    print(f"Items with both seed & rewritten GPT evals: {num_with_both}")
    if check_improve:
        print(f"Items with improved metric: {num_any_improved}")
    else:
        print(f"Items with worse metric: {num_with_both - num_any_improved}")

    # print(f"Qualified items missing at least one GPT eval: {num_missing}")

    return list_res


def save_pair_csvs(qualified, mishard, missoft, postfix='_self',
                     out_dir="scripts/data/analysis",):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 1. QUALIFIED
    # -------------------------
    rows = []
    for item in qualified:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if_improved = gpt_compare_res.get("if_improved", None)
        if if_improved or if_improved is None:
            continue

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
            "increased_aspect": ", ".join(item.get("increased_aspect", [])),
        })

    df_quality = pd.DataFrame(rows)
    df_quality.to_csv(os.path.join(out_dir, f"pair_qualified{postfix}.csv"),
                      sep="\t", index=False)

    print(f"[qualified] Saved {len(df_quality)} rows.")


    # -------------------------
    # 2. MIS-HARD (only improved)
    # -------------------------
    rows = []
    for item in mishard:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if_improved = gpt_compare_res.get("if_improved", None)
        if not if_improved or if_improved is None:
            continue

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
        })

    df_mishard = pd.DataFrame(rows)
    df_mishard.to_csv(os.path.join(out_dir, f"pair_mishard{postfix}.csv"),
                      sep="\t", index=False)

    print(f"[mis-hard] Saved {len(df_mishard)} rows.")


    # -------------------------
    # 3. MIS-SOFT (only improved)
    # -------------------------
    rows = []
    for item in missoft:
        gpt_compare_res = item.get("gpt_eval_compare", {})
        if_improved = gpt_compare_res.get("if_improved", None)
        if not if_improved or if_improved is None:
            continue

        rows.append({
            "seed_question": item.get("seed_question", ""),
            "rewritten_question": item.get("rewritten_question", ""),
        })

    df_missoft = pd.DataFrame(rows)
    df_missoft.to_csv(os.path.join(out_dir, f"pair_missoft{postfix}.csv"),
                      sep="\t", index=False)

    print(f"[mis-soft] Saved {len(df_missoft)} rows.")

    return df_quality, df_mishard, df_missoft
