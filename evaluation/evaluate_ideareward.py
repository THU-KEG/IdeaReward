#!/usr/bin/env python3
"""Evaluate both IdeaReward checkpoints with the paper's pointwise protocol."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train" / "OpenRLHF"))

CARD_FIELDS = ("title", "research_question", "motivations", "methods", "experiment_plan")
QUAD_ORDER = ("strong_accept", "standard_accept", "standard_reject", "strong_reject")
USER_REQUEST = """You are an expert researcher in computer science and artificial intelligence. Propose a self-contained research idea that is novel, technically sound, significant, clearly specified, and realistically executable, with the potential to yield an acceptance-worthy research paper if executed as proposed.

Present the research idea proposal in the following JSON format:
Required JSON shape:
{
  "title": "a concise research-paper-style title",
  "research_question": "a clear and specific research question",
  "motivations": "a clear explanation of the background and motivations of the research question",
  "methods": "a detailed description of the proposed approach",
  "experiment_plan": "a concrete plan for evaluating the proposed approach, including tasks or datasets, metrics, baselines, ablations, and other analysis experiments"
}
"""


def category(candidate: dict[str, Any], n_candidates: int) -> str:
    decision = candidate.get("final_decision")
    if n_candidates == 2 and isinstance(decision, str):
        if decision.endswith("accept"):
            return "standard_accept"
        if decision.endswith("reject"):
            return "standard_reject"
    if decision in QUAD_ORDER:
        return str(decision)
    rank = candidate.get("impact_rank")
    if rank is not None:
        rank = int(rank)
        if n_candidates == 2:
            return "standard_accept" if rank == 1 else "standard_reject"
        return QUAD_ORDER[rank - 1]
    raise ValueError(f"Cannot determine gold category for candidate {candidate.get('idea_id')}")


def task_name(path: Path, benchmark: Path) -> str:
    relative = path.relative_to(benchmark)
    return relative.parts[0] if len(relative.parts) > 1 else "benchmark"


def load_groups(benchmark: Path) -> list[dict[str, Any]]:
    groups = []
    benchmark_dirs = [benchmark / name for name in ("review_grounded", "impact_grounded", "annotation_grounded")]
    benchmark_dirs = [path for path in benchmark_dirs if path.is_dir()] or [benchmark]
    json_paths = sorted(path for directory in benchmark_dirs for path in directory.rglob("*.json"))
    for path in json_paths:
        if path.name == "manifest.json":
            continue
        try:
            group = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        candidates = group.get("candidates") if isinstance(group, dict) else None
        if not isinstance(candidates, list) or len(candidates) not in {2, 4}:
            continue
        seen_categories = set()
        normalized = []
        for candidate in candidates:
            card = candidate.get("anonymized_idea")
            if not isinstance(card, dict):
                raise ValueError(f"{path}: candidate lacks anonymized_idea")
            try:
                gold = category(candidate, len(candidates))
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            seen_categories.add(gold)
            normalized.append({
                "idea_id": str(candidate["idea_id"]),
                "gold_category": gold,
                "card": {field: str(card.get(field) or "") for field in CARD_FIELDS},
            })
        expected = {"standard_accept", "standard_reject"} if len(candidates) == 2 else set(QUAD_ORDER)
        if seen_categories != expected:
            raise ValueError(f"{path}: expected categories {sorted(expected)}, got {sorted(seen_categories)}")
        groups.append({
            "task": task_name(path, benchmark),
            "group_id": str(group.get("group_id") or path.stem),
            "split": "pair" if len(candidates) == 2 else "quadruple",
            "candidates": normalized,
        })
    if not groups:
        raise ValueError(f"No benchmark pair or quadruple JSON files found under {benchmark}")
    return groups


def messages(card: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": USER_REQUEST},
        {"role": "assistant", "content": json.dumps(card, ensure_ascii=False, indent=2)},
    ]


def score_checkpoint(checkpoint: Path, groups: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer
    from openrlhf.models import get_llm_for_sequence_regression

    if not torch.cuda.is_available():
        raise RuntimeError("IdeaReward evaluation requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = get_llm_for_sequence_regression(
        str(checkpoint), "reward", param_dtype="bf16", attn_implementation="flash_attention_2",
        init_value_head=False, packing_samples=False,
    ).cuda().eval()

    items = [(group, candidate) for group in groups for candidate in group["candidates"]]
    predictions = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        texts = []
        for _, candidate in batch:
            text = tokenizer.apply_chat_template(messages(candidate["card"]), tokenize=False).rstrip("\n")
            if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                text += " " + tokenizer.eos_token
            texts.append(text)
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=16384, add_special_tokens=False,
        ).to("cuda")
        encoded["input_ids"][:, -1] = tokenizer.eos_token_id
        encoded["attention_mask"][:, -1] = True
        with torch.inference_mode():
            scores = model(**encoded).float().view(-1).cpu().tolist()
        for (group, candidate), score in zip(batch, scores):
            valid = math.isfinite(score)
            predictions.append({
                "task": group["task"], "group_id": group["group_id"], "split": group["split"],
                "idea_id": candidate["idea_id"], "gold_category": candidate["gold_category"],
                "score": score if valid else None, "invalid": not valid,
            })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def preference_credit(better: float, worse: float) -> float:
    return 1.0 if better > worse else 0.5 if better == worse else 0.0


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record)
    pair_scores, top1_scores, pra_scores = [], [], []
    for group in grouped.values():
        by_category = {record["gold_category"]: record for record in group}
        invalid = any(record["invalid"] or record["score"] is None for record in group)
        if group[0]["split"] == "pair":
            pair_scores.append(0.0 if invalid else preference_credit(
                float(by_category["standard_accept"]["score"]),
                float(by_category["standard_reject"]["score"]),
            ))
            continue
        if invalid:
            top1_scores.append(0.0)
            pra_scores.append(0.0)
            continue
        scores = {name: float(by_category[name]["score"]) for name in QUAD_ORDER}
        best = max(scores.values())
        tied_top = [name for name, score in scores.items() if score == best]
        top1_scores.append(1.0 / len(tied_top) if "strong_accept" in tied_top else 0.0)
        credits = [preference_credit(scores[better], scores[worse])
                   for better, worse in combinations(QUAD_ORDER, 2)]
        pra_scores.append(sum(credits) / 6.0)
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "pair_accuracy": mean(pair_scores),
        "quadruple_top1_accuracy": mean(top1_scores),
        "quadruple_pairwise_ranking_accuracy": mean(pra_scores),
        "pair_groups": len(pair_scores),
        "quadruple_groups": len(top1_scores),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--llama-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    groups = load_groups(args.benchmark.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in (("IdeaReward-Llama", args.llama_checkpoint), ("IdeaReward-Qwen", args.qwen_checkpoint)):
        predictions = score_checkpoint(checkpoint.resolve(), groups, args.batch_size)
        write_jsonl(args.output_dir / f"{name}.jsonl", predictions)
        tasks = sorted({record["task"] for record in predictions})
        report = {task: metrics([record for record in predictions if record["task"] == task]) for task in tasks}
        (args.output_dir / f"{name}-metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {json.dumps(report, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
