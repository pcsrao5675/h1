"""
find_weak_spots.py - Person C (Adversarial Feedback Generator)
Mastercard Innovation Challenge 2026

Identifies false negatives (fraud cases misclassified as legitimate),
ranks attack types by lowest recall, and exports weak_spots.json
for Person B to generate targeted adversarial round-2 mutations.
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any

def extract_weak_spots(results_file: str, dataset_files: List[str], output_file: str, top_k_examples: int = 10):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file '{results_file}' not found.")

    # 1. Map case_id to narrative and original metadata
    case_meta = {}
    for df in dataset_files:
        if os.path.exists(df):
            with open(df, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cid = rec.get("case_id")
                        if cid:
                            case_meta[cid] = rec
                    except json.JSONDecodeError:
                        continue

    # 2. Read detection results
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 3. Compute per-attack-type stats & collect false negatives
    attack_totals = defaultdict(int)
    attack_correct = defaultdict(int)
    missed_by_type = defaultdict(list)

    for r in results:
        cid = r.get("case_id")
        actual_label = str(r.get("actual_label", "")).strip().lower()
        pred_label = str(r.get("predicted_label", "")).strip().lower()
        
        # Get actual attack type from dataset metadata if available
        meta = case_meta.get(cid, {})
        attack_type = meta.get("attack_type") or r.get("predicted_attack_type") or "none"

        if actual_label == "fraud":
            attack_totals[attack_type] += 1
            if pred_label == "fraud":
                attack_correct[attack_type] += 1
            else:
                # False negative (Missed fraud)
                narrative = meta.get("narrative") or r.get("reason", "")
                missed_by_type[attack_type].append({
                    "case_id": cid,
                    "narrative": narrative,
                    "attack_type": attack_type,
                    "confidence": r.get("confidence", 0.0),
                    "detector_reason": r.get("reason", "")
                })

    # 4. Rank attack types by lowest recall (weakest first)
    recall_by_type = []
    for atk, total in attack_totals.items():
        if atk == "none":
            continue
        corr = attack_correct[atk]
        recall = corr / total if total > 0 else 0.0
        recall_by_type.append((atk, recall, total, corr))

    recall_by_type.sort(key=lambda x: x[1])

    # Identify weak attack types (recall < 1.0 or lowest 2)
    weak_attack_types = [atk for atk, rec, _, _ in recall_by_type if rec < 1.0]
    if not weak_attack_types and recall_by_type:
        weak_attack_types = [recall_by_type[0][0]]

    # Collect example missed cases for the weak attack types
    example_missed_cases = []
    for atk in weak_attack_types:
        cases = missed_by_type.get(atk, [])
        for c in cases[:top_k_examples]:
            example_missed_cases.append({
                "case_id": c["case_id"],
                "narrative": c["narrative"],
                "attack_type": c["attack_type"]
            })

    output_data = {
        "weak_attack_types": weak_attack_types,
        "example_missed_cases": example_missed_cases
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("        PERSON C — WEAK SPOTS IDENTIFIED FOR PERSON B")
    print("=" * 60)
    print(f"Weak Attack Types Ranked: {weak_attack_types}")
    print(f"Total Example Missed Cases Exported: {len(example_missed_cases)}")
    print(f"Exported feedback file: {output_file}")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Find weak spots and export feedback for Person B")
    parser.add_argument("--results", "-r", default="detection_results.jsonl")
    parser.add_argument("--datasets", "-d", nargs="+", default=["seed_dataset.jsonl", "augmented_dataset.jsonl"])
    parser.add_argument("--output", "-o", default="weak_spots.json")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    extract_weak_spots(args.results, args.datasets, args.output, args.top_k)

if __name__ == "__main__":
    main()
