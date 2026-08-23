"""
evaluate.py - Evaluation & Metrics Reporter
Mastercard Innovation Challenge 2026

Reads detection_results.jsonl, computes classification metrics (Accuracy,
Precision, Recall, F1, Confusion Matrix, and Per-Attack-Type breakdowns),
prints the report to stdout, and writes evaluation_report.md.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    # Fallback in case scikit-learn is not installed in standard env
    accuracy_score = None


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Loads detection results from JSONL file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file '{results_file}' not found.")

    records = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line {line_num} in {results_file}")
    return records


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall and per-attack-type performance metrics."""
    if not records:
        return {"total": 0}

    total = len(records)
    y_true = [str(r.get("actual_label", "")).strip().lower() for r in records]
    y_pred = [str(r.get("predicted_label", "")).strip().lower() for r in records]
    
    # Calculate basic counts
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "fraud" and yp == "fraud")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "legitimate" and yp == "legitimate")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "legitimate" and yp == "fraud")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "fraud" and yp == "legitimate")

    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0

    # Fraud class metrics (Positive = Fraud)
    precision_fraud = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_fraud = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_fraud = (2 * precision_fraud * recall_fraud / (precision_fraud + recall_fraud)) if (precision_fraud + recall_fraud) > 0 else 0.0

    # Legit class metrics (Positive = Legitimate)
    precision_legit = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_legit = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_legit = (2 * precision_legit * recall_legit / (precision_legit + recall_legit)) if (precision_legit + recall_legit) > 0 else 0.0

    # Macro averages
    macro_f1 = (f1_fraud + f1_legit) / 2.0
    macro_precision = (precision_fraud + precision_legit) / 2.0
    macro_recall = (recall_fraud + recall_legit) / 2.0

    # Per-attack-type breakdown
    # Attack type is recorded in predicted_attack_type or inferred from case records
    attack_type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "missed": 0, "confidences": []})
    
    for r in records:
        # Check actual or predicted attack type
        atk = r.get("predicted_attack_type") or r.get("attack_type") or "none"
        is_corr = r.get("correct", False)
        conf = float(r.get("confidence", 0.0))
        
        attack_type_stats[atk]["total"] += 1
        attack_type_stats[atk]["confidences"].append(conf)
        if is_corr:
            attack_type_stats[atk]["correct"] += 1
        else:
            attack_type_stats[atk]["missed"] += 1

    per_attack_metrics = {}
    for atk, stats in sorted(attack_type_stats.items()):
        tot = stats["total"]
        cor = stats["correct"]
        acc = cor / tot if tot > 0 else 0.0
        avg_conf = sum(stats["confidences"]) / tot if tot > 0 else 0.0
        per_attack_metrics[atk] = {
            "total_cases": tot,
            "correct_predictions": cor,
            "accuracy": acc,
            "avg_confidence": avg_conf
        }

    # False Negatives (Fraud cases classified as Legit)
    false_negatives = [r for r in records if str(r.get("actual_label", "")).lower() == "fraud" and str(r.get("predicted_label", "")).lower() == "legitimate"]
    
    # False Positives (Legit cases classified as Fraud)
    false_positives = [r for r in records if str(r.get("actual_label", "")).lower() == "legitimate" and str(r.get("predicted_label", "")).lower() == "fraud"]

    return {
        "total_evaluated": total,
        "accuracy": accuracy,
        "precision_fraud": precision_fraud,
        "recall_fraud": recall_fraud,
        "f1_fraud": f1_fraud,
        "precision_legitimate": precision_legit,
        "recall_legitimate": recall_legit,
        "f1_legitimate": f1_legit,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": {
            "true_positives_fraud": tp,
            "true_negatives_legit": tn,
            "false_positives_fraud": fp,
            "false_negatives_fraud": fn
        },
        "per_attack_type": per_attack_metrics,
        "false_negatives": false_negatives,
        "false_positives": false_positives
    }


def generate_markdown_report(metrics: Dict[str, Any], output_md_path: str):
    """Formats metrics as a rich markdown report."""
    cm = metrics["confusion_matrix"]
    
    md = []
    md.append("# Mastercard Innovation Challenge 2026 — Blue Team Detector Evaluation Report")
    md.append("\n**Component**: Person C — GenAI Fraud Detector")
    md.append(f"**Total Cases Evaluated**: `{metrics['total_evaluated']}`")
    md.append("\n---\n")

    md.append("## 1. Overall Performance Summary\n")
    md.append("| Metric | Score | Percentage |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Overall Accuracy** | `{metrics['accuracy']:.4f}` | **{metrics['accuracy']*100:.2f}%** |")
    md.append(f"| **Fraud F1-Score** | `{metrics['f1_fraud']:.4f}` | **{metrics['f1_fraud']*100:.2f}%** |")
    md.append(f"| **Fraud Precision** | `{metrics['precision_fraud']:.4f}` | {metrics['precision_fraud']*100:.2f}% |")
    md.append(f"| **Fraud Recall (Detection Rate)** | `{metrics['recall_fraud']:.4f}` | {metrics['recall_fraud']*100:.2f}% |")
    md.append(f"| **Legitimate Precision** | `{metrics['precision_legitimate']:.4f}` | {metrics['precision_legitimate']*100:.2f}% |")
    md.append(f"| **Legitimate Recall** | `{metrics['recall_legitimate']:.4f}` | {metrics['recall_legitimate']*100:.2f}% |")
    md.append(f"| **Macro F1-Score** | `{metrics['macro_f1']:.4f}` | {metrics['macro_f1']*100:.2f}% |")

    md.append("\n## 2. Confusion Matrix\n")
    md.append("| | Predicted Fraud | Predicted Legitimate | Total Actual |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Actual Fraud** | `{cm['true_positives_fraud']}` (TP) | `{cm['false_negatives_fraud']}` (FN) | `{cm['true_positives_fraud'] + cm['false_negatives_fraud']}` |")
    md.append(f"| **Actual Legitimate** | `{cm['false_positives_fraud']}` (FP) | `{cm['true_negatives_legit']}` (TN) | `{cm['false_positives_fraud'] + cm['true_negatives_legit']}` |")
    md.append(f"| **Total Predicted** | `{cm['true_positives_fraud'] + cm['false_positives_fraud']}` | `{cm['false_negatives_fraud'] + cm['true_negatives_legit']}` | `{metrics['total_evaluated']}` |")

    md.append("\n## 3. Per-Attack-Type Performance Breakdown\n")
    md.append("| Attack Type | Total Cases | Correct | Accuracy / Recall | Avg Confidence |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for atk, st in metrics["per_attack_type"].items():
        md.append(f"| `{atk}` | {st['total_cases']} | {st['correct_predictions']} | **{st['accuracy']*100:.1f}%** | {st['avg_confidence']:.2f} |")

    md.append("\n## 4. Error Analysis & Weak Spots\n")
    fn_list = metrics["false_negatives"]
    fp_list = metrics["false_positives"]

    md.append(f"- **False Negatives (Missed Fraud)**: `{len(fn_list)}` cases")
    if fn_list:
        md.append("\n| Case ID | Predicted Attack Type | Confidence | Reason Snippet |")
        md.append("| :--- | :--- | :--- | :--- |")
        for fn in fn_list[:10]:
            md.append(f"| `{fn.get('case_id')}` | `{fn.get('predicted_attack_type')}` | `{fn.get('confidence', 0.0):.2f}` | {fn.get('reason', '')[:80]}... |")

    md.append(f"\n- **False Positives (False Alarms)**: `{len(fp_list)}` cases")
    if fp_list:
        md.append("\n| Case ID | Predicted Attack Type | Confidence | Reason Snippet |")
        md.append("| :--- | :--- | :--- | :--- |")
        for fp in fp_list[:10]:
            md.append(f"| `{fp.get('case_id')}` | `{fp.get('predicted_attack_type')}` | `{fp.get('confidence', 0.0):.2f}` | {fp.get('reason', '')[:80]}... |")

    md.append("\n---\n*Report automatically generated by `evaluate.py`.*")

    report_content = "\n".join(md)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


def print_console_summary(metrics: Dict[str, Any]):
    """Prints a clean summary table to console."""
    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 65)
    print("      MASTERCARD BLUE TEAM FRAUD DETECTOR EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Cases Evaluated  : {metrics['total_evaluated']}")
    print(f"Overall Accuracy       : {metrics['accuracy']*100:.2f}% ({metrics['accuracy']:.4f})")
    print(f"Fraud F1 Score         : {metrics['f1_fraud']*100:.2f}% ({metrics['f1_fraud']:.4f})")
    print(f"Fraud Precision        : {metrics['precision_fraud']*100:.2f}% ({metrics['precision_fraud']:.4f})")
    print(f"Fraud Recall (Coverage): {metrics['recall_fraud']*100:.2f}% ({metrics['recall_fraud']:.4f})")
    print(f"Macro F1 Score         : {metrics['macro_f1']*100:.2f}% ({metrics['macro_f1']:.4f})")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (Fraud correctly identified) : {cm['true_positives_fraud']}")
    print(f"  True Negatives  (Legit correctly identified) : {cm['true_negatives_legit']}")
    print(f"  False Positives (Legit flagged as Fraud)     : {cm['false_positives_fraud']}")
    print(f"  False Negatives (Fraud missed as Legit)      : {cm['false_negatives_fraud']}")
    print("-" * 65)
    print("PER-ATTACK-TYPE BREAKDOWN:")
    print(f"{'Attack Type':<26} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10} | {'Confidence':<10}")
    print("-" * 65)
    for atk, st in metrics["per_attack_type"].items():
        print(f"{atk:<26} | {st['total_cases']:<6} | {st['correct_predictions']:<8} | {st['accuracy']*100:>7.1f}%  | {st['avg_confidence']:>9.2f}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Person C Fraud Detection Results")
    parser.add_argument(
        "--results", "-r",
        default="detection_results.jsonl",
        help="Path to detection_results.jsonl (default: detection_results.jsonl)"
    )
    parser.add_argument(
        "--report", "-o",
        default="evaluation_report.md",
        help="Path to save evaluation_report.md (default: evaluation_report.md)"
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file '{args.results}' not found. Run detector.py first.")
        sys.exit(1)

    records = load_results(args.results)
    if not records:
        print(f"Error: No records found in '{args.results}'.")
        sys.exit(1)

    metrics = compute_metrics(records)
    print_console_summary(metrics)
    generate_markdown_report(metrics, args.report)
    print(f"Full markdown evaluation report saved to: {args.report}")


if __name__ == "__main__":
    main()
