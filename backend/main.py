"""
Mastercard Hackathon - Fraud Detection Backend
FastAPI server with endpoints:
  GET  /health          -> health check
  GET  /sample-attacks  -> sample fraud cases from datasets
  POST /predict         -> run detector on a case_id or raw narrative
  GET  /metrics         -> round1 vs final recall per attack type
  GET  /summary         -> pipeline summary stats
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# App setup
app = FastAPI(
    title="Mastercard Fraud Detection API",
    description="Red-team / Blue-team GenAI fraud detection pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths (backend/ is inside FullRepo/)
BASE          = Path(__file__).parent.parent
SEED_PATH     = BASE / "seed_dataset.jsonl"
AUG_PATH      = BASE / "augmented_dataset.jsonl"
RESULTS_PATH  = BASE / "detection_results.jsonl"
METRICS_PATH  = BASE / "metrics_report_final.json"

# Helpers
def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_results_index() -> dict:
    results = load_jsonl(RESULTS_PATH)
    return {r["case_id"]: r for r in results}

# Pydantic models
class PredictRequest(BaseModel):
    case_id: Optional[str] = None
    narrative: Optional[str] = None
    transaction: Optional[dict] = None

# GET /health
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

# GET /sample-attacks
@app.get("/sample-attacks")
def sample_attacks(limit: int = 10):
    seed = load_jsonl(SEED_PATH)
    if not seed:
        raise HTTPException(status_code=500, detail="seed_dataset.jsonl not found")

    ATTACK_TYPES = [
        "account_takeover", "synthetic_identity", "refund_fraud",
        "card_testing", "prompt_injection_scam", "deepfake_kyc",
    ]
    selected = []
    seen_attacks = set()
    legit_count = 0

    for case in seed:
        at  = case.get("attack_type", "none")
        lbl = case.get("label", "")
        if at in ATTACK_TYPES and at not in seen_attacks:
            selected.append(case)
            seen_attacks.add(at)
        elif lbl == "legitimate" and legit_count < 2:
            selected.append(case)
            legit_count += 1

    out = []
    for c in selected[:limit]:
        txn     = c.get("transaction", {})
        full_n  = c.get("narrative", "")
        snippet = full_n[:300].replace("\n", " ").strip()
        if len(full_n) > 300:
            snippet += "..."
        out.append({
            "case_id":            c["case_id"],
            "attack_type":        c.get("attack_type", "none"),
            "label":              c.get("label", "unknown"),
            "narrative_snippet":  snippet,
            "amount":             txn.get("amount"),
            "currency":           txn.get("currency"),
            "channel":            txn.get("channel"),
            "cardholder_country": txn.get("cardholder_country"),
            "merchant_category":  txn.get("merchant_category"),
            "generation_method":  c.get("generation_method", "seed"),
        })
    return out

# POST /predict
@app.post("/predict")
def predict(req: PredictRequest):
    results_idx = load_results_index()
    seed        = load_jsonl(SEED_PATH)
    aug         = load_jsonl(AUG_PATH)
    all_cases   = {c["case_id"]: c for c in seed + aug}

    # Case A: lookup by case_id
    if req.case_id:
        cid = req.case_id
        if cid not in results_idx:
            raise HTTPException(status_code=404, detail=f"case_id '{cid}' not found")
        result   = results_idx[cid]
        case     = all_cases.get(cid, {})
        txn      = case.get("transaction", {})
        full_n   = case.get("narrative", "")
        snippet  = full_n[:300].replace("\n", " ").strip()
        if len(full_n) > 300:
            snippet += "..."
        return {
            "case_id":              cid,
            "predicted_label":      result["predicted_label"],
            "predicted_attack_type":result["predicted_attack_type"],
            "confidence":           result["confidence"],
            "reason":               result["reason"],
            "actual_label":         result.get("actual_label"),
            "correct":              result.get("correct"),
            "narrative_snippet":    snippet,
            "transaction":          txn,
        }

    # Case B: raw narrative - keyword heuristic (no API cost for demo)
    if req.narrative:
        narrative = req.narrative.lower()
        txn       = req.transaction or {}

        KEYWORDS = {
            "account_takeover":      ["otp","password","reset","login","verify","credential","hijack","locked out"],
            "synthetic_identity":    ["identity","kyc","document","ssn","synthetic","fabricated","fake id"],
            "refund_fraud":          ["refund","chargeback","dispute","return","not received","never arrived"],
            "card_testing":          ["small charge","micro","test transaction","0.99","1.00","verify card"],
            "prompt_injection_scam": ["ignore previous","prompt","injection","system:","disregard","chatbot","override"],
            "deepfake_kyc":          ["deepfake","voice clone","synthetic video","liveness","biometric","face swap"],
        }

        scores = {k: sum(1 for kw in kws if kw in narrative) for k, kws in KEYWORDS.items()}
        best_attack = max(scores, key=lambda k: scores[k])
        best_score  = scores[best_attack]

        if best_score == 0:
            label, attack_type = "legitimate", "none"
            confidence = 0.72
            reason = "No significant fraud indicators detected in the narrative."
        else:
            label, attack_type = "fraud", best_attack
            confidence = min(0.60 + best_score * 0.08, 0.97)
            reason = f"Narrative contains {best_score} indicator(s) consistent with {attack_type.replace('_',' ')} pattern."

        snippet = req.narrative[:300].replace("\n", " ").strip()
        if len(req.narrative) > 300:
            snippet += "..."

        return {
            "case_id":               "live_input",
            "predicted_label":       label,
            "predicted_attack_type": attack_type,
            "confidence":            round(confidence, 2),
            "reason":                reason,
            "actual_label":          None,
            "correct":               None,
            "narrative_snippet":     snippet,
            "transaction":           txn,
        }

    raise HTTPException(status_code=400, detail="Provide 'case_id' or 'narrative'")

# GET /metrics
@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=500, detail="metrics_report_final.json not found")
    with open(METRICS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    chart_data = []
    for attack_type, vals in raw.get("per_attack_type", {}).items():
        chart_data.append({
            "attack_type":   attack_type.replace("_", " ").title(),
            "round1_recall": round(vals["r1_recall"] * 100, 1),
            "final_recall":  round(vals["final_recall"] * 100, 1),
            "improvement":   round((vals["final_recall"] - vals["r1_recall"]) * 100, 1),
            "cases":         vals["cases"],
        })

    return {
        "overall": {
            "round1_recall": round(raw["overall_r1_recall"] * 100, 1),
            "final_recall":  round(raw["overall_final_recall"] * 100, 1),
            "recall_lift":   round(raw["overall_recall_lift"] * 100, 1),
        },
        "per_attack_type": chart_data,
        "chart_labels":    [d["attack_type"] for d in chart_data],
        "round1_values":   [d["round1_recall"] for d in chart_data],
        "final_values":    [d["final_recall"] for d in chart_data],
    }

# GET /summary
@app.get("/summary")
def summary():
    seed    = load_jsonl(SEED_PATH)
    aug     = load_jsonl(AUG_PATH)
    results = load_jsonl(RESULTS_PATH)
    correct = sum(1 for r in results if r.get("correct") is True)
    total   = len(results)
    return {
        "seed_cases":      len(seed),
        "augmented_cases": len(aug),
        "total_evaluated": total,
        "accuracy_pct":    round(correct / total * 100, 2) if total else 0,
        "fraud_cases":     sum(1 for c in seed + aug if c.get("label") == "fraud"),
        "legit_cases":     sum(1 for c in seed + aug if c.get("label") == "legitimate"),
        "attack_types":    6,
    }
