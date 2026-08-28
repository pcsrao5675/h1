# FraudLens

> **Mastercard Innovation Challenge 2026 @ GFF** — AI Defense Lab for Payment Security
> Red-team / blue-team pipeline that identifies emerging GenAI-powered payment fraud, simulates it at scale, and defends against it with a closed feedback loop.

---

## What this is

FraudLens is a closed-loop adversarial pipeline built for the Mastercard Innovation Challenge 2026 at the Global Fintech Fest. It owns the full red-team/blue-team cycle in three pillars:

1. **Identify** — map the landscape of emerging GenAI-native payment fraud (6 attack categories, including prompt-injection scams and deepfake KYC).
2. **Generate** — simulate realistic fraud cases and transactions at scale, including adversarial round-2 mutations targeted at detector gaps.
3. **Defend** — detect fraud using a Gemini-powered classifier with multi-key rotation and 3-model fallback.

The same loop runs forward and backward: detector false-negatives from round 1 (`weak_spots.json`) feed round-2 augmentation, which is then evaluated again to measure recall lift.

```
              ┌────────────────────────────────────────────────┐
              │  PILLAR 1 — Identify                          │
              │  Research & categorise emerging GenAI fraud    │
              │  6 categories, 79 hand-curated seed cases      │
              └─────────────────┬──────────────────────────────┘
                                ▼
              ┌────────────────────────────────────────────────┐
              │  PILLAR 2 — Generate                           │
              │  Gemini synthesises cases (seed + augmented)   │
              │  Adversarial round 2 targets weak_spots.json   │
              └─────────────────┬──────────────────────────────┘
                                ▼
              ┌────────────────────────────────────────────────┐
              │  PILLAR 3 — Defend                             │
              │  detector.py (Gemini 2.5/2.0/1.5 fallback,     │
              │  multi-key rotation, JSON-mode response)       │
              └─────────────────┬──────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────────────────┐
              │  CLOSED LOOP                                   │
              │  find_weak_spots.py → weak_spots.json           │
              │  scale_up_dataset.py → augmented_dataset.jsonl  │
              │  compare_rounds.py → recall-lift report         │
              └────────────────────────────────────────────────┘
```

## Live prototype

The closed-loop dashboard is deployed as a FastAPI service on Render:
**https://fraudlens.onrender.com/**

The prototype shows a 3-panel operational view: live pipeline ticker, case browser, and detector playground, plus a metrics view that surfaces the closed-loop recall lift.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set up API keys (Gemini)
cp .env.example .env
# edit .env and add GEMINI_API_KEY_1 (or more, up to 3)

# 3. Run the closed loop end-to-end
python detector.py --input seed_dataset.jsonl augmented_dataset.jsonl
python evaluate.py
python find_weak_spots.py
python scale_up_dataset.py      # consumes weak_spots.json, produces round 2
python compare_rounds.py        # writes round_comparison_report.md + metrics_report_final.json

# 4. Start the prototype
uvicorn backend.main:app --reload
# open http://localhost:8000
```

## Repo layout

| File | Role |
| :--- | :--- |
| `generate_seed_attacks.py` | **Person A** — Gemini-driven seed generator (79 cases across 6 attack types) |
| `scale_up_dataset.py` | **Person B** — round-2 adversarial augmenter (consumes `weak_spots.json`) |
| `detector.py` | **Person C** — fraud detector (Gemini, multi-key, fallback) |
| `evaluate.py` | Person C — accuracy / precision / recall / F1 / confusion-matrix report |
| `find_weak_spots.py` | Person C — exports false-negatives to `weak_spots.json` for the next round |
| `compare_rounds.py` | Person C — round-1 vs round-2 recall-lift report (the closed-loop number) |
| `seed_dataset.jsonl` | 79 hand-curated seed cases |
| `augmented_dataset.jsonl` | 520 augmented round-2 cases |
| `detection_results.jsonl` | detector output (one row per case) |
| `weak_spots.json` | false-negatives fed back into augmentation |
| `metrics_report_final.json` | computed metrics for the prototype and walkthrough |
| `round_comparison_report.md` | round-1 → round-2 recall-lift table |
| `evaluation_report.md` | full per-class metrics, confusion matrix, error analysis |
| `backend/` | FastAPI prototype (Stitch-style dashboard, served from `index.html`) |
| `walkthrough.docx` | judges' writeup (see `submission/`) |
| `submission/` | submission package — `walkthrough.docx`, `solution-summary.md`, screenshots, links |

## Headline numbers

| Metric | Value | Source |
| :--- | :--- | :--- |
| Cases evaluated | **599** (79 seed + 520 aug) | `detection_results.jsonl` |
| Overall accuracy | **93.16%** | `evaluation_report.md` |
| Fraud F1 | **93.64%** | `evaluation_report.md` |
| Fraud recall | **93.50%** | `evaluation_report.md` |
| Closed-loop recall lift | see `round_comparison_report.md` | `compare_rounds.py` |
| Attack categories | **6** | `generate_seed_attacks.py` |
| False positives | **20** | `evaluation_report.md` |
| False negatives | **21** | `evaluation_report.md` |

The single most important number is the recall lift in `round_comparison_report.md` — it directly answers the challenge's "did the closed loop work?" question.

## The 6 attack categories

| Category | What it is | Why GenAI amplifies it |
| :--- | :--- | :--- |
| `account_takeover` | Credential theft / session hijack | AI-generated phishing, voice cloning for IVR bypass |
| `synthetic_identity` | Real + fake identity blended to open accounts | AI-generated KYC documents, synthetic faces |
| `refund_fraud` | Fake chargebacks / "item not received" | AI-edited delivery screenshots, fabricated narratives |
| `card_testing` | Micro-transactions to validate stolen cards | Bot scripts generated and tuned by LLMs |
| `prompt_injection_scam` | Adversarial prompts against bank chatbots | Direct attack vector — the AI itself is the target |
| `deepfake_kyc` | Synthetic video / voice during onboarding | Diffusion-model faces, neural voice clones |

`prompt_injection_scam` and `deepfake_kyc` are the **novel GenAI-native vectors** at the heart of the challenge.

## How the closed loop works

1. **Person A** generates 79 hand-curated seed cases covering all 6 attack types using Gemini.
2. **Person C** runs `detector.py` on the seeds, producing `detection_results.jsonl`.
3. **Person C** runs `find_weak_spots.py` → exports false-negatives to `weak_spots.json`.
4. **Person B** runs `scale_up_dataset.py`, which mutates the seed cases and targets the attack types where the detector is weakest, producing 520 augmented round-2 cases.
5. **Person C** runs `detector.py` on the combined dataset, then `compare_rounds.py` to produce the lift report.

The lift report (`round_comparison_report.md`) is the proof that the loop closes — the same detector, run on adversarial round-2 data, measurably improves on the attack types the red team targeted.

## License & team

Built for the Mastercard Innovation Challenge 2026 at GFF Mumbai. Team: see `submission/links.txt`.
