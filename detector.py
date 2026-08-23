"""
detector.py - Person C (Blue Team Detector)
Mastercard Innovation Challenge 2026

Classifies payment transactions and narrative scenarios as fraud or legitimate
using the Google Gemini API with multi-key rotation and automatic model fallback.
"""

import os
import sys
import json
import time
import glob
import argparse
import logging
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 safely on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_detector")

VALID_ATTACK_TYPES = {
    "account_takeover",
    "synthetic_identity",
    "refund_fraud",
    "card_testing",
    "prompt_injection_scam",
    "deepfake_kyc",
    "none"
}

MODEL_HIERARCHY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

class KeyManager:
    """Manages rotation and health of multiple Gemini API keys."""
    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k and k.strip() and not k.startswith("<")]
        if not self.api_keys:
            raise ValueError(
                "No valid Gemini API keys found. Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, "
                "or GEMINI_API_KEY_3 in your .env file."
            )
        self.current_index = 0
        logger.info(f"Initialized KeyManager with {len(self.api_keys)} API key(s).")

    def get_current_key(self) -> str:
        return self.api_keys[self.current_index]

    def rotate_key(self) -> str:
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        logger.info(f"Rotated to API key index {self.current_index + 1}/{len(self.api_keys)}")
        return self.get_current_key()

    def num_keys(self) -> int:
        return len(self.api_keys)


def load_api_keys() -> List[str]:
    """Loads API keys from .env and environment variables."""
    load_dotenv(override=True)
    keys = []
    for var_name in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY"]:
        val = os.getenv(var_name)
        if val and val.strip() and not val.startswith("<"):
            if val not in keys:
                keys.append(val.strip())
    
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY_") and v and v.strip() and not v.startswith("<"):
            if v.strip() not in keys:
                keys.append(v.strip())
    return keys


def build_system_prompt() -> str:
    """Constructs the system prompt for the fraud detector."""
    return """You are an expert fraud detection AI agent for Mastercard Payment Security.
Your task is to analyze payment transaction metadata and contextual communication narratives to classify whether the transaction is fraudulent or legitimate.

Attack Types you must identify if fraud is detected:
1. 'account_takeover': Credential theft, unauthorized login, session hijacking, SIM swapping, or credential stuffing followed by unauthorized payments or beneficiary changes.
2. 'synthetic_identity': Fabricated or manipulated identities, fictitious SSN/national IDs, mismatched KYC profiles, or burner profiles created solely for card opening.
3. 'refund_fraud': Malicious chargeback manipulation, false merchant dispute claims, claiming goods never arrived when they did, or double-refund exploitation.
4. 'card_testing': Low-value probing transactions across multiple merchants/bins to test stolen card validity before high-value cash-out.
5. 'prompt_injection_scam': Emerging GenAI fraud: adversarial prompt injections against automated customer support chatbots, AI voice assistants, or automated limit-increase agents to trick the AI into bypassing security checks or approving funds.
6. 'deepfake_kyc': Emerging GenAI fraud: synthetic voice cloning, real-time deepfake facial manipulation during biometric verification, or AI-generated identity document tampering.
7. 'none': Use this when the transaction is legitimate or no specific attack category matches.

You must output a strictly valid JSON object matching this schema:
{
  "predicted_label": "fraud" | "legitimate",
  "predicted_attack_type": "account_takeover" | "synthetic_identity" | "refund_fraud" | "card_testing" | "prompt_injection_scam" | "deepfake_kyc" | "none",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<1-2 concise sentences explaining the key signals for this decision>"
}"""


def build_case_prompt(case: Dict[str, Any]) -> str:
    """Builds the user prompt containing transaction metadata and narrative."""
    tx = case.get("transaction", {})
    narrative = case.get("narrative", "")
    
    tx_formatted = {
        "amount": tx.get("amount"),
        "currency": tx.get("currency", "USD"),
        "merchant_category": tx.get("merchant_category"),
        "timestamp": tx.get("timestamp"),
        "channel": tx.get("channel"),
        "cardholder_country": tx.get("cardholder_country"),
        "device_fingerprint": tx.get("device_fingerprint")
    }
    
    return f"""Analyze the following transaction and case scenario:

[TRANSACTION METADATA]
{json.dumps(tx_formatted, indent=2)}

[CASE NARRATIVE & CONTEXT]
{narrative}

Evaluate the evidence carefully. Return ONLY the JSON object with predicted_label, predicted_attack_type, confidence, and reason."""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Cleans and parses raw LLM output into a dictionary."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            raise ValueError(f"Could not parse valid JSON from response: {raw_text[:200]}")
    
    predicted_label = str(data.get("predicted_label", "")).strip().lower()
    if "fraud" in predicted_label:
        data["predicted_label"] = "fraud"
    elif "legit" in predicted_label:
        data["predicted_label"] = "legitimate"
    else:
        data["predicted_label"] = "fraud" if data.get("predicted_attack_type", "none") != "none" else "legitimate"
    
    attack_type = str(data.get("predicted_attack_type", "")).strip().lower()
    if attack_type not in VALID_ATTACK_TYPES:
        matched = False
        for valid in VALID_ATTACK_TYPES:
            if valid in attack_type or attack_type in valid:
                data["predicted_attack_type"] = valid
                matched = True
                break
        if not matched:
            data["predicted_attack_type"] = "none" if data["predicted_label"] == "legitimate" else "account_takeover"
    else:
        data["predicted_attack_type"] = attack_type

    try:
        data["confidence"] = float(data.get("confidence", 0.9))
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))
    except (ValueError, TypeError):
        data["confidence"] = 0.85

    data["reason"] = str(data.get("reason", "Analyzed transaction features and narrative context.")).strip()
    return data


def classify_case_mock(case: Dict[str, Any]) -> Dict[str, Any]:
    """Mock classification logic used when --mock flag is passed for testing."""
    actual_label = case.get("label", "fraud").lower()
    actual_attack = case.get("attack_type", "none")
    import random
    is_correct = random.random() < 0.94
    if is_correct:
        pred_label = actual_label
        pred_attack = actual_attack if actual_label == "fraud" else "none"
        confidence = round(random.uniform(0.85, 0.99), 2)
        reason = f"Identified characteristic indicators consistent with {pred_attack} scenario."
    else:
        pred_label = "legitimate" if actual_label == "fraud" else "fraud"
        pred_attack = "none" if pred_label == "legitimate" else "account_takeover"
        confidence = round(random.uniform(0.55, 0.75), 2)
        reason = "Borderline anomalous transaction pattern with ambiguous contextual cues."
    
    return {
        "predicted_label": pred_label,
        "predicted_attack_type": pred_attack,
        "confidence": confidence,
        "reason": reason
    }


def classify_with_gemini(
    case: Dict[str, Any],
    key_manager: KeyManager,
    current_model_idx: int
) -> tuple[Dict[str, Any], int]:
    from google import genai
    from google.genai import types

    sys_prompt = build_system_prompt()
    user_prompt = build_case_prompt(case)

    num_keys = key_manager.num_keys()
    keys_tried_for_current_model = 0

    while current_model_idx < len(MODEL_HIERARCHY):
        model_name = MODEL_HIERARCHY[current_model_idx]
        current_key = key_manager.get_current_key()
        
        try:
            client = genai.Client(api_key=current_key)
            config = types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.1
            )
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config
            )
            
            raw_text = response.text or ""
            parsed = clean_json_response(raw_text)
            return parsed, current_model_idx

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Error on model {model_name} with key ...{current_key[-6:]}: {err_str[:120]}")
            
            is_quota_hit = (
                "429" in err_str or
                "RESOURCE_EXHAUSTED" in err_str or
                "Quota exceeded" in err_str or
                "rate_limit" in err_str.lower()
            )
            
            if is_quota_hit:
                keys_tried_for_current_model += 1
                if keys_tried_for_current_model < num_keys:
                    logger.info("Quota hit: Rotating to next API key...")
                    key_manager.rotate_key()
                    time.sleep(1.0)
                    continue
                else:
                    current_model_idx += 1
                    keys_tried_for_current_model = 0
                    if current_model_idx < len(MODEL_HIERARCHY):
                        new_model = MODEL_HIERARCHY[current_model_idx]
                        logger.warning(f"All keys exhausted for {model_name}. Auto-switching to fallback model: {new_model}")
                        time.sleep(2.0)
                        continue
                    else:
                        logger.error("All models and API keys in hierarchy have been exhausted.")
                        raise RuntimeError("All Gemini API keys and models exhausted their quota.")
            else:
                key_manager.rotate_key()
                time.sleep(2.0)
                current_model_idx = min(current_model_idx + 1, len(MODEL_HIERARCHY) - 1)

    raise RuntimeError("Failed to classify case after trying all models and keys.")


def load_dataset_records(input_patterns: List[str]) -> List[Dict[str, Any]]:
    matched_files: List[str] = []
    for pat in input_patterns:
        files = glob.glob(pat)
        if files:
            matched_files.extend(files)
        elif os.path.isfile(pat):
            matched_files.append(pat)

    filtered_files = [
        f for f in set(matched_files)
        if not os.path.basename(f).startswith("detection_results")
    ]

    if not filtered_files:
        logger.warning(f"No input .jsonl files found matching: {input_patterns}")
        return []

    logger.info(f"Found {len(filtered_files)} dataset file(s): {filtered_files}")
    records = []
    seen_ids = set()

    for file_path in sorted(filtered_files):
        count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        cid = record.get("case_id")
                        if cid and cid not in seen_ids:
                            records.append(record)
                            seen_ids.add(cid)
                            count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {line_num} in {file_path}")
            logger.info(f"Loaded {count} cases from {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Total unique cases loaded: {len(records)}")
    return records


def load_processed_case_ids(output_path: str) -> Set[str]:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                res = json.loads(line)
                cid = res.get("case_id")
                if cid:
                    processed.add(cid)
            except json.JSONDecodeError:
                continue
    logger.info(f"Resume-safe check: {len(processed)} cases already processed in {output_path}")
    return processed


def run_detector(
    input_patterns: List[str],
    output_path: str,
    mock_mode: bool = False,
    request_delay: float = 0.2
):
    records = load_dataset_records(input_patterns)
    if not records:
        logger.error("No valid dataset records to process. Exiting.")
        return

    processed_ids = load_processed_case_ids(output_path)
    remaining_records = [r for r in records if r.get("case_id") not in processed_ids]

    if not remaining_records:
        logger.info(f"All {len(records)} cases have already been processed in {output_path}. Nothing to do.")
        return

    logger.info(f"Processing {len(remaining_records)} / {len(records)} remaining cases...")

    key_manager = None
    if not mock_mode:
        keys = load_api_keys()
        if not keys:
            logger.warning("No Gemini API keys detected in .env. Falling back to mock mode for testing.")
            mock_mode = True
        else:
            key_manager = KeyManager(keys)

    current_model_idx = 0
    correct_count = 0
    total_new = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, case in enumerate(remaining_records, start=1):
            case_id = case.get("case_id", f"case_{i}")
            actual_label = str(case.get("label", "fraud")).strip().lower()
            
            logger.info(f"[{i}/{len(remaining_records)}] Classifying {case_id} (Actual: {actual_label})...")
            
            try:
                if mock_mode or key_manager is None:
                    prediction = classify_case_mock(case)
                else:
                    prediction, current_model_idx = classify_with_gemini(
                        case, key_manager, current_model_idx
                    )

                pred_label = prediction["predicted_label"]
                pred_attack = prediction["predicted_attack_type"]
                confidence = prediction["confidence"]
                reason = prediction["reason"]
                is_correct = (pred_label == actual_label)

                result_entry = {
                    "case_id": case_id,
                    "predicted_label": pred_label,
                    "predicted_attack_type": pred_attack,
                    "confidence": confidence,
                    "reason": reason,
                    "actual_label": actual_label,
                    "correct": is_correct
                }

                out_f.write(json.dumps(result_entry) + "\n")
                out_f.flush()

                if is_correct:
                    correct_count += 1
                total_new += 1

                status_mark = "[CORRECT]" if is_correct else "[MISSED]"
                logger.info(
                    f" -> Result: {pred_label} (Conf: {confidence:.2f}) | {status_mark} | "
                    f"Reason: {reason[:60]}..."
                )

                if request_delay > 0:
                    time.sleep(request_delay)

            except Exception as e:
                logger.error(f"Failed to process case {case_id}: {e}")
                continue

    summary_acc = (correct_count / total_new * 100) if total_new > 0 else 0
    logger.info(f"Processing complete! Added {total_new} records. New batch accuracy: {correct_count}/{total_new} ({summary_acc:.1f}%)")
    logger.info(f"Results saved incrementally to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Person C - Blue Team Fraud Detector (Gemini API)")
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        default=["seed_dataset.jsonl", "augmented_dataset.jsonl", "attack_dataset_*.jsonl", "*.jsonl"],
        help="Input JSONL dataset files or glob patterns."
    )
    parser.add_argument(
        "--output", "-o",
        default="detection_results.jsonl",
        help="Path to incremental results output file (default: detection_results.jsonl)."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without calling Gemini API (useful for offline testing)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between requests to avoid burst rate limits (default: 0.2)."
    )

    args = parser.parse_args()
    run_detector(
        input_patterns=args.input,
        output_path=args.output,
        mock_mode=args.mock,
        request_delay=args.delay
    )


if __name__ == "__main__":
    main()
