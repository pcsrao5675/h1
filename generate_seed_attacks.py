"""
generate_seed_attacks.py
=========================
Person A — Attack Generator (Red Team, seed attacks)
Mastercard Innovation Challenge 2026 — AI Defense Lab for Payment Security

Generates a seed set of fraud + legitimate "cases" across 6 GenAI-era fraud
categories, using Google's Gemini API (free tier). Output is JSONL matching
the team's shared schema so it plugs directly into Person B's scale-up
pipeline and Person C's detector.

SETUP
-----
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Set it as an environment variable before running:
       export GEMINI_API_KEY="your-key-here"
   (On Windows PowerShell:  $env:GEMINI_API_KEY="your-key-here")
3. pip install google-generativeai
4. python generate_seed_attacks.py

OUTPUT
------
- seed_dataset.jsonl   -> the full seed dataset (fraud + legitimate cases)
- README.md            -> auto-generated explanation of each attack category
"""

import os
import json
import time
import random
import re
from datetime import datetime, timedelta

import google.genai as genai
from google.genai import types

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

def _load_env_file():
    for env_path in [".env", "../.env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

# Candidate models to try in order if MODEL_NAME fails with 404/not available
CANDIDATE_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
MODEL_NAME = "gemini-3.7-flash"  # Default active model
FAILED_MODELS = set()

CASES_PER_ATTACK_TYPE = 9        # ~9 x 6 categories = ~54 fraud cases
LEGIT_CASES = 25

OUTPUT_FILE = "seed_dataset.jsonl"
README_FILE = "README.md"

# --- MULTI-KEY ROTATION -----------------------------------------------
# Collect all configured GEMINI_API_KEY_1..20 and GEMINI_API_KEY
_keys_found = []
for i in range(1, 21):
    k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
    if k and k not in _keys_found:
        _keys_found.append(k)

single_key = os.environ.get("GEMINI_API_KEY", "").strip()
if single_key and single_key not in _keys_found:
    _keys_found.append(single_key)

API_KEYS = _keys_found
_current_key_index = 0

STOP_ON_QUOTA_EXHAUSTED = True   # only relevant once ALL keys are exhausted

if not API_KEYS:
    raise SystemExit(
        "ERROR: Set at least one API key first.\n"
        "Either GEMINI_API_KEY=\"...\" for a single key, or\n"
        "GEMINI_API_KEY_1=\"...\", GEMINI_API_KEY_2=\"...\" etc. for rotation.\n"
        "Alternatively, put them in a .env file in this directory.\n"
        "Get free keys at https://aistudio.google.com/apikey (one per Google account)"
    )

print(f"Loaded {len(API_KEYS)} API key(s) for rotation.", flush=True)
_client = genai.Client(api_key=API_KEYS[_current_key_index])


def switch_to_next_key():
    """Rotate to the next available API key. Returns False if we've
    exhausted every key we have (all daily quotas used up)."""
    global _current_key_index, _client
    _current_key_index += 1
    if _current_key_index >= len(API_KEYS):
        return False
    print(f"\n  >> Switching to API key #{_current_key_index + 1} of {len(API_KEYS)}...\n", flush=True)
    _client = genai.Client(api_key=API_KEYS[_current_key_index])
    return True


# ---------------------------------------------------------------------------
# 1. SWAPPABLE LLM CALL — keep this function signature identical across the
#    team (Person B reuses it) so everyone's generation code is consistent.
# ---------------------------------------------------------------------------

class DailyQuotaExhausted(Exception):
    """Raised only when ALL available API keys have hit their daily quota."""
    pass


def try_switch_fallback_model(err_str: str) -> bool:
    """Attempt to switch to an alternative working model if the current one is retired/404."""
    global MODEL_NAME
    FAILED_MODELS.add(MODEL_NAME)
    print(f"\n[!] Model '{MODEL_NAME}' returned error: {err_str[:120]}...", flush=True)
    
    match = re.search(r"models/(gemini-[0-9a-zA-Z\.\-]+)", err_str)
    if match:
        suggested = match.group(1)
        if suggested not in FAILED_MODELS and suggested not in CANDIDATE_MODELS:
            CANDIDATE_MODELS.insert(0, suggested)
        
    for alt in CANDIDATE_MODELS:
        if alt not in FAILED_MODELS:
            print(f"  -> Switching MODEL_NAME to '{alt}' and retrying...", flush=True)
            MODEL_NAME = alt
            return True
    return False


def handle_daily_quota_exhausted():
    """When a key hits daily quota for current model, try the next key.
    If all keys are exhausted on current model, switch to the next candidate model and reset keys!"""
    global _current_key_index, _client, MODEL_NAME
    if switch_to_next_key():
        return True
    # All keys exhausted for current model
    print(f"\n[!] All {len(API_KEYS)} API key(s) exhausted daily quota for model '{MODEL_NAME}'. Checking next model...", flush=True)
    if try_switch_fallback_model("Daily quota exhausted for model"):
        _current_key_index = 0
        _client = genai.Client(api_key=API_KEYS[_current_key_index])
        print(f"  -> Resetting to Key #1 with new model '{MODEL_NAME}'.\n", flush=True)
        return True
    return False


def call_llm(prompt: str, retries: int = 6, temperature: float = 1.0) -> str:
    """Single entry point for all LLM calls. Returns raw text response.
    Automatically rotates to the next API key/model if the current one's daily
    quota is exhausted, and only gives up once every key and model is exhausted."""
    global MODEL_NAME
    for attempt in range(retries):
        try:
            response = _client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                ),
            )
            if response.text:
                return response.text
            raise ValueError("Empty response text from Gemini API")
        except Exception as e:
            err_str = str(e)
            
            # Check for model not available / 404
            if "is no longer available" in err_str or "404" in err_str or "not found" in err_str.lower():
                if try_switch_fallback_model(err_str):
                    continue
                else:
                    raise SystemExit(
                        f"\nERROR: Model '{MODEL_NAME}' is no longer available.\n"
                        f"Full error: {err_str}\n"
                    )

            # Check for 503 unavailable spikes
            if "503" in err_str or "unavailable" in err_str.lower():
                if attempt >= 2 and try_switch_fallback_model(err_str):
                    continue

            # Check for invalid API key
            if "API_KEY_INVALID" in err_str or "api key not valid" in err_str.lower():
                print(f"  [!] Key #{_current_key_index + 1} appears invalid.", flush=True)
                if switch_to_next_key():
                    continue
                else:
                    raise RuntimeError("All configured API keys are invalid.")

            # Check for daily quota exhaustion
            if (
                "perday" in err_str.lower()
                or "per day" in err_str.lower()
                or "daily" in err_str.lower()
                or "generaterequestsperday" in err_str.lower()
                or "free_tier_requests" in err_str.lower()
            ):
                print(f"  [!] Key #{_current_key_index + 1} daily quota hit on '{MODEL_NAME}'.", flush=True)
                if handle_daily_quota_exhausted():
                    continue  # retry with next key/model
                else:
                    raise DailyQuotaExhausted(err_str)

            # Short per-minute rate limit (429 / RESOURCE_EXHAUSTED)
            if "429" in err_str or "resource_exhausted" in err_str.lower() or "quota" in err_str.lower():
                wait = min(40, (attempt + 1) * 6)
                print(f"  [call_llm] rate limit encountered (attempt {attempt+1}/{retries}). Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue

            wait = 2 ** attempt
            print(f"  [call_llm] attempt {attempt+1} failed: {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("call_llm failed after all retries")


def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM responses."""
    if not text:
        raise ValueError("Empty text received for JSON extraction")
    
    cleaned = text.strip()
    # Try direct parse
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Extract outermost JSON object { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str, strict=False)
        except Exception:
            # Strip trailing commas
            cleaned_json = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                return json.loads(cleaned_json, strict=False)
            except Exception:
                pass

    # Field-level regex fallback if JSON structure is slightly malformed
    amt_match = re.search(r'"amount"\s*:\s*([\d\.]+)', text)
    narr_match = re.search(r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    fp_match = re.search(r'"device_fingerprint"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|null)', text)
    if amt_match and narr_match:
        try:
            narr = narr_match.group(1).encode().decode('unicode_escape', 'ignore')
        except Exception:
            narr = narr_match.group(1)
        return {
            "amount": float(amt_match.group(1)),
            "narrative": narr,
            "device_fingerprint": fp_match.group(1) if fp_match and fp_match.group(1) else None,
        }

    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ---------------------------------------------------------------------------
# 2. ATTACK CATEGORY DEFINITIONS
# ---------------------------------------------------------------------------

ATTACK_CATEGORIES = {
    "account_takeover": {
        "description": "Credential theft or session hijacking used to seize "
                        "control of a legitimate cardholder's account.",
        "channels": ["app", "ecom", "call-center"],
    },
    "synthetic_identity": {
        "description": "A fabricated identity (real + fake data blended) used "
                        "to open or operate a card account.",
        "channels": ["ecom", "app"],
    },
    "refund_fraud": {
        "description": "Fake or manipulated dispute/chargeback claims to "
                        "extract refunds without legitimate cause.",
        "channels": ["ecom", "call-center"],
    },
    "card_testing": {
        "description": "Small, rapid transactions used to validate stolen "
                        "card numbers before a larger fraudulent purchase.",
        "channels": ["ecom", "card-present"],
    },
    "prompt_injection_scam": {
        "description": "GenAI-era social engineering: attacker manipulates a "
                        "bank chatbot, voice assistant, or customer via "
                        "AI-generated scripts/prompt injection to authorize "
                        "fraudulent actions. This is the hackathon's core "
                        "'novel emerging GenAI fraud' focus.",
        "channels": ["app", "call-center", "ecom"],
    },
    "deepfake_kyc": {
        "description": "Deepfake audio/video or AI-generated documents used "
                        "to bypass identity verification (KYC) checks.",
        "channels": ["app", "ecom"],
    },
}

MERCHANT_CATEGORIES = [
    "electronics", "travel", "groceries", "digital_goods", "fashion",
    "food_delivery", "utilities", "gaming", "jewelry", "subscription_services",
]

COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "BR", "DE"]
CURRENCIES = {"IN": "INR", "US": "USD", "GB": "GBP", "AE": "AED",
              "SG": "SGD", "NG": "NGN", "BR": "BRL", "DE": "EUR"}


# ---------------------------------------------------------------------------
# 3. PROMPT TEMPLATES
# ---------------------------------------------------------------------------

def build_fraud_prompt(attack_type: str, info: dict, case_num: int) -> str:
    channel = random.choice(info["channels"])
    country = random.choice(COUNTRIES)
    merchant = random.choice(MERCHANT_CATEGORIES)

    return f"""You are generating realistic synthetic training data for a payment
fraud detection system. Generate ONE fictional fraud case for the category
"{attack_type}" ({info['description']}).

Requirements:
- Write a 200-500 word "narrative" field: a realistic log-style artifact.
  Depending on the channel ("{channel}"), make it a chat transcript, a call
  center transcript, an email exchange, or a first-person incident
  description. Make it feel like a real captured log, not a summary — use
  natural dialogue, timestamps within the text if relevant, and realistic
  (but fictional) names/details.
- Merchant category: "{merchant}". Country: "{country}".
- This is fictional/synthetic data for defensive AI research only. Do not
  include any real personal data, real account numbers, or real company
  confidential info — everything must be clearly fictional.
- Vary the specific tactics used so this case #{case_num} feels distinct
  from other cases in the same category.

Return ONLY a single JSON object (no markdown fences, no explanation) with
exactly these fields:
{{
  "amount": <float, realistic for the merchant category and country>,
  "narrative": "<the 200-500 word narrative>",
  "device_fingerprint": "<a short fake device id string, or null>"
}}
"""


def build_legit_prompt(case_num: int) -> str:
    channel = random.choice(["app", "ecom", "call-center", "card-present"])
    country = random.choice(COUNTRIES)
    merchant = random.choice(MERCHANT_CATEGORIES)

    return f"""You are generating realistic synthetic training data for a payment
fraud detection system. Generate ONE fictional LEGITIMATE (non-fraud)
customer interaction — completely normal banking/payment activity.

Requirements:
- Write a 150-350 word "narrative" field: a realistic log-style artifact
  (chat transcript, call center transcript, email, or short interaction
  description) for channel "{channel}". Examples of normal activity: a
  routine purchase, updating contact info, asking about a transaction,
  requesting a card replacement for a lost card, a normal subscription
  renewal, etc. Nothing suspicious should be happening.
- Merchant category: "{merchant}". Country: "{country}".
- All details fictional. Case #{case_num} should feel distinct from others.

Return ONLY a single JSON object (no markdown fences, no explanation) with
exactly these fields:
{{
  "amount": <float, realistic for the merchant category and country>,
  "narrative": "<the 150-350 word narrative>",
  "device_fingerprint": "<a short fake device id string, or null>"
}}
"""


# ---------------------------------------------------------------------------
# 4. GENERATION LOOP
# ---------------------------------------------------------------------------

def random_timestamp():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 8, 1)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=random_seconds)).isoformat() + "Z"


def parse_amount(val, country="US") -> float:
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.]", "", val)
        if cleaned:
            try:
                return round(float(cleaned), 2)
            except ValueError:
                pass
    return round(random.uniform(25.0, 750.0), 2)


def make_case(case_id, attack_type, generation_method, label,
              channel, country, merchant, llm_fields):
    return {
        "case_id": case_id,
        "transaction": {
            "amount": parse_amount(llm_fields.get("amount"), country),
            "currency": CURRENCIES.get(country, "USD"),
            "merchant_category": merchant,
            "timestamp": random_timestamp(),
            "channel": channel,
            "cardholder_country": country,
            "device_fingerprint": llm_fields.get("device_fingerprint"),
        },
        "narrative": llm_fields.get("narrative", ""),
        "attack_type": attack_type,
        "generation_method": generation_method,
        "label": label,
        "source_round": 1,
    }


def load_existing():
    """Resume support: read whatever's already in OUTPUT_FILE so we don't
    regenerate (and burn quota on) cases we already have."""
    if not os.path.exists(OUTPUT_FILE):
        return [], 0, {}
    cases = []
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    max_num = 0
    for c in cases:
        n = int(c["case_id"].split("_")[1])
        max_num = max(max_num, n)
    per_type_counts = {}
    for c in cases:
        if c["label"] == "fraud":
            per_type_counts[c["attack_type"]] = per_type_counts.get(c["attack_type"], 0) + 1
    return cases, max_num, per_type_counts


def append_case(case):
    """Write a single case immediately so progress survives a crash/quota cutoff."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(case) + "\n")


def generate_fraud_cases(existing_counts, counter_start):
    counter = counter_start
    quota_hit = False
    new_cases = []
    for attack_type, info in ATTACK_CATEGORIES.items():
        already = existing_counts.get(attack_type, 0)
        remaining = CASES_PER_ATTACK_TYPE - already
        if remaining <= 0:
            print(f"\n'{attack_type}' already has {already} cases, skipping.")
            continue
        print(f"\nGenerating '{attack_type}' cases ({already} already done, {remaining} to go)...")
        for i in range(1, remaining + 1):
            channel = random.choice(info["channels"])
            country = random.choice(COUNTRIES)
            merchant = random.choice(MERCHANT_CATEGORIES)
            prompt = build_fraud_prompt(attack_type, info, already + i)

            try:
                raw = call_llm(prompt, temperature=1.1)
                fields = extract_json(raw)
            except DailyQuotaExhausted:
                print("\n  [!] Daily free-tier quota exhausted. Stopping cleanly — "
                      "progress so far is already saved. Re-run this script "
                      "tomorrow (or with a new key) to continue.")
                quota_hit = True
                break
            except Exception as e:
                print(f"  [!] Skipping case {i} for {attack_type}: {e}")
                continue

            case_id = f"seed_{counter:03d}"
            case = make_case(case_id, attack_type, "seed", "fraud",
                              channel, country, merchant, fields)
            append_case(case)
            new_cases.append(case)
            counter += 1
            print(f"  -> {case_id} ({channel}, {country}, {merchant}) [saved]")
            time.sleep(1.5)
        if quota_hit:
            break
    return new_cases, counter, quota_hit


def generate_legit_cases(counter_start, already_have):
    counter = counter_start
    remaining = LEGIT_CASES - already_have
    new_cases = []
    quota_hit = False
    if remaining <= 0:
        print(f"\nAlready have {already_have} legitimate cases, skipping.")
        return new_cases, counter, False
    print(f"\nGenerating {remaining} more legitimate cases "
          f"({already_have} already done)...")
    for i in range(1, remaining + 1):
        channel = random.choice(["app", "ecom", "call-center", "card-present"])
        country = random.choice(COUNTRIES)
        merchant = random.choice(MERCHANT_CATEGORIES)
        prompt = build_legit_prompt(already_have + i)

        try:
            raw = call_llm(prompt, temperature=0.9)
            fields = extract_json(raw)
        except DailyQuotaExhausted:
            print("\n  [!] Daily free-tier quota exhausted. Stopping cleanly — "
                  "progress so far is already saved. Re-run this script "
                  "tomorrow (or with a new key) to continue.")
            quota_hit = True
            break
        except Exception as e:
            print(f"  [!] Skipping legit case {i}: {e}")
            continue

        case_id = f"seed_{counter:03d}"
        case = make_case(case_id, "none", "seed", "legitimate",
                          channel, country, merchant, fields)
        append_case(case)
        new_cases.append(case)
        counter += 1
        print(f"  -> {case_id} ({channel}, {country}, {merchant}) [saved]")
        time.sleep(1.5)
    return new_cases, counter, quota_hit


# ---------------------------------------------------------------------------
# 5. README GENERATION
# ---------------------------------------------------------------------------

def write_readme(fraud_count, legit_count):
    lines = [
        "# Seed Dataset — Attack Category Reference\n",
        "Generated by `generate_seed_attacks.py` using the Gemini API "
        f"(model: `{MODEL_NAME}`).\n",
        f"**Total cases:** {fraud_count} fraud + {legit_count} legitimate = "
        f"{fraud_count + legit_count}\n",
        "## Attack categories\n",
    ]
    for attack_type, info in ATTACK_CATEGORIES.items():
        lines.append(f"### {attack_type}")
        lines.append(info["description"])
        lines.append(
            "**Why it matters for this challenge:** " +
            ("This is the hackathon's core focus — GenAI tools let attackers "
             "generate convincing, personalized social-engineering scripts "
             "at a scale and quality that wasn't possible before, and can "
             "even attempt to manipulate AI-powered bank chatbots directly "
             "via prompt injection."
             if attack_type == "prompt_injection_scam" else
             "GenAI amplifies this by making the fabricated narratives, "
             "documents, or identity signals far more convincing and "
             "cheaper to produce at scale, compared to manual fraud."
             )
        )
        lines.append("")
    with open(README_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {README_FILE}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    existing_cases, max_num, fraud_counts_by_type = load_existing()
    existing_legit_count = sum(1 for c in existing_cases if c["label"] == "legitimate")
    existing_fraud_count = sum(1 for c in existing_cases if c["label"] == "fraud")

    if existing_cases:
        print(f"Resuming: found {len(existing_cases)} existing cases in "
              f"{OUTPUT_FILE} ({existing_fraud_count} fraud, "
              f"{existing_legit_count} legit). Continuing from there.\n")

    counter = max_num + 1

    new_fraud, counter, quota_hit_1 = generate_fraud_cases(fraud_counts_by_type, counter)

    quota_hit_2 = False
    new_legit = []
    if not quota_hit_1:
        new_legit, counter, quota_hit_2 = generate_legit_cases(counter, existing_legit_count)

    total_fraud = existing_fraud_count + len(new_fraud)
    total_legit = existing_legit_count + len(new_legit)
    total = total_fraud + total_legit

    write_readme(total_fraud, total_legit)

    print(f"\n{'='*60}")
    if quota_hit_1 or quota_hit_2:
        target = sum(CASES_PER_ATTACK_TYPE for _ in ATTACK_CATEGORIES) + LEGIT_CASES
        print(f"PAUSED (all {len(API_KEYS)} API key(s) hit their daily quota).")
        print(f"Progress saved to {OUTPUT_FILE}.")
        print(f"  So far: {total} / {target} target cases "
              f"({total_fraud} fraud, {total_legit} legit)")
        print(f"  Add more keys as GEMINI_API_KEY_{len(API_KEYS)+1}, or just "
              f"re-run this script again in ~24h once quotas reset — "
              f"it will pick up exactly where it left off.")
    else:
        print(f"DONE. Wrote {total} total cases to {OUTPUT_FILE}")
        print(f"  Fraud: {total_fraud}  |  Legitimate: {total_legit}")


if __name__ == "__main__":
    main()
