import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


RECOMMENDATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bnot recommended\b", "NOT_RECOMMENDED"),
    (r"\bno recommendation(?: possible)?\b", "NO_RECOMMENDATION"),
    (r"\binsufficient evidence\b", "NO_RECOMMENDATION"),
    (r"\bmay be considered\b", "CONDITIONAL_CONSIDER"),
    (r"\bonly (?:within|in) (?:a )?clinical[- ]trial\b", "CONDITIONAL_CONSIDER"),
    (r"\brecommended\b", "RECOMMENDED"),
]

SCENARIO_PATTERNS: List[Tuple[str, str]] = [
    (r"initial diagnosis|primary diagnosis", "INITIAL_DIAGNOSIS"),
    (r"pre[- ]operative staging|\bstaging\b", "PREOP_STAGING"),
    (r"assessment of treatment response|treatment response|response", "TREATMENT_RESPONSE"),
    (r"recurrence|restaging", "RECURRENCE_OR_RESTAGING"),
    (r"solitary metastasis", "SOLITARY_METASTASIS_AT_RECURRENCE"),
]

CONDITION_HINT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bonly when\b(.+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"\bwhen\b(.+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"\bif\b(.+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"\bfor patients?\b(.+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"\bin patients?\b(.+?)(?:\.|;|$)", re.IGNORECASE),
]

ALLOWED_SCENARIOS = {
    "INITIAL_DIAGNOSIS",
    "PREOP_STAGING",
    "TREATMENT_RESPONSE",
    "RECURRENCE_OR_RESTAGING",
    "SOLITARY_METASTASIS_AT_RECURRENCE",
    "GENERAL",
    "GLOBAL",
}

ALLOWED_RECOMMENDATION_CLASSES = {
    "RECOMMENDED",
    "NOT_RECOMMENDED",
    "CONDITIONAL_CONSIDER",
    "NO_RECOMMENDATION",
    "APPEND_QUALIFIER",
    "APPEND_SAFETY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Hugging Face guideline dataset, convert first N records into "
            "activation-condition/execution rule specs, and save as extended JSON."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="epfl-llm/guidelines",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=50,
        help="Number of records to process from the beginning of the split.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guideline_policy/epfl_guideline_rule_specs.extended.jsonl",
        help="Output path. Use .jsonl for JSON Lines or .json for array JSON.",
    )
    parser.add_argument(
        "--text-fields",
        type=str,
        default="guideline,text,content,body,recommendation,recommendations",
        help="Comma-separated candidate text fields in priority order.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="llm",
        choices=["llm", "hybrid", "regex"],
        help="Rule extraction engine. 'llm' is primary. 'hybrid' uses regex fallback.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="LLM model used for rule extraction.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="",
        help="OpenAI API key. If empty, OPENAI_API_KEY env var is used.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default="",
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--max-llm-rules",
        type=int,
        default=30,
        help="Upper bound of rules requested from LLM per record.",
    )
    parser.add_argument(
        "--allow-regex-fallback",
        action="store_true",
        help="If LLM parsing fails, fallback to regex extraction for predictable patterns.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_sentence_chunks(text: str) -> List[str]:
    # Keep line and sentence boundaries to preserve recommendation statements.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: List[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        # Split long blocks into sentence-like chunks.
        for sent in re.split(r"(?<=[\.!?])\s+", block):
            sent = sent.strip(" -\t")
            if sent:
                chunks.append(sent)
    return chunks


def infer_scenario(text: str) -> str:
    lower = text.lower()
    for pattern, scenario in SCENARIO_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return scenario
    return "GENERAL"


def infer_recommendation_class(text: str) -> Optional[str]:
    lower = text.lower()
    for pattern, rec_class in RECOMMENDATION_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return rec_class
    return None


def extract_condition_hint(text: str) -> Optional[str]:
    for pattern in CONDITION_HINT_PATTERNS:
        match = pattern.search(text)
        if match:
            clause = match.group(1).strip(" ,;:")
            if clause:
                return clause
    return None


def derive_execute_action(text: str, rec_class: str) -> str:
    if rec_class == "NOT_RECOMMENDED":
        return f"Do not perform as routine practice. Basis: {text}"
    if rec_class == "RECOMMENDED":
        return f"Recommend use in this scenario. Basis: {text}"
    if rec_class == "CONDITIONAL_CONSIDER":
        return f"Consider only under stated conditions. Basis: {text}"
    if rec_class == "NO_RECOMMENDATION":
        return f"No recommendation due to limited or absent evidence. Basis: {text}"
    return f"Review clinically. Basis: {text}"


def priority_for_class(rec_class: str) -> int:
    if rec_class == "NOT_RECOMMENDED":
        return 80
    if rec_class == "RECOMMENDED":
        return 70
    if rec_class == "CONDITIONAL_CONSIDER":
        return 60
    if rec_class == "NO_RECOMMENDATION":
        return 50
    return 40


def build_llm_prompt(guideline_text: str, max_rules: int) -> List[Dict[str, str]]:
    system_prompt = (
        "You convert clinical guideline text into deterministic policy rules. "
        "Return only valid JSON with no markdown. "
        "Each rule must include: scenario, activation_condition, recommendation_class, execute, source_statement. "
        "Allowed recommendation_class: RECOMMENDED, NOT_RECOMMENDED, CONDITIONAL_CONSIDER, NO_RECOMMENDATION. "
        "Allowed scenario: INITIAL_DIAGNOSIS, PREOP_STAGING, TREATMENT_RESPONSE, RECURRENCE_OR_RESTAGING, "
        "SOLITARY_METASTASIS_AT_RECURRENCE, GENERAL. "
        "Use concise activation conditions in plain language. "
        "Do not invent evidence beyond the given text."
    )
    user_prompt = (
        f"Extract up to {max_rules} rules from this guideline text.\n"
        "Output JSON object with this schema:\n"
        "{\n"
        "  \"rules\": [\n"
        "    {\n"
        "      \"scenario\": \"...\",\n"
        "      \"activation_condition\": \"...\",\n"
        "      \"recommendation_class\": \"...\",\n"
        "      \"execute\": \"...\",\n"
        "      \"source_statement\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guideline text:\n"
        f"{guideline_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Predictable fallback: if model wraps JSON in code fences or extra text.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        snippet = fence_match.group(1)
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        snippet = brace_match.group(1)
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def call_llm_extract_rules(
    client: Any,
    model: str,
    guideline_text: str,
    max_rules: int,
) -> Dict[str, Any]:
    messages = build_llm_prompt(guideline_text, max_rules=max_rules)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    content = response.choices[0].message.content
    if content is None:
        return {}

    if not isinstance(content, str):
        content = str(content)

    parsed = extract_json_object(content)
    return parsed if parsed is not None else {}


def normalize_rule(rule: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    scenario = str(rule.get("scenario", "GENERAL")).strip().upper()
    if scenario not in ALLOWED_SCENARIOS:
        scenario = infer_scenario(str(rule.get("source_statement", "")))
        if scenario not in ALLOWED_SCENARIOS:
            scenario = "GENERAL"

    rec_class = str(rule.get("recommendation_class", "")).strip().upper()
    if rec_class not in ALLOWED_RECOMMENDATION_CLASSES:
        rec_class = infer_recommendation_class(str(rule.get("source_statement", "")))
        if rec_class is None:
            return None

    activation_condition = str(rule.get("activation_condition", "")).strip()
    if not activation_condition:
        activation_condition = f"scenario={scenario}"

    execute = str(rule.get("execute", "")).strip()
    source_statement = str(rule.get("source_statement", "")).strip()
    if not source_statement:
        source_statement = "llm_generated_rule"
    if not execute:
        execute = derive_execute_action(source_statement, rec_class)

    return {
        "rule_id": f"R{idx:03d}",
        "scenario": scenario,
        "priority": priority_for_class(rec_class),
        "activation_condition": activation_condition,
        "recommendation_class": rec_class,
        "execute": execute,
        "source_statement": source_statement,
    }


def validate_llm_rules(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        return []

    validated: List[Dict[str, Any]] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        normalized = normalize_rule(raw, idx=len(validated) + 1)
        if normalized is not None:
            validated.append(normalized)
    return validated


def extract_text_from_record(record: Dict[str, Any], preferred_fields: List[str]) -> str:
    for field in preferred_fields:
        if field in record and record[field] is not None:
            value = record[field]
            if isinstance(value, str) and value.strip():
                return normalize_text(value)
            if isinstance(value, list):
                lines = [str(v).strip() for v in value if str(v).strip()]
                if lines:
                    return normalize_text("\n".join(lines))

    # Fallback: concatenate textual values from the record.
    text_values: List[str] = []
    for _, value in record.items():
        if isinstance(value, str) and value.strip():
            text_values.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    text_values.append(item.strip())

    return normalize_text("\n".join(text_values))


def regex_rule_extraction(text: str) -> List[Dict[str, Any]]:
    chunks = to_sentence_chunks(text)
    rules: List[Dict[str, Any]] = []

    for chunk in chunks:
        rec_class = infer_recommendation_class(chunk)
        if rec_class is None:
            continue

        scenario = infer_scenario(chunk)
        condition_hint = extract_condition_hint(chunk)
        activation_condition = (
            f"scenario={scenario} AND {condition_hint}"
            if condition_hint
            else f"scenario={scenario}"
        )

        rules.append(
            {
                "rule_id": f"R{len(rules) + 1:03d}",
                "scenario": scenario,
                "priority": priority_for_class(rec_class),
                "activation_condition": activation_condition,
                "recommendation_class": rec_class,
                "execute": derive_execute_action(chunk, rec_class),
                "source_statement": chunk,
            }
        )

    return rules


def append_global_wrapper_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rules = list(rules)

    # Add global wrappers so every output includes disclosure and safety behavior.
    out_rules.append(
        {
            "rule_id": f"R{len(out_rules) + 1:03d}",
            "scenario": "GLOBAL",
            "priority": 20,
            "activation_condition": "always=true",
            "recommendation_class": "APPEND_QUALIFIER",
            "execute": (
                "Append evidence-quality qualifier: recommendations may be limited by "
                "heterogeneity, small sample sizes, and limited randomized evidence."
            ),
            "source_statement": "system_generated_global_evidence_note",
        }
    )
    out_rules.append(
        {
            "rule_id": f"R{len(out_rules) + 1:03d}",
            "scenario": "GLOBAL",
            "priority": 10,
            "activation_condition": "always=true",
            "recommendation_class": "APPEND_SAFETY",
            "execute": (
                "Append safety notice: this is decision support and does not replace "
                "specialist clinical judgment."
            ),
            "source_statement": "system_generated_global_safety_note",
        }
    )

    # Re-label sequentially in final sorted order for stable downstream processing.
    out_rules = sorted(out_rules, key=lambda r: (-r["priority"], r["rule_id"]))
    for i, rule in enumerate(out_rules, start=1):
        rule["rule_id"] = f"R{i:03d}"
    return out_rules


def build_rule_spec_from_text(
    text: str,
    engine: str,
    llm_client: Optional[Any],
    llm_model: str,
    max_llm_rules: int,
    allow_regex_fallback: bool,
) -> Dict[str, Any]:
    extraction_method = "regex"
    extraction_errors: List[str] = []

    rules: List[Dict[str, Any]] = []
    if engine in {"llm", "hybrid"}:
        if llm_client is None:
            extraction_errors.append("LLM client unavailable")
        else:
            try:
                llm_payload = call_llm_extract_rules(
                    client=llm_client,
                    model=llm_model,
                    guideline_text=text,
                    max_rules=max_llm_rules,
                )
                rules = validate_llm_rules(llm_payload)
                if rules:
                    extraction_method = "llm"
                else:
                    extraction_errors.append("LLM returned no valid rules")
            except Exception as exc:  # Defensive, keeps batch processing robust.
                extraction_errors.append(f"LLM extraction failed: {exc}")

    if not rules and (engine == "regex" or allow_regex_fallback or engine == "hybrid"):
        rules = regex_rule_extraction(text)
        extraction_method = "regex_fallback" if engine != "regex" else "regex"

    final_rules = append_global_wrapper_rules(rules)

    return {
        "rule_count": len(final_rules),
        "rules": final_rules,
        "engine_metadata": {
            "evaluation_mode": "priority_desc",
            "tie_breaker": "stable_order",
            "default_recommendation": "NO_RECOMMENDATION",
            "always_append_rules": ["APPEND_QUALIFIER", "APPEND_SAFETY"],
            "extraction_method": extraction_method,
            "extraction_errors": extraction_errors,
        },
    }


def ensure_output_parent(path: str) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_extended_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_extended_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from exc

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'tqdm'. Install it with: pip install tqdm"
        ) from exc

    preferred_fields = [f.strip() for f in args.text_fields.split(",") if f.strip()]

    llm_client: Optional[Any] = None
    if args.engine in {"llm", "hybrid"}:
        api_key = args.openai_api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise SystemExit(
                    "Missing dependency 'openai'. Install it with: pip install openai"
                ) from exc

            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if args.openai_base_url.strip():
                client_kwargs["base_url"] = args.openai_base_url.strip()
            llm_client = OpenAI(**client_kwargs)
        elif args.engine == "llm":
            raise SystemExit(
                "LLM engine requested but no API key found. Provide --openai-api-key or set OPENAI_API_KEY."
            )

    dataset = load_dataset(args.dataset, split=args.split)
    n = min(args.first_n, len(dataset))

    enriched_rows: List[Dict[str, Any]] = []
    method_counts: Dict[str, int] = {"llm": 0, "regex": 0, "regex_fallback": 0}
    for i in tqdm(range(n), total=n, desc="Converting records"):
        record = dict(dataset[i])
        guideline_text = extract_text_from_record(record, preferred_fields)
        rule_spec = build_rule_spec_from_text(
            text=guideline_text,
            engine=args.engine,
            llm_client=llm_client,
            llm_model=args.llm_model,
            max_llm_rules=safe_int(args.max_llm_rules, 30),
            allow_regex_fallback=args.allow_regex_fallback,
        )
        method = rule_spec.get("engine_metadata", {}).get("extraction_method", "regex")
        if method in method_counts:
            method_counts[method] += 1

        enriched_rows.append(
            {
                "record_index": i,
                "dataset": args.dataset,
                "split": args.split,
                "source_record": record,
                "extracted_guideline_text": guideline_text,
                "rule_spec": rule_spec,
            }
        )

    ensure_output_parent(args.output)
    if args.output.lower().endswith(".json"):
        write_extended_json(args.output, enriched_rows)
    else:
        write_extended_jsonl(args.output, enriched_rows)

    print(f"Processed {len(enriched_rows)} records from {args.dataset}:{args.split}")
    print(f"Saved extended rule specs to: {args.output}")
    print(
        "Extraction method counts: "
        + ", ".join(f"{k}={v}" for k, v in method_counts.items())
    )


if __name__ == "__main__":
    main()
