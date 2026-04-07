import argparse
import json
import os
from typing import Any, Dict, List, Optional


NON_ACTIONABLE_CLASSES = {"APPEND_QUALIFIER", "APPEND_SAFETY"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate over guideline records from an extended rule-spec JSONL file "
            "and generate a plausible patient vignette plus recommended actions for each record."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="guideline_policy/epfl_guideline_rule_specs.extended.jsonl",
        help="Path to the extended JSONL guideline file.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First record_index to process, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last record_index to process, inclusive. Defaults to the last record in the file.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="LLM model used to generate the vignette.",
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
        "--max-rules",
        type=int,
        default=8,
        help="Maximum number of actionable rules to include in the prompt context.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guideline_policy/epfl_guideline_vignettes.jsonl",
        help="Path to save generated results. Use .jsonl for one record per line or .json for an array.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_actionable_rules(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = record.get("rule_spec", {}).get("rules", [])
    return [
        rule
        for rule in rules
        if isinstance(rule, dict)
        and rule.get("recommendation_class") not in NON_ACTIONABLE_CLASSES
    ]


def select_records(
    rows: List[Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
) -> List[Dict[str, Any]]:
    if end_index is not None and end_index < start_index:
        raise ValueError("end-index must be greater than or equal to start-index")

    selected: List[Dict[str, Any]] = []
    for row in rows:
        record_index = row.get("record_index")
        if not isinstance(record_index, int):
            continue
        if record_index < start_index:
            continue
        if end_index is not None and record_index > end_index:
            continue
        selected.append(row)
    return selected


def summarize_guideline(record: Dict[str, Any]) -> Dict[str, Any]:
    source_record = record.get("source_record", {})
    rule_spec = record.get("rule_spec", {})
    actionable_rules = get_actionable_rules(record)
    return {
        "record_index": record.get("record_index"),
        "source": source_record.get("source", "unknown"),
        "id": source_record.get("id", "unknown"),
        "title": source_record.get("title") or source_record.get("overview") or "Untitled guideline",
        "extraction_method": rule_spec.get("engine_metadata", {}).get("extraction_method", "unknown"),
        "actionable_rules": actionable_rules,
    }


def build_rule_context(actionable_rules: List[Dict[str, Any]], max_rules: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        actionable_rules,
        key=lambda rule: (-int(rule.get("priority", 0)), str(rule.get("rule_id", ""))),
    )
    return ordered[:max_rules]


def build_messages(guideline: Dict[str, Any], selected_rules: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_prompt = (
        "You generate realistic but fictional patient vignettes from structured clinical rules. "
        "Use only the supplied rules. Do not invent recommendations that contradict them. "
        "Return only valid JSON with no markdown."
    )

    user_payload = {
        "task": (
            "Generate one plausible patient vignette that matches one or more of the rules below, "
            "then list the recommended actions that follow from the triggered rules."
        ),
        "requirements": [
            "The vignette must be clinically plausible and internally consistent.",
            "Trigger at least one rule and at most three rules from the provided list.",
            "Do not use GLOBAL wrapper rules as triggered rules.",
            "Keep the vignette concise: 120-220 words.",
            "Recommended actions must be directly grounded in the triggered rules.",
            "Mention uncertainty or evidence limits only if supported by the selected rules.",
        ],
        "output_schema": {
            "patient_vignette": "string",
            "triggered_rule_ids": ["R001"],
            "recommended_actions": ["string"],
            "rationale": "string",
        },
        "guideline_metadata": {
            "record_index": guideline["record_index"],
            "source": guideline["source"],
            "title": guideline["title"],
            "extraction_method": guideline["extraction_method"],
        },
        "rules": selected_rules,
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def call_llm(client: Any, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if content is None:
        return {}
    if not isinstance(content, str):
        content = str(content)
    parsed = extract_json_object(content)
    return parsed if parsed is not None else {}


def validate_generation(result: Dict[str, Any], selected_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rule_ids = {rule.get("rule_id") for rule in selected_rules}

    triggered_rule_ids = result.get("triggered_rule_ids", [])
    if not isinstance(triggered_rule_ids, list):
        triggered_rule_ids = []
    filtered_rule_ids = [rule_id for rule_id in triggered_rule_ids if rule_id in valid_rule_ids]

    recommended_actions = result.get("recommended_actions", [])
    if not isinstance(recommended_actions, list):
        recommended_actions = []

    return {
        "patient_vignette": str(result.get("patient_vignette", "")).strip(),
        "triggered_rule_ids": filtered_rule_ids,
        "recommended_actions": [str(action).strip() for action in recommended_actions if str(action).strip()],
        "rationale": str(result.get("rationale", "")).strip(),
    }


def ensure_output_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json_output(path: str, payload: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl_output(path: str, payload: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_with_progress(records: List[Dict[str, Any]]):
    try:
        from tqdm import tqdm
    except ImportError:
        return records

    return tqdm(records, desc="Generating vignettes", unit="record")


def main() -> None:
    args = parse_args()

    api_key = args.openai_api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "No OpenAI API key found. Provide --openai-api-key or set OPENAI_API_KEY."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency 'openai'. Install it with: pip install openai") from exc

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if args.openai_base_url.strip():
        client_kwargs["base_url"] = args.openai_base_url.strip()
    client = OpenAI(**client_kwargs)

    rows = load_jsonl(args.input_jsonl)
    records_to_process = select_records(rows, args.start_index, args.end_index)
    if not records_to_process:
        raise SystemExit("No records found in the requested index range.")

    output_payload: List[Dict[str, Any]] = []
    for record in iter_with_progress(records_to_process):
        guideline = summarize_guideline(record)
        selected_rules = build_rule_context(guideline["actionable_rules"], max_rules=args.max_rules)
        if not selected_rules:
            output_payload.append(
                {
                    "guideline": {
                        "record_index": guideline["record_index"],
                        "id": guideline["id"],
                        "source": guideline["source"],
                        "title": guideline["title"],
                        "extraction_method": guideline["extraction_method"],
                    },
                    "selected_rules": [],
                    "generated_case": {
                        "patient_vignette": "",
                        "triggered_rule_ids": [],
                        "recommended_actions": [],
                        "rationale": "No actionable rules available for this record.",
                    },
                    "status": "skipped_no_actionable_rules",
                }
            )
            continue

        messages = build_messages(guideline, selected_rules)
        raw_result = call_llm(
            client=client,
            messages=messages,
            model=args.llm_model,
        )
        generation = validate_generation(raw_result, selected_rules)

        output_payload.append(
            {
                "guideline": {
                    "record_index": guideline["record_index"],
                    "id": guideline["id"],
                    "source": guideline["source"],
                    "title": guideline["title"],
                    "extraction_method": guideline["extraction_method"],
                },
                "selected_rules": selected_rules,
                "generated_case": generation,
                "status": "generated",
            }
        )

    ensure_output_parent(args.output)
    if args.output.lower().endswith(".json"):
        write_json_output(args.output, output_payload)
    else:
        write_jsonl_output(args.output, output_payload)

    print(f"Processed {len(output_payload)} guideline records")
    print(f"Saved vignette generations to: {args.output}")


if __name__ == "__main__":
    main()