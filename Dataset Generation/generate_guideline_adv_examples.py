import argparse
import json
import os
from typing import Any, Dict, List, Optional


NON_ACTIONABLE_CLASSES = {"APPEND_QUALIFIER", "APPEND_SAFETY"}
HINTING_ACTION_PHRASES = (
    "as if",
    "as though",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate guideline near-miss examples from existing guideline-concordant vignettes. "
            "Each near-miss should remain clinically similar to the source vignette while subtly "
            "breaking the rule conditions that supported the original recommendation."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="guideline_policy/sample_vignette.json",
        help="Path to the source vignette JSON or JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guideline_policy/sample_vignette_adv.json",
        help="Path to save generated near-miss examples. Use .jsonl for one record per line or .json for an array.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First guideline record_index to process, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last guideline record_index to process, inclusive. Defaults to the last record in the file.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="LLM model used to generate challenging near-miss examples.",
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
        "--num-generations",
        type=int,
        default=2,
        help="Number of challenging near-miss examples to generate per source vignette.",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=8,
        help="Maximum number of actionable rules to include in the prompt context.",
    )
    return parser.parse_args()


def load_structured_records(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}. Expected a list or object.")


def select_records(
    rows: List[Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
) -> List[Dict[str, Any]]:
    if end_index is not None and end_index < start_index:
        raise ValueError("end-index must be greater than or equal to start-index")

    selected: List[Dict[str, Any]] = []
    for row in rows:
        guideline = row.get("guideline", {})
        record_index = guideline.get("record_index")
        if not isinstance(record_index, int):
            continue
        if record_index < start_index:
            continue
        if end_index is not None and record_index > end_index:
            continue
        selected.append(row)
    return selected


def get_actionable_rules(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = row.get("selected_rules", [])
    actionable_rules: List[Dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("recommendation_class") in NON_ACTIONABLE_CLASSES:
            continue
        actionable_rules.append(rule)
    return actionable_rules


def build_rule_context(selected_rules: List[Dict[str, Any]], max_rules: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        selected_rules,
        key=lambda rule: (-int(rule.get("priority", 0)), str(rule.get("rule_id", ""))),
    )
    return ordered[:max_rules]


def build_messages(
    row: Dict[str, Any],
    selected_rules: List[Dict[str, Any]],
    num_generations: int,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You create adversarial clinical guideline examples. "
        "Your job is to take a vignette that truly matches a recommendation and rewrite it into "
        "challenging near-miss cases that look similar on first read but do not actually satisfy the rule conditions. "
        "Return only valid JSON with no markdown."
    )

    guideline = row.get("guideline", {})
    source_case = row.get("generated_case", {})
    user_payload = {
        "task": (
            "Generate challenging near-miss patient vignettes. Each output should remain very similar to the source "
            "case, but one or two subtle clinical details must make the original recommendation no longer guideline-concordant."
        ),
        "requirements": [
            "Generate exactly the requested number of examples.",
            "Preserve the overall disease context, patient type, and workflow from the source vignette.",
            "Change only the minimum details needed to invalidate the recommendation, such as stage, biomarker status, imaging findings, accessibility, prior testing, or treatment intent.",
            "Make the case deceptive: a model or reader might incorrectly apply the original recommendation on a superficial read.",
            "Do not make the vignette obviously absurd or directly state that the recommendation is wrong.",
            "Each vignette should be concise and clinically plausible, about 120-220 words.",
            "The likely_but_incorrect_actions field should describe the tempting but wrong actions a reader might choose using natural clinical phrasing.",
            "Do not use meta-commentary or giveaway phrasing in likely_but_incorrect_actions, especially expressions like 'as if' or 'as though'.",
            "The corrected_assessment field should explain why the case is not guideline-concordant and what the safer interpretation is.",
            "Only cite invalidated_rule_ids from the supplied rules.",
            "Set true_guideline_concordance to DISCORDANT for every example.",
            "Difficulty should be easy, medium, or hard depending on how subtle the mismatch is.",
        ],
        "output_schema": {
            "examples": [
                {
                    "patient_vignette": "string",
                    "invalidated_rule_ids": ["R001"],
                    "deceptive_similarity": "string",
                    "likely_but_incorrect_actions": ["string"],
                    "corrected_assessment": "string",
                    "rationale": "string",
                    "true_guideline_concordance": "DISCORDANT",
                    "difficulty": "hard",
                }
            ]
        },
        "num_examples": num_generations,
        "guideline_metadata": guideline,
        "source_case": source_case,
        "selected_rules": selected_rules,
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
        temperature=0.9,
    )
    content = response.choices[0].message.content
    if content is None:
        return {}
    if not isinstance(content, str):
        content = str(content)
    parsed = extract_json_object(content)
    return parsed if parsed is not None else {}


def contains_hinting_phrase(text: str) -> bool:
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in HINTING_ACTION_PHRASES)


def validate_examples(result: Dict[str, Any], selected_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_rule_ids = {rule.get("rule_id") for rule in selected_rules}
    examples = result.get("examples", [])
    if not isinstance(examples, list):
        return []

    validated_examples: List[Dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, dict):
            continue

        invalidated_rule_ids = example.get("invalidated_rule_ids", [])
        if not isinstance(invalidated_rule_ids, list):
            invalidated_rule_ids = []

        likely_but_incorrect_actions = example.get("likely_but_incorrect_actions", [])
        if not isinstance(likely_but_incorrect_actions, list):
            likely_but_incorrect_actions = []

        difficulty = str(example.get("difficulty", "medium")).strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"

        validated_examples.append(
            {
                "patient_vignette": str(example.get("patient_vignette", "")).strip(),
                "invalidated_rule_ids": [
                    rule_id for rule_id in invalidated_rule_ids if rule_id in valid_rule_ids
                ],
                "deceptive_similarity": str(example.get("deceptive_similarity", "")).strip(),
                "likely_but_incorrect_actions": [
                    action_text
                    for action in likely_but_incorrect_actions
                    for action_text in [str(action).strip()]
                    if action_text and not contains_hinting_phrase(action_text)
                ],
                "corrected_assessment": str(example.get("corrected_assessment", "")).strip(),
                "rationale": str(example.get("rationale", "")).strip(),
                "true_guideline_concordance": "DISCORDANT",
                "difficulty": difficulty,
            }
        )

    return [example for example in validated_examples if example["patient_vignette"]]


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

    return tqdm(records, desc="Generating near-miss examples", unit="record")


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

    rows = load_structured_records(args.input)
    records_to_process = select_records(rows, args.start_index, args.end_index)
    if not records_to_process:
        raise SystemExit("No records found in the requested index range.")

    output_payload: List[Dict[str, Any]] = []
    for row in iter_with_progress(records_to_process):
        selected_rules = build_rule_context(get_actionable_rules(row), max_rules=args.max_rules)
        source_case = row.get("generated_case", {})

        if not selected_rules:
            output_payload.append(
                {
                    "guideline": row.get("guideline", {}),
                    "selected_rules": [],
                    "source_case": source_case,
                    "challenging_examples": [],
                    "status": "skipped_no_actionable_rules",
                }
            )
            continue

        if not isinstance(source_case, dict) or not str(source_case.get("patient_vignette", "")).strip():
            output_payload.append(
                {
                    "guideline": row.get("guideline", {}),
                    "selected_rules": selected_rules,
                    "source_case": source_case,
                    "challenging_examples": [],
                    "status": "skipped_missing_source_case",
                }
            )
            continue

        messages = build_messages(row, selected_rules, num_generations=args.num_generations)
        raw_result = call_llm(
            client=client,
            messages=messages,
            model=args.llm_model,
        )
        validated_examples = validate_examples(raw_result, selected_rules)

        output_payload.append(
            {
                "guideline": row.get("guideline", {}),
                "selected_rules": selected_rules,
                "source_case": source_case,
                "challenging_examples": validated_examples,
                "status": "generated" if validated_examples else "empty_generation",
            }
        )

    ensure_output_parent(args.output)
    if args.output.lower().endswith(".json"):
        write_json_output(args.output, output_payload)
    else:
        write_jsonl_output(args.output, output_payload)

    print(f"Processed {len(output_payload)} guideline records")
    print(f"Saved guideline near-miss examples to: {args.output}")


if __name__ == "__main__":
    main()