#!/usr/bin/env python3
import argparse
import gc
import json
import multiprocessing
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams


VLLM_ENABLE_V1_MULTIPROCESSING = "0"


model_names = [
    {"type": "hf", "model_name": "Qwen/Qwen3-4B-Instruct-2507"},
    {"type": "hf", "model_name": "google/medgemma-1.5-4b-it"},
    {"type": "hf", "model_name": "google/medgemma-27b-text-it"},
    # {"type": "openai", "model_name": "gpt-5-mini"},
]


system_prompt = """
You are an expert clinical guideline assistant.
You will be given structured guideline rules and a patient vignette.
Generate the clinical actions that are supported by the supplied rules for that exact vignette.

Requirements:
- Use only the supplied rules.
- Include both recommended and not-recommended actions when the rules support them.
- Do not invent actions that are not justified by the rules.
- If no supplied rule clearly supports an action, return an empty list.
- Do not explain your answer.

Return only valid JSON in this format:
{"recommended_actions": ["action 1", "action 2"]}
""".strip()


class EntailmentScorer:
    def __init__(self, device_preference: str) -> None:
        self.model_name = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if device_preference == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if self.device.type == "cuda":
            clear_gpu_memory()

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        if not premise or not hypothesis:
            return 0.0

        inputs = self.tokenizer(
            str(premise),
            str(hypothesis),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)

        return float(predictions[0][2].item())

    def bidirectional_score(self, text1: str, text2: str) -> float:
        return min(
            self.check_entailment(text1, text2),
            self.check_entailment(text2, text1),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate guideline-supported clinical actions for each case and score them against reference actions."
    )
    parser.add_argument(
        "--cases-path",
        default="guideline_policy/sample_vignette.json",
        help="Path to a JSON or JSONL file containing guideline cases with reference recommended_actions.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="guideline_policy/generative_guideline_action_predictions.csv",
        help="Where to save per-example generations and case-level action metrics.",
    )
    parser.add_argument(
        "--results-csv",
        default="guideline_policy/generative_guideline_action_results.csv",
        help="Where to save summary metrics.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Bidirectional entailment threshold used to count a generated action as matching a reference action.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on the number of cases to evaluate for debugging.",
    )
    parser.add_argument(
        "--entailment-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for semantic action matching. 'auto' uses CPU for local HF generation and CUDA otherwise when available.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key, required only when OpenAI models are enabled.",
    )
    parser.add_argument(
        "--subprocess-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout for each model subprocess. Use 0 to wait indefinitely.",
    )
    return parser.parse_args()


def load_structured_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input dataset not found: {path}")

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


def format_rules(selected_rules: List[Dict[str, Any]]) -> str:
    formatted_rules: List[str] = []
    for rule in selected_rules:
        if not isinstance(rule, dict):
            continue
        formatted_rules.append(
            "\n".join(
                [
                    f"Rule ID: {rule.get('rule_id', 'UNKNOWN')}",
                    f"Scenario: {rule.get('scenario', 'UNKNOWN')}",
                    f"Priority: {rule.get('priority', 'UNKNOWN')}",
                    f"Recommendation Class: {rule.get('recommendation_class', 'UNKNOWN')}",
                    f"Activation Condition: {rule.get('activation_condition', '')}",
                    f"Guideline Action: {rule.get('execute', '')}",
                    f"Source Statement: {rule.get('source_statement', '')}",
                ]
            )
        )
    return "\n\n".join(formatted_rules)


def deduplicate_actions(actions: List[str]) -> List[str]:
    seen = set()
    deduplicated: List[str] = []
    for action in actions:
        normalized = normalize_text(action)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(str(action).strip())
    return deduplicated


def flatten_reference_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for row in rows:
        guideline = row.get("guideline", {})
        selected_rules = row.get("selected_rules", [])

        case_payload = row.get("generated_case")
        case_kind = "generated_case"
        if not isinstance(case_payload, dict):
            case_payload = row.get("source_case")
            case_kind = "source_case"
        if not isinstance(case_payload, dict):
            continue

        vignette = str(case_payload.get("patient_vignette", "")).strip()
        reference_actions = case_payload.get("recommended_actions", [])
        if not vignette or not isinstance(reference_actions, list):
            continue

        cleaned_reference_actions = deduplicate_actions(
            [str(action).strip() for action in reference_actions if str(action).strip()]
        )
        if not cleaned_reference_actions:
            continue

        flattened.append(
            {
                "record_index": guideline.get("record_index"),
                "guideline_id": guideline.get("id", ""),
                "guideline_source": guideline.get("source", ""),
                "guideline_title": guideline.get("title", ""),
                "case_id": f"{guideline.get('record_index', 'unknown')}_{case_kind}",
                "case_kind": case_kind,
                "patient_vignette": vignette,
                "reference_actions": cleaned_reference_actions,
                "reference_rule_ids": case_payload.get("triggered_rule_ids", []),
                "rules_text": format_rules(selected_rules),
            }
        )

        challenging_examples = row.get("challenging_examples", [])
        if not isinstance(challenging_examples, list):
            continue

        for example_index, example in enumerate(challenging_examples):
            if not isinstance(example, dict):
                continue

            vignette = str(example.get("patient_vignette", "")).strip()
            reference_actions = example.get("recommended_actions", [])
            if not vignette or not isinstance(reference_actions, list):
                continue

            cleaned_reference_actions = deduplicate_actions(
                [str(action).strip() for action in reference_actions if str(action).strip()]
            )

            flattened.append(
                {
                    "record_index": guideline.get("record_index"),
                    "guideline_id": guideline.get("id", ""),
                    "guideline_source": guideline.get("source", ""),
                    "guideline_title": guideline.get("title", ""),
                    "case_id": f"{guideline.get('record_index', 'unknown')}_challenging_example_{example_index}",
                    "case_kind": "challenging_example",
                    "patient_vignette": vignette,
                    "reference_actions": cleaned_reference_actions,
                    "reference_rule_ids": example.get("triggered_rule_ids", []),
                    "rules_text": format_rules(selected_rules),
                }
            )
    return flattened


def prepare_eval_df(cases_path: str, max_cases: int) -> pd.DataFrame:
    flattened_rows = flatten_reference_cases(load_structured_records(cases_path))
    if not flattened_rows:
        raise ValueError("No valid cases with reference recommended_actions were found in the input file.")

    if max_cases > 0:
        flattened_rows = flattened_rows[:max_cases]

    df = pd.DataFrame(flattened_rows)
    df["reference_actions_text"] = df["reference_actions"].apply(
        lambda actions: "\n".join(f"- {action}" for action in actions)
    )
    df["reference_rule_ids_text"] = df["reference_rule_ids"].apply(
        lambda rule_ids: ", ".join(str(rule_id) for rule_id in rule_ids)
    )
    return df


def create_prompt(row: pd.Series) -> List[Dict[str, str]]:
    user_prompt = f"""
Guideline Metadata:
- Record Index: {row['record_index']}
- Guideline ID: {row['guideline_id']}
- Source: {row['guideline_source']}
- Title: {row['guideline_title']}

Structured Guideline Rules:
{row['rules_text']}

Patient Vignette:
{row['patient_vignette']}

Return only JSON in the format:
{{"recommended_actions": ["action 1", "action 2"]}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_payload(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None

    for candidate in [text]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        snippet = text[object_start : object_end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end != -1 and list_end > list_start:
        snippet = text[list_start : list_end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def parse_generated_actions(raw_text: str) -> List[str]:
    payload = extract_json_payload(raw_text)
    actions: List[str] = []

    if isinstance(payload, dict):
        for key in ["recommended_actions", "actions", "recommendations"]:
            value = payload.get(key)
            if isinstance(value, list):
                actions = [str(item).strip() for item in value if str(item).strip()]
                break
            if isinstance(value, str) and value.strip():
                actions = [value.strip()]
                break
    elif isinstance(payload, list):
        actions = [str(item).strip() for item in payload if str(item).strip()]

    if actions:
        return deduplicate_actions(actions)

    cleaned_lines: List[str] = []
    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^recommended_actions\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[-*•]+\s*", "", line)
        line = re.sub(r"^\d+[\).]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if line and line not in {"[", "]", "{", "}"}:
            cleaned_lines.append(line)

    return deduplicate_actions(cleaned_lines)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def match_action_lists(
    generated_actions: List[str],
    reference_actions: List[str],
    scorer: EntailmentScorer,
    similarity_threshold: float,
) -> Dict[str, Any]:
    if not generated_actions and not reference_actions:
        return {
            "matched_pairs": [],
            "matched_generated_actions": [],
            "matched_reference_actions": [],
            "unmatched_generated_actions": [],
            "unmatched_reference_actions": [],
            "num_matched_actions": 0,
            "action_precision": 1.0,
            "action_recall": 1.0,
            "action_f1": 1.0,
            "exact_match": 1,
            "average_match_score": 1.0,
        }

    if not generated_actions or not reference_actions:
        precision = 1.0 if not generated_actions and not reference_actions else 0.0
        recall = 1.0 if not generated_actions and not reference_actions else 0.0
        f1 = 1.0 if not generated_actions and not reference_actions else 0.0
        return {
            "matched_pairs": [],
            "matched_generated_actions": [],
            "matched_reference_actions": [],
            "unmatched_generated_actions": generated_actions,
            "unmatched_reference_actions": reference_actions,
            "num_matched_actions": 0,
            "action_precision": precision,
            "action_recall": recall,
            "action_f1": f1,
            "exact_match": int(not generated_actions and not reference_actions),
            "average_match_score": 0.0,
        }

    score_matrix: List[List[float]] = []
    for generated_action in generated_actions:
        row_scores: List[float] = []
        for reference_action in reference_actions:
            row_scores.append(scorer.bidirectional_score(generated_action, reference_action))
        score_matrix.append(row_scores)

    remaining_generated = set(range(len(generated_actions)))
    remaining_reference = set(range(len(reference_actions)))
    matched_pairs: List[Dict[str, Any]] = []

    while remaining_generated and remaining_reference:
        best_pair: Optional[Tuple[int, int, float]] = None
        for generated_index in remaining_generated:
            for reference_index in remaining_reference:
                score = score_matrix[generated_index][reference_index]
                if best_pair is None or score > best_pair[2]:
                    best_pair = (generated_index, reference_index, score)

        if best_pair is None or best_pair[2] < similarity_threshold:
            break

        generated_index, reference_index, score = best_pair
        matched_pairs.append(
            {
                "generated_action": generated_actions[generated_index],
                "reference_action": reference_actions[reference_index],
                "score": score,
            }
        )
        remaining_generated.remove(generated_index)
        remaining_reference.remove(reference_index)

    num_matched_actions = len(matched_pairs)
    action_precision = safe_divide(num_matched_actions, len(generated_actions))
    action_recall = safe_divide(num_matched_actions, len(reference_actions))
    action_f1 = safe_divide(2 * action_precision * action_recall, action_precision + action_recall)

    return {
        "matched_pairs": matched_pairs,
        "matched_generated_actions": [pair["generated_action"] for pair in matched_pairs],
        "matched_reference_actions": [pair["reference_action"] for pair in matched_pairs],
        "unmatched_generated_actions": [generated_actions[index] for index in sorted(remaining_generated)],
        "unmatched_reference_actions": [reference_actions[index] for index in sorted(remaining_reference)],
        "num_matched_actions": num_matched_actions,
        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,
        "exact_match": int(
            num_matched_actions == len(generated_actions) == len(reference_actions)
        ),
        "average_match_score": float(np.mean([pair["score"] for pair in matched_pairs])) if matched_pairs else 0.0,
    }


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError as exc:
            print(f"Skipping CUDA cleanup due to runtime error: {exc}")


def shutdown_torch_distributed() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    try:
        torch.distributed.destroy_process_group()
    except Exception as exc:
        print(f"Skipping torch.distributed cleanup due to runtime error: {exc}")


def shutdown_vllm_engine(llm: Optional[LLM]) -> None:
    if llm is None:
        return

    llm_engine = getattr(llm, "llm_engine", None)
    output_processor = getattr(llm_engine, "output_processor", None)
    if output_processor is not None and hasattr(output_processor, "close"):
        try:
            output_processor.close()
        except Exception as exc:
            print(f"Skipping vLLM output processor cleanup due to runtime error: {exc}")

    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        try:
            engine_core.shutdown(timeout=5.0)
        except Exception as exc:
            print(f"Skipping vLLM engine core cleanup due to runtime error: {exc}")

    renderer = getattr(llm_engine, "renderer", None)
    if renderer is not None and hasattr(renderer, "shutdown"):
        try:
            renderer.shutdown()
        except Exception as exc:
            print(f"Skipping vLLM renderer cleanup due to runtime error: {exc}")


def log_progress(message: str) -> None:
    print(message, flush=True)


def evaluate_with_hf(model_name: str, prompts: List[List[Dict[str, str]]]) -> List[str]:
    llm: Optional[LLM] = None
    tokenizer = None
    previous_vllm_multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    try:
        log_progress(f"[{model_name}] Initializing vLLM engine")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = VLLM_ENABLE_V1_MULTIPROCESSING
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype=torch.bfloat16,
            async_scheduling=False,
        )
        log_progress(f"[{model_name}] vLLM engine ready")
        tokenizer = llm.get_tokenizer()

        stop_tok_id: List[int] = []
        if tokenizer.eos_token_id is not None:
            stop_tok_id.append(tokenizer.eos_token_id)
        for special_token in ["<|eot_id|>", "<|eos_token|>", "<end_of_turn>", "</s>"]:
            try:
                sid = tokenizer.convert_tokens_to_ids(special_token)
                if isinstance(sid, int) and sid not in stop_tok_id:
                    stop_tok_id.append(sid)
            except Exception:
                pass

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            stop_token_ids=stop_tok_id,
        )

        formatted_prompts: List[str] = []
        for prompt in prompts:
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )

        log_progress(f"[{model_name}] Starting generation for {len(formatted_prompts)} prompts")
        outputs = llm.generate(formatted_prompts, sampling_params)
        log_progress(f"[{model_name}] Generation finished")
        return [output.outputs[0].text.strip() for output in outputs]
    finally:
        shutdown_vllm_engine(llm)
        shutdown_torch_distributed()
        if previous_vllm_multiprocessing is None:
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = previous_vllm_multiprocessing
        del tokenizer
        del llm


def evaluate_with_openai(model_name: str, prompts: List[List[Dict[str, str]]], openai_api_key: str) -> List[str]:
    if not openai_api_key:
        raise ValueError("OpenAI API key is required for OpenAI models. Pass --openai-api-key.")

    client = OpenAI(api_key=openai_api_key)
    responses: List[str] = []
    for prompt in tqdm(prompts, desc=f"OpenAI requests ({model_name})", leave=False):
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            max_completion_tokens=256,
            n=1,
        )
        content = response.choices[0].message.content
        responses.append(content.strip() if isinstance(content, str) else str(content))
    return responses


def resolve_entailment_device(device_preference: str, model_type: str) -> str:
    if device_preference in {"cpu", "cuda"}:
        return device_preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def truncate_numeric_values(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    factor = 10 ** digits
    truncated_df = df.copy()
    numeric_cols = truncated_df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(numeric_cols) > 0:
        truncated_df[numeric_cols] = truncated_df[numeric_cols].apply(
            lambda col: np.trunc(col * factor) / factor
        )
    return truncated_df


def summarize_slice(slice_df: pd.DataFrame, model_name: str, slice_name: str) -> Dict[str, Any]:
    exact_match_rate = float(np.mean(slice_df["exact_match"])) if len(slice_df) else None
    return {
        "model_name": model_name,
        "slice_name": slice_name,
        "case_accuracy": exact_match_rate,
        "exact_match_rate": exact_match_rate,
        "mean_action_precision": float(np.mean(slice_df["action_precision"])) if len(slice_df) else None,
        "mean_action_recall": float(np.mean(slice_df["action_recall"])) if len(slice_df) else None,
        "mean_action_f1": float(np.mean(slice_df["action_f1"])) if len(slice_df) else None,
        "mean_match_score": float(np.mean(slice_df["average_match_score"])) if len(slice_df) else None,
        "hallucination_free_rate": float(np.mean(slice_df["num_unmatched_generated_actions"] == 0)) if len(slice_df) else None,
        "full_recall_rate": float(np.mean(slice_df["num_unmatched_reference_actions"] == 0)) if len(slice_df) else None,
        "mean_generated_actions": float(np.mean(slice_df["num_generated_actions"])) if len(slice_df) else None,
        "mean_reference_actions": float(np.mean(slice_df["num_reference_actions"])) if len(slice_df) else None,
        "support": int(len(slice_df)),
    }


def evaluate_model(
    model_config: Dict[str, str],
    df: pd.DataFrame,
    openai_api_key: str,
    similarity_threshold: float,
    entailment_device: str,
) -> Dict[str, pd.DataFrame]:
    scorer: Optional[EntailmentScorer] = None
    try:
        log_progress(f"[{model_config['model_name']}] Building prompts")
        prompts = [create_prompt(row) for _, row in df.iterrows()]
        if model_config["type"] == "hf":
            raw_outputs = evaluate_with_hf(model_config["model_name"], prompts)
        else:
            raw_outputs = evaluate_with_openai(model_config["model_name"], prompts, openai_api_key)

        log_progress(f"[{model_config['model_name']}] Loading entailment scorer")
        clear_gpu_memory()
        resolved_entailment_device = resolve_entailment_device(
            entailment_device,
            model_config["type"],
        )
        scorer = EntailmentScorer(resolved_entailment_device)
        log_progress(
            f"[{model_config['model_name']}] Entailment scorer ready on {resolved_entailment_device}"
        )

        prediction_df = df.copy()
        prediction_df["model_name"] = model_config["model_name"]
        prediction_df["raw_model_output"] = raw_outputs
        prediction_df["generated_actions"] = [parse_generated_actions(text) for text in raw_outputs]

        log_progress(f"[{model_config['model_name']}] Matching generated and reference actions")
        match_results = [
            match_action_lists(generated, reference, scorer, similarity_threshold)
            for generated, reference in tqdm(
                zip(
                    prediction_df["generated_actions"].tolist(),
                    prediction_df["reference_actions"].tolist(),
                ),
                total=len(prediction_df),
                desc=f"Action matching ({model_config['model_name']})",
                leave=False,
            )
        ]
        log_progress(f"[{model_config['model_name']}] Action matching finished")

        prediction_df["matched_pairs"] = [result["matched_pairs"] for result in match_results]
        prediction_df["matched_generated_actions"] = [result["matched_generated_actions"] for result in match_results]
        prediction_df["matched_reference_actions"] = [result["matched_reference_actions"] for result in match_results]
        prediction_df["unmatched_generated_actions"] = [result["unmatched_generated_actions"] for result in match_results]
        prediction_df["unmatched_reference_actions"] = [result["unmatched_reference_actions"] for result in match_results]
        prediction_df["num_matched_actions"] = [result["num_matched_actions"] for result in match_results]
        prediction_df["action_precision"] = [result["action_precision"] for result in match_results]
        prediction_df["action_recall"] = [result["action_recall"] for result in match_results]
        prediction_df["action_f1"] = [result["action_f1"] for result in match_results]
        prediction_df["exact_match"] = [result["exact_match"] for result in match_results]
        prediction_df["average_match_score"] = [result["average_match_score"] for result in match_results]
        prediction_df["num_generated_actions"] = prediction_df["generated_actions"].apply(len)
        prediction_df["num_reference_actions"] = prediction_df["reference_actions"].apply(len)
        prediction_df["num_unmatched_generated_actions"] = prediction_df["unmatched_generated_actions"].apply(len)
        prediction_df["num_unmatched_reference_actions"] = prediction_df["unmatched_reference_actions"].apply(len)
        prediction_df["decision"] = prediction_df["exact_match"].apply(
            lambda exact_match: "Correct" if exact_match else "Incorrect"
        )

        for column in [
            "generated_actions",
            "reference_actions",
            "matched_pairs",
            "matched_generated_actions",
            "matched_reference_actions",
            "unmatched_generated_actions",
            "unmatched_reference_actions",
            "reference_rule_ids",
        ]:
            prediction_df[column] = prediction_df[column].apply(json.dumps)

        summary_rows = [summarize_slice(prediction_df, model_config["model_name"], "overall")]

        for case_kind, slice_df in prediction_df.groupby("case_kind", dropna=False):
            summary_rows.append(
                summarize_slice(slice_df, model_config["model_name"], str(case_kind))
            )

        for action_count, slice_df in prediction_df.groupby("num_reference_actions", dropna=False):
            summary_rows.append(
                summarize_slice(slice_df, model_config["model_name"], f"reference_actions_{action_count}")
            )

        summary_df = truncate_numeric_values(pd.DataFrame(summary_rows), digits=3)
        log_progress(f"[{model_config['model_name']}] Evaluation finished")
        return {
            "predictions": prediction_df,
            "summary": summary_df,
        }
    finally:
        if scorer is not None:
            scorer.close()
        shutdown_torch_distributed()
        clear_gpu_memory()


def run_model_subprocess(
    model_config: Dict[str, str],
    eval_df_path: str,
    predictions_csv: str,
    results_csv: str,
    openai_api_key: str,
    similarity_threshold: float,
    entailment_device: str,
) -> None:
    try:
        log_progress(f"[{model_config['model_name']}] Loading evaluation dataset")
        df = pd.read_json(eval_df_path, orient="records")
        outputs = evaluate_model(
            model_config,
            df,
            openai_api_key,
            similarity_threshold,
            entailment_device,
        )

        predictions = outputs["predictions"]
        summary = outputs["summary"]
        log_progress(f"[{model_config['model_name']}] Writing CSV outputs")

        if os.path.exists(predictions_csv):
            predictions.to_csv(predictions_csv, mode="a", header=False, index=False)
        else:
            predictions.to_csv(predictions_csv, mode="w", header=True, index=False)

        if os.path.exists(results_csv):
            summary.to_csv(results_csv, mode="a", header=False, index=False)
        else:
            summary.to_csv(results_csv, mode="w", header=True, index=False)
        log_progress(f"[{model_config['model_name']}] Subprocess work completed")
    except Exception as exc:
        print(f"Error evaluating {model_config['model_name']}: {exc}")
    finally:
        shutdown_torch_distributed()
        clear_gpu_memory()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("--similarity-threshold must be between 0 and 1.")
    if args.subprocess_timeout_seconds < 0.0:
        raise ValueError("--subprocess-timeout-seconds must be greater than or equal to 0.")

    openai_api_key = args.openai_api_key.strip()
    requires_openai_key = any(model_config.get("type") == "openai" for model_config in model_names)
    if requires_openai_key and not openai_api_key:
        raise ValueError("This run includes OpenAI models. Please provide --openai-api-key.")

    eval_df = prepare_eval_df(args.cases_path.strip(), args.max_cases)
    ensure_parent_dir(args.predictions_csv)
    ensure_parent_dir(args.results_csv)

    if os.path.exists(args.predictions_csv):
        os.remove(args.predictions_csv)
    if os.path.exists(args.results_csv):
        os.remove(args.results_csv)

    temp_eval_df_path = os.path.join(
        os.path.dirname(args.results_csv) or ".",
        "_guideline_generation_eval_cases.json",
    )
    eval_df.to_json(temp_eval_df_path, orient="records")

    ctx = multiprocessing.get_context("spawn")
    for model_config in model_names:
        proc = ctx.Process(
            target=run_model_subprocess,
            args=(
                model_config,
                temp_eval_df_path,
                args.predictions_csv,
                args.results_csv,
                openai_api_key,
                args.similarity_threshold,
                args.entailment_device,
            ),
        )
        proc.start()
        if args.subprocess_timeout_seconds > 0.0:
            proc.join(timeout=args.subprocess_timeout_seconds)
        else:
            proc.join()
        if args.subprocess_timeout_seconds > 0.0 and proc.is_alive():
            print(
                f"Timed out waiting for {model_config['model_name']} to exit cleanly; terminating subprocess."
            )
            proc.terminate()
            proc.join()
        exit_code = proc.exitcode
        proc.close()
        if exit_code not in {0, None}:
            print(f"Subprocess for {model_config['model_name']} exited with code {exit_code}")
        print(f"Completed {model_config['model_name']}")

    if os.path.exists(temp_eval_df_path):
        os.remove(temp_eval_df_path)

    print(f"Saved per-example predictions to: {args.predictions_csv}")
    print(f"Saved summary results to: {args.results_csv}")


if __name__ == "__main__":
    main()