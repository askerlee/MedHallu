#!/usr/bin/env python3
import argparse
import gc
import json
import multiprocessing
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from vllm import LLM, SamplingParams


model_names = [
    {"type": "hf", "model_name": "Qwen/Qwen3-4B-Instruct-2507"},
    {"type": "hf", "model_name": "google/medgemma-1.5-4b-it"},
    {"type": "hf", "model_name": "google/medgemma-27b-text-it"},
    # {"type": "openai", "model_name": "gpt-5-mini"},
]


system_prompt = """
You are an expert clinical guideline judge.
You will be given structured guideline rules, a patient vignette, and proposed clinical actions.
Your job is to decide whether the proposed actions are guideline-concordant for that vignette using only the supplied guideline rules.

Answer '0' if the proposed actions follow the guideline rules for the case.
Answer '1' if the proposed actions violate or are not supported by the guideline rules for the case.
Answer '2' if the rules are insufficient to decide or you are genuinely unsure.

Do not return any explanation or extra text. Return only one character: 0, 1, or 2.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether LLMs can distinguish guideline-concordant cases from guideline-violating adversarial cases."
    )
    parser.add_argument(
        "--groundtruth-path",
        default="",
        help="Optional path to a separate guideline-concordant vignette JSON or JSONL file. If omitted, concordant source cases are read from --adversarial-path source_case entries.",
    )
    parser.add_argument(
        "--adversarial-path",
        default="guideline_policy/sample_vignette_adv.json",
        help="Path to the adversarial near-miss vignette JSON or JSONL file.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="guideline_policy/guideline_case_predictions.csv",
        help="Where to save per-example predictions.",
    )
    parser.add_argument(
        "--results-csv",
        default="guideline_policy/guideline_case_results.csv",
        help="Where to save summary metrics.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key, required only when OpenAI models are enabled.",
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


def flatten_groundtruth_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for row in rows:
        guideline = row.get("guideline", {})
        selected_rules = row.get("selected_rules", [])
        source_case = row.get("generated_case", {})
        vignette = str(source_case.get("patient_vignette", "")).strip()
        proposed_actions = source_case.get("recommended_actions", [])
        if not vignette or not isinstance(proposed_actions, list) or not proposed_actions:
            continue

        flattened.append(
            {
                "record_index": guideline.get("record_index"),
                "guideline_id": guideline.get("id", ""),
                "guideline_source": guideline.get("source", ""),
                "guideline_title": guideline.get("title", ""),
                "case_id": f"{guideline.get('record_index', 'unknown')}_source",
                "case_kind": "source_case",
                "difficulty": "concordant",
                "label": 0,
                "patient_vignette": vignette,
                "proposed_actions": proposed_actions,
                "reference_rule_ids": source_case.get("triggered_rule_ids", []),
                "rules_text": format_rules(selected_rules),
            }
        )
    return flattened


def flatten_groundtruth_cases_from_adversarial(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for row in rows:
        guideline = row.get("guideline", {})
        selected_rules = row.get("selected_rules", [])
        source_case = row.get("source_case", {})
        vignette = str(source_case.get("patient_vignette", "")).strip()
        proposed_actions = source_case.get("recommended_actions", [])
        if not vignette or not isinstance(proposed_actions, list) or not proposed_actions:
            continue

        flattened.append(
            {
                "record_index": guideline.get("record_index"),
                "guideline_id": guideline.get("id", ""),
                "guideline_source": guideline.get("source", ""),
                "guideline_title": guideline.get("title", ""),
                "case_id": f"{guideline.get('record_index', 'unknown')}_source",
                "case_kind": "source_case",
                "difficulty": "concordant",
                "label": 0,
                "patient_vignette": vignette,
                "proposed_actions": proposed_actions,
                "reference_rule_ids": source_case.get("triggered_rule_ids", []),
                "rules_text": format_rules(selected_rules),
            }
        )
    return flattened


def flatten_adversarial_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for row in rows:
        guideline = row.get("guideline", {})
        selected_rules = row.get("selected_rules", [])
        challenging_examples = row.get("challenging_examples", [])
        if not isinstance(challenging_examples, list):
            continue

        for example_index, example in enumerate(challenging_examples):
            if not isinstance(example, dict):
                continue
            vignette = str(example.get("patient_vignette", "")).strip()
            proposed_actions = example.get("likely_but_incorrect_actions", [])
            if not vignette or not isinstance(proposed_actions, list) or not proposed_actions:
                continue

            flattened.append(
                {
                    "record_index": guideline.get("record_index"),
                    "guideline_id": guideline.get("id", ""),
                    "guideline_source": guideline.get("source", ""),
                    "guideline_title": guideline.get("title", ""),
                    "case_id": f"{guideline.get('record_index', 'unknown')}_adv_{example_index}",
                    "case_kind": "challenging_example",
                    "difficulty": str(example.get("difficulty", "medium")).strip().lower() or "medium",
                    "label": 1,
                    "patient_vignette": vignette,
                    "proposed_actions": proposed_actions,
                    "reference_rule_ids": example.get("invalidated_rule_ids", []),
                    "rules_text": format_rules(selected_rules),
                }
            )
    return flattened


def prepare_eval_df(groundtruth_path: str, adversarial_path: str) -> pd.DataFrame:
    adversarial_rows = load_structured_records(adversarial_path)

    if groundtruth_path:
        groundtruth_rows = load_structured_records(groundtruth_path)
        flattened_groundtruth = flatten_groundtruth_cases(groundtruth_rows)
    else:
        flattened_groundtruth = flatten_groundtruth_cases_from_adversarial(adversarial_rows)

    flattened_rows = flattened_groundtruth + flatten_adversarial_cases(adversarial_rows)
    if not flattened_rows:
        raise ValueError("No valid guideline cases were found after flattening the inputs.")

    df = pd.DataFrame(flattened_rows)
    df["proposed_actions_text"] = df["proposed_actions"].apply(
        lambda actions: "\n".join(f"- {str(action).strip()}" for action in actions if str(action).strip())
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

Proposed Clinical Actions:
{row['proposed_actions_text']}

Return just one token:
- 0 if the proposed actions are guideline-concordant for this vignette.
- 1 if the proposed actions violate or are not supported by the supplied guideline rules for this vignette.
- 2 if you are unsure.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_model_response(raw_text: str) -> int:
    text = (raw_text or "").strip().lower()
    digits = [char for char in text if char in {"0", "1", "2"}]
    if digits:
        return int(digits[0])
    if any(token in text for token in ["unsure", "not sure", "cannot determine", "can't determine", "insufficient"]):
        return 2
    if any(token in text for token in ["violate", "discordant", "not supported", "incorrect"]):
        return 1
    if any(token in text for token in ["concordant", "follow", "supported", "correct"]):
        return 0
    return 2


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError as exc:
            print(f"Skipping CUDA cleanup due to runtime error: {exc}")


def evaluate_with_hf(model_name: str, prompts: List[List[Dict[str, str]]]) -> List[str]:
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        dtype=torch.bfloat16,
    )
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
        max_tokens=8,
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

    outputs = llm.generate(formatted_prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]

    del llm, tokenizer
    return responses


def evaluate_with_openai(model_name: str, prompts: List[List[Dict[str, str]]], openai_api_key: str) -> List[str]:
    if not openai_api_key:
        raise ValueError("OpenAI API key is required for OpenAI models. Pass --openai-api-key.")

    client = OpenAI(api_key=openai_api_key)
    responses: List[str] = []
    for prompt in tqdm(prompts, desc=f"OpenAI requests ({model_name})", leave=False):
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            max_completion_tokens=8,
            n=1,
        )
        content = response.choices[0].message.content
        responses.append(content.strip() if isinstance(content, str) else str(content))
    return responses


def calculate_binary_metrics(true_labels: List[int], predicted_labels: List[int]) -> Dict[str, Optional[float]]:
    valid_indices = [index for index, prediction in enumerate(predicted_labels) if prediction != 2]
    abstain_rate = (sum(1 for prediction in predicted_labels if prediction == 2) / len(predicted_labels)) if predicted_labels else None
    coverage = (len(valid_indices) / len(predicted_labels)) if predicted_labels else None

    if not valid_indices:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "abstain_rate": abstain_rate,
            "coverage": coverage,
            "support": len(predicted_labels),
        }

    filtered_true = [true_labels[index] for index in valid_indices]
    filtered_pred = [predicted_labels[index] for index in valid_indices]
    return {
        "accuracy": accuracy_score(filtered_true, filtered_pred),
        "precision": precision_score(filtered_true, filtered_pred, zero_division=0),
        "recall": recall_score(filtered_true, filtered_pred, zero_division=0),
        "f1": f1_score(filtered_true, filtered_pred, zero_division=0),
        "abstain_rate": abstain_rate,
        "coverage": coverage,
        "support": len(predicted_labels),
    }


def truncate_numeric_values(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    factor = 10 ** digits
    truncated_df = df.copy()
    numeric_cols = truncated_df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(numeric_cols) > 0:
        truncated_df[numeric_cols] = truncated_df[numeric_cols].apply(
            lambda col: np.trunc(col * factor) / factor
        )
    return truncated_df


def evaluate_model(model_config: Dict[str, str], df: pd.DataFrame, openai_api_key: str) -> Dict[str, pd.DataFrame]:
    prompts = [create_prompt(row) for _, row in df.iterrows()]
    if model_config["type"] == "hf":
        raw_predictions = evaluate_with_hf(model_config["model_name"], prompts)
    else:
        raw_predictions = evaluate_with_openai(model_config["model_name"], prompts, openai_api_key)

    prediction_df = df.copy()
    prediction_df["model_name"] = model_config["model_name"]
    prediction_df["raw_model_output"] = raw_predictions
    prediction_df["predicted_label"] = [parse_model_response(text) for text in raw_predictions]
    prediction_df["decision"] = prediction_df.apply(
        lambda row: "Correct"
        if row["predicted_label"] == row["label"]
        else "Not Sure"
        if row["predicted_label"] == 2
        else "Incorrect",
        axis=1,
    )

    summary_rows: List[Dict[str, Any]] = []
    overall_metrics = calculate_binary_metrics(
        prediction_df["label"].tolist(),
        prediction_df["predicted_label"].tolist(),
    )
    summary_rows.append(
        {
            "model_name": model_config["model_name"],
            "slice_name": "overall",
            **overall_metrics,
        }
    )

    for difficulty, slice_df in prediction_df.groupby("difficulty", dropna=False):
        metrics = calculate_binary_metrics(
            slice_df["label"].tolist(),
            slice_df["predicted_label"].tolist(),
        )
        summary_rows.append(
            {
                "model_name": model_config["model_name"],
                "slice_name": str(difficulty),
                **metrics,
            }
        )

    for case_kind, slice_df in prediction_df.groupby("case_kind", dropna=False):
        metrics = calculate_binary_metrics(
            slice_df["label"].tolist(),
            slice_df["predicted_label"].tolist(),
        )
        summary_rows.append(
            {
                "model_name": model_config["model_name"],
                "slice_name": str(case_kind),
                **metrics,
            }
        )

    summary_df = truncate_numeric_values(pd.DataFrame(summary_rows), digits=3)
    return {
        "predictions": prediction_df,
        "summary": summary_df,
    }


def run_model_subprocess(
    model_config: Dict[str, str],
    eval_df_path: str,
    predictions_csv: str,
    results_csv: str,
    openai_api_key: str,
) -> None:
    try:
        df = pd.read_json(eval_df_path, orient="records")
        outputs = evaluate_model(model_config, df, openai_api_key)

        predictions = outputs["predictions"]
        summary = outputs["summary"]

        if os.path.exists(predictions_csv):
            predictions.to_csv(predictions_csv, mode="a", header=False, index=False)
        else:
            predictions.to_csv(predictions_csv, mode="w", header=True, index=False)

        if os.path.exists(results_csv):
            summary.to_csv(results_csv, mode="a", header=False, index=False)
        else:
            summary.to_csv(results_csv, mode="w", header=True, index=False)
    except Exception as exc:
        print(f"Error evaluating {model_config['model_name']}: {exc}")
    finally:
        clear_gpu_memory()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()

    openai_api_key = args.openai_api_key.strip()
    requires_openai_key = any(model_config.get("type") == "openai" for model_config in model_names)
    if requires_openai_key and not openai_api_key:
        raise ValueError("This run includes OpenAI models. Please provide --openai-api-key.")

    groundtruth_path = args.groundtruth_path.strip()
    adversarial_path = args.adversarial_path.strip()
    if not adversarial_path:
        raise ValueError("--adversarial-path cannot be empty.")

    eval_df = prepare_eval_df(groundtruth_path, adversarial_path)
    ensure_parent_dir(args.predictions_csv)
    ensure_parent_dir(args.results_csv)

    if os.path.exists(args.predictions_csv):
        os.remove(args.predictions_csv)
    if os.path.exists(args.results_csv):
        os.remove(args.results_csv)

    temp_eval_df_path = os.path.join(
        os.path.dirname(args.results_csv) or ".",
        "_guideline_eval_cases.json",
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
            ),
        )
        proc.start()
        proc.join()
        print(f"Completed {model_config['model_name']}")

    if os.path.exists(temp_eval_df_path):
        os.remove(temp_eval_df_path)

    print(f"Saved per-example predictions to: {args.predictions_csv}")
    print(f"Saved summary results to: {args.results_csv}")


if __name__ == "__main__":
    main()