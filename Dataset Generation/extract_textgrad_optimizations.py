#!/usr/bin/env python3

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


VARIABLE_TAG_PATTERN = re.compile(r"<VARIABLE>\s*(.*?)\s*</VARIABLE>", re.DOTALL)
HALLUCINATED_ANSWER_MARKER = "#Hallucinated Answer#:"
QUERY_MARKER = "\nQuery: "
RESPONSE_MARKER = "\nResponse: "
QUESTION_PATTERNS = [
    re.compile(
        r"#Question#:\s*(.*?)\n\n(?:#Knowledge#:|#Answer#:|#Ground truth answer#:|#Hallucinated Answer#:)" ,
        re.DOTALL,
    ),
    re.compile(
        r"(?:^|\n)Question:\s*(.*?)(?:\n\n(?:#|Short answer|Brief summary|One-line takeaway|Quick patient takeaway|Clinician note|Key points|Practical advice|Practical guidance|What to watch for|Evidence and limitations|References|Bottom line)|\n|$)",
        re.DOTALL,
    ),
    re.compile(r"for this question:\s*(.*?)(?:\n|$)", re.IGNORECASE),
]


@dataclass
class OptimizationStep:
    evaluation_trace: Optional[str] = None
    backward_prompt: Optional[str] = None
    backward_gradient: Optional[str] = None
    optimizer_prompt: Optional[str] = None
    optimizer_response: Optional[str] = None
    updated_text: Optional[str] = None

    def to_dict(self, step_index: int) -> Dict[str, Any]:
        return {
            "step_index": step_index,
            "evaluation_trace": self.evaluation_trace,
            "backward_prompt": self.backward_prompt,
            "backward_gradient": self.backward_gradient,
            "optimizer_prompt": self.optimizer_prompt,
            "optimizer_response": self.optimizer_response,
            "updated_text": self.updated_text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final TextGrad optimization outputs from a TextGrad JSONL log."
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to the TextGrad JSONL log file.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path for the extracted JSONL output. Defaults next to the log file.",
    )
    parser.add_argument(
        "--steps-per-optimization",
        type=int,
        default=3,
        help="Number of TextGrad update steps per optimization run. Defaults to 3 to match generation.py.",
    )
    parser.add_argument(
        "--include-steps",
        action="store_true",
        help="Include all intermediate step details in each output record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of optimization runs to write.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_variable_text(optimizer_prompt: Optional[str]) -> Optional[str]:
    if not optimizer_prompt:
        return None
    match = VARIABLE_TAG_PATTERN.search(optimizer_prompt)
    if not match:
        return None
    return match.group(1).strip()


def extract_query_text(evaluation_trace: Optional[str]) -> Optional[str]:
    if not evaluation_trace or QUERY_MARKER not in evaluation_trace or RESPONSE_MARKER not in evaluation_trace:
        return None

    _, prompt_and_response = evaluation_trace.split(QUERY_MARKER, 1)
    prompt, _ = prompt_and_response.rsplit(RESPONSE_MARKER, 1)
    return prompt.strip()


def extract_question(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for pattern in QUESTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def extract_question_from_candidates(candidates: List[Optional[str]]) -> Optional[str]:
    for candidate in candidates:
        question = extract_question(candidate)
        if question:
            return question
    return None


def reconstruct_follow_up_prompt(initial_text: Optional[str], improved_text: Optional[str]) -> Optional[str]:
    if not initial_text or not improved_text:
        return None
    if HALLUCINATED_ANSWER_MARKER not in initial_text:
        return None

    prompt_prefix = initial_text.split(HALLUCINATED_ANSWER_MARKER, 1)[0] + HALLUCINATED_ANSWER_MARKER
    return f"{prompt_prefix}\n\nPrevious attempt improved by TextGrad: {improved_text}"


def build_output_record(
    optimization_index: int,
    steps: List[OptimizationStep],
    steps_per_optimization: int,
    include_steps: bool,
) -> Dict[str, Any]:
    first_step = steps[0]
    last_step = steps[-1]
    initial_text = extract_query_text(first_step.evaluation_trace)
    if not initial_text:
        initial_text = extract_variable_text(first_step.optimizer_prompt)
    final_updated_text = normalize_text(last_step.updated_text) or None
    question = extract_question_from_candidates(
        [
            initial_text,
            final_updated_text,
            first_step.evaluation_trace,
            first_step.optimizer_prompt,
            first_step.backward_prompt,
            first_step.backward_gradient,
            last_step.optimizer_response,
        ]
    )

    record: Dict[str, Any] = {
        "optimization_index": optimization_index,
        "step_count": len(steps),
        "is_partial": len(steps) != steps_per_optimization,
        "question": question,
        "initial_text": initial_text,
        "final_optimizer_response": normalize_text(last_step.optimizer_response) or None,
        "final_updated_text": final_updated_text,
        "reconstructed_follow_up_prompt": reconstruct_follow_up_prompt(initial_text, final_updated_text),
    }

    if include_steps:
        record["steps"] = [step.to_dict(i + 1) for i, step in enumerate(steps)]

    return record


def iter_optimization_records(
    log_file: str,
    steps_per_optimization: int,
    include_steps: bool,
) -> Iterator[Dict[str, Any]]:
    current_step: Optional[OptimizationStep] = None
    current_steps: List[OptimizationStep] = []
    optimization_index = 0

    with open(log_file, 'r', encoding='utf-8') as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = normalize_text(record.get("msg") or record.get("message"))

            if message == "LLMCall function forward":
                current_step = OptimizationStep(
                    evaluation_trace=normalize_text(record.get("text")) or None,
                )
                continue

            if current_step is None:
                continue

            if message == "_backward_through_llm prompt":
                current_step.backward_prompt = normalize_text(record.get("_backward_through_llm")) or None
                continue

            if message == "_backward_through_llm gradient":
                current_step.backward_gradient = normalize_text(record.get("_backward_through_llm")) or None
                continue

            if message == "TextualGradientDescent prompt for update":
                current_step.optimizer_prompt = normalize_text(record.get("prompt")) or None
                continue

            if message == "TextualGradientDescent optimizer response":
                current_step.optimizer_response = normalize_text(record.get("optimizer.response")) or None
                continue

            if message == "TextualGradientDescent updated text":
                current_step.updated_text = normalize_text(record.get("parameter.value")) or None
                current_steps.append(current_step)
                current_step = None

                if len(current_steps) == steps_per_optimization:
                    optimization_index += 1
                    yield build_output_record(
                        optimization_index,
                        current_steps,
                        steps_per_optimization,
                        include_steps,
                    )
                    current_steps = []

    if current_steps:
        optimization_index += 1
        yield build_output_record(
            optimization_index,
            current_steps,
            steps_per_optimization,
            include_steps,
        )


def default_output_path(log_file: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(log_file))
    return f"{base}.optimizations.jsonl"


def main() -> None:
    args = parse_args()
    log_file = os.path.abspath(args.log_file)
    output_file = os.path.abspath(args.output_file) if args.output_file else default_output_path(log_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    written = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        for record in iter_optimization_records(
            log_file=log_file,
            steps_per_optimization=args.steps_per_optimization,
            include_steps=args.include_steps,
        ):
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(f"Wrote {written} optimization records to {output_file}")


if __name__ == "__main__":
    main()