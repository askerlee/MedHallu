#!/usr/bin/env python3
import torch
import gc
import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_from_disk
# vLLM imports
from vllm import LLM, SamplingParams
import torch
import multiprocessing
import ast
import numpy as np
import re

# (If using the OpenAI branch, uncomment and set your API key)
from openai import OpenAI

model_names = [
    #{'type': 'hf', 'model_name': 'Qwen/Qwen3.5-4B'},
    #{'type': 'hf', 'model_name': 'Qwen/Qwen3.5-9B'},
    # {'type': 'hf', 'model_name': 'Qwen/Qwen3-30B-A3B-Instruct-2507'},
    # {'type': 'hf', 'model_name': 'deepseek-ai/DeepSeek-V3.2'},
    {'type': 'hf', 'model_name': 'Qwen/Qwen3-4B-Instruct-2507'},
    {'type': 'hf', 'model_name': 'google/medgemma-1.5-4b-it'},
    {'type': 'hf', 'model_name': 'google/medgemma-27b-text-it'},
    # {'type': 'openai', 'model_name': 'gpt-5-mini'},
]

# ---------------------
# PROMPT CREATION
# ---------------------
system_prompt = """
You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the
provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based
on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes


You are trying to determine if there is a factual contradiction between the answer and the world knowledge.
Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the factuality and want to pass. If you are unsure about the answer choose `2` instead of guessing and making a mistake. Do not return anything else, just the answer.
"""

def create_prompt(question, option1):
    prompt = f"""
Question: {question}
Answer: {option1}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt

def create_prompt_withknowledge(question, option1, knowledge):
    prompt = f"""
World Knowledge: {knowledge}
Question: {question}
Answer: {option1}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt

# ---------------------
# GPU MEMORY CLEARING
# ---------------------
def clear_gpu_memory():
    if torch.cuda.is_available():
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError as e:
            print(f"Skipping CUDA cleanup due to runtime error: {e}")


def prepare_df_from_hf_dataset(dataset_dir):
    dataset = load_from_disk(dataset_dir)
    if isinstance(dataset, dict):
        split_name = 'train' if 'train' in dataset else next(iter(dataset.keys()))
        split = dataset[split_name]
    else:
        split = dataset

    df = split.to_pandas()
    column_map = {
        'Question': 'question',
        'Knowledge': 'knowledge',
        'Ground Truth': 'ground_truth',
        'Hallucinated Answer': 'least_similar_answer',
        'Difficulty Level': 'final_difficulty_level',
    }
    df = df.rename(columns=column_map)

    required_columns = [
        'question',
        'knowledge',
        'ground_truth',
        'least_similar_answer',
        'final_difficulty_level',
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {dataset_dir} is missing required columns: {missing}. "
            "Expected a MedHallu pqa_labeled-style schema."
        )

    def _normalize_contexts(x):
        if isinstance(x, np.ndarray):
            return [str(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [str(v) for v in x]
        if pd.isna(x):
            return []
        return [str(x)]

    df['knowledge'] = df['knowledge'].apply(lambda x: str({'contexts': _normalize_contexts(x)}))
    return df[required_columns]


def parse_knowledge_field(raw_knowledge):
    if raw_knowledge is None or (isinstance(raw_knowledge, float) and pd.isna(raw_knowledge)):
        return ""

    text = str(raw_knowledge)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            contexts = parsed.get('contexts', "")
            if isinstance(contexts, list):
                return "\n".join(str(item) for item in contexts)
            return str(contexts)
        return str(parsed)
    except Exception:
        legacy_match = re.search(r"\{'contexts':\s*array\((.*),\s*dtype=.*\)\}\s*$", text, flags=re.DOTALL)
        if legacy_match:
            try:
                arr_like = ast.literal_eval(legacy_match.group(1).strip())
                if isinstance(arr_like, list):
                    return "\n".join(str(item) for item in arr_like)
                return str(arr_like)
            except Exception:
                return text
        return text

# ---------------------
# METRICS CALCULATION
# ---------------------
def calculate_metrics(answer_list, llm_answers, df, model_name, use_knowledge):
    # Parse llm_answers into integers (0, 1, 2)
    llm_answers_int = []
    for i in llm_answers:
        i_lower = i.lower()
        if any(x in i_lower for x in ['1', 'not', 'non']):
            llm_answers_int.append(1)
        elif any(x in i_lower for x in ['not sure', 'pass', 'skip', '2']):
            llm_answers_int.append(2)
        else:
            llm_answers_int.append(0)

    answer_int = [int(i) for i in answer_list]

    df['llm_answers_int'] = llm_answers_int
    df['answer_int'] = answer_int
    df['Decision'] = [
        'Correct' if llm_answers_int[i] == answer_int[i]
        else 'Not Sure' if llm_answers_int[i] == 2
        else 'Incorrect'
        for i in range(len(llm_answers_int))
    ]

    # Difficulty-level metrics (filter out "Not Sure" predictions for metric calculation)
    difficulty_indices = {
        'easy':   [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'easy'],
        'medium': [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'medium'],
        'hard':   [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'hard']
    }

    metrics = {}
    for difficulty in ['easy', 'medium', 'hard']:
        indices = difficulty_indices[difficulty]
        if not indices:
            metrics[f'{difficulty}_accuracy']  = None
            metrics[f'{difficulty}_precision'] = None
            metrics[f'{difficulty}_recall']    = None
            metrics[f'{difficulty}_f1']        = None
            metrics[f'{difficulty}_percent_of_time_not_sure_chosen'] = None
            continue

        diff_answers = [answer_int[i] for i in indices]
        diff_llm = [llm_answers_int[i] for i in indices]

        # Calculate percentage of "Not Sure" responses
        not_sure_count = sum(1 for ans in diff_llm if ans == 2)
        metrics[f'{difficulty}_percent_of_time_not_sure_chosen'] = (not_sure_count / len(diff_llm))

        # Filter out cases where the model was not sure (i.e., where prediction is 2)
        valid_idx = [j for j, pred in enumerate(diff_llm) if pred != 2]
        if valid_idx:
            filtered_true = [diff_answers[j] for j in valid_idx]
            filtered_pred = [diff_llm[j] for j in valid_idx]
            metrics[f'{difficulty}_accuracy']  = accuracy_score(filtered_true, filtered_pred)
            metrics[f'{difficulty}_precision'] = precision_score(filtered_true, filtered_pred, zero_division=0)
            metrics[f'{difficulty}_recall']    = recall_score(filtered_true, filtered_pred, zero_division=0)
            metrics[f'{difficulty}_f1']        = f1_score(filtered_true, filtered_pred, zero_division=0)
        else:
            metrics[f'{difficulty}_accuracy']  = None
            metrics[f'{difficulty}_precision'] = None
            metrics[f'{difficulty}_recall']    = None
            metrics[f'{difficulty}_f1']        = None

    # Overall metrics (filtering out "Not Sure" predictions)
    valid_indices = [i for i, ans in enumerate(llm_answers_int) if ans != 2]
    if valid_indices:
        filtered_answers = [answer_int[i] for i in valid_indices]
        filtered_llm = [llm_answers_int[i] for i in valid_indices]
        metrics.update({
            'Model Name': model_name['model_name'],
            'Knowledge': 'Yes' if use_knowledge else 'No',
            'precision': precision_score(filtered_answers, filtered_llm, zero_division=0),
            'recall':    recall_score(filtered_answers, filtered_llm, zero_division=0),
            'f1':        f1_score(filtered_answers, filtered_llm, zero_division=0)
        })
    else:
        metrics.update({
            'Model Name': model_name['model_name'],
            'Knowledge': 'Yes' if use_knowledge else 'No',
            'precision': 0,
            'recall': 0,
            'f1': 0
        })

    not_sure_total = sum(1 for ans in llm_answers_int if ans == 2)
    metrics['overall_percent_of_time_not_sure_chosen'] = (not_sure_total / len(llm_answers_int)) if llm_answers_int else None

    return pd.DataFrame([metrics])


def truncate_numeric_values(df, digits=3):
    factor = 10 ** digits
    truncated_df = df.copy()
    numeric_cols = truncated_df.select_dtypes(include=['float', 'float32', 'float64']).columns
    if len(numeric_cols) > 0:
        truncated_df[numeric_cols] = truncated_df[numeric_cols].apply(
            lambda col: np.trunc(col * factor) / factor
        )
    return truncated_df

# ---------------------
# EVALUATION FUNCTION
# ---------------------
def run_evaluation(model_name, df, use_knowledge=False, openai_api_key=""):
    chosen_answer_indices = []
    prompts = []
    answer_list = []   # ground-truth 0 or 1 for which answer is chosen

    for i in range(len(df)):
        question = df.loc[i, 'question']
        ground_truth = df.loc[i, 'ground_truth']
        hallucinated_answer = df.loc[i, 'least_similar_answer']

        if use_knowledge:
            knowledge = parse_knowledge_field(df.loc[i, 'knowledge'])
        else:
            knowledge = None

        answers = [ground_truth, hallucinated_answer]
        random_val = random.randint(0, 1)
        chosen = answers[random_val]
        answer_list.append(random_val)

        if use_knowledge:
            user_prompt = create_prompt_withknowledge(question, chosen, knowledge)
        else:
            user_prompt = create_prompt(question, chosen)

        prompt_chat = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{system_prompt} {user_prompt}"},
        ]
        prompts.append(prompt_chat)

    llm_answers = []

    if model_name['type'] == 'hf':
        # Initialize vLLM model
        llm = LLM(
            model=model_name['model_name'],
            tensor_parallel_size=1,  # adjust based on your GPU setup
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype=torch.bfloat16,
        )
        tokenizer = llm.get_tokenizer()

        # Determine stop token IDs
        stop_tok_id = []
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
            temperature=0.3,
            top_p=0.95,
            max_tokens=512,
            stop_token_ids=stop_tok_id
        )

        # Format prompts for batch generation
        batch_formatted_prompts = []
        for chat_prompt in prompts:
            formatted_prompt = tokenizer.apply_chat_template(
                chat_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            batch_formatted_prompts.append(formatted_prompt)

        outputs = llm.generate(batch_formatted_prompts, sampling_params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            llm_answers.append(text)

        # Explicitly delete the model objects to free GPU memory
        del llm, tokenizer

    else:
        # Example for OpenAI calls (if needed)
        from openai import OpenAI

        if not openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI models. Pass --openai-api-key.")

        client = OpenAI(api_key=openai_api_key)
        progress_desc = f"OpenAI requests ({model_name['model_name']}, knowledge={use_knowledge})"
        for chat_prompt in tqdm(prompts, desc=progress_desc, leave=False):
            response = client.chat.completions.create(model=model_name['model_name'],
            messages=chat_prompt,
            max_completion_tokens=4,
            n=1)
            content = response.choices[0].message.content.strip()
            llm_answers.append(content)

    result_df = calculate_metrics(answer_list, llm_answers, df, model_name, use_knowledge)
    return result_df

# ---------------------
# FUNCTION TO RUN ONE EVALUATION IN A SUBPROCESS
# ---------------------
def evaluate_model_subprocess(model_name, use_knowledge, df_path, csv_path, openai_api_key):
    try:
        # Each subprocess loads its own copy of the data
        df = pd.read_csv(df_path)
        print(f"Running {model_name['model_name']} with knowledge = {use_knowledge}")
        result = run_evaluation(model_name, df, use_knowledge, openai_api_key)
        result = truncate_numeric_values(result, digits=3)
        # Append results to CSV (create file with header if it does not exist)
        if os.path.exists(csv_path):
            result.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            result.to_csv(csv_path, mode='w', header=True, index=False)
    except Exception as e:
        print(f"Error evaluating {model_name['model_name']} with knowledge={use_knowledge}: {e}")
    finally:
        clear_gpu_memory()

# ---------------------
# MAIN FUNCTION
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection models with vLLM.")
    parser.add_argument("--df-path", help="Path to input CSV dataset.")
    parser.add_argument(
        "--hf-dataset-dir",
        help="Path to local Hugging Face dataset saved by load_from_disk/save_to_disk (e.g., Detection/MedHallu_pqa_labeled).",
    )
    parser.add_argument(
        "--prepared-csv-path",
        default="prepared_detection_input.csv",
        help="Where to save prepared CSV when using --hf-dataset-dir.",
    )
    parser.add_argument("--csv-path", default="detection_results.csv", help="Path to output CSV results.")
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key (required when evaluating OpenAI models).",
    )
    args = parser.parse_args()

    df_path = args.df_path.strip() if args.df_path else ""
    hf_dataset_dir = args.hf_dataset_dir.strip() if args.hf_dataset_dir else ""
    prepared_csv_path = args.prepared_csv_path.strip()
    csv_path = args.csv_path.strip()
    openai_api_key = args.openai_api_key.strip()

    if not df_path and not hf_dataset_dir:
        raise ValueError("Provide either --df-path or --hf-dataset-dir.")

    if hf_dataset_dir:
        if not os.path.isdir(hf_dataset_dir):
            raise FileNotFoundError(f"Hugging Face dataset directory not found: {hf_dataset_dir}")
        if not prepared_csv_path:
            raise ValueError("--prepared-csv-path cannot be empty when using --hf-dataset-dir.")
        prepared_df = prepare_df_from_hf_dataset(hf_dataset_dir)
        prepared_df.to_csv(prepared_csv_path, index=False)
        df_path = prepared_csv_path
        print(f"Prepared input CSV from local dataset: {df_path}")

    if not df_path:
        raise ValueError("--df-path cannot be empty.")
    if not os.path.isfile(df_path):
        raise FileNotFoundError(f"Input dataset not found: {df_path}")
    if not csv_path:
        raise ValueError("--csv-path cannot be empty.")

    requires_openai_key = any(model_name.get('type') == 'openai' for model_name in model_names)
    if requires_openai_key and not openai_api_key:
        raise ValueError("This run includes OpenAI models. Please provide --openai-api-key.")

    # For each model, run evaluation without knowledge and with knowledge sequentially in separate processes.
    ctx = multiprocessing.get_context("spawn")
    for model_name in model_names:
        for use_knowledge in [False, True]:
            proc = ctx.Process(
                target=evaluate_model_subprocess,
                args=(model_name, use_knowledge, df_path, csv_path, openai_api_key)
            )
            proc.start()
            proc.join()  # Wait for the subprocess to finish before moving on
            print(f"Completed {model_name['model_name']} with knowledge = {use_knowledge}\n")

    print(f"All results saved to {csv_path}")

if __name__ == "__main__":
    main()
