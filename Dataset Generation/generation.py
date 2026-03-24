import os
import random
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
import openai
from openai import OpenAI
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
import re
import torch
from transformers import pipeline, set_seed
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
import textgrad as tg
multiprocessing.set_start_method('spawn', force=True)

# ============ Model Type Enums and Configs ============
class ModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class ModelConfig:
    model_type: ModelType
    model_id: str
    temperature: float = 0.3
    max_tokens: int = 64
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    top_p: float = 0.9
    do_sample: bool = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ============ Hyperparameters and Configuration ============
NUM_GENERATIONS = 5
BATCH_SIZE = 9000
GENERATOR_MODEL_ID = "Qwen/Qwen3.5-27B"
# TEMPERATURE = 0.8
GENERATOR_TEMPERATURE = 0.8
DISCRIMINATOR_TEMPERATURE = 0.3
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
OUTPUT_FILE = os.path.join(RESULTS_DIR, "medhallu_output.csv")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "medhallu_checkpoint.csv")
DEFAULT_TEXTGRAD_OPTIMIZATIONS_FILE = os.path.join(
    SCRIPT_DIR,
    "logs",
    "2026-03-17_17-40-27.optimizations.jsonl",
)
TEXTGRAD_REPLAY_ENGINE = None
TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION: Dict[str, List[str]] = defaultdict(list)
TEXTGRAD_OPTIMIZATION_CURSOR_BY_QUESTION: Dict[str, int] = defaultdict(int)
DETERMINISTIC_MODE = False
RANDOM_SEED: Optional[int] = None

GENERATOR_CONFIG = ModelConfig(
    model_type=ModelType.HUGGINGFACE,
    model_id=GENERATOR_MODEL_ID,
    temperature=GENERATOR_TEMPERATURE,
    max_tokens=MAX_NEW_TOKENS,
    top_p=TOP_P
)

# Discriminator configurations
DISCRIMINATOR_CONFIGS = [
    ModelConfig(
        model_type=ModelType.OPENAI, 
        model_id="gpt-5-mini",
        temperature=DISCRIMINATOR_TEMPERATURE
    ),
    ModelConfig(
        model_type=ModelType.HUGGINGFACE, 
        model_id="google/gemma-3-4b-it",
        temperature=DISCRIMINATOR_TEMPERATURE,
        top_p=TOP_P,
        max_tokens = 4,
    ),
    ModelConfig(
        model_type=ModelType.HUGGINGFACE, 
        model_id="Qwen/Qwen3.5-9B",
        temperature=DISCRIMINATOR_TEMPERATURE,
        top_p=TOP_P,
        max_tokens = 4
    )
]

# ============ System Prompts ============
def load_prompt(file_path):
    """Load system prompt from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read().strip()
        return system_prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found at: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading system prompt file: {str(e)}")


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments."""
    parser = argparse.ArgumentParser(description="Generate MedHallu samples from PubMedQA or local MedQA JSON.")
    parser.add_argument(
        "--medqa-json-path",
        type=str,
        default=None,
        help="Optional local path to a MedQA-style JSON file.",
    )
    parser.add_argument(
        "--medqa-split",
        type=str,
        default='test',
        help="Optional split value from the MedQA JSON to keep, for example 'test'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Maximum number of examples to process.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=OUTPUT_FILE,
        help="Path to the final generated CSV file.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=CHECKPOINT_FILE,
        help="Path to the checkpoint CSV file written during generation.",
    )
    parser.add_argument(
        "--textgrad-replay-log",
        type=str,
        default=None,
        help="Optional TextGrad JSONL log file to replay optimization calls without contacting the TextGrad model.",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool, nargs='?', const=True, default=True, 
        help="Disable sampling for local Hugging Face models and seed local RNGs. External API calls are unchanged.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. If --deterministic is set without a seed, seed 0 is used.",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=" ",  # Placeholder for OpenAI API key
        help="OpenAI API key for using OpenAI models as discriminators.",
    )
    
    return parser.parse_args()


def configure_runtime_randomness(seed: int, deterministic: bool) -> None:
    """Seed local RNGs and enable deterministic kernels when requested."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def configure_model_sampling(deterministic: bool) -> None:
    """Apply runtime sampling settings to generator and discriminator configs."""
    GENERATOR_CONFIG.temperature = 0.0 if deterministic else GENERATOR_TEMPERATURE
    GENERATOR_CONFIG.top_p = 1.0 if deterministic else TOP_P
    GENERATOR_CONFIG.do_sample = not deterministic

    for config in DISCRIMINATOR_CONFIGS:
        if config.model_type == ModelType.HUGGINGFACE:
            config.temperature = 0.0 if deterministic else DISCRIMINATOR_TEMPERATURE
            config.top_p = 1.0 if deterministic else TOP_P
            config.do_sample = not deterministic
        else:
            config.temperature = DISCRIMINATOR_TEMPERATURE
            config.top_p = TOP_P
            config.do_sample = True


def _extract_text_field(value: Any) -> str:
    """Extract text from a plain string or nested {'text': ...} object."""
    if isinstance(value, dict):
        value = value.get("text", "")
    if value is None:
        return ""
    return str(value).strip()


def normalize_question_key(question: Optional[str]) -> str:
    """Normalize a question string for cache lookups."""
    if question is None:
        return ""
    return re.sub(r"\s+", " ", str(question)).strip().lower()


def load_textgrad_optimization_results(file_path: str) -> Dict[str, List[str]]:
    """Load extracted TextGrad optimization outputs keyed by normalized question."""
    results_by_question: Dict[str, List[str]] = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_key = normalize_question_key(record.get("question"))
            optimized_text = _extract_text_field(record.get("final_updated_text"))
            if question_key and optimized_text:
                results_by_question[question_key].append(optimized_text)

    return results_by_question


def get_cached_textgrad_result(question: str) -> Optional[str]:
    """Return the next cached TextGrad optimization result for a question, if available."""
    question_key = normalize_question_key(question)
    cached_results = TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION.get(question_key)
    if not cached_results:
        return None

    cursor = TEXTGRAD_OPTIMIZATION_CURSOR_BY_QUESTION[question_key]
    if cursor >= len(cached_results):
        return None

    TEXTGRAD_OPTIMIZATION_CURSOR_BY_QUESTION[question_key] += 1
    question_excerpt = question.strip().replace("\n", " ")[:120]
    print(
        f"Using cached TextGrad optimization {cursor + 1}/{len(cached_results)} "
        f"for question: {question_excerpt}"
    )
    return cached_results[cursor]


def _extract_optional_context_text(example: Dict[str, Any]) -> str:
    """Extract a best-effort context field without leaking the gold answer."""
    candidate_values = [
        example.get("context"),
        example.get("knowledge"),
        example.get("evidence"),
        example.get("support"),
        example.get("explanation"),
        example.get("rationale"),
    ]

    input_value = example.get("input")
    if isinstance(input_value, dict):
        candidate_values.extend([
            input_value.get("context"),
            input_value.get("knowledge"),
            input_value.get("evidence"),
            input_value.get("support"),
        ])

    for value in candidate_values:
        if isinstance(value, list):
            joined_value = "\n".join(
                text for text in (_extract_text_field(item) for item in value) if text
            )
            if joined_value:
                return joined_value
            continue

        text = _extract_text_field(value)
        if text:
            return text

    return ""


def _extract_correct_reference_text(references: Any) -> str:
    """Return the tagged correct answer text, or the first reference as a fallback."""
    if not isinstance(references, list):
        return ""

    fallback_text = ""
    for reference in references:
        if not isinstance(reference, dict):
            continue

        reference_text = _extract_text_field(reference.get("output"))
        if reference_text and not fallback_text:
            fallback_text = reference_text

        tags = reference.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        if any(str(tag).lower() == "correct" for tag in tags):
            return reference_text

    return fallback_text


def load_medqa_json_dataset(file_path: str, split_name: Optional[str] = None, limit: Optional[int] = None) -> Dataset:
    """Load a local MedQA JSON file and adapt it to the columns expected by generation.py."""
    raw_dataset = load_dataset("json", data_files={"train": file_path}, split="train")

    adapted_rows = []
    for example in raw_dataset:
        example_split = example.get("split")
        if split_name and example_split != split_name:
            continue

        question = _extract_text_field(example.get("input"))
        context = _extract_optional_context_text(example)
        correct_answer = _extract_correct_reference_text(example.get("references"))

        if not question or not correct_answer:
            continue

        adapted_rows.append({
            "question": question,
            "context": context,
            "long_answer": correct_answer,
        })

        if limit is not None and len(adapted_rows) >= limit:
            break

    if not adapted_rows:
        split_message = f" for split '{split_name}'" if split_name else ""
        raise ValueError(f"No valid MedQA records were found in {file_path}{split_message}.")

    return Dataset.from_list(adapted_rows)


def load_generation_dataset(medqa_json_path: Optional[str], medqa_split: Optional[str], batch_size: int):
    """Load the source dataset in the flat schema expected by the generator."""
    if medqa_json_path:
        print(f"Loading MedQA JSON from {medqa_json_path}...")
        return load_medqa_json_dataset(
            medqa_json_path,
            split_name=medqa_split,
            limit=batch_size,
        )

    print("Loading PubMedQA from the Hugging Face Hub...")
    return load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{batch_size}]")

# ============ LLM Wrapper Class ============
class LLMWrapper:
    def __init__(self, config: ModelConfig, openai_client=None):
        self.config = config
        self.client = openai_client
        self.pipe = None
        self.terminators = None
        
        if config.model_type == ModelType.HUGGINGFACE:
            self._setup_hf_pipeline()
    
    def _setup_hf_pipeline(self):
        try:
            self.pipe = pipeline(
                'text-generation',
                model=self.config.model_id,
                tokenizer=self.config.model_id,
                model_kwargs={"torch_dtype": self.config.torch_dtype},
                device_map="auto"
            )
            if "llama" in self.config.model_id.lower():
                self.terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
            elif "gemma" in self.config.model_id.lower():
                self.terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ]
            else:
                self.terminators = [self.pipe.tokenizer.eos_token_id]
        except Exception as e:
            print(f"Error setting up HF pipeline for {self.config.model_id}: {e}")
            raise

    def generate(self, messages: Union[str, List[Dict[str, str]]], max_new_tokens: int = None) -> str:
        try:
            if self.config.model_type == ModelType.OPENAI:
                return self._generate_openai(messages)
            else:
                return self._generate_hf(messages, max_new_tokens)
        except Exception as e:
            print(f"Error in generate for {self.config.model_id}: {e}")
            return ""

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.responses.create(
                model=self.config.model_id,
                input=messages,
                max_output_tokens=self.config.max_tokens,
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""
        
    def _generate_hf(self, messages: str, max_new_tokens: int = None) -> str:
        try:
            tokens = max_new_tokens if max_new_tokens else self.config.max_tokens
            generation_kwargs = {
                "max_new_tokens": tokens,
                "eos_token_id": self.terminators,
                "return_full_text": False,
                "do_sample": self.config.do_sample,
            }
            if self.config.do_sample:
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p

            outputs = self.pipe(messages, **generation_kwargs)
            
            # HuggingFace pipeline returns a list of dictionaries with 'generated_text' key
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            return ""
            
        except Exception as e:
            print(f"Error in HuggingFace generation: {e}")
            return ""


class TextGradOpenAIEngine(tg.EngineLM):
    """TextGrad engine wrapper that uses an explicit OpenAI API key."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        api_key: str,
        model_string: str = "gpt-5-mini",
    ):
        cleaned_api_key = api_key.strip() if api_key else ""
        if not cleaned_api_key:
            raise ValueError("An OpenAI API key is required for TextGrad optimization.")

        self.client = OpenAI(api_key=cleaned_api_key)
        self.model_string = model_string
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        response = self.client.responses.create(
            model=self.model_string,
            input=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_tokens,
        )
        return response.output_text.strip()

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


class TextGradReplayEngine(tg.EngineLM):
    """Replay TextGrad model outputs from a JSONL log produced by TextGrad."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(self, log_path: str):
        resolved_log_path = os.path.abspath(log_path)
        if not os.path.exists(resolved_log_path):
            raise FileNotFoundError(f"TextGrad replay log not found: {resolved_log_path}")

        self.log_path = resolved_log_path
        self.model_string = f"textgrad-replay:{os.path.basename(resolved_log_path)}"
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self._responses_by_key: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._keys_by_prompt: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._cursor_by_key: Dict[Tuple[str, str], int] = defaultdict(int)
        self.fallback_engine: Optional[tg.EngineLM] = None
        self.live_fallback_call_count = 0
        self.recorded_call_count = 0
        self._load_log()

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_llm_forward_text(text: str) -> Tuple[str, str, str]:
        query_marker = "\nQuery: "
        response_marker = "\nResponse: "
        if not text.startswith("System:") or query_marker not in text or response_marker not in text:
            raise ValueError("Unsupported TextGrad forward log format.")

        remainder = text[len("System:"):]
        system_prompt, prompt_and_response = remainder.split(query_marker, 1)
        prompt, response = prompt_and_response.rsplit(response_marker, 1)
        return system_prompt, prompt, response

    def _store_response(
        self,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
        *,
        count_recorded: bool = True,
    ) -> None:
        normalized_prompt = self._normalize_text(prompt)
        normalized_system_prompt = self._normalize_text(system_prompt)
        key = (normalized_system_prompt, normalized_prompt)
        self._responses_by_key[key].append(response)
        if key not in self._keys_by_prompt[normalized_prompt]:
            self._keys_by_prompt[normalized_prompt].append(key)
        if count_recorded:
            self.recorded_call_count += 1

    def set_fallback_engine(self, engine: Optional[tg.EngineLM]) -> None:
        self.fallback_engine = engine

    def _load_log(self) -> None:
        pending_backward_prompt: Optional[str] = None
        pending_optimizer_prompt: Optional[str] = None

        with open(self.log_path, 'r', encoding='utf-8') as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = str(record.get('msg') or record.get('message') or '').strip()

                if message == 'LLMCall function forward':
                    llm_text = record.get('text')
                    if not llm_text:
                        continue
                    try:
                        system_prompt, prompt, response = self._parse_llm_forward_text(str(llm_text))
                    except ValueError:
                        continue
                    self._store_response(prompt, response, system_prompt=system_prompt)
                    continue

                if message == '_backward_through_llm prompt':
                    pending_backward_prompt = self._normalize_text(record.get('_backward_through_llm'))
                    continue

                if message == '_backward_through_llm gradient':
                    if pending_backward_prompt:
                        self._store_response(
                            pending_backward_prompt,
                            self._normalize_text(record.get('_backward_through_llm')),
                        )
                        pending_backward_prompt = None
                    continue

                if message == 'TextualGradientDescent prompt for update':
                    pending_optimizer_prompt = self._normalize_text(record.get('prompt'))
                    continue

                if message == 'TextualGradientDescent optimizer response':
                    if pending_optimizer_prompt:
                        self._store_response(
                            pending_optimizer_prompt,
                            self._normalize_text(record.get('optimizer.response')),
                        )
                        pending_optimizer_prompt = None

        if self.recorded_call_count == 0:
            raise ValueError(f"No replayable TextGrad calls were found in {self.log_path}")

    def _resolve_key(self, prompt: str, system_prompt: Optional[str]) -> Tuple[str, str]:
        normalized_prompt = self._normalize_text(prompt)
        normalized_system_prompt = self._normalize_text(system_prompt)
        exact_key = (normalized_system_prompt, normalized_prompt)
        if exact_key in self._responses_by_key:
            return exact_key

        candidate_keys = self._keys_by_prompt.get(normalized_prompt, [])
        if len(candidate_keys) == 1:
            return candidate_keys[0]

        if not candidate_keys:
            raise KeyError(
                "No recorded TextGrad response matched the current replay prompt. "
                f"Prompt excerpt: {normalized_prompt[:160]}"
            )

        raise KeyError(
            "Multiple recorded TextGrad responses matched the current replay prompt. "
            "Add a more specific replay log or re-run with exact prompt parity."
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        try:
            key = self._resolve_key(prompt, system_prompt)
            response_index = self._cursor_by_key[key]
            responses = self._responses_by_key[key]

            if response_index >= len(responses):
                raise RuntimeError(
                    "TextGrad replay log does not contain enough recorded responses for this prompt. "
                    f"Prompt excerpt: {key[1][:160]}"
                )

            self._cursor_by_key[key] += 1
            return responses[response_index]
        except (KeyError, RuntimeError) as replay_error:
            if self.fallback_engine is None:
                raise

            print(
                "TextGrad replay miss. Falling back to live TextGrad engine. "
                f"Reason: {replay_error}"
            )
            response = self.fallback_engine.generate(
                prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
            self._store_response(
                prompt,
                response,
                system_prompt=system_prompt,
                count_recorded=False,
            )
            self.live_fallback_call_count += 1
            return response

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

# ============ Utility Functions ============
def calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate semantic similarity between two texts using sentence transformers."""
    try:
        if not text1 or not text2:
            return 0.0
        
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return float(cosine_similarity.item())
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 0.0
    
def create_prompt(question: str, option1: str, option2: str, justification: str) -> str:
    """Create a prompt for the discriminator model."""
    prompt = f"""
    Question: {question}
    Option 1: {option1}
    Option 2: {option2} + {justification}
    
    Return just the answer, for example: "Option 1" or "Option 2". Don't return anything else, just the answer.
    """
    return prompt

def get_sections(text: str) -> Tuple[str, str, str, str, str]:
    """Extract different sections from the generated text."""
    try:
        pattern = r'#([^#]+)#:\s*([^#]+)'
        matches = re.findall(pattern, text)
        sections = {}
        for label, content in matches:
            sections[label.strip()] = content.strip()
        
        return (sections.get('Question', ''),
                sections.get('Knowledge', ''),
                sections.get('Ground truth answer', ''),
                sections.get('Hallucinated Answer', ''),
                sections.get('Justification of Hallucinated answer', ''))
    except Exception as e:
        print(f"Error parsing sections: {e}")
        return '', '', '', '', ''

def determine_difficulty(discriminator_results: List[bool]) -> DifficultyLevel:
    """Determine the difficulty level based on which discriminators were fooled."""
    num_fooled = sum(discriminator_results)
    
    if num_fooled == len(discriminator_results):
        return DifficultyLevel.HARD
    elif num_fooled > 1:
        return DifficultyLevel.MEDIUM
    else:
        return DifficultyLevel.EASY

def create_empty_results_df() -> pd.DataFrame:
    """Create an empty DataFrame with the required columns including per-discriminator results."""
    columns = ['question', 'knowledge', 'ground_truth']
    
    for i in range(NUM_GENERATIONS):
        columns.extend([
            f'hallucinated_answer_{i+1}', 
            f'justification_{i+1}'
        ])
        for j, config in enumerate(DISCRIMINATOR_CONFIGS):
            columns.append(f'fooled_discriminator_{j+1}_{i+1}')
        columns.append(f'difficulty_level_{i+1}')
    
    columns.append('least_similar_answer')
    columns.append('final_difficulty_level')
    return pd.DataFrame(columns=columns)

def ensure_parent_directory(file_path: str) -> None:
    """Create the parent directory for a file if it does not already exist."""
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

def save_checkpoint(df: pd.DataFrame, filename: Optional[str] = None):
    """Save the current state of the DataFrame to a checkpoint file."""
    try:
        checkpoint_path = filename or CHECKPOINT_FILE
        ensure_parent_directory(checkpoint_path)
        df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        
# ============ Model Initialization ============
def initialize_models(openai_api_key: str = "", hf_model_id: str = GENERATOR_MODEL_ID):
    """Initialize all required models."""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize sentence transformer model
        sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize generator (HuggingFace wrapper)
        # generator_config = ModelConfig(
        #     model_type=ModelType.HUGGINGFACE,
        #     model_id=hf_model_id,
        #     temperature=TEMPERATURE,
        #     top_p=TOP_P
        # )
        
        generator_wrapper = LLMWrapper(GENERATOR_CONFIG)
        
        # Initialize discriminator wrappers
        discriminator_wrappers = []
        for config in DISCRIMINATOR_CONFIGS:
            if config.model_type == ModelType.OPENAI:
                wrapper = LLMWrapper(config, openai_client=openai_client)
            else:
                wrapper = LLMWrapper(config)
            discriminator_wrappers.append(wrapper)
        
        return generator_wrapper, discriminator_wrappers, sent_model
    
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

def setup_textgrad_optimization(
    failed_text: str,
    question: str,
    knowledge: str,
    openai_api_key: str = "",
) -> Tuple[tg.Variable, tg.Variable]:
    """Setup TextGrad optimization for failed hallucination."""
    if openai_api_key and openai_api_key.strip():
        live_textgrad_engine = TextGradOpenAIEngine(
            api_key=openai_api_key,
            model_string="gpt-5-mini",
        )
    else:
        live_textgrad_engine = tg.get_engine("gpt-4o-mini")

    if TEXTGRAD_REPLAY_ENGINE is not None:
        TEXTGRAD_REPLAY_ENGINE.set_fallback_engine(live_textgrad_engine)
        tg.set_backward_engine(TEXTGRAD_REPLAY_ENGINE, override=True)
    else:
        tg.set_backward_engine(live_textgrad_engine, override=True)
    
    # Create variable for the failed attempt
    failed_attempt = tg.Variable(
        failed_text,
        requires_grad=True,
        role_description="hallucinated answer that needs improvement"
    )
    
    # Create context variables
    context = tg.Variable(
        f"Question: {question}\nKnowledge Context: {knowledge}",
        requires_grad=False,
        role_description="context for hallucination generation"
    )
    
    return failed_attempt, context

def create_textgrad_loss(question: str) -> tg.TextLoss:
    """Create a TextGrad loss function for hallucination evaluation."""
    evaluation_instruction = tg.Variable(
        f"""Evaluate the given hallucinated answer for this question: {question}
        Consider:
        1. Plausibility given the context, is it on-topic and relevant?
        2. Internal consistency, does the answer make sense and is it logically coherent?
        3. Use of domain-specific terminology
        4. How easy is it for the discriminator to detect the answer as a hallucination?
        
        Provide specific feedback on how to improve the answer to make it more convincing.
        Be concise and direct.""",
        requires_grad=False,
        role_description="evaluation instructions"
    )
    
    return tg.TextLoss(evaluation_instruction)

def improve_hallucination_with_textgrad(
    failed_text: str,
    question: str,
    knowledge: str,
    openai_api_key: str = "",
) -> str:
    """Improve failed hallucination using TextGrad optimization."""
    try:
        cached_result = get_cached_textgrad_result(question)
        if cached_result is not None:
            return cached_result

        # Setup TextGrad variables
        failed_attempt, context = setup_textgrad_optimization(
            failed_text,
            question,
            knowledge,
            openai_api_key=openai_api_key,
        )
        
        # Create optimizer
        optimizer = tg.TGD(parameters=[failed_attempt])
        
        # Create loss function
        loss_fn = create_textgrad_loss(question)
        
        # Optimization loop
        for _ in range(3):  # Multiple optimization steps
            # Compute loss
            loss = loss_fn(failed_attempt)
            
            # Backward pass
            loss.backward()
            
            # Update the text
            optimizer.step()
            
            # Clear gradients
            optimizer.zero_grad()
        
        return failed_attempt.value
        
    except Exception as e:
        print(f"Error in TextGrad optimization: {e}")
        return failed_text
    
def check_hallu_multiple(
    answer: str, 
    discriminator_wrappers: List[LLMWrapper]
) -> Tuple[List[bool], Optional[str], Optional[str]]:
    """Check if the hallucinated answer can fool multiple discriminators."""
    print("\nChecking hallucination against multiple discriminators...")
    try:
        question, knowledge, ground_truth_answer, hallucinated_answer, justification = get_sections(answer)
        
        if not all([hallucinated_answer, justification]):
            print("Missing required sections in generated text")
            return [False] * len(discriminator_wrappers), None, None
        
        option1 = ground_truth_answer
        option2 = hallucinated_answer
        prompt = create_prompt(question, option1, option2, justification)
        
        results = []
        for i, wrapper in enumerate(discriminator_wrappers):
            try:
                if wrapper.config.model_type == ModelType.OPENAI:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_DETECTION},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT_DETECTION}\n\nUser: {prompt}"}]
                
                pred_answer = wrapper.generate(messages)
                print(f"Discriminator {i+1} answer: {pred_answer}")
                
                results.append("2" in pred_answer.lower())
            except Exception as e:
                print(f"Error with discriminator {i+1}: {e}")
                results.append(False)
        
        return results, hallucinated_answer, justification
    except Exception as e:
        print(f"Error in check_hallu_multiple: {e}")
        return [False] * len(discriminator_wrappers), None, None

def generate_hallucinations(
    question: str, 
    knowledge: str, 
    ground_truth: str, 
    generator_wrapper: LLMWrapper,
    discriminator_wrappers: List[LLMWrapper],
    sent_model: SentenceTransformer,
    openai_api_key: str = "",
) -> Tuple[List[Dict], Optional[str], DifficultyLevel]:
    """Generate and evaluate hallucinations with TextGrad-based improvement."""
    hallucinations = []
    current_system_prompt = SYSTEM_PROMPT
    best_difficulty = DifficultyLevel.EASY
    attempts_on_current_datapoint = 0
    
    while attempts_on_current_datapoint < NUM_GENERATIONS:
        try:
            print(f"\nAttempt {attempts_on_current_datapoint + 1}/{NUM_GENERATIONS} for current datapoint")
            
            # 1. Fresh generation for each attempt
            prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
            
            if generator_wrapper.config.model_type == ModelType.OPENAI:
                messages = [{"role": "system", "content": current_system_prompt},
                          {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": f"{current_system_prompt}\n\n{prompt}"}]
            
            generated_text = generator_wrapper.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
            
            if not generated_text:
                print(f"No text generated in attempt {attempts_on_current_datapoint+1}")
                attempts_on_current_datapoint += 1
                continue
            
            current_text = generated_text
            if generator_wrapper.config.model_type == ModelType.HUGGINGFACE:
                long_answer = f"{prompt} {current_text}"
            else:
                long_answer = current_text
            
            # 2. Check this generation
            discriminator_results, hallucinated_answer, justification = check_hallu_multiple(
                long_answer, discriminator_wrappers
            )
            
            # If any discriminator is fooled, record and break immediately
            if any(discriminator_results):
                difficulty = determine_difficulty(discriminator_results)
                current_hallucination = {
                    'answer': hallucinated_answer if hallucinated_answer else current_text,
                    'justification': justification,
                    'discriminator_results': discriminator_results,
                    'difficulty': difficulty,
                    'attempt_number': attempts_on_current_datapoint + 1
                }
                hallucinations.append(current_hallucination)
                if difficulty.value > best_difficulty.value:
                    best_difficulty = difficulty
                break  # Exit immediately as we've fooled at least one discriminator
            
            # 3. If no discriminator was fooled, try TextGrad improvement
            print(f"Attempt failed. Using TextGrad to improve...")
            try:
                improved_text = improve_hallucination_with_textgrad(
                    long_answer,
                    question,
                    knowledge,
                    openai_api_key=openai_api_key,
                )
                
                # Generate based on TextGrad's improvements
                if generator_wrapper.config.model_type == ModelType.OPENAI:
                    messages = [
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": f"{prompt}\n\nPrevious attempt improved by TextGrad: {improved_text}"}
                    ]
                else:
                    messages = [{"role": "user", "content": f"{current_system_prompt}\n\n{prompt}\n\nPrevious attempt improved by TextGrad: {improved_text}"}]
                
                improved_generated = generator_wrapper.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
                improved_long_answer = improved_generated if generator_wrapper.config.model_type == ModelType.OPENAI else f"{prompt} {improved_generated}"
                
                # Check the improved generation
                discriminator_results, hallucinated_answer, justification = check_hallu_multiple(
                    improved_long_answer, discriminator_wrappers
                )
                
                # If TextGrad improvement fooled any discriminator, record and break
                if any(discriminator_results):
                    difficulty = determine_difficulty(discriminator_results)
                    current_hallucination = {
                        'answer': hallucinated_answer if hallucinated_answer else improved_generated,
                        'justification': justification,
                        'discriminator_results': discriminator_results,
                        'difficulty': difficulty,
                        'attempt_number': attempts_on_current_datapoint + 1
                    }
                    hallucinations.append(current_hallucination)
                    if difficulty.value > best_difficulty.value:
                        best_difficulty = difficulty
                    break  # Exit as TextGrad improvement fooled a discriminator
                    
            except Exception as e:
                print(f"Error in TextGrad optimization: {e}")
            
            # 4. Record failed attempt
            difficulty = determine_difficulty(discriminator_results)
            current_hallucination = {
                'answer': hallucinated_answer if hallucinated_answer else current_text,
                'justification': justification,
                'discriminator_results': discriminator_results,
                'difficulty': difficulty,
                'attempt_number': attempts_on_current_datapoint + 1
            }
            hallucinations.append(current_hallucination)
            
            # Move to next attempt only if we haven't fooled any discriminator
            attempts_on_current_datapoint += 1
                
        except Exception as e:
            print(f"Error in generation attempt {attempts_on_current_datapoint+1}: {e}")
            attempts_on_current_datapoint += 1
            continue
    
    # If we never fooled any discriminator, find least similar answer
    if not any(any(h['discriminator_results']) for h in hallucinations):
        try:
            similarities = [calculate_semantic_similarity(h['answer'], ground_truth, sent_model) 
                          for h in hallucinations if h['answer']]
            if similarities:
                max_index = similarities.index(max(similarities))
                final_answer = hallucinations[max_index]['answer']
            else:
                final_answer = None
        except Exception as e:
            print(f"Error calculating similarities: {e}")
            final_answer = None
    else:
        # Use the first successful hallucination
        successful = next(h for h in hallucinations if any(h['discriminator_results']))
        final_answer = successful['answer']
    
    return hallucinations, final_answer, best_difficulty
    
# ============ Main Function ============
def main():
    """Main execution function."""
    args = parse_args()
    
    # Update global variables based on arguments
    global SYSTEM_PROMPT, SYSTEM_PROMPT_DETECTION, BATCH_SIZE, OUTPUT_FILE, CHECKPOINT_FILE, TEXTGRAD_REPLAY_ENGINE, TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION, TEXTGRAD_OPTIMIZATION_CURSOR_BY_QUESTION, DETERMINISTIC_MODE, RANDOM_SEED
    SYSTEM_PROMPT = load_prompt("./Prompts/system_prompt_medical.txt")
    SYSTEM_PROMPT_DETECTION = load_prompt("./Prompts/system_prompt_detection.txt")
    BATCH_SIZE = args.batch_size
    OUTPUT_FILE = os.path.abspath(args.output_file)
    CHECKPOINT_FILE = os.path.abspath(args.checkpoint_file)
    DETERMINISTIC_MODE = args.deterministic
    RANDOM_SEED = args.seed if args.seed is not None else (0 if args.deterministic else None)
    configure_model_sampling(args.deterministic)
    if RANDOM_SEED is not None:
        configure_runtime_randomness(RANDOM_SEED, args.deterministic)
    TEXTGRAD_REPLAY_ENGINE = None
    TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION = defaultdict(list)
    TEXTGRAD_OPTIMIZATION_CURSOR_BY_QUESTION = defaultdict(int)
    if args.textgrad_replay_log:
        TEXTGRAD_REPLAY_ENGINE = TextGradReplayEngine(args.textgrad_replay_log)
    if args.medqa_json_path and os.path.exists(DEFAULT_TEXTGRAD_OPTIMIZATIONS_FILE):
        TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION = load_textgrad_optimization_results(
            DEFAULT_TEXTGRAD_OPTIMIZATIONS_FILE,
        )
        cached_result_count = sum(
            len(results) for results in TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION.values()
        )
        print(
            f"Loaded cached TextGrad optimizations for "
            f"{len(TEXTGRAD_OPTIMIZATION_RESULTS_BY_QUESTION)} questions "
            f"({cached_result_count} results) from {DEFAULT_TEXTGRAD_OPTIMIZATIONS_FILE}"
        )
    ensure_parent_directory(OUTPUT_FILE)
    ensure_parent_directory(CHECKPOINT_FILE)

    if args.deterministic:
        print(
            f"Deterministic mode enabled with seed {RANDOM_SEED}. "
            "Hugging Face sampling is disabled and local RNGs are seeded. External API calls are unchanged."
        )

    if TEXTGRAD_REPLAY_ENGINE is not None:
        print(
            f"Loaded TextGrad replay log from {TEXTGRAD_REPLAY_ENGINE.log_path} "
            f"with {TEXTGRAD_REPLAY_ENGINE.recorded_call_count} recorded responses. "
            "Unmatched prompts will fall back to live TextGrad."
        )
    
    print("Initializing models...")
    generator_wrapper, discriminator_wrappers, sent_model = initialize_models(
        openai_api_key=args.openai_key,
    )
    
    results_df = create_empty_results_df()
    
    print("Loading dataset...")
    df = load_generation_dataset(args.medqa_json_path, args.medqa_split, BATCH_SIZE)
    
    print(f"Processing {len(df['question'])} questions...")
    for i in tqdm(range(len(df['question']))):
        try:
            question = df['question'][i]
            knowledge = df['context'][i]
            ground_truth = df['long_answer'][i]
            
            print(f"\nProcessing question {i+1}/{len(df['question'])}...")
            hallucinations, least_similar_answer, final_difficulty = generate_hallucinations(
                question, knowledge, ground_truth, 
                generator_wrapper,
                discriminator_wrappers,
                sent_model,
                openai_api_key=args.openai_key,
            )
            
            row_data = {
                'question': question,
                'knowledge': knowledge,
                'ground_truth': ground_truth,
                'least_similar_answer': least_similar_answer,
                'final_difficulty_level': final_difficulty.value
            }
            
            for j, hall in enumerate(hallucinations):
                if hall:
                    row_data[f'hallucinated_answer_{j+1}'] = hall['answer']
                    row_data[f'justification_{j+1}'] = hall['justification']
                    for k, result in enumerate(hall['discriminator_results']):
                        row_data[f'fooled_discriminator_{k+1}_{j+1}'] = result
                    row_data[f'difficulty_level_{j+1}'] = hall['difficulty'].value
                else:
                    row_data[f'hallucinated_answer_{j+1}'] = None
                    row_data[f'justification_{j+1}'] = None
                    for k in range(len(discriminator_wrappers)):
                        row_data[f'fooled_discriminator_{k+1}_{j+1}'] = False
                    row_data[f'difficulty_level_{j+1}'] = DifficultyLevel.EASY.value
            
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            save_checkpoint(results_df)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
    
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()