import copy
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
from transformers import GenerationConfig, pipeline, set_seed
import multiprocessing
from dataclasses import dataclass
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
GENERATOR_MODEL_ID = "Qwen/Qwen3.5-9B"
# TEMPERATURE = 0.8
GENERATOR_TEMPERATURE = 0.8
DISCRIMINATOR_TEMPERATURE = 0.3
TOP_P = 0.95
MAX_NEW_TOKENS = 512
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
OUTPUT_FILE = os.path.join(RESULTS_DIR, "medhallu_output.csv")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "medhallu_checkpoint.csv")

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
        model_id="Qwen/Qwen3-4B",
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
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
        help="Number of hallucination attempts to generate per question.",
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
        default="",
        help="OpenAI API key for using OpenAI models as discriminators.",
    )
    
    return parser.parse_args()


def configure_runtime_randomness(seed: int, deterministic: bool) -> None:
    """Seed local RNGs and enable deterministic kernels when requested."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE)

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
            default_generation_config = GenerationConfig.from_model_config(self.pipe.model.config)
            default_generation_config.max_length = None
            self.pipe.model.generation_config = default_generation_config
            self.pipe.generation_config = default_generation_config
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

    def _should_disable_thinking(self) -> bool:
        model_id = self.config.model_id.lower()
        return model_id.startswith("qwen/qwen3")

    def generate(self, messages: Union[str, List[Dict[str, str]]], max_new_tokens: int = None) -> str:
        try:
            if self.config.model_type == ModelType.OPENAI:
                return self._generate_openai(messages)
            else:
                return self._generate_hf(messages, max_new_tokens)
        except Exception as e:
            print(f"Error in generate for {self.config.model_id}: {e}")
            return ""

    @staticmethod
    def _get_response_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _prepare_openai_request(
        self,
        messages: Union[str, List[Dict[str, str]]],
    ) -> Tuple[Optional[str], Union[str, List[Dict[str, str]]]]:
        if isinstance(messages, str):
            return None, messages

        instructions = None
        normalized_messages: List[Dict[str, str]] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")

            if role in {"system", "developer"} and instructions is None:
                instructions = str(content)
                continue

            if role == "system":
                role = "developer"

            normalized_messages.append({
                "role": role,
                "content": content,
            })

        return instructions, normalized_messages

    def _extract_openai_response_text(self, response: Any) -> str:
        output_text = self._get_response_value(response, "output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        text_parts = []
        for output_item in self._get_response_value(response, "output", []) or []:
            if self._get_response_value(output_item, "type") != "message":
                continue

            for content_item in self._get_response_value(output_item, "content", []) or []:
                if self._get_response_value(content_item, "type") != "output_text":
                    continue

                text = self._get_response_value(content_item, "text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        if text_parts:
            return "\n".join(text_parts)

        print(
            "OpenAI response contained no text output "
            f"(status={self._get_response_value(response, 'status')}, "
            f"error={self._get_response_value(response, 'error')}, "
            f"incomplete_details={self._get_response_value(response, 'incomplete_details')})"
        )
        return ""

    def _openai_reasoning_config(self) -> Optional[Dict[str, str]]:
        model_id = self.config.model_id.lower()
        if model_id.startswith("gpt-5") or model_id.startswith("o"):
            return {"effort": "minimal"}
        return None

    def _create_openai_response(
        self,
        instructions: Optional[str],
        input_payload: Union[str, List[Dict[str, str]]],
        max_output_tokens: int,
    ) -> Any:
        request_kwargs: Dict[str, Any] = {
            "model": self.config.model_id,
            "instructions": instructions,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
            "text": {"verbosity": "low"},
        }

        reasoning = self._openai_reasoning_config()
        if reasoning is not None:
            request_kwargs["reasoning"] = reasoning

        return self.client.responses.create(**request_kwargs)

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self.client is None:
                raise ValueError("OpenAI client is not configured. Pass --openai-key.")

            instructions, input_payload = self._prepare_openai_request(messages)
            response = self._create_openai_response(
                instructions=instructions,
                input_payload=input_payload,
                max_output_tokens=self.config.max_tokens,
            )
            text = self._extract_openai_response_text(response)
            if text:
                return text

            incomplete_details = self._get_response_value(response, "incomplete_details")
            incomplete_reason = self._get_response_value(incomplete_details, "reason")
            if incomplete_reason == "max_output_tokens":
                retry_max_tokens = max(self.config.max_tokens * 4, 256)
                print(
                    f"Retrying OpenAI generation for {self.config.model_id} "
                    f"with max_output_tokens={retry_max_tokens}"
                )
                retry_response = self._create_openai_response(
                    instructions=instructions,
                    input_payload=input_payload,
                    max_output_tokens=retry_max_tokens,
                )
                return self._extract_openai_response_text(retry_response)

            return ""
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""
        
    def _generate_hf(self, messages: Union[str, List[Dict[str, str]]], 
                     max_new_tokens: int = None) -> str:
        try:
            max_new_tokens = max_new_tokens if max_new_tokens else self.config.max_tokens
            generation_config = copy.deepcopy(self.pipe.model.generation_config)
            generation_config.max_new_tokens = max_new_tokens
            generation_config.max_length = None
            generation_config.eos_token_id = self.terminators
            generation_config.do_sample = self.config.do_sample
            if self.config.do_sample:
                generation_config.temperature = self.config.temperature
                generation_config.top_p = self.config.top_p
            else:
                generation_config.temperature = None
                generation_config.top_p = None

            prompt_input = messages
            if self._should_disable_thinking() and isinstance(messages, list):
                try:
                    prompt_input = self.pipe.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt_input = self.pipe.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            outputs = self.pipe(
                prompt_input,
                generation_config=generation_config,
                return_full_text=False,
            )
            
            # HuggingFace pipeline returns a list of dictionaries with 'generated_text' key
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            return ""
            
        except Exception as e:
            print(f"Error in HuggingFace generation: {e}")
            return ""


class TextGradOptimizer(tg.EngineLM):
    """Encapsulate TextGrad engine setup and hallucination optimization."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(self, openai_api_key: str = "", model_string: str = "gpt-5-mini"):
        self.openai_api_key = openai_api_key
        self.model_string = model_string
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.client = None

        cleaned_api_key = openai_api_key.strip() if openai_api_key else ""
        if cleaned_api_key:
            self.client = OpenAI(api_key=cleaned_api_key)

    def _create_engine(self) -> tg.EngineLM:
        if self.openai_api_key and self.openai_api_key.strip():
            return self
        return tg.get_engine("gpt-4o-mini")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        if self.client is None:
            raise ValueError("An OpenAI API key is required for TextGrad optimization.")

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

    def _setup_optimization(
        self,
        failed_text: str,
        question: str,
        knowledge: str,
    ) -> tg.Variable:
        """Configure TextGrad state for a failed hallucination attempt."""
        live_textgrad_engine = self._create_engine()
        tg.set_backward_engine(live_textgrad_engine, override=True)

        failed_attempt = tg.Variable(
            failed_text,
            requires_grad=True,
            role_description="hallucinated answer that needs improvement"
        )

        tg.Variable(
            f"Question: {question}\nKnowledge Context: {knowledge}",
            requires_grad=False,
            role_description="context for hallucination generation"
        )

        return failed_attempt

    def create_loss(self, question: str) -> tg.TextLoss:
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

    def improve_hallucination(
        self,
        failed_text: str,
        question: str,
        knowledge: str,
    ) -> str:
        """Improve a failed hallucination attempt with TextGrad."""
        try:
            failed_attempt = self._setup_optimization(
                failed_text,
                question,
                knowledge,
            )

            optimizer = tg.TGD(parameters=[failed_attempt])
            loss_fn = self.create_loss(question)

            for _ in range(3):
                loss = loss_fn(failed_attempt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            return failed_attempt.value

        except Exception as e:
            print(f"Error in TextGrad optimization: {e}")
            return failed_text


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

# option1: GT answer, option2: hallucinated answer + justification.
# Check if the LLM discriminator prefers the hallucinated answer or the ground truth answer, 
# given the question and knowledge context.
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

def create_empty_results_df(
    num_generations: int,
    discriminator_configs: List[ModelConfig],
) -> pd.DataFrame:
    """Create an empty DataFrame with the required columns including per-discriminator results."""
    columns = ['question', 'knowledge', 'ground_truth']
    
    for i in range(num_generations):
        columns.extend([
            f'hallucinated_answer_{i+1}', 
            f'justification_{i+1}'
        ])
        for j, config in enumerate(discriminator_configs):
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

def save_checkpoint(df: pd.DataFrame, checkpoint_path: str):
    """Save the current state of the DataFrame to a checkpoint file."""
    try:
        ensure_parent_directory(checkpoint_path)
        df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        
# ============ Model Initialization ============
def initialize_models(openai_api_key: str = "", hf_model_id: str = GENERATOR_MODEL_ID):
    """Initialize all required models."""
    try:
        cleaned_api_key = openai_api_key.strip() if openai_api_key else ""

        # Initialize OpenAI client
        requires_openai = any(
            config.model_type == ModelType.OPENAI for config in DISCRIMINATOR_CONFIGS
        )
        if requires_openai and not cleaned_api_key:
            raise ValueError("This run includes OpenAI discriminators. Please provide --openai-key.")

        openai_client = OpenAI(api_key=cleaned_api_key) if cleaned_api_key else None
        
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
    
def check_hallu_multiple(
    answer: str, 
    discriminator_wrappers: List[LLMWrapper],
    detection_system_prompt: str,
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
                        {"role": "system", "content": detection_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": f"{detection_system_prompt}\n\nUser: {prompt}"}]
                
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
    num_generations: int,
    system_prompt: str,
    detection_system_prompt: str,
    textgrad_optimizer: TextGradOptimizer,
) -> Tuple[List[Dict], Optional[str], DifficultyLevel]:
    """Generate and evaluate hallucinations with TextGrad-based improvement."""
    hallucinations = []
    current_system_prompt = system_prompt
    best_difficulty = DifficultyLevel.EASY
    attempts_on_current_datapoint = 0
    
    while attempts_on_current_datapoint < num_generations:
        try:
            print(f"\nAttempt {attempts_on_current_datapoint + 1}/{num_generations} for current datapoint")
            
            # 1. Fresh generation for each attempt
            # knowledge: 'context' in the row.
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
                long_answer,
                discriminator_wrappers,
                detection_system_prompt,
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
                improved_text = textgrad_optimizer.improve_hallucination(
                    long_answer,
                    question,
                    knowledge,
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
                    improved_long_answer,
                    discriminator_wrappers,
                    detection_system_prompt,
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

    system_prompt = load_prompt("./Prompts/system_prompt_medical.txt")
    detection_system_prompt = load_prompt("./Prompts/system_prompt_detection.txt")
    batch_size = args.batch_size
    num_generations = args.num_generations
    output_file = os.path.abspath(args.output_file)
    checkpoint_file = os.path.abspath(args.checkpoint_file)
    random_seed = args.seed if args.seed is not None else (0 if args.deterministic else None)
    configure_model_sampling(args.deterministic)
    if random_seed is not None:
        configure_runtime_randomness(random_seed, args.deterministic)

    ensure_parent_directory(output_file)
    ensure_parent_directory(checkpoint_file)

    if args.deterministic:
        print(
            f"Deterministic mode enabled with seed {random_seed}. "
            "Hugging Face sampling is disabled, local RNGs are seeded, and "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', CUBLAS_WORKSPACE_CONFIG_VALUE)}. "
            "External API calls are unchanged."
        )

    print("Initializing models...")
    generator_wrapper, discriminator_wrappers, sent_model = initialize_models(
        openai_api_key=args.openai_key,
    )
    textgrad_optimizer = TextGradOptimizer(openai_api_key=args.openai_key)
    
    results_df = create_empty_results_df(num_generations, DISCRIMINATOR_CONFIGS)
    
    print("Loading dataset...")
    df = load_generation_dataset(args.medqa_json_path, args.medqa_split, batch_size)
    
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
                num_generations,
                system_prompt,
                detection_system_prompt,
                textgrad_optimizer,
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
            save_checkpoint(results_df, checkpoint_file)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue

    results_df.to_csv(output_file, index=False)
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    main()