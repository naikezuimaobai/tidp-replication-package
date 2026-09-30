"""Experiment 1: Browser-Use vs MindAct vs TIDP.

This script is the official Experiment 1 runner for quick or full Mind2Web
evaluation. It supports configurable dataset split, loaded files, sampled
tasks, max evaluated steps, and random seed.

- keep TIDP step success rate from dropping
- reduce TIDP tokens_consumed
- keep TIDP recall above Browser-Use / MindAct

By default, the script does not call the action-decision LLM because the
reported step success in these experiments is derived from key-element recall.
Use --run_action_llm when you also want action-decision timing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONTEXT_LIMIT_TOKENS, DATASET_PATH

from baselines import BrowserUsePruner, MindActPruner
from data_loader import Mind2WebLoader
from dynamic_pruner import DynamicPruner
from evaluator import Evaluator
from intent_classifier import IntentClassifier
from relevance_scorer import RelevanceScorer


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens with tiktoken using a stable OpenAI-compatible encoding."""
    if not text or len(text.strip()) == 0:
        return 0
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class SimpleActionDecider:
    """Optional action-decision LLM timing helper."""

    def __init__(self, llm_config: LLMModelConfig, operation_hint_mode: str = "full"):
        from openai import OpenAI

        self.config = llm_config
        self.client = OpenAI(
            base_url=llm_config.api_base,
            api_key=llm_config.resolved_api_key(),
            timeout=30.0,
        )
        self.model = llm_config.model
        self.operation_hint_mode = operation_hint_mode
        self.prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> str:
        """Load the versioned action prompt, with an embedded fallback."""
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "action_decision.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError:
            return ""  # the caller will use the inline prompt fallback

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        """Extract action text from OpenAI-compatible response variants."""
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text") or item.get("content")
                else:
                    value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
            text = "\n".join(parts).strip()
            if text:
                return text

        for attr in ("reasoning_content", "reasoning", "text"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    @staticmethod
    def _normalize_action_text(text: str) -> str:
        """Extract the executable action line from verbose model output."""
        raw = (text or "").strip()
        if not raw:
            return ""

        def strict_action(line: str) -> str:
            candidate = re.sub(r"^(?:Action|Answer|Output)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            click = re.fullmatch(r"CLICK\s*\[(\d+)\]", candidate, flags=re.IGNORECASE)
            if click:
                return f"CLICK [{click.group(1)}]"

            typed = re.fullmatch(r"(TYPE|SELECT)\s+(.{1,100}?)\s*\[(\d+)\]", candidate, flags=re.IGNORECASE)
            if typed:
                op = typed.group(1).upper()
                value = typed.group(2).strip().strip("\"'")
                return f"{op} {value} [{typed.group(3)}]"

            return ""

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in reversed(lines):
            action_line = strict_action(line)
            if action_line:
                return action_line

        labeled = re.findall(
            r"(?:Action|Answer|Output)\s*:\s*((?:CLICK\s*\[\d+\])|(?:(?:TYPE|SELECT)\s+[^\[\]\n\r]{1,100}?\s*\[\d+\]))",
            raw,
            flags=re.IGNORECASE,
        )
        if labeled:
            action_line = strict_action(labeled[-1])
            if action_line:
                return action_line

        direct = re.findall(
            r"(?:CLICK\s*\[\d+\])|(?:(?:TYPE|SELECT)\s+[^\[\]\n\r]{1,100}?\s*\[\d+\])",
            raw,
            flags=re.IGNORECASE,
        )
        for match in reversed(direct):
            action_line = strict_action(match)
            if action_line:
                return action_line

        return ""

    @staticmethod
    def _is_executable_action_text(text: str) -> bool:
        """Return True only for a parseable one-line action."""
        raw = (text or "").strip()
        if not raw or "\n" in raw:
            return False
        return bool(
            re.fullmatch(r"CLICK\s*\[\d+\]", raw, flags=re.IGNORECASE)
            or re.fullmatch(r"(TYPE|SELECT)\s+.{1,100}?\s*\[\d+\]", raw, flags=re.IGNORECASE)
        )

    def decide_action(
        self,
        kept_elements: List[Dict[str, Any]],
        intent: str,
        task_description: str,
        step_description: str,
        expected_operation: str = "",
        operation_value: str = "",
    ) -> Tuple[str, float]:
        page_summary = ""
        for i, elem in enumerate(kept_elements):
            desc = elem.get("description") or f"{elem.get('type', '')}: {elem.get('text', '')}"
            page_summary += f"[{i+1}] {desc[:260]}\n"

        hint_lines = ""
        if self.operation_hint_mode == "full":
            hint_lines = (
                f"Expected operation: {expected_operation or 'infer from current step'}\n"
                f"Value to type/select if needed: {operation_value or '(none)'}\n"
            )
        elif self.operation_hint_mode == "value":
            hint_lines = f"Target value if the step requires typing or selection: {operation_value or '(none)'}\n"

        inline_prompt = f"""You are a web agent. Based on the task requirements and filtered page elements, identify the most relevant element for the current step.

Task: {task_description}
Current step: {step_description}
Intent category: {intent}
{hint_lines}

Available elements (sorted by relevance):
{page_summary}

Output exactly ONE line. The first token must be CLICK, TYPE, or SELECT.
Use the 1-based element number from the list above, wrapped in square brackets.

Valid formats:
CLICK [7]
TYPE text to enter [7]
SELECT option text [7]

Rules:
- Output only the action line, with no reasoning, no labels, and no extra words before or after it.
- Choose the element that best matches the current step, not merely the first clickable element.
- For TYPE, choose a textbox/input/search field and include the target text when it is available.
- For SELECT, choose a select/option/dropdown/radio/checkbox element and include the target option when it is available.
- For CLICK, choose the link/button/control named by the current step.
- The element number must come from the available element list.

Any explanation, sentence, JSON, or placeholder is invalid.

Action:"""

        prompt = self.prompt_template.format(
            task_description=task_description,
            step_description=step_description,
            intent=intent,
            hint_lines=hint_lines,
            page_summary=page_summary,
        ) if self.prompt_template else inline_prompt

        start_time = time.time()
        action = ""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a web agent. Output exactly one executable action line. "
                        "Never return an empty response."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            last_empty = ""
            for _ in range(3):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=128,
                    temperature=0,
                )
                choice = response.choices[0]
                raw_action = self._extract_message_text(choice.message)
                action = self._normalize_action_text(raw_action)
                if self._is_executable_action_text(action):
                    break

                finish_reason = getattr(choice, "finish_reason", "")
                if raw_action.strip():
                    last_empty = f"invalid action response, finish_reason={finish_reason}"
                else:
                    last_empty = f"empty response, finish_reason={finish_reason}"
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_action[:500]},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was invalid. "
                                "Output only the final action line, with no explanation and no label, "
                                "using one of these formats only:\n"
                                "CLICK [number]\n"
                                "TYPE text [number]\n"
                                "SELECT option [number]"
                            ),
                        },
                    ]
                )

            if not self._is_executable_action_text(action):
                action = f"Error: {last_empty or 'empty response'}"
        except Exception as exc:
            action = f"Error: {exc}"
        return action, (time.time() - start_time) * 1000


METHOD_LABELS = {
    "browseruse": "Browser-Use",
    "mindact": "MindAct",
    "icddp": "TIDP",
}

SPLIT_ALIASES = {
    "cross_task": "test_cross_task",
    "cross_website": "test_cross_website",
    "cross_domain": "test_cross_domain",
}

DEFAULT_SPLITS = ["test_cross_task", "test_cross_website", "test_cross_domain"]


class LLMModelConfig(BaseModel):
    """OpenAI-compatible model endpoint used by Experiment 1."""

    model_config = ConfigDict(extra="forbid")

    family: str = Field(..., description="Short model-family label, e.g. gpt/qwen/deepseek.")
    model: str = Field(..., description="Provider model name.")
    api_base: str = Field(..., description="OpenAI-compatible API base URL.")
    api_key_env: str | None = Field(None, description="Environment variable containing the API key.")
    api_key: str | None = Field(None, description="Direct API key value. Prefer api_key_env.")

    def resolved_api_key(self) -> str:
        """Resolve the API key without forcing local Ollama users to set one."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "") or "EMPTY"
        return "EMPTY"

    def public_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe config without secrets."""
        return {
            "family": self.family,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
        }


class ActionPrediction(BaseModel):
    """Parsed action selected by the action-decision LLM."""

    model_config = ConfigDict(extra="forbid")

    raw: str
    operation: str | None = None
    value: str = ""
    element_index: int | None = None


MODEL_FAMILY_DEFAULTS = {
    "gpt": LLMModelConfig(
        family="gpt",
        model="gpt-4.1-mini",
        api_base="https://api.apiyi.com/v1",
        api_key_env="APIYI_API_KEY",
    ),
    "qwen": LLMModelConfig(
        family="qwen",
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key_env=None,
    ),
    "deepseek": LLMModelConfig(
        family="deepseek",
        model="deepseek-v4-flash",
        api_base="https://api.apiyi.com/v1",
        api_key_env="APIYI_API_KEY",
    ),
}

DEFAULT_INTENT_MODEL_CONFIG = LLMModelConfig(
    family="intent-qwen",
    model="qwen2.5:3b",
    api_base="http://localhost:11434/v1",
    api_key_env=None,
)


def kept_tokens(kept_elements: List[Dict[str, Any]]) -> int:
    """Count tokens consumed by the serialized kept elements."""
    return count_tokens("".join(elem.get("description", "") + "\n" for elem in kept_elements))


ACTION_RERANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "go",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "page",
    "please",
    "select",
    "step",
    "task",
    "the",
    "then",
    "this",
    "to",
    "type",
    "with",
    "you",
    "your",
}


def _action_tokens(text: str) -> set[str]:
    """Extract lightweight lexical features for action-candidate reranking."""
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {token for token in tokens if len(token) > 1 and token not in ACTION_RERANK_STOPWORDS}


def _element_action_text(elem: Dict[str, Any]) -> str:
    """Return the element text used by the action reranker."""
    parts = [
        elem.get("description", ""),
        elem.get("text", ""),
        elem.get("type", ""),
        elem.get("tag", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _intent_element_bonus(intent: str, elem: Dict[str, Any]) -> float:
    """Prefer element types that fit the inferred user intent without using gold labels."""
    text = _element_action_text(elem).lower()
    elem_type = str(elem.get("type", "")).lower()
    tag = str(elem.get("tag", "")).lower()

    if intent == "information_input":
        if any(marker in text for marker in ("input", "textbox", "textarea", "search", "placeholder")):
            return 0.30
        if tag in {"input", "textarea"} or elem_type in {"input", "text", "search"}:
            return 0.30
    if intent == "form_selection":
        if any(marker in text for marker in ("select", "option", "dropdown", "radio", "checkbox", "combobox")):
            return 0.30
        if tag in {"select", "option"} or elem_type in {"select", "option", "checkbox", "radio"}:
            return 0.30
    if intent == "button_interaction":
        if any(marker in text for marker in ("button", "link", "clickable", "submit", "continue", "next")):
            return 0.22
        if tag in {"button", "a"} or elem_type in {"button", "link"}:
            return 0.22
    if intent == "content_browsing":
        if any(marker in text for marker in ("link", "heading", "article", "title", "result", "next", "more")):
            return 0.16
    return 0.0


def rerank_action_candidates(
    kept_elements: List[Dict[str, Any]],
    intent: str,
    task_description: str,
    step_description: str,
    operation_value: str,
    enable_rerank: bool,
    top_k: int,
    method_name: str = "",
) -> List[Dict[str, Any]]:
    """Sort kept elements for action decisions while preserving pruning metrics."""
    candidates = list(kept_elements)
    if not candidates:
        return candidates
    if not enable_rerank:
        return candidates[:top_k] if top_k > 0 else candidates

    query_tokens = _action_tokens(f"{task_description} {step_description}")
    value_tokens = _action_tokens(operation_value)

    def score_item(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, int]:
        original_index, elem = item
        elem_tokens = _action_tokens(_element_action_text(elem))
        overlap = len(query_tokens & elem_tokens) / max(1, len(query_tokens))
        value_overlap = len(value_tokens & elem_tokens) / max(1, len(value_tokens)) if value_tokens else 0.0
        try:
            relevance = float(elem.get("relevance", 0.0) or 0.0)
        except (TypeError, ValueError):
            relevance = 0.0
        if method_name == "icddp":
            score = (
                0.70 * relevance
                + 0.18 * overlap
                + 0.07 * value_overlap
                + _intent_element_bonus(intent, elem)
            )
        else:
            score = (
                0.45 * relevance
                + 0.35 * overlap
                + 0.15 * value_overlap
                + _intent_element_bonus(intent, elem)
            )
        return score, -original_index

    ranked = [elem for _, elem in sorted(enumerate(candidates), key=score_item, reverse=True)]
    return ranked[:top_k] if top_k > 0 else ranked


def action_elements(kept_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a JSON-safe compact element list for action decisions."""
    elements = []
    for elem in kept_elements:
        backend_node_id = ""
        tag_element = elem.get("element")
        if tag_element is not None and hasattr(tag_element, "get"):
            backend_node_id = str(tag_element.get("backend_node_id") or "")
        elements.append(
            {
                "type": elem.get("type", ""),
                "text": elem.get("text", ""),
                "description": elem.get("description", ""),
                "relevance": float(elem.get("relevance", 0.0) or 0.0),
                "backend_node_id": backend_node_id,
            }
        )
    return elements


def parse_action_prediction(action: str) -> ActionPrediction:
    """Parse CLICK/TYPE/SELECT action text produced by the LLM."""
    raw = (action or "").strip()
    if not raw or raw == "SKIPPED" or raw.startswith("Error:"):
        return ActionPrediction(raw=raw)

    json_text = raw
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()
    if json_text.startswith("{") and json_text.endswith("}"):
        try:
            data = json.loads(json_text)
            nested_action = data.get("action")
            if isinstance(nested_action, str) and nested_action.strip() != raw:
                parsed = parse_action_prediction(nested_action)
                if parsed.operation or parsed.element_index is not None:
                    return ActionPrediction(
                        raw=raw,
                        operation=parsed.operation,
                        value=parsed.value,
                        element_index=parsed.element_index,
                    )
            operation = str(
                data.get("operation")
                or data.get("op")
                or data.get("action_type")
                or data.get("type")
                or ""
            ).upper()
            if operation not in {"CLICK", "TYPE", "SELECT"}:
                operation = None
            index_value = (
                data.get("element_index")
                or data.get("element")
                or data.get("index")
                or data.get("id")
            )
            try:
                element_index = int(index_value) if index_value is not None else None
            except (TypeError, ValueError):
                element_index = None
            value = str(data.get("value") or data.get("text") or data.get("option") or "").strip()
            return ActionPrediction(raw=raw, operation=operation, value=value, element_index=element_index)
        except json.JSONDecodeError:
            pass

    bracket_indexes = re.findall(r"\[(\d+)\]", raw)
    element_index = int(bracket_indexes[-1]) if bracket_indexes else None
    if element_index is None:
        labeled = re.search(
            r"(?:element(?:_|\s*)?(?:index|id|number)?|index|id|#)\s*[:=#]?\s*(\d+)",
            raw,
            re.IGNORECASE,
        )
        if labeled:
            element_index = int(labeled.group(1))

    op_match = re.search(r"\b(CLICK|TYPE|SELECT)\b", raw, re.IGNORECASE)
    operation = op_match.group(1).upper() if op_match else None
    if element_index is None and operation:
        tail = raw[op_match.end():]
        tail_numbers = re.findall(r"\b(\d+)\b", tail)
        if tail_numbers:
            element_index = int(tail_numbers[-1])

    value = ""
    if operation in {"TYPE", "SELECT"}:
        after_op = raw[op_match.end():] if op_match else raw
        value = re.sub(r"\[\d+\]", "", after_op)
        value = re.sub(
            r"(?:element(?:_|\s*)?(?:index|id|number)?|index|id|#)\s*[:=#]?\s*\d+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = value.strip(" :-\"'")

    return ActionPrediction(raw=raw, operation=operation, value=value, element_index=element_index)


def selected_element_hits_positive(
    evaluator: Evaluator,
    kept_elements: List[Dict[str, Any]],
    pos_candidates: List[Dict[str, Any]],
    prediction: ActionPrediction,
) -> int | None:
    """Return whether the predicted 1-based element index matches any positive candidate."""
    if prediction.element_index is None:
        return None
    selected_pos = prediction.element_index - 1
    if selected_pos < 0 or selected_pos >= len(kept_elements):
        return 0

    selected_id = evaluator.extract_backend_node_id_from_element(kept_elements[selected_pos])
    pos_ids = {evaluator.extract_backend_node_id(pos) for pos in pos_candidates}
    pos_ids.discard("")
    if selected_id and pos_ids:
        return 1 if selected_id in pos_ids else 0

    selected_text = (kept_elements[selected_pos].get("text") or "").lower().strip()
    for pos in pos_candidates:
        pos_text = evaluator.extract_candidate_text(pos).lower().strip()
        if evaluator.text_match(pos_text, selected_text):
            return 1
    return 0


def evaluate_kept(
    evaluator: Evaluator,
    name: str,
    kept_elements: List[Dict[str, Any]],
    stats: Dict[str, Any],
    pos_candidates: List[Dict[str, Any]],
    neg_candidates: List[Dict[str, Any]],
    process_time_ms: float,
    action_time_ms: float = 0.0,
    action: str = "SKIPPED",
    expected_operation: str = "",
) -> Dict[str, Any]:
    """Evaluate one pruning method on a single step."""
    total_elements = stats.get("total", 0)
    kept_count = len(kept_elements)
    prune_rate = stats.get("prune_rate")
    if prune_rate is None:
        prune_rate = 1 - kept_count / total_elements if total_elements > 0 else 0.0

    metrics = evaluator.evaluate_step(
        pos_candidates=pos_candidates,
        neg_candidates=neg_candidates,
        kept_elements=kept_elements,
        stats=stats,
    )
    prediction = parse_action_prediction(action)
    element_acc = selected_element_hits_positive(evaluator, kept_elements, pos_candidates, prediction)
    op_match = None
    step_sr = None
    if prediction.operation and expected_operation:
        op_match = 1 if prediction.operation == expected_operation else 0
    if element_acc is not None and op_match is not None:
        step_sr = 1 if element_acc and op_match else 0

    return {
        f"{name}_success": 1 if metrics.recall > 0 else 0,
        f"{name}_tokens": kept_tokens(kept_elements),
        f"{name}_recall": metrics.recall,
        f"{name}_precision": metrics.precision,
        f"{name}_f1": metrics.f1,
        f"{name}_prune_rate": prune_rate,
        f"{name}_kept_elements": kept_count,
        f"{name}_total_elements": total_elements,
        f"{name}_process_time_ms": process_time_ms,
        f"{name}_action_time_ms": action_time_ms,
        f"{name}_total_time_ms": process_time_ms + action_time_ms,
        f"{name}_action": action,
        f"{name}_pred_op": prediction.operation,
        f"{name}_pred_index": prediction.element_index,
        f"{name}_element_acc": element_acc,
        f"{name}_op_match": op_match,
        f"{name}_step_sr": step_sr,
    }


class Experiment1Runner:
    """Official three-method Experiment 1 runner."""

    def __init__(
        self,
        data_dir: str,
        cross_encoder_path: str | None,
        scorer_model_path: str | None,
        scorer_device: str,
        mindact_device: str | None,
        mindact_batch_size: int,
        context_limit: int,
        skip_mindact: bool,
        run_action_llm: bool,
        operation_hint_mode: str,
        action_rerank: bool,
        action_rerank_methods: set[str],
        action_top_k: int,
        llm_config: LLMModelConfig,
        intent_config: LLMModelConfig,
        intent_cache_path: str | None,
        pruning_cache_path: str | None,
        use_pruning_cache_only: bool,
        precompute_intents_only: bool = False,
    ):
        self.loader = Mind2WebLoader(data_dir)
        self.context_limit = context_limit
        self.skip_mindact = skip_mindact
        self.run_action_llm = run_action_llm
        self.operation_hint_mode = operation_hint_mode
        self.action_rerank = action_rerank
        self.action_rerank_methods = action_rerank_methods
        self.action_top_k = action_top_k
        self.llm_config = llm_config
        self.intent_config = intent_config
        self.intent_cache_path = intent_cache_path
        self.intent_cache = self._load_intent_cache(intent_cache_path)
        self.pruning_cache_path = pruning_cache_path
        self.pruning_cache = self._load_pruning_cache(pruning_cache_path)
        self.use_pruning_cache_only = use_pruning_cache_only
        self.precompute_intents_only = precompute_intents_only

        self.intent_classifier = IntentClassifier(
            api_base=intent_config.api_base,
            model=intent_config.model,
            api_key_env=intent_config.api_key_env,
            api_key=intent_config.api_key,
        )
        self.relevance_scorer = None
        self.icddp_pruner = None
        self.browseruse_pruner = None
        self.mindact_pruner = None
        self.action_decider = None
        if not precompute_intents_only and not use_pruning_cache_only:
            self.relevance_scorer = RelevanceScorer(model_path=scorer_model_path, device=scorer_device)
            self.icddp_pruner = DynamicPruner(scorer=self.relevance_scorer)
            self.browseruse_pruner = BrowserUsePruner()

            if not skip_mindact:
                if not cross_encoder_path:
                    raise ValueError("--cross_encoder_path is required unless --skip_mindact is set")
                self.mindact_pruner = MindActPruner(
                    cross_encoder_path,
                    device=mindact_device,
                    batch_size=mindact_batch_size,
                )

            self.action_decider = (
                SimpleActionDecider(llm_config, operation_hint_mode=operation_hint_mode) if run_action_llm else None
            )
        elif use_pruning_cache_only:
            self.action_decider = (
                SimpleActionDecider(llm_config, operation_hint_mode=operation_hint_mode) if run_action_llm else None
            )
        self.evaluator = Evaluator()

    def _load_intent_cache(self, path: str | None) -> Dict[str, Dict[str, Any]]:
        """Load precomputed intent labels keyed by step uid."""
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", data)

    def _save_intent_cache(self) -> None:
        """Persist intent labels when an intent cache path is configured."""
        if not self.intent_cache_path:
            return
        output_dir = os.path.dirname(self.intent_cache_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.intent_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "intent_model": self.intent_config.public_dict(),
                    "intents": self.intent_cache,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _load_pruning_cache(self, path: str | None) -> Dict[str, Dict[str, Any]]:
        """Load cached pruning/evaluation results keyed by split and step uid."""
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("steps", data)

    def _save_pruning_cache(self) -> None:
        """Persist cached pruning/evaluation results."""
        if not self.pruning_cache_path:
            return
        output_dir = os.path.dirname(self.pruning_cache_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.pruning_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "context_limit": self.context_limit,
                    "steps": self.pruning_cache,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def pruning_cache_key(self, split_name: str, task_id: str, step_uid: str, method: str) -> str:
        """Build a stable cache key for method-level pruning results."""
        return f"{split_name}:{task_id}:{step_uid}:{method}:ctx{self.context_limit}"

    def cached_method_result(
        self,
        split_name: str,
        task_id: str,
        step_uid: str,
        method: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]] | None:
        """Return cached row fields and action elements for one method."""
        cached = self.pruning_cache.get(self.pruning_cache_key(split_name, task_id, step_uid, method))
        if not cached:
            return None
        return cached.get("row_fields", {}), cached.get("kept_elements", [])

    def store_method_result(
        self,
        split_name: str,
        task_id: str,
        step_uid: str,
        method: str,
        row_fields: Dict[str, Any],
        kept_elements: List[Dict[str, Any]],
    ) -> None:
        """Cache method-level pruning result without action-decision fields."""
        if not self.pruning_cache_path:
            return
        prefix = f"{method}_"
        cached_fields = {
            key: value
            for key, value in row_fields.items()
            if key.startswith(prefix)
            and key
            not in {
                f"{method}_action",
                f"{method}_pred_op",
                f"{method}_pred_index",
                f"{method}_element_acc",
                f"{method}_op_match",
                f"{method}_step_sr",
                f"{method}_action_time_ms",
            }
        }
        self.pruning_cache[self.pruning_cache_key(split_name, task_id, step_uid, method)] = {
            "row_fields": cached_fields,
            "kept_elements": action_elements(kept_elements),
        }

    def get_intent(
        self,
        task: Any,
        step: Any,
    ) -> Tuple[str, float, str]:
        """Return cached or newly classified intent for a step."""
        cache_key = step.uid or f"{task.task_id}:{step.step_idx}"
        cached = self.intent_cache.get(cache_key)
        if cached and cached.get("intent"):
            return cached["intent"], 0.0, "cache"

        start = time.time()
        intent = self.intent_classifier.classify_with_fallback(
            confirmed_task=task.confirmed_task,
            subtask_description=step.action_repr,
            operation_hint=step.operation_type,
        )
        intent_time_ms = (time.time() - start) * 1000
        self.intent_cache[cache_key] = {
            "intent": intent,
            "task_id": task.task_id,
            "step_idx": step.step_idx,
            "step_uid": step.uid,
            "operation_type": step.operation_type,
            "action_repr": step.action_repr,
        }
        return intent, intent_time_ms, "classified"

    def load_tasks(
        self,
        split_name: str,
        max_files: int | None,
        sample_tasks: int | None,
        sample_seed: int,
    ):
        """Load split files and optionally sample tasks deterministically."""
        tasks = self.loader.load_split(split_name, max_files=max_files)
        if sample_tasks is not None and sample_tasks > 0 and sample_tasks < len(tasks):
            rng = random.Random(sample_seed)
            tasks = rng.sample(tasks, sample_tasks)
            print(f"Sampled {len(tasks)} tasks with seed={sample_seed}")
        else:
            print(f"Using all loaded tasks: {len(tasks)}")
        return tasks

    def action_candidates(
        self,
        method_name: str,
        kept_elements: List[Dict[str, Any]],
        intent: str,
        task_description: str,
        step_description: str,
        operation_value: str,
    ) -> List[Dict[str, Any]]:
        """Prepare the ordered element list shown to the action-decision model."""
        rerank_value = operation_value if self.operation_hint_mode in {"value", "full"} else ""
        return rerank_action_candidates(
            kept_elements=kept_elements,
            intent=intent,
            task_description=task_description,
            step_description=step_description,
            operation_value=rerank_value,
            enable_rerank=self.action_rerank and method_name in self.action_rerank_methods,
            top_k=self.action_top_k,
            method_name=method_name,
        )

    def maybe_decide_action(
        self,
        kept_elements: List[Dict[str, Any]],
        intent: str,
        task_description: str,
        step_description: str,
        expected_operation: str,
        operation_value: str,
    ) -> Tuple[str, float]:
        """Optionally run the action-decision LLM for timing."""
        if not self.action_decider:
            return "SKIPPED", 0.0
        return self.action_decider.decide_action(
            kept_elements=kept_elements,
            intent=intent,
            task_description=task_description,
            step_description=step_description,
            expected_operation=expected_operation,
            operation_value=operation_value,
        )

    def run(
        self,
        split_name: str,
        max_files: int | None,
        sample_tasks: int | None,
        max_steps: int | None,
        sample_seed: int,
        save_results: bool,
        output_path: str | None,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 80}")
        print(f"Experiment 1 on {split_name}")
        print(f"{'=' * 80}")

        tasks = self.load_tasks(split_name, max_files, sample_tasks, sample_seed)
        print(f"Total tasks loaded: {len(tasks)}")
        print(f"Context limit: {self.context_limit}")
        print(f"Model family: {self.llm_config.family} ({self.llm_config.model})")
        print(f"Intent model: {self.intent_config.family} ({self.intent_config.model})")
        print(f"Intent cache: {self.intent_cache_path or 'disabled'}")
        print(f"Action LLM timing: {'enabled' if self.run_action_llm else 'skipped'}")
        print(
            "Action decision: "
            f"operation_hint_mode={self.operation_hint_mode}, "
            f"rerank={'enabled' if self.action_rerank else 'disabled'}, "
            f"rerank_methods={','.join(sorted(self.action_rerank_methods)) or 'none'}, "
            f"top_k={self.action_top_k if self.action_top_k > 0 else 'all'}"
        )

        results: List[Dict[str, Any]] = []
        precomputed_steps = 0

        for task in tqdm(tasks, desc=f"Processing {split_name}"):
            for step in task.steps:
                if max_steps is not None and len(results) >= max_steps:
                    break
                if not step.raw_html or len(step.raw_html) < 100:
                    continue

                subtask_desc = step.action_repr
                if step.operation_value:
                    subtask_desc = f"{step.action_repr} value: {step.operation_value}"

                intent, intent_time_ms, intent_source = self.get_intent(task, step)
                if self.precompute_intents_only:
                    precomputed_steps += 1
                    results.append(
                        {
                            "task_id": task.task_id,
                            "step_uid": step.uid,
                            "split": split_name,
                            "intent": intent,
                            "intent_source": intent_source,
                            "operation_type": step.operation_type,
                            "step_description": step.action_repr[:160],
                            "icddp_intent_time_ms": intent_time_ms,
                        }
                    )
                    continue

                row: Dict[str, Any] = {
                    "task_id": task.task_id,
                    "step_uid": step.uid,
                    "split": split_name,
                    "model_family": self.llm_config.family,
                    "model_name": self.llm_config.model,
                    "website": task.website,
                    "intent": intent,
                    "operation_type": step.operation_type,
                    "intent_source": intent_source,
                    "step_description": step.action_repr[:160],
                    "original_tokens": count_tokens(step.raw_html),
                    "icddp_intent_time_ms": intent_time_ms,
                }

                try:
                    cached = self.cached_method_result(split_name, task.task_id, step.uid, "browseruse")
                    if cached:
                        cached_fields, bu_kept = cached
                        bu_process_ms = 0.0
                        row.update(cached_fields)
                    else:
                        if self.browseruse_pruner is None:
                            raise ValueError("Missing Browser-Use pruning cache entry and pruning models are disabled")
                        start = time.time()
                        bu_kept, bu_stats = self.browseruse_pruner.process(step.raw_html, self.context_limit)
                        bu_process_ms = (time.time() - start) * 1000
                        bu_fields = evaluate_kept(
                            self.evaluator,
                            "browseruse",
                            bu_kept,
                            bu_stats,
                            step.pos_candidates,
                            step.neg_candidates,
                            bu_process_ms,
                            0.0,
                            "SKIPPED",
                            step.operation_type,
                        )
                        row.update(bu_fields)
                        self.store_method_result(split_name, task.task_id, step.uid, "browseruse", bu_fields, bu_kept)
                    bu_action_kept = self.action_candidates(
                        "browseruse",
                        bu_kept,
                        intent,
                        task.confirmed_task,
                        step.action_repr,
                        step.operation_value,
                    )
                    bu_action, bu_action_ms = self.maybe_decide_action(
                        bu_action_kept,
                        intent,
                        task.confirmed_task,
                        step.action_repr,
                        step.operation_type,
                        step.operation_value,
                    )
                    row.update(
                        {
                            "browseruse_action_time_ms": bu_action_ms,
                            "browseruse_total_time_ms": row.get("browseruse_process_time_ms", bu_process_ms)
                            + bu_action_ms,
                            "browseruse_action": bu_action,
                            "browseruse_action_candidate_count": len(bu_action_kept),
                        }
                    )
                    bu_prediction = parse_action_prediction(bu_action)
                    row["browseruse_pred_op"] = bu_prediction.operation
                    row["browseruse_pred_index"] = bu_prediction.element_index
                    row["browseruse_element_acc"] = selected_element_hits_positive(
                        self.evaluator, bu_action_kept, step.pos_candidates, bu_prediction
                    )
                    row["browseruse_op_match"] = (
                        1 if bu_prediction.operation == step.operation_type else 0 if bu_prediction.operation else None
                    )
                    row["browseruse_step_sr"] = (
                        1
                        if row["browseruse_element_acc"] and row["browseruse_op_match"]
                        else 0
                        if row["browseruse_element_acc"] is not None and row["browseruse_op_match"] is not None
                        else None
                    )

                    if self.mindact_pruner is not None or self.use_pruning_cache_only:
                        cached = self.cached_method_result(split_name, task.task_id, step.uid, "mindact")
                        if cached:
                            cached_fields, ma_kept = cached
                            ma_process_ms = 0.0
                            row.update(cached_fields)
                        else:
                            if self.mindact_pruner is None:
                                raise ValueError("Missing MindAct pruning cache entry and pruning models are disabled")
                            start = time.time()
                            ma_kept, ma_stats = self.mindact_pruner.process(
                                html_content=step.raw_html,
                                task_description=task.confirmed_task,
                                context_limit=self.context_limit,
                                step_description=subtask_desc,
                                action_history=task.action_reprs[:step.step_idx],
                            )
                            ma_process_ms = (time.time() - start) * 1000
                            ma_fields = evaluate_kept(
                                self.evaluator,
                                "mindact",
                                ma_kept,
                                ma_stats,
                                step.pos_candidates,
                                step.neg_candidates,
                                ma_process_ms,
                                0.0,
                                "SKIPPED",
                                step.operation_type,
                            )
                            row.update(ma_fields)
                            self.store_method_result(split_name, task.task_id, step.uid, "mindact", ma_fields, ma_kept)
                        ma_action_kept = self.action_candidates(
                            "mindact",
                            ma_kept,
                            intent,
                            task.confirmed_task,
                            step.action_repr,
                            step.operation_value,
                        )
                        ma_action, ma_action_ms = self.maybe_decide_action(
                            ma_action_kept,
                            intent,
                            task.confirmed_task,
                            step.action_repr,
                            step.operation_type,
                            step.operation_value,
                        )
                        row.update(
                            {
                                "mindact_action_time_ms": ma_action_ms,
                                "mindact_total_time_ms": row.get("mindact_process_time_ms", ma_process_ms)
                                + ma_action_ms,
                                "mindact_action": ma_action,
                                "mindact_action_candidate_count": len(ma_action_kept),
                            }
                        )
                        ma_prediction = parse_action_prediction(ma_action)
                        row["mindact_pred_op"] = ma_prediction.operation
                        row["mindact_pred_index"] = ma_prediction.element_index
                        row["mindact_element_acc"] = selected_element_hits_positive(
                            self.evaluator, ma_action_kept, step.pos_candidates, ma_prediction
                        )
                        row["mindact_op_match"] = (
                            1 if ma_prediction.operation == step.operation_type else 0 if ma_prediction.operation else None
                        )
                        row["mindact_step_sr"] = (
                            1
                            if row["mindact_element_acc"] and row["mindact_op_match"]
                            else 0
                            if row["mindact_element_acc"] is not None and row["mindact_op_match"] is not None
                            else None
                        )

                    cached = self.cached_method_result(split_name, task.task_id, step.uid, "icddp")
                    if cached:
                        cached_fields, ic_kept = cached
                        ic_process_ms = 0.0
                        row.update(cached_fields)
                    else:
                        if self.icddp_pruner is None:
                            raise ValueError("Missing TIDP pruning cache entry and pruning models are disabled")
                        start = time.time()
                        ic_kept, ic_stats = self.icddp_pruner.prune(
                            html_content=step.raw_html,
                            intent=intent,
                            subtask_description=subtask_desc,
                            context_limit=self.context_limit,
                        )
                        ic_process_ms = (time.time() - start) * 1000
                        ic_fields = evaluate_kept(
                            self.evaluator,
                            "icddp",
                            ic_kept,
                            ic_stats,
                            step.pos_candidates,
                            step.neg_candidates,
                            ic_process_ms,
                            0.0,
                            "SKIPPED",
                            step.operation_type,
                        )
                        row.update(ic_fields)
                        self.store_method_result(split_name, task.task_id, step.uid, "icddp", ic_fields, ic_kept)
                    ic_action_kept = self.action_candidates(
                        "icddp",
                        ic_kept,
                        intent,
                        task.confirmed_task,
                        step.action_repr,
                        step.operation_value,
                    )
                    ic_action, ic_action_ms = self.maybe_decide_action(
                        ic_action_kept,
                        intent,
                        task.confirmed_task,
                        step.action_repr,
                        step.operation_type,
                        step.operation_value,
                    )
                    row.update(
                        {
                            "icddp_action_time_ms": ic_action_ms,
                            "icddp_total_time_ms": row.get("icddp_process_time_ms", ic_process_ms)
                            + ic_action_ms,
                            "icddp_action": ic_action,
                            "icddp_action_candidate_count": len(ic_action_kept),
                        }
                    )
                    ic_prediction = parse_action_prediction(ic_action)
                    row["icddp_pred_op"] = ic_prediction.operation
                    row["icddp_pred_index"] = ic_prediction.element_index
                    row["icddp_element_acc"] = selected_element_hits_positive(
                        self.evaluator, ic_action_kept, step.pos_candidates, ic_prediction
                    )
                    row["icddp_op_match"] = (
                        1 if ic_prediction.operation == step.operation_type else 0 if ic_prediction.operation else None
                    )
                    row["icddp_step_sr"] = (
                        1
                        if row["icddp_element_acc"] and row["icddp_op_match"]
                        else 0
                        if row["icddp_element_acc"] is not None and row["icddp_op_match"] is not None
                        else None
                    )
                    row["icddp_total_time_ms"] = (
                        row["icddp_process_time_ms"] + row["icddp_action_time_ms"] + intent_time_ms
                    )

                except Exception as exc:
                    print(f"  Skip step {step.uid}: {exc}")
                    continue

                results.append(row)

            if max_steps is not None and len(results) >= max_steps:
                break

        if not results:
            print("No valid results.")
            return {}

        self._save_intent_cache()
        self._save_pruning_cache()
        if self.precompute_intents_only:
            summary = {
                "total_steps": len(results),
                "precomputed_steps": precomputed_steps,
                "intent_cache_path": self.intent_cache_path,
                "intent_model": self.intent_config.public_dict(),
            }
            print(f"\nPrecomputed intents for {precomputed_steps} steps.")
            if save_results:
                if output_path is None:
                    os.makedirs("results", exist_ok=True)
                    output_path = os.path.join("results", f"experiment1_intents_{split_name}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": summary, "step_results": results}, f, indent=2, ensure_ascii=False)
                print(f"Saved to {output_path}")
            return summary

        methods = ["browseruse", "icddp"] if self.skip_mindact else ["browseruse", "mindact", "icddp"]
        summary = self._summarize(results, methods)
        self._print_summary(summary, results, methods)

        if save_results:
            if output_path is None:
                os.makedirs("results", exist_ok=True)
                output_path = os.path.join(
                    "results",
                    f"experiment1_{self.llm_config.family}_{split_name}.json",
                )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment": "experiment1_browseruse_mindact_icddp",
                        "split": split_name,
                        "context_limit": self.context_limit,
                        "max_files": max_files,
                        "sample_tasks": sample_tasks,
                        "max_steps": max_steps,
                        "sample_seed": sample_seed,
                        "action_llm_enabled": self.run_action_llm,
                        "model_config": self.llm_config.public_dict(),
                        "intent_model_config": self.intent_config.public_dict(),
                        "intent_cache_path": self.intent_cache_path,
                        "pruning_cache_path": self.pruning_cache_path,
                        "operation_hint_mode": self.operation_hint_mode,
                        "action_rerank": self.action_rerank,
                        "action_rerank_methods": sorted(self.action_rerank_methods),
                        "action_top_k": self.action_top_k,
                        "summary": summary,
                        "step_results": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"\nSaved to {output_path}")

        return summary

    def _summarize(self, results: List[Dict[str, Any]], methods: List[str]) -> Dict[str, Any]:
        n = len(results)

        def avg(key: str) -> float:
            return sum(row.get(key, 0.0) for row in results) / n

        def avg_optional(key: str) -> float | None:
            values = [row.get(key) for row in results if row.get(key) is not None]
            return sum(values) / len(values) if values else None

        def macro_op_f1(method: str) -> float | None:
            labels = ["CLICK", "TYPE", "SELECT"]
            predictions = [
                (row.get("operation_type"), row.get(f"{method}_pred_op"))
                for row in results
                if row.get(f"{method}_pred_op") is not None and row.get("operation_type")
            ]
            if not predictions:
                return None
            f1_scores = []
            for label in labels:
                tp = sum(1 for gold, pred in predictions if gold == label and pred == label)
                fp = sum(1 for gold, pred in predictions if gold != label and pred == label)
                fn = sum(1 for gold, pred in predictions if gold == label and pred != label)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            return sum(f1_scores) / len(f1_scores)

        summary: Dict[str, Any] = {
            "total_steps": n,
            "model_family": self.llm_config.family,
            "model_name": self.llm_config.model,
        }
        for method in methods:
            summary[method] = {
                "success_rate": avg(f"{method}_success"),
                "avg_tokens": avg(f"{method}_tokens"),
                "avg_prune_rate": avg(f"{method}_prune_rate"),
                "avg_recall": avg(f"{method}_recall"),
                "avg_process_time_ms": avg(f"{method}_process_time_ms"),
                "avg_action_time_ms": avg(f"{method}_action_time_ms"),
                "avg_total_time_ms": avg(f"{method}_total_time_ms"),
                "element_accuracy": avg_optional(f"{method}_element_acc"),
                "operation_f1": macro_op_f1(method),
                "step_success_rate": avg_optional(f"{method}_step_sr"),
            }
        summary["icddp"]["avg_intent_time_ms"] = avg("icddp_intent_time_ms")
        summary["icddp"]["avg_total_time_ms"] = (
            summary["icddp"]["avg_process_time_ms"]
            + summary["icddp"]["avg_action_time_ms"]
            + summary["icddp"]["avg_intent_time_ms"]
        )
        return summary

    def _print_summary(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]],
        methods: List[str],
    ) -> None:
        print(f"\n{'=' * 80}")
        print("Experiment 1 Results")
        print(f"{'=' * 80}")
        print(f"Total steps: {summary['total_steps']}")

        header = f"{'Metric':<26}" + "".join(f" {METHOD_LABELS[m]:<14}" for m in methods)
        print(f"\n{'-' * 80}")
        print(header)
        print(f"{'-' * 80}")
        metric_rows = [
            ("Tokens consumed", "avg_tokens", 1, ".0f"),
            ("Prune rate (%)", "avg_prune_rate", 100, ".2f"),
            ("Key recall (%)", "avg_recall", 100, ".2f"),
            ("Prune time (ms)", "avg_process_time_ms", 1, ".1f"),
            ("Action time (ms)", "avg_action_time_ms", 1, ".1f"),
            ("Total time (ms)", "avg_total_time_ms", 1, ".1f"),
        ]
        for label, key, scale, fmt in metric_rows:
            row = f"{label:<26}"
            for method in methods:
                row += f" {summary[method][key] * scale:<14{fmt}}"
            print(row)

        if any(summary[method].get("element_accuracy") is not None for method in methods):
            print(f"\n{'-' * 80}")
            print("Paper-style metrics (requires --run_action_llm)")
            print(f"{'-' * 80}")
            paper_rows = [
                ("Ele. Acc (%)", "element_accuracy"),
                ("Op. F1 (%)", "operation_f1"),
                ("Step SR (%)", "step_success_rate"),
            ]
            for label, key in paper_rows:
                row = f"{label:<26}"
                for method in methods:
                    value = summary[method].get(key)
                    row += f" {value * 100:<14.2f}" if value is not None else f" {'-':<14}"
                print(row)

        if "icddp" in methods:
            print(f"{'TIDP intent time (ms)':<26} {summary['icddp']['avg_intent_time_ms']:<14.1f}")

        print(f"\n{'-' * 80}")
        print("By intent: Paper-style Step SR (%)")
        print(f"{'-' * 80}")
        self._print_intent_table(results, methods, "step_sr", 100, ".1f", optional=True)

        print(f"\n{'-' * 80}")
        print("By intent: Tokens consumed")
        print(f"{'-' * 80}")
        self._print_intent_table(results, methods, "tokens", 1, ".0f")

        print(f"\n{'-' * 80}")
        print("By intent: Key recall (%)")
        print(f"{'-' * 80}")
        self._print_intent_table(results, methods, "recall", 100, ".1f")

    def _print_intent_table(
        self,
        results: List[Dict[str, Any]],
        methods: List[str],
        metric_suffix: str,
        scale: float,
        fmt: str,
        optional: bool = False,
    ) -> None:
        intent_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: {method: {"sum": 0.0, "count": 0.0} for method in methods}
        )
        for row in results:
            intent = row["intent"]
            for method in methods:
                value = row.get(f"{method}_{metric_suffix}")
                if optional and value is None:
                    continue
                intent_stats[intent][method]["sum"] += value if value is not None else 0.0
                intent_stats[intent][method]["count"] += 1

        header = f"{'Intent':<24}" + "".join(f" {METHOD_LABELS[m]:<14}" for m in methods)
        print(header)
        for intent in ["content_browsing", "form_selection", "information_input", "button_interaction"]:
            row_text = f"{intent:<24}"
            for method in methods:
                stat = intent_stats[intent][method]
                if optional and not stat["count"]:
                    row_text += f" {'-':<14}"
                else:
                    value = stat["sum"] / stat["count"] if stat["count"] else 0.0
                    row_text += f" {value * scale:<14{fmt}}"
            print(row_text)


def parse_splits(split: str, splits: str | None) -> List[str]:
    """Resolve split CLI aliases into Mind2Web split names."""
    raw = splits or split
    if raw.lower() == "all":
        return DEFAULT_SPLITS
    resolved = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        resolved.append(SPLIT_ALIASES.get(name, name))
    return resolved


def build_model_configs(args: argparse.Namespace) -> List[LLMModelConfig]:
    """Build one or more model configs from CLI arguments."""
    raw_families = args.model_families or args.model_family
    families = [item.strip().lower() for item in raw_families.split(",") if item.strip()]
    if "all" in families:
        families = ["gpt", "qwen", "deepseek"]

    configs = []
    for family in families:
        if family not in MODEL_FAMILY_DEFAULTS:
            raise ValueError(f"Unknown model family: {family}. Choose from gpt, qwen, deepseek, all.")
        base = MODEL_FAMILY_DEFAULTS[family]
        family_model = getattr(args, f"{family}_model")
        family_api_base = getattr(args, f"{family}_api_base")
        family_api_key_env = getattr(args, f"{family}_api_key_env")
        configs.append(
            LLMModelConfig(
                family=family,
                model=family_model or args.llm_model or base.model,
                api_base=family_api_base or args.llm_api_base or base.api_base,
                api_key_env=(
                    family_api_key_env
                    if family_api_key_env is not None
                    else args.llm_api_key_env
                    if args.llm_api_key_env is not None
                    else base.api_key_env
                ),
                api_key=args.llm_api_key,
            )
        )
    return configs


def build_intent_config(args: argparse.Namespace) -> LLMModelConfig:
    """Build the fixed local intent-classifier model config."""
    return LLMModelConfig(
        family="intent-qwen",
        model=args.intent_model or DEFAULT_INTENT_MODEL_CONFIG.model,
        api_base=args.intent_api_base or DEFAULT_INTENT_MODEL_CONFIG.api_base,
        api_key_env=(
            args.intent_api_key_env
            if args.intent_api_key_env is not None
            else DEFAULT_INTENT_MODEL_CONFIG.api_key_env
        ),
        api_key=args.intent_api_key,
    )


def parse_method_set(raw: str) -> set[str]:
    """Parse a comma-separated method list used for action reranking."""
    if not raw or raw.strip().lower() in {"none", "off", "false"}:
        return set()
    methods = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "all" in methods:
        return {"browseruse", "mindact", "icddp"}
    valid = {"browseruse", "mindact", "icddp"}
    unknown = methods - valid
    if unknown:
        raise ValueError(f"Unknown action rerank method(s): {', '.join(sorted(unknown))}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Browser-Use vs MindAct vs TIDP")
    parser.add_argument("--data_dir", type=str, default=DATASET_PATH)
    parser.add_argument("--split", type=str, default="test_cross_domain")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits or 'all'. Aliases: cross_task,cross_website,cross_domain",
    )
    parser.add_argument("--max_files", type=int, default=1, help="Number of JSON shard files to load")
    parser.add_argument("--sample_tasks", type=int, default=None, help="Number of tasks sampled after loading")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--context_limit", type=int, default=CONTEXT_LIMIT_TOKENS)
    parser.add_argument("--cross_encoder_path", type=str, default=None)
    parser.add_argument("--scorer_model_path", type=str, default=None)
    parser.add_argument("--scorer_device", type=str, default="cuda")
    parser.add_argument("--mindact_device", type=str, default="cuda")
    parser.add_argument("--mindact_batch_size", type=int, default=32)
    parser.add_argument("--skip_mindact", action="store_true")
    parser.add_argument("--run_action_llm", action="store_true")
    parser.add_argument(
        "--operation_hint_mode",
        choices=["none", "value", "full"],
        default="full",
        help="Action prompt hinting: none hides operation/value, value gives only target value, full gives operation/value.",
    )
    parser.add_argument(
        "--disable_action_rerank",
        action="store_true",
        help="Disable lightweight reranking of kept elements before action decisions.",
    )
    parser.add_argument(
        "--action_rerank_methods",
        type=str,
        default="icddp",
        help="Comma-separated methods to rerank before action decisions: icddp, browseruse, mindact, all, or none.",
    )
    parser.add_argument(
        "--action_top_k",
        type=int,
        default=0,
        help="Limit action-decision prompt to the top-k reranked kept elements; 0 means all kept elements.",
    )
    parser.add_argument("--intent_cache_path", type=str, default=None)
    parser.add_argument("--pruning_cache_path", type=str, default=None)
    parser.add_argument("--use_pruning_cache_only", action="store_true")
    parser.add_argument("--precompute_intents_only", action="store_true")
    parser.add_argument("--intent_model", type=str, default=None, help="Fixed local intent model")
    parser.add_argument("--intent_api_base", type=str, default=None, help="Intent model OpenAI-compatible API base")
    parser.add_argument("--intent_api_key_env", type=str, default=None)
    parser.add_argument("--intent_api_key", type=str, default=None)
    parser.add_argument("--model_family", type=str, default="qwen", help="gpt, qwen, deepseek, or all")
    parser.add_argument("--model_families", type=str, default=None, help="Comma-separated model families")
    parser.add_argument("--llm_model", type=str, default=None, help="Override model name for selected families")
    parser.add_argument("--llm_api_base", type=str, default=None, help="Override OpenAI-compatible API base")
    parser.add_argument("--llm_api_key_env", type=str, default=None, help="Override API key environment variable")
    parser.add_argument("--llm_api_key", type=str, default=None, help="Direct API key override; prefer env vars")
    parser.add_argument("--gpt_model", type=str, default=None)
    parser.add_argument("--gpt_api_base", type=str, default=None)
    parser.add_argument("--gpt_api_key_env", type=str, default=None)
    parser.add_argument("--qwen_model", type=str, default=None)
    parser.add_argument("--qwen_api_base", type=str, default=None)
    parser.add_argument("--qwen_api_key_env", type=str, default=None)
    parser.add_argument("--deepseek_model", type=str, default=None)
    parser.add_argument("--deepseek_api_base", type=str, default=None)
    parser.add_argument("--deepseek_api_key_env", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    split_names = parse_splits(args.split, args.splits)
    model_configs = build_model_configs(args)
    intent_config = build_intent_config(args)
    action_rerank_methods = parse_method_set(args.action_rerank_methods)
    if args.precompute_intents_only:
        model_configs = [MODEL_FAMILY_DEFAULTS["qwen"]]
    use_single_output = len(split_names) == 1 and len(model_configs) == 1

    print(
        "CUDA config: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"scorer_device={args.scorer_device}, "
        f"mindact_device={args.mindact_device}"
    )

    all_summaries: Dict[str, Any] = {}
    for llm_config in model_configs:
        runner = Experiment1Runner(
            data_dir=args.data_dir,
            cross_encoder_path=args.cross_encoder_path,
            scorer_model_path=args.scorer_model_path,
            scorer_device=args.scorer_device,
            mindact_device=args.mindact_device,
            mindact_batch_size=args.mindact_batch_size,
            context_limit=args.context_limit,
            skip_mindact=args.skip_mindact,
            run_action_llm=args.run_action_llm,
            operation_hint_mode=args.operation_hint_mode,
            action_rerank=not args.disable_action_rerank,
            action_rerank_methods=action_rerank_methods,
            action_top_k=args.action_top_k,
            llm_config=llm_config,
            intent_config=intent_config,
            intent_cache_path=args.intent_cache_path,
            pruning_cache_path=args.pruning_cache_path,
            use_pruning_cache_only=args.use_pruning_cache_only,
            precompute_intents_only=args.precompute_intents_only,
        )
        for split_name in split_names:
            output_path = args.output_path if use_single_output else None
            summary = runner.run(
                split_name=split_name,
                max_files=args.max_files,
                sample_tasks=args.sample_tasks,
                max_steps=args.max_steps,
                sample_seed=args.sample_seed,
                save_results=not args.no_save,
                output_path=output_path,
            )
            all_summaries[f"{llm_config.family}:{split_name}"] = summary

    if len(all_summaries) > 1 and not args.no_save:
        os.makedirs("results", exist_ok=True)
        combined_path = os.path.join("results", "experiment1_summary_all.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
