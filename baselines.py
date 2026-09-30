"""Faithful offline baselines for Browser-Use and MindAct-style pruning.

The real Browser-Use selector map is produced from a live browser state with
CDP, accessibility data, paint-order filtering, and event-listener detection.
Mind2Web offline HTML does not contain all of that runtime state, so this file
implements the closest static equivalent over the saved HTML.

MindAct is implemented as a candidate-generation plus cross-encoder reranking
baseline. It uses task context, previous action history, the current step, and
rich element descriptions instead of text-only pairs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup, Comment, Tag

from config import USABLE_CONTEXT_RATIO
from utils import (
    MAX_ELEMENT_TEXT_LEN,
    estimate_tokens,
    extract_all_elements,
    get_element_text,
    get_element_type,
    parse_html,
)


def _attr_value(elem: Tag, attr: str) -> str:
    value = elem.get(attr, "")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_bounds(elem: Tag) -> Tuple[float, float, float, float] | None:
    raw = elem.get("bounding_box_rect")
    if not raw:
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = (float(p) for p in parts)
    except ValueError:
        return None
    return x, y, width, height


def _clean_soup(html_content: str) -> BeautifulSoup:
    soup = parse_html(html_content)
    for tag in soup(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()
    return soup


def _has_form_control_descendant(elem: Tag, max_depth: int = 2) -> bool:
    if max_depth <= 0:
        return False
    for child in elem.children:
        if not isinstance(child, Tag):
            continue
        if child.name and child.name.lower() in {"input", "select", "textarea"}:
            return True
        if _has_form_control_descendant(child, max_depth - 1):
            return True
    return False


def _element_label(elem: Tag, max_len: int = MAX_ELEMENT_TEXT_LEN) -> str:
    parts: List[str] = []
    for attr in ("aria-label", "aria_label", "placeholder", "alt", "title", "value", "name"):
        value = _normalize_space(_attr_value(elem, attr))
        if value and value.lower() not in {"none", "null"}:
            parts.append(value)

    text = _normalize_space(get_element_text(elem, max_len))
    if text and text.lower() not in {"none", "null"}:
        parts.append(text)

    seen = set()
    deduped = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    return " | ".join(deduped)[:max_len]


def compact_description(elem: Tag, include_backend_id: bool = False) -> str:
    """Build a compact but target-identifying element description."""
    tag = elem.name.lower() if elem.name else "text"
    label = _element_label(elem, 120)
    attrs: Dict[str, str] = {}
    label_lower = label.lower()

    important_attrs = (
        "type",
        "role",
        "name",
        "placeholder",
        "aria-label",
        "aria_label",
        "value",
        "alt",
        "title",
        "href",
        "is_clickable",
    )
    if include_backend_id:
        important_attrs = ("backend_node_id",) + important_attrs

    for attr in important_attrs:
        if not elem.has_attr(attr):
            continue
        value = _normalize_space(_attr_value(elem, attr))
        if not value or value.lower() in {"none", "null"}:
            continue
        if attr not in {"type", "role", "is_clickable", "backend_node_id"} and value.lower() in label_lower:
            continue
        if attr == "href":
            value = value[:80]
        attrs[attr.replace("aria_label", "aria-label")] = value[:100]

    if attrs:
        attr_str = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        return f"<{tag} {attr_str}>{label}</{tag}>"
    return f"<{tag}>{label}</{tag}>"


def rich_element_description(elem: Tag) -> str:
    """Element representation for MindAct-style cross-encoder reranking."""
    tag = elem.name.lower() if elem.name else "text"
    elem_type = get_element_type(elem)
    label = _element_label(elem, 160)
    attrs = []
    for attr in (
        "type",
        "role",
        "name",
        "placeholder",
        "aria-label",
        "aria_label",
        "value",
        "alt",
        "title",
        "id",
        "class",
        "is_clickable",
    ):
        if elem.has_attr(attr):
            value = _normalize_space(_attr_value(elem, attr))
            if value and value.lower() not in {"none", "null"}:
                attrs.append(f"{attr}={value[:80]}")

    parent_text = ""
    if isinstance(elem.parent, Tag):
        parent_text = _normalize_space(get_element_text(elem.parent, 120))
    sibling_texts = []
    for sibling in list(elem.previous_siblings)[-1:] + list(elem.next_siblings)[:1]:
        if isinstance(sibling, Tag):
            text = _normalize_space(get_element_text(sibling, 80))
            if text:
                sibling_texts.append(text)

    fields = [
        f"tag: {tag}",
        f"type: {elem_type}",
        f"text: {label}",
    ]
    if attrs:
        fields.append(f"attrs: {'; '.join(attrs)}")
    if parent_text and parent_text.lower() != label.lower():
        fields.append(f"parent: {parent_text}")
    if sibling_texts:
        fields.append(f"nearby: {' | '.join(sibling_texts)}")
    return " ; ".join(fields)


class BrowserUsePruner:
    """Static Browser-Use selector-map approximation for offline Mind2Web HTML."""

    INTERACTIVE_TAGS = {
        "button",
        "input",
        "select",
        "textarea",
        "a",
        "details",
        "summary",
        "option",
        "optgroup",
    }
    INTERACTIVE_ROLES = {
        "button",
        "link",
        "menuitem",
        "option",
        "radio",
        "checkbox",
        "tab",
        "textbox",
        "combobox",
        "slider",
        "spinbutton",
        "search",
        "searchbox",
        "row",
        "cell",
        "gridcell",
        "listbox",
        "switch",
    }
    INTERACTIVE_ATTRIBUTES = {
        "onclick",
        "onmousedown",
        "onmouseup",
        "onkeydown",
        "onkeyup",
        "onchange",
        "onsubmit",
        "tabindex",
        "data-action",
    }
    SEARCH_INDICATORS = {
        "search",
        "magnify",
        "glass",
        "lookup",
        "find",
        "query",
        "search-icon",
        "search-btn",
        "search-button",
        "searchbox",
    }
    STATE_ATTRIBUTES = {
        "checked",
        "selected",
        "expanded",
        "pressed",
        "required",
        "autocomplete",
        "contenteditable",
    }

    def _is_disabled_or_hidden(self, elem: Tag) -> bool:
        if elem.has_attr("disabled"):
            return True
        aria_disabled = _attr_value(elem, "aria-disabled").lower()
        if aria_disabled == "true":
            return True
        aria_hidden = _attr_value(elem, "aria-hidden").lower()
        if aria_hidden == "true" and elem.get("is_clickable") != "true":
            return True
        return False

    def _has_search_signal(self, elem: Tag) -> bool:
        class_text = _attr_value(elem, "class").lower()
        elem_id = _attr_value(elem, "id").lower()
        if any(indicator in class_text or indicator in elem_id for indicator in self.SEARCH_INDICATORS):
            return True
        for attr_name, attr_value in elem.attrs.items():
            if attr_name.startswith("data-"):
                value = _attr_value(elem, attr_name).lower()
                if any(indicator in value for indicator in self.SEARCH_INDICATORS):
                    return True
        return False

    def is_interactive(self, elem: Tag) -> bool:
        if elem.name is None:
            return False
        tag = elem.name.lower()
        if tag in {"html", "body"}:
            return False
        if self._is_disabled_or_hidden(elem):
            return False

        if elem.get("is_clickable") == "true":
            return True

        if tag in {"iframe", "frame"}:
            bounds = _parse_bounds(elem)
            if bounds is None:
                return True
            _, _, width, height = bounds
            return width > 100 and height > 100

        if tag == "label":
            if elem.get("for"):
                return False
            if _has_form_control_descendant(elem, 2):
                return True

        if tag == "span" and _has_form_control_descendant(elem, 2):
            return True

        if self._has_search_signal(elem):
            return True

        if tag in self.INTERACTIVE_TAGS:
            return True

        if any(elem.has_attr(attr) for attr in self.INTERACTIVE_ATTRIBUTES):
            return True

        role = _attr_value(elem, "role").lower()
        if role in self.INTERACTIVE_ROLES:
            return True

        if any(elem.has_attr(attr) for attr in self.STATE_ATTRIBUTES):
            return True

        style = _attr_value(elem, "style").lower()
        if "cursor: pointer" in style or "cursor:pointer" in style:
            return True

        bounds = _parse_bounds(elem)
        if bounds is not None:
            _, _, width, height = bounds
            if 10 <= width <= 50 and 10 <= height <= 50:
                icon_attrs = {"class", "role", "onclick", "data-action", "aria-label", "aria_label", "title"}
                if any(elem.has_attr(attr) for attr in icon_attrs):
                    return True

        return False

    def process(self, html_content: str, context_limit: int = 8000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        soup = _clean_soup(html_content)
        total_elements = len(soup.find_all())
        all_elements = [elem for elem in soup.find_all() if self.is_interactive(elem)]
        if not all_elements:
            return [], {"total": total_elements, "kept": 0, "prune_rate": 1.0}

        usable_tokens = int(context_limit * USABLE_CONTEXT_RATIO)
        kept: List[Dict[str, Any]] = []
        tokens_used = 0
        for elem in all_elements:
            desc = compact_description(elem)
            elem_tokens = estimate_tokens(desc)
            if tokens_used + elem_tokens <= usable_tokens or not kept:
                kept.append(
                    {
                        "element": elem,
                        "tag": elem.name.lower() if elem.name else "text",
                        "type": get_element_type(elem),
                        "text": _element_label(elem, MAX_ELEMENT_TEXT_LEN),
                        "relevance": 0.0,
                        "description": desc,
                    }
                )
                tokens_used += elem_tokens
            else:
                break

        return kept, {
            "total": total_elements,
            "kept": len(kept),
            "prune_rate": 1 - len(kept) / total_elements if total_elements > 0 else 0.0,
        }


class MindActPruner:
    """MindAct-style candidate generation plus cross-encoder reranking."""

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 32):
        print(f"Loading MindAct cross-encoder from {model_path}...")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, device=device)
        self.batch_size = batch_size
        print("MindAct cross-encoder loaded.")

    def build_query(
        self,
        task_description: str,
        step_description: str | None = None,
        action_history: List[str] | None = None,
    ) -> str:
        parts = [f"Task: {task_description}"]
        if action_history:
            parts.append("Previous actions: " + " ; ".join(action_history[-5:]))
        if step_description:
            parts.append(f"Current step: {step_description}")
        return " ".join(parts)

    def process(
        self,
        html_content: str,
        task_description: str,
        context_limit: int = 8000,
        step_description: str | None = None,
        action_history: List[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        soup = _clean_soup(html_content)
        for elem in soup.select(
            '[hidden], [style*="display: none"], [style*="display:none"], '
            '[style*="visibility: hidden"], [style*="visibility:hidden"], [aria-hidden="true"]'
        ):
            if elem.get("is_clickable") != "true":
                elem.decompose()

        all_elements = extract_all_elements(soup)
        total_elements = len(all_elements)
        if not all_elements:
            return [], {"total": total_elements, "kept": 0, "prune_rate": 1.0}

        query = self.build_query(task_description, step_description, action_history)
        descriptions = [rich_element_description(elem) for elem in all_elements]
        pairs = [[query, description] for description in descriptions]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored = sorted(zip(all_elements, descriptions, scores), key=lambda item: item[2], reverse=True)
        usable_tokens = int(context_limit * USABLE_CONTEXT_RATIO)
        kept: List[Dict[str, Any]] = []
        tokens_used = 0
        for elem, rich_desc, score in scored:
            desc = compact_description(elem)
            elem_tokens = estimate_tokens(desc)
            if tokens_used + elem_tokens <= usable_tokens or not kept:
                kept.append(
                    {
                        "element": elem,
                        "tag": elem.name.lower() if elem.name else "text",
                        "type": get_element_type(elem),
                        "text": _element_label(elem, MAX_ELEMENT_TEXT_LEN),
                        "relevance": float(score),
                        "description": desc,
                        "rerank_description": rich_desc,
                    }
                )
                tokens_used += elem_tokens
            else:
                break

        return kept, {
            "total": total_elements,
            "kept": len(kept),
            "prune_rate": 1 - len(kept) / total_elements if total_elements > 0 else 0.0,
        }
