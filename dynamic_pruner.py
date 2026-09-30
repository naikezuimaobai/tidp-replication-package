"""Dynamic pruning with category-specific rules - Optimized"""

from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Any, Optional, Tuple
import re

from config import *
from utils import (
    get_element_text, 
    get_element_type, 
    get_relevant_attributes,
    get_dom_depth, 
    is_near_element, 
    extract_all_elements,
    estimate_tokens, 
    compress_element,
    MAX_ELEMENT_TEXT_LEN,
    basic_node_filtering,
    parse_html
)


class DynamicPruner:
    """Intent-category-driven dynamic DOM pruner."""

    INTERACTIVE_TAGS = {'a', 'button', 'input', 'select', 'textarea', 'option', 'summary'}
    INTERACTIVE_ROLES = {
        'button', 'link', 'textbox', 'searchbox', 'checkbox', 'radio',
        'combobox', 'listbox', 'menuitem', 'option', 'tab', 'switch'
    }
    EVENT_ATTRIBUTES = {'onclick', 'onchange', 'onsubmit', 'onmousedown', 'onmouseup', 'onkeydown', 'onkeyup'}
    COMPACT_ATTRS = ('type', 'role', 'name', 'placeholder', 'aria-label', 'aria_label', 'value', 'alt', 'title')
    
    def __init__(
        self,
        scorer=None,
        enable_basic_filter: bool = True,
        enable_intent_filter: bool = True,
        enable_relevance_filter: bool = True,
        enable_inheritance_recovery: bool = True,
    ):
        self.scorer = scorer
        self.enable_basic_filter = enable_basic_filter
        self.enable_intent_filter = enable_intent_filter
        self.enable_relevance_filter = enable_relevance_filter
        self.enable_inheritance_recovery = enable_inheritance_recovery
        # Navigation keywords used for content-browsing exceptions.
        self.browsing_keywords = [
            '下一页', '上一页', '加载更多', '展开全文', '显示更多', '查看更多',
            '翻页', 'next', 'load more', 'show more', 'view more', 'see more',
            'expand', 'read more', '继续', '加载', '更多'
        ]

    def _attr_text(self, elem: Tag, attr: str) -> str:
        value = elem.get(attr, '')
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        return str(value).strip()

    def _element_label(self, elem: Tag, max_len: int = MAX_ELEMENT_TEXT_LEN) -> str:
        """Extract a compact action-oriented label for icons, inputs, and unlabeled buttons."""
        parts = []
        for attr in ('aria-label', 'aria_label', 'placeholder', 'alt', 'title', 'value', 'name'):
            value = self._attr_text(elem, attr)
            if value and value.lower() not in {'none', 'null'}:
                parts.append(value)

        text = get_element_text(elem, max_len)
        if text and text.lower() not in {'none', 'null'}:
            parts.append(text)

        seen = set()
        deduped = []
        for part in parts:
            normalized = re.sub(r'\s+', ' ', part).strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                deduped.append(normalized)

        return ' | '.join(deduped)[:max_len]

    def _is_interactive_candidate(self, elem: Tag) -> bool:
        tag = elem.name.lower() if elem.name else ''
        role = elem.get('role', '').lower()
        style = elem.get('style', '').lower()

        if tag in self.INTERACTIVE_TAGS:
            return True
        if role in self.INTERACTIVE_ROLES:
            return True
        if elem.get('is_clickable') == 'true' or elem.has_attr('tabindex'):
            return True
        if any(elem.has_attr(attr) for attr in self.EVENT_ATTRIBUTES):
            return True

        if 'cursor: pointer' in style or 'cursor:pointer' in style:
            return True

        parent = elem.parent if isinstance(elem.parent, Tag) else None
        if tag in {'svg', 'img', 'span', 'div'} and parent is not None:
            parent_tag = parent.name.lower() if parent.name else ''
            parent_role = parent.get('role', '').lower()
            if parent_tag in {'a', 'button'} or parent_role in self.INTERACTIVE_ROLES:
                return True

        return False

    def _generic_keep_condition(self, elem: Tag, text: str) -> bool:
        """Unified non-intent-specific keep rule for ablation."""
        tag = elem.name.lower() if elem.name else ''
        if self._is_interactive_candidate(elem):
            return True
        if tag in {'label', 'fieldset', 'legend'}:
            return True
        if tag in {'div', 'span', 'p', 'li', 'section', 'article'} and len(text) > 5:
            return True
        return False
    
    def check_keep_conditions(self, elem: Tag, intent: str) -> bool:
        """Check whether an element satisfies the retention rules."""
        tag = elem.name.lower() if elem.name else ''
        class_str = ' '.join(elem.get('class', [])) if elem.get('class') else ''
        text = self._element_label(elem, MAX_ELEMENT_TEXT_LEN).lower()

        if intent == 'generic':
            return self._generic_keep_condition(elem, text)
        
        # ====== General retention rules for all intents ======
        # 1. Elements with accessible labels or alternative text are usually important.
        if elem.get('aria-label') or elem.get('aria_label') or elem.get('alt') or elem.get('title'):
            return True
        
        # 2. Interactive elements with an explicit ARIA role.
        role = elem.get('role', '').lower()
        if role in self.INTERACTIVE_ROLES:
            return True
        
        # 3. Elements with onclick or similar event handlers.
        if any(elem.has_attr(attr) for attr in self.EVENT_ATTRIBUTES):
            return True
        
        # ====== Intent-specific rules ======
        if intent == 'content_browsing':
            # 1. Keep links and buttons.
            if tag == 'a' or tag == 'button':
                return True
            
            # 2. Keep block elements with sufficiently long text.
            if tag in ['div', 'section', 'article', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li']:
                if len(text) > 2:  # Retain the current threshold.
                    return True
            
            # 3. Keep meaningful images and media.
            if tag in ['img', 'figure', 'video', 'svg']:
                return True
            
            # 4. Keep navigation-related elements with relaxed conditions.
            nav_indicators = ['nav', 'menu', 'breadcrumb', 'tab', 'pagination']
            if any(ind in class_str for ind in nav_indicators):
                return True
            
            # 5. Keep textual elements as a safety net.
            if len(text) > 5:
                return True
            
            return False
        
        elif intent == 'information_input':
            # Information input: retain input controls with a permissive policy.
            
            # 1. Retain all input elements except radio and checkbox controls.
            if tag == 'input':
                input_type = elem.get('type', '').lower()
                if input_type not in ['radio', 'checkbox']:
                    return True
            
            # 2. Retain textarea and contenteditable elements.
            if tag in ['textarea'] or elem.get('contenteditable') == 'true':
                return True
            
            # 3. Retain elements with a placeholder.
            if elem.get('placeholder'):
                return True
            
            # 4. Retain elements with a name attribute.
            if elem.get('name'):
                return True
            
            # 5. Retain IDs containing input or search keywords.
            elem_id = elem.get('id', '').lower()
            if elem_id and any(kw in elem_id for kw in ['search', 'input', 'text', 'lookup', 'query', 'find', 'email', 'password', 'username', 'login']):
                return True
            
            # 6. Retain labels for input controls.
            if tag == 'label':
                return True
            
            # 7. Retain search and submit buttons with relaxed keywords.
            if tag == 'button' or (tag == 'input' and elem.get('type') in ['submit', 'button']):
                return True
            
            # 8. Retain input-area containers and descriptions as a safety net.
            if len(text) > 3 and tag in ['span', 'div', 'p', 'label']:
                return True
            
            return False
        
        elif intent == 'information_input':
            # Information input: retain input controls.
            if tag == 'input':
                input_type = elem.get('type', '').lower()
                if input_type not in ['radio', 'checkbox']:
                    return True
            if tag in ['textarea'] or elem.get('contenteditable') == 'true':
                return True
            
            # Search and submit buttons with relaxed keywords.
            if tag == 'button' or (tag == 'input' and elem.get('type') in ['submit', 'button']):
                btn_text = text
                if any(kw in btn_text for kw in ['search', 'submit', 'send', '搜索', '提交', 'go', 'next', 'login', 'sign']):
                    return True
            
            # Labels and descriptions for input fields.
            if tag == 'label' and elem.get('for'):
                return True
            if len(text) > 3 and tag in ['span', 'div', 'p']:
                return True
            
            return False
        
        elif intent == 'button_interaction':
            # Button interaction: retain clickable elements with a permissive policy.
            if self._is_interactive_candidate(elem):
                return True
            
            # Short, highly relevant text blocks may be JavaScript-bound click targets.
            if len(text) > 3 and tag in {'div', 'span', 'li'}:
                return True
            
            return False
        
        return False
    
    def check_delete_conditions(self, elem: Tag, intent: str, all_form_controls: List[Tag] = None) -> bool:
        """Check whether an element satisfies the deletion rules."""
        tag = elem.name.lower() if elem.name else ''
        class_str = ' '.join(elem.get('class', [])) if elem.get('class') else ''
        style = elem.get('style', '')
        
        if intent == 'content_browsing':
            # Remove advertisements and pop-ups.
            ad_indicators = ['ad', 'advertisement', 'banner', 'popup', 'modal']
            if any(ind in class_str.lower() for ind in ad_indicators):
                return True
            # Remove unrelated social elements.
            social_indicators = ['social', 'share', 'like', 'comment']
            if any(ind in class_str.lower() for ind in social_indicators):
                return True
            return False
        
        elif intent == 'form_selection':
            # Remove elements outside form regions.
            if all_form_controls is not None:
                # Proximity: DOM depth difference <= 3 or a common ancestor.
                is_near = any(is_near_element(elem, ctrl, MAX_DOM_DEPTH_DIFF) for ctrl in all_form_controls)
                in_form = elem.find_parent('form') is not None
                
                # Delete only when the element is outside forms and far from controls.
                if not is_near and not in_form:
                    # Preserve form labels and descriptive text.
                    tag = elem.name.lower() if elem.name else ''
                    text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
                    if tag not in ['label', 'fieldset', 'legend'] and len(text) < 10:
                        return True
            
            # Remove unrelated navigation regions.
            layout_indicators = ['navbar', 'sidebar', 'footer', 'header']
            if any(ind in class_str.lower() for ind in layout_indicators):
                return True
            return False

        elif intent == 'information_input':
            # Remove elements outside input regions.
            if all_form_controls is not None:
                # Proximity: DOM depth difference <= 3 or a common ancestor.
                is_near = any(is_near_element(elem, ctrl, MAX_DOM_DEPTH_DIFF) for ctrl in all_form_controls)
                
                # Delete only when the element is far from input controls.
                if not is_near:
                    # Preserve input labels and hint text.
                    tag = elem.name.lower() if elem.name else ''
                    text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
                    if tag not in ['label', 'span', 'div', 'p'] or len(text) < 5:
                        return True
            
            # Remove advertisements.
            ad_indicators = ['ad', 'promotion', 'recommendation']
            if any(ind in class_str.lower() for ind in ad_indicators):
                return True
            return False
        # elif intent == 'form_selection':
        # Implementation note.
        #     if all_form_controls is not None:
        #         is_near = any(is_near_element(elem, ctrl, MAX_DOM_DEPTH_DIFF) for ctrl in all_form_controls)
        #         if not is_near and elem.find_parent('form') is None:
        #             return True
        # Implementation note.
        #     layout_indicators = ['navbar', 'sidebar', 'footer', 'header']
        #     if any(ind in class_str.lower() for ind in layout_indicators):
        #         return True
        #     return False
        
        # elif intent == 'information_input':
        # Implementation note.
        #     if all_form_controls is not None:
        #         is_near = any(is_near_element(elem, ctrl, MAX_DOM_DEPTH_DIFF) for ctrl in all_form_controls)
        #         if not is_near:
        #             return True
        # Implementation note.
        #     ad_indicators = ['ad', 'promotion', 'recommendation']
        #     if any(ind in class_str.lower() for ind in ad_indicators):
        #         return True
        #     return False
        
        elif intent == 'button_interaction':
            # Remove advertisements and decorative elements.
            ad_indicators = ['ad', 'banner', 'decoration']
            if any(ind in class_str.lower() for ind in ad_indicators):
                return True
            return False
        
        return False
    

    def get_thresholds(self, intent: str, total_elements: int) -> Dict[str, Any]:
        """Get dynamic thresholds with the current scaling policy."""
        if intent == 'generic':
            # No intent-aware filtering: use a deliberately broad, task-agnostic
            # keep policy. This preserves more potentially relevant and irrelevant
            # nodes before the final context-budget truncation, making the ablation
            # reflect the loss of intent-specific constraints instead of falling
            # back to an averaged intent profile.
            base = {'high': 0.45, 'low': 0.20, 'p1': 0.25, 'p2': 0.85}
        else:
            base = INTENT_BASE_PARAMS[intent].copy()
        
        # Compute the scaling factor.
        if total_elements <= N_MIN:
            s = 0
        elif total_elements >= N_MAX:
            s = 1
        else:
            s = (total_elements - N_MIN) / (N_MAX - N_MIN)
        
        # Apply the configured scaling policy.
        thresholds = {
            'high': min(0.80, base['high'] + s * 0.10),  # Implementation note.
            'low': max(0.05, base['low'] - s * 0.05),    # Implementation note.
            'p1': max(0.10, base['p1'] * (1 - s * 0.2)), # Implementation note.
            'p2': max(0.15, base['p2'] * (1 - s * 0.15)) # Implementation note.
        }
        
        thresholds['p1_count'] = max(1, int(total_elements * thresholds['p1']))
        thresholds['p2_count'] = max(2, int(total_elements * thresholds['p2']))
        
        return thresholds
    # def get_thresholds(self, intent: str, total_elements: int) -> Dict[str, Any]:
    # Implementation note.
    #     base = INTENT_BASE_PARAMS[intent].copy()
        
    # Implementation note.
    #     if total_elements <= N_MIN:
    #         s = 0
    #     elif total_elements >= N_MAX:
    #         s = 1
    #     else:
    #         s = (total_elements - N_MIN) / (N_MAX - N_MIN)
        
    # Implementation note.
    #     thresholds = {
    #         'high': min(0.95, base['high'] + s * 0.15),
    #         'low': max(0.10, base['low'] - s * 0.15),
    #         'p1': max(0.05, base['p1'] * (1 - s * 0.5)),
    #         'p2': max(0.10, base['p2'] * (1 - s * 0.3))
    #     }
        
    #     thresholds['p1_count'] = max(1, int(total_elements * thresholds['p1']))
    #     thresholds['p2_count'] = max(2, int(total_elements * thresholds['p2']))
        
    #     return thresholds
    
    def prune(self, html_content: str, intent: str, subtask_description: str,
              context_limit: int = CONTEXT_LIMIT_TOKENS) -> Tuple[List[Dict], Dict]:
        """
        Execute the complete pruning pipeline.
        """
        # Step 1: parse HTML.
        soup = parse_html(html_content)
        
        # Step 2: apply basic filtering.
        if self.enable_basic_filter:
            soup = basic_node_filtering(soup)
        
        # Step 3: extract candidate elements.
        all_elements = extract_all_elements(soup)
        total_elements = len(all_elements)
        
        if total_elements == 0:
            return [], {'total': 0, 'kept': 0, 'fallback_triggered': True}
        
        # Step 4: set the query.
        if self.scorer and self.enable_relevance_filter:
            self.scorer.set_query(subtask_description)
        
        # Step 5: batch-compute relevance for all elements.
        relevance_scores = {}
        if self.scorer and self.enable_relevance_filter and all_elements:
            # Extract text in batches.
            element_texts = [self._element_label(elem, MAX_ELEMENT_TEXT_LEN) for elem in all_elements]
            
            # Compute scores in batches.
            scores = self.scorer.compute_batch_relevance(element_texts)
            
            for elem, score in zip(all_elements, scores):
                relevance_scores[elem] = score
        
        # Step 6: obtain dynamic thresholds.
        effective_intent = intent if self.enable_intent_filter else 'generic'
        thresholds = self.get_thresholds(effective_intent, total_elements)

        # Step 6.5: collect form controls for proximity checks.
        all_form_controls = []
        if self.enable_intent_filter and intent in ['form_selection', 'information_input']:
            for elem in all_elements:
                tag = elem.name.lower() if elem.name else ''
                if tag in ['input', 'select', 'textarea', 'button']:
                    all_form_controls.append(elem)
        
        # Step 7: sort by relevance.
        if self.enable_relevance_filter:
            sorted_elements = sorted(all_elements, key=lambda e: relevance_scores.get(e, 0.5), reverse=True)
        else:
            sorted_elements = list(all_elements)
        
        # Step 8: decide which elements to retain.
        kept_elements = []
        for idx, elem in enumerate(sorted_elements):
            rank_pct = (idx + 1) / total_elements
            score = relevance_scores.get(elem, 0.5)
            
            # Retention decision.
            if not self.enable_relevance_filter:
                keep = self.check_keep_conditions(elem, effective_intent)
            elif score >= thresholds['high'] or rank_pct <= thresholds['p1']:
                keep = True
            elif score < thresholds['low'] and rank_pct > thresholds['p2']:
                keep = False
            else:
                # Use intent rules for the middle-relevance region.
                keep = self.check_keep_conditions(elem, effective_intent)
            
            # Implementation note.
            # if intent == 'information_input':
            #     tag_name = elem.name.lower() if elem.name else ''
            # Implementation note.
            #     if tag_name == 'input':
            #         input_type = elem.get('type', '').lower()
            #         if input_type not in ['radio', 'checkbox']:
            # Implementation note.
            #             if elem.get('placeholder') or elem.get('name') or elem.get('aria-label'):
            #                 keep = True
            # Implementation note.
            #             elif score >= 0.3:
            #                 keep = True
            # ====== Refined information-input protection ======
            if self.enable_intent_filter and intent == 'information_input':
                tag_name = elem.name.lower() if elem.name else ''
                if tag_name == 'input':
                    input_type = elem.get('type', '').lower()
                    if input_type not in ['radio', 'checkbox']:
                        # Force retention only for clearly identified inputs.
                        if elem.get('placeholder') or elem.get('name') or elem.get('aria-label'):
                            keep = True
                        # Retain other inputs only when sufficiently relevant.
                        elif score >= 0.4:
                            keep = True
                        # Do not force retention for other inputs.
                elif tag_name == 'textarea':
                    if elem.get('placeholder') or elem.get('name') or score >= 0.4:
                        keep = True
            
            if self.enable_intent_filter and intent == 'button_interaction':
                tag_name = elem.name.lower() if elem.name else ''
                if self._is_interactive_candidate(elem):
                    keep = True
                elif tag_name in {'div', 'span', 'li'} and score < 0.45:
                    keep = False

            if keep:
                kept_elements.append({
                    'element': elem,
                    'tag': elem.name.lower() if elem.name else 'text',
                    'type': get_element_type(elem),
                    'text': self._element_label(elem, MAX_ELEMENT_TEXT_LEN),
                    'relevance': score,
                    'description': self._format_element_description(elem)
                })
        # kept_elements = []
        # for idx, elem in enumerate(sorted_elements):
        #     rank_pct = (idx + 1) / total_elements
        #     score = relevance_scores.get(elem, 0.5)
            
        # Implementation note.
        #     if score >= thresholds['high'] or rank_pct <= thresholds['p1']:
        #         keep = True
        #     elif score < thresholds['low'] and rank_pct > thresholds['p2']:
        #         keep = False
        #     else:
        # Implementation note.
        #         keep = self.check_keep_conditions(elem, intent)
            
        #     if keep:
        #         kept_elements.append({
        #             'element': elem,
        #             'tag': elem.name.lower() if elem.name else 'text',
        #             'type': get_element_type(elem),
        #             'text': get_element_text(elem, MAX_ELEMENT_TEXT_LEN),
        #             'relevance': score,
        #             'description': self._format_element_description(elem)
        #         })
        
        # Step 9: run the fallback check.
        fallback_triggered = len(kept_elements) < FALLBACK_MIN_ELEMENTS
        if fallback_triggered and self.enable_inheritance_recovery:
            kept_elements = self._apply_fallback(all_elements, effective_intent, relevance_scores)
        
        # Step 10: enforce the context limit.
        kept_elements = self._enforce_context_limit(kept_elements, context_limit)
        
        # Collect statistics.
        stats = {
            'total': total_elements,
            'kept': len(kept_elements),
            'prune_rate': 1 - len(kept_elements) / total_elements if total_elements > 0 else 0,
            'fallback_triggered': fallback_triggered,
            'thresholds': thresholds,
            'ablation': {
                'enable_basic_filter': self.enable_basic_filter,
                'enable_intent_filter': self.enable_intent_filter,
                'enable_relevance_filter': self.enable_relevance_filter,
                'enable_inheritance_recovery': self.enable_inheritance_recovery,
            },
        }
        
        # Print cache statistics.
        if self.scorer and hasattr(self.scorer, 'get_cache_stats'):
            cache_stats = self.scorer.get_cache_stats()
            if cache_stats['total'] > 0:
                print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%} ({cache_stats['hits']}/{cache_stats['total']})")
        
        return kept_elements, stats
    
    def _format_element_description(self, elem: Tag) -> str:
        """Format an element as compact action-decision input."""
        tag = elem.name.lower() if elem.name else 'text'
        text = self._element_label(elem, 80)
        attrs = {}
        text_lower = text.lower()
        for attr in self.COMPACT_ATTRS:
            if elem.has_attr(attr):
                value = self._attr_text(elem, attr)
                if value and value.lower() not in {'none', 'null'} and value.lower() not in text_lower:
                    attrs[attr.replace('aria_label', 'aria-label')] = value[:60]
        
        if attrs:
            attr_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items() if v])
            return f"<{tag} {attr_str}>{text}</{tag}>"
        else:
            return f"<{tag}>{text}</{tag}>"
    
    def _apply_fallback(self, all_elements: List[Tag], intent: str,
                    relevance_scores: Dict[Tag, float]) -> List[Dict]:
        """Fallback logic that retains additional elements."""
        fallback_results = []
        
        # First attempt: apply retention conditions.
        for elem in all_elements:
            if self.check_keep_conditions(elem, intent):
                fallback_results.append({
                    'element': elem,
                    'tag': elem.name.lower() if elem.name else 'text',
                    'type': get_element_type(elem),
                    'text': self._element_label(elem, MAX_ELEMENT_TEXT_LEN),
                    'relevance': relevance_scores.get(elem, 0.5),
                    'description': self._format_element_description(elem)
                })
        
        # If necessary, add all elements with text.
        if len(fallback_results) < FALLBACK_MIN_ELEMENTS:
            for elem in all_elements:
                text = self._element_label(elem, MAX_ELEMENT_TEXT_LEN)
                if len(text) > 2:  # Retain elements with meaningful text.
                    # Avoid duplicates.
                    if not any(r['element'] == elem for r in fallback_results):
                        fallback_results.append({
                            'element': elem,
                            'tag': elem.name.lower() if elem.name else 'text',
                            'type': get_element_type(elem),
                            'text': text,
                            'relevance': relevance_scores.get(elem, 0.5),
                            'description': self._format_element_description(elem)
                        })
        
        # If necessary, retain all interactive elements.
        if len(fallback_results) < FALLBACK_MIN_ELEMENTS:
            interactive_tags = {'a', 'button', 'input', 'select', 'textarea', 'option'}
            for elem in all_elements:
                if elem.name and elem.name.lower() in interactive_tags:
                    if not any(r['element'] == elem for r in fallback_results):
                        text = self._element_label(elem, MAX_ELEMENT_TEXT_LEN)
                        fallback_results.append({
                            'element': elem,
                            'tag': elem.name.lower() if elem.name else 'text',
                            'type': get_element_type(elem),
                            'text': text,
                            'relevance': relevance_scores.get(elem, 0.5),
                            'description': self._format_element_description(elem)
                        })
        
        # Sort by relevance while retaining more elements.
        fallback_results.sort(key=lambda x: x['relevance'], reverse=True)
        max_fallback = max(FALLBACK_MIN_ELEMENTS * 5, 50)  # Retain at least 50 when possible.
        if len(fallback_results) > max_fallback:
            fallback_results = fallback_results[:max_fallback]
        
        return fallback_results
    # def _apply_fallback(self, all_elements: List[Tag], intent: str,
    #                     relevance_scores: Dict[Tag, float]) -> List[Dict]:
    # Implementation note.
    #     fallback_results = []
    #     for elem in all_elements:
    #         if self.check_keep_conditions(elem, intent):
    #             fallback_results.append({
    #                 'element': elem,
    #                 'tag': elem.name.lower() if elem.name else 'text',
    #                 'type': get_element_type(elem),
    #                 'text': get_element_text(elem, MAX_ELEMENT_TEXT_LEN),
    #                 'relevance': relevance_scores.get(elem, 0.5),
    #                 'description': self._format_element_description(elem)
    #             })
        
    #     if len(fallback_results) == 0:
    #         for elem in all_elements:
    #             text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
    #             if len(text) > 5:
    #                 fallback_results.append({
    #                     'element': elem,
    #                     'tag': elem.name.lower() if elem.name else 'text',
    #                     'type': get_element_type(elem),
    #                     'text': text,
    #                     'relevance': relevance_scores.get(elem, 0.5),
    #                     'description': self._format_element_description(elem)
    #                 })
        
    #     fallback_results.sort(key=lambda x: x['relevance'], reverse=True)
    #     max_fallback = FALLBACK_MIN_ELEMENTS * 3
    #     if len(fallback_results) > max_fallback:
    #         fallback_results = fallback_results[:max_fallback]
        
    #     return fallback_results
    
    def _enforce_context_limit(self, kept_elements: List[Dict], context_limit: int) -> List[Dict]:
        """Enforce the context-size limit."""
        if not kept_elements:
            return kept_elements
        
        usable_tokens = int(context_limit * USABLE_CONTEXT_RATIO)
        current_tokens = sum(estimate_tokens(elem['description']) for elem in kept_elements)
        
        if current_tokens <= usable_tokens:
            return kept_elements
        
        kept_elements.sort(key=lambda x: x['relevance'], reverse=True)
        
        result = []
        tokens_used = 0
        for elem in kept_elements:
            elem_tokens = estimate_tokens(elem['description'])
            if tokens_used + elem_tokens <= usable_tokens:
                result.append(elem)
                tokens_used += elem_tokens
            elif len(result) == 0:
                result.append(elem)
                break
        
        return result

