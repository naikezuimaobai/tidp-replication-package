"""RQ1: Five Methods Comprehensive Comparison (Complete Metrics)"""

import argparse
import json
import os
import time
import sys
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiktoken
from config import *
from baselines import BrowserUsePruner, MindActPruner
from data_loader import Mind2WebLoader
from intent_classifier import IntentClassifier
from relevance_scorer import RelevanceScorer
from dynamic_pruner import DynamicPruner
from evaluator import Evaluator
from utils import (
    parse_html, get_element_text, get_relevant_attributes, get_element_type,
    estimate_tokens, extract_all_elements, MAX_ELEMENT_TEXT_LEN
)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    if not text or len(text.strip()) == 0:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def build_description(tag, attrs, text):
    if attrs:
        attr_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items() if v])
        return f"<{tag} {attr_str}>{text}</{tag}>"
    return f"<{tag}>{text}</{tag}>"


class SimpleActionDecider:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=LLM_API_BASE, api_key="EMPTY", timeout=30.0)
        self.model = LLM_MODEL

    def decide_action(self, kept_elements: List[Dict], intent: str,
                      task_description: str, step_description: str) -> Tuple[str, float]:
        page_summary = ""
        for i, elem in enumerate(kept_elements[:50]):
            page_summary += f"[{i+1}] {elem['type']}: {elem['text'][:100]}\n"

        prompt = f"""You are a web agent. Based on the task requirements and filtered page elements, identify the most relevant element for the current step.

Task: {task_description}
Current step: {step_description}
Operation type: {intent}

Available elements (sorted by relevance):
{page_summary}

Instructions:
- Focus on elements ranked [1]-[10] first
- Match the operation type: {intent} actions require appropriate element types
- Consider element text, type, and attributes

Output exactly ONE action using the matching format:
- For clicking: CLICK [element_index]
- For typing: TYPE [your_text] [element_index]
- For selecting: SELECT [your_option] [element_index]

Examples:
- CLICK [3]
- TYPE iowa [5]
- SELECT Lowest Price [9]

Action:"""

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web agent. Output exactly one action."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=128,
                temperature=0
            )
            action = response.choices[0].message.content.strip()
        except Exception as e:
            action = f"Error: {e}"
        return action, (time.time() - start_time) * 1000


# ==================== Visible Truncation (32K) ====================
class VisibleTruncation:
    def process(self, html_content: str, context_limit: int = 32000) -> Tuple[List[Dict], Dict]:
        soup = parse_html(html_content)
        all_elements = extract_all_elements(soup)
        total_elements = len(all_elements)
        if total_elements == 0:
            return [], {'total': 0, 'kept': 0}

        usable_tokens = int(context_limit * USABLE_CONTEXT_RATIO)
        kept, tokens_used = [], 0
        for elem in all_elements:
            tag = elem.name.lower() if elem.name else 'text'
            text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
            attrs = get_relevant_attributes(elem)
            desc = build_description(tag, attrs, text)
            et = estimate_tokens(desc)
            if tokens_used + et <= usable_tokens:
                kept.append({'element': elem, 'tag': tag, 'type': get_element_type(elem),
                             'text': text, 'relevance': 0.0, 'description': desc})
                tokens_used += et
            else:
                if not kept:
                    kept.append({'element': elem, 'tag': tag, 'type': get_element_type(elem),
                                 'text': text, 'relevance': 0.0, 'description': desc})
                break
        return kept, {'total': total_elements, 'kept': len(kept)}


# ==================== Uniform Pruning (8K) ====================
class UniformPruner:
    INTERACTIVE_TAGS = {'a', 'button', 'input', 'select', 'textarea', 'option'}
    INTERACTIVE_ROLES = {
        'button', 'link', 'textbox', 'searchbox', 'checkbox', 'radio',
        'combobox', 'listbox', 'menuitem', 'option', 'tab', 'switch'
    }

    def process(self, html_content: str, context_limit: int = 8000) -> Tuple[List[Dict], Dict]:
        soup = parse_html(html_content)
        all_tags = len(soup.find_all())
        for tag in soup(['script', 'style', 'link', 'meta', 'noscript']):
            tag.decompose()
        for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
            comment.extract()
        for elem in soup.select('[hidden], [style*="display: none"], [style*="display:none"], '
                                '[style*="visibility: hidden"], [style*="visibility:hidden"], '
                                '[aria-hidden="true"]'):
            elem.decompose()

        all_elems = []
        for elem in soup.find_all():
            if elem.name is None:
                continue
            tag = elem.name.lower()
            if tag in self.INTERACTIVE_TAGS:
                all_elems.append(elem)
            elif elem.has_attr('onclick') or elem.has_attr('onchange') or elem.has_attr('onsubmit'):
                all_elems.append(elem)
            elif elem.get('role', '') in self.INTERACTIVE_ROLES:
                all_elems.append(elem)

        if not all_elems:
            return [], {'total': all_tags, 'kept': 0}

        usable_tokens = int(context_limit * USABLE_CONTEXT_RATIO)
        kept, tokens_used = [], 0
        for elem in all_elems:
            tag = elem.name.lower() if elem.name else 'text'
            text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
            attrs = get_relevant_attributes(elem)
            desc = build_description(tag, attrs, text)
            et = estimate_tokens(desc)
            if tokens_used + et <= usable_tokens:
                kept.append({'element': elem, 'tag': tag, 'type': get_element_type(elem),
                             'text': text, 'relevance': 0.0, 'description': desc})
                tokens_used += et
            else:
                if not kept:
                    kept.append({'element': elem, 'tag': tag, 'type': get_element_type(elem),
                                 'text': text, 'relevance': 0.0, 'description': desc})
                break
        return kept, {'total': all_tags, 'kept': len(kept)}


# ==================== Runner ====================
class FiveMethodRunner:
    def __init__(self, data_dir: str, cross_encoder_path: str):
        self.data_dir = data_dir
        self.loader = Mind2WebLoader(data_dir)
        self.intent_classifier = IntentClassifier()
        self.relevance_scorer = RelevanceScorer()
        self.pruner = DynamicPruner(scorer=self.relevance_scorer)
        self.visible = VisibleTruncation()
        self.uniform = UniformPruner()
        self.browseruse = BrowserUsePruner()
        self.mindact = MindActPruner(cross_encoder_path)
        self.action_decider = SimpleActionDecider()
        self.evaluator = Evaluator()

    def estimate_kept_tokens(self, kept):
        return count_tokens(''.join(e.get('description', '') + '\n' for e in kept))

    def run(self, split_name: str, max_tasks: int = None, save_results: bool = True):
        print(f"\n{'='*70}")
        print(f"Five Methods Comparison on {split_name}")
        print(f"{'='*70}")

        tasks = self.loader.load_split(split_name, max_files=max_tasks)
        if max_tasks and max_tasks < len(tasks):
            tasks = tasks[:max_tasks]
        print(f"Total tasks: {len(tasks)}")

        results = []
        total_steps = 0

        for task in tqdm(tasks, desc=f"Processing {split_name}"):
            for step in task.steps:
                if not step.raw_html or len(step.raw_html) < 100:
                    continue
                total_steps += 1
                subtask_desc = step.action_repr
                if step.operation_value:
                    subtask_desc = f"{step.action_repr} value: {step.operation_value}"
                original_tokens = count_tokens(step.raw_html)

                # Implementation note.
                t_intent = time.time()
                intent = self.intent_classifier.classify_with_fallback(
                    confirmed_task=task.confirmed_task,
                    subtask_description=step.action_repr,
                    operation_hint=step.operation_type)
                intent_time = (time.time() - t_intent) * 1000

                def eval_method(name, kept, stats, proc_time, llm_time):
                    metrics = self.evaluator.evaluate_step(
                        pos_candidates=step.pos_candidates,
                        neg_candidates=step.neg_candidates,
                        kept_elements=kept, stats=stats)
                    return {
                        f'{name}_success': 1 if metrics.recall > 0 else 0,
                        f'{name}_tokens': self.estimate_kept_tokens(kept),
                        f'{name}_recall': metrics.recall,
                        f'{name}_proc_ms': proc_time,
                        f'{name}_llm_ms': llm_time,
                        f'{name}_total_ms': proc_time + llm_time,
                    }

                row = {
                    'task_id': task.task_id, 'step_uid': step.uid,
                    'intent': intent, 'original_tokens': original_tokens,
                }

                # --- Visible Truncation (32K) ---
                t0 = time.time()
                try:
                    v_kept, v_stats = self.visible.process(step.raw_html, 32000)
                except Exception:
                    continue
                v_proc = (time.time() - t0) * 1000
                v_action, v_llm = self.action_decider.decide_action(
                    v_kept, 'content_browsing', task.confirmed_task, step.action_repr)
                row.update(eval_method('visible', v_kept, v_stats, v_proc, v_llm))

                # --- Uniform Pruning (8K) ---
                t0 = time.time()
                try:
                    u_kept, u_stats = self.uniform.process(step.raw_html, 8000)
                except Exception:
                    continue
                u_proc = (time.time() - t0) * 1000
                u_action, u_llm = self.action_decider.decide_action(
                    u_kept, 'button_interaction', task.confirmed_task, step.action_repr)
                row.update(eval_method('uniform', u_kept, u_stats, u_proc, u_llm))

                # --- Browser-Use (8K) ---
                t0 = time.time()
                try:
                    b_kept, b_stats = self.browseruse.process(step.raw_html, 8000)
                except Exception:
                    continue
                b_proc = (time.time() - t0) * 1000
                b_action, b_llm = self.action_decider.decide_action(
                    b_kept, 'button_interaction', task.confirmed_task, step.action_repr)
                row.update(eval_method('browseruse', b_kept, b_stats, b_proc, b_llm))

                # --- MindAct (8K) ---
                t0 = time.time()
                try:
                    m_kept, m_stats = self.mindact.process(
                        html_content=step.raw_html,
                        task_description=task.confirmed_task,
                        context_limit=8000,
                        step_description=subtask_desc,
                        action_history=task.action_reprs[:step.step_idx],
                    )
                except Exception:
                    continue
                m_proc = (time.time() - t0) * 1000
                m_action, m_llm = self.action_decider.decide_action(
                    m_kept, 'button_interaction', task.confirmed_task, step.action_repr)
                row.update(eval_method('mindact', m_kept, m_stats, m_proc, m_llm))

                # --- TIDP (8K) ---
                ic_start = time.time()
                try:
                    ic_kept, ic_stats = self.pruner.prune(step.raw_html, intent, subtask_desc, 8000)
                except Exception:
                    continue
                ic_proc = (time.time() - ic_start) * 1000
                ic_action, ic_llm = self.action_decider.decide_action(
                    ic_kept, intent, task.confirmed_task, step.action_repr)
                row.update(eval_method('icddp', ic_kept, ic_stats, ic_proc, ic_llm))
                row['icddp_intent_ms'] = intent_time
                row['icddp_total_ms'] = ic_proc + ic_llm + intent_time

                results.append(row)

        if total_steps == 0:
            print("No valid steps!"); return {}

        n = total_steps
        methods = ['visible', 'uniform', 'browseruse', 'mindact', 'icddp']

        def avg(k):
            return sum(r[k] for r in results) / n

        summary = {'total_steps': n}
        for m in methods:
            summary[m] = {
                'success_rate': avg(f'{m}_success'),
                'avg_tokens': avg(f'{m}_tokens'),
                'avg_recall': avg(f'{m}_recall'),
                'avg_proc_ms': avg(f'{m}_proc_ms'),
                'avg_llm_ms': avg(f'{m}_llm_ms'),
                'avg_total_ms': avg(f'{m}_total_ms'),
            }
        # Implementation note.
        summary['icddp']['avg_intent_ms'] = avg('icddp_intent_ms')
        summary['icddp']['avg_total_ms'] = avg('icddp_total_ms')

        self._print_results(summary, results)
        if save_results:
            self._save_results(split_name, summary, results)
        return summary

    def _print_results(self, summary, step_results):
        methods = ['visible', 'uniform', 'browseruse', 'mindact', 'icddp']
        labels = ['Visible Trunc.', 'Uniform Prune', 'Browser-Use', 'MindAct', 'TIDP']

        print(f"\n{'='*85}")
        print("Five Methods Comparison - Results")
        print(f"{'='*85}")
        print(f"Total steps: {summary['total_steps']}")

        print(f"\n{'─'*85}")
        header = f"{'Metric':<22}"
        for lb in labels:
            header += f" {lb:<14}"
        print(header)
        print(f"{'─'*85}")

        metrics = [
            ('Step success rate (%)', 'success_rate', 100, '.2f'),
            ('Token consumption', 'avg_tokens', 1, '.0f'),
            ('Key-element recall (%)', 'avg_recall', 100, '.2f'),
        ]
        for name, key, scale, fmt in metrics:
            row = f"{name:<22}"
            for m in methods:
                row += f" {summary[m][key]*scale:<14{fmt}}"
            print(row)

        # Implementation note.
        print(f"{'─'*85}")
        print(f"{'Intent classification (ms)':<22} {'─':<14} {'─':<14} {'─':<14} {'─':<14} {summary['icddp']['avg_intent_ms']:<14.1f}")
        row = f"{'Action decision (ms)':<22}"
        for m in methods:
            row += f" {summary[m]['avg_llm_ms']:<14.1f}"
        print(row)
        row = f"{'Pruning time (ms)':<22}"
        for m in methods:
            row += f" {summary[m]['avg_proc_ms']:<14.1f}"
        print(row)
        row = f"{'Total time (ms)':<22}"
        for m in methods:
            row += f" {summary[m]['avg_total_ms']:<14.1f}"
        print(row)
        print(f"{'─'*85}")

        # Implementation note.
        intent_stats = defaultdict(lambda: {m: {'success': 0, 'tokens': 0, 'cnt': 0} for m in methods})
        for r in step_results:
            intent = r['intent']
            for m in methods:
                intent_stats[intent][m]['cnt'] += 1
                intent_stats[intent][m]['success'] += r[f'{m}_success']
                intent_stats[intent][m]['tokens'] += r[f'{m}_tokens']

        print(f"\n{'─'*85}")
        print("Step success rate by intent (%)")
        print(f"{'─'*85}")
        header = f"{'Intent':<22}"
        for lb in labels:
            header += f" {lb:<14}"
        print(header)
        for intent in ['content_browsing', 'form_selection', 'information_input', 'button_interaction']:
            row = f"{intent:<22}"
            for m in methods:
                s = intent_stats[intent][m]
                row += f" {s['success']/s['cnt']*100 if s['cnt']>0 else 0:<14.1f}"
            print(row)

        print(f"\n{'─'*85}")
        print("Token consumption by intent")
        print(f"{'─'*85}")
        print(header)
        for intent in ['content_browsing', 'form_selection', 'information_input', 'button_interaction']:
            row = f"{intent:<22}"
            for m in methods:
                s = intent_stats[intent][m]
                row += f" {s['tokens']/s['cnt'] if s['cnt']>0 else 0:<14.0f}"
            print(row)
        print(f"{'─'*85}")

    def _save_results(self, split_name, summary, step_results):
        os.makedirs('results', exist_ok=True)
        path = os.path.join('results', f'five_methods_{split_name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'split': split_name, 'summary': summary, 'step_results': step_results},
                      f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATASET_PATH)
    parser.add_argument('--split', type=str, default='test_cross_domain')
    parser.add_argument('--cross_encoder_path', type=str, required=True)
    parser.add_argument('--max_tasks', type=int, default=None)
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print("pip install tiktoken"); return

    runner = FiveMethodRunner(args.data_dir, args.cross_encoder_path)
    runner.run(args.split, args.max_tasks)


if __name__ == '__main__':
    main()
