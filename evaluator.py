"""Element-level evaluation for TIDP using backend_node_id for exact matching."""

from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import json
import re
from bs4 import Tag


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    prune_rate: float
    pos_preserved_rate: float
    neg_filtered_rate: float
    total_elements: int
    kept_elements: int
    tp: int
    fp: int
    tn: int
    fn: int


class Evaluator:
    """Element-level evaluator using exact backend_node_id matching."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0
        self.total_pos = 0
        self.total_neg = 0
        self.total_kept = 0
        self.total_elements = 0
    
    def extract_backend_node_id(self, candidate: Dict) -> str:
        """Extract backend_node_id from a candidate element."""
        # Method 1: read the direct field.
        node_id = candidate.get('backend_node_id', '')
        if node_id:
            return str(node_id)
        
        # Method 2: read the attributes field, which may be a string or dict.
        attrs = candidate.get('attributes', {})
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except:
                attrs = {}
        
        if isinstance(attrs, dict) and 'backend_node_id' in attrs:
            return str(attrs['backend_node_id'])
        
        return ''
    
    def extract_backend_node_id_from_element(self, elem) -> str:
        """Extract backend_node_id from a BeautifulSoup tag or element dict."""
        # Handle dictionaries used in kept_elements.
        if isinstance(elem, dict):
            node_id = elem.get('backend_node_id', '')
            if node_id:
                return str(node_id)
            # Read the BeautifulSoup tag from the element field.
            tag_elem = elem.get('element')
            if tag_elem and isinstance(tag_elem, Tag):
                if tag_elem.get('backend_node_id'):
                    return str(tag_elem.get('backend_node_id'))
            # Recover the identifier from the description.
            desc = elem.get('description', '')
            match = re.search(r'backend_node_id="([^"]+)"', desc)
            if match:
                return match.group(1)
            # Recover the identifier from the text field when present.
            text = elem.get('text', '')
            match = re.search(r'backend_node_id["\']?\s*[:=]\s*["\']?(\d+)', text)
            if match:
                return match.group(1)
        
        # Handle BeautifulSoup tags.
        elif isinstance(elem, Tag):
            if elem.get('backend_node_id'):
                return str(elem.get('backend_node_id'))
        
        return ''
    
    def extract_candidate_text(self, candidate: Dict) -> str:
        """Extract useful candidate text for fallback matching and debugging."""
        attrs = candidate.get('attributes', {})
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except:
                attrs = {}
        
        texts = []
        
        # Collect text in priority order.
        if isinstance(attrs, dict):
            if 'aria_label' in attrs and attrs['aria_label']:
                texts.append(attrs['aria_label'].strip())
            if 'placeholder' in attrs and attrs['placeholder']:
                texts.append(attrs['placeholder'].strip())
            if 'name' in attrs and attrs['name']:
                texts.append(attrs['name'].strip())
            if 'title' in attrs and attrs['title']:
                texts.append(attrs['title'].strip())
            if 'value' in attrs and attrs['value'] and attrs['value'].strip():
                texts.append(attrs['value'].strip())
            if 'id' in attrs and attrs['id']:
                texts.append(attrs['id'].strip())
        
        # Add the tag name.
        if 'tag' in candidate:
            texts.append(candidate['tag'])
        
        return ' '.join(texts) if texts else ''
    
    def text_match(self, text1: str, text2: str) -> bool:
        """Match text using the fallback heuristic."""
        if not text1 or not text2:
            return False
        
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Exact match.
        if t1 == t2:
            return True
        
        # Match after punctuation normalization.
        t1_clean = re.sub(r'[^\w\s]', ' ', t1)
        t2_clean = re.sub(r'[^\w\s]', ' ', t2)
        
        if t1_clean == t2_clean:
            return True
        
        # Subset match.
        words1 = set(t1_clean.split())
        words2 = set(t2_clean.split())
        
        if words1 and words2:
            common = words1 & words2
            if len(common) >= 2 and (len(common) / len(words1) >= 0.8 or len(common) / len(words2) >= 0.8):
                return True
        
        # Containment match.
        if len(t1) >= 5 and len(t2) >= 5:
            if t1 in t2 or t2 in t1:
                return True
        
        return False
    
    def evaluate_step(self, pos_candidates: List[Dict], neg_candidates: List[Dict],
                      kept_elements: List[Dict], stats: Dict) -> EvaluationMetrics:
        """Evaluate one step, preferring exact backend_node_id matching."""
        
        # ========== Method 1: exact backend_node_id matching ==========
        pos_ids = set()
        for pos in pos_candidates:
            node_id = self.extract_backend_node_id(pos)
            if node_id:
                pos_ids.add(node_id)
        
        neg_ids = set()
        for neg in neg_candidates:
            node_id = self.extract_backend_node_id(neg)
            if node_id:
                neg_ids.add(node_id)
        
        kept_ids = set()
        for elem_dict in kept_elements:
            node_id = self.extract_backend_node_id_from_element(elem_dict)
            if node_id:
                kept_ids.add(node_id)
        
        # Implementation note.
        if pos_ids and kept_ids:
            tp = len(pos_ids & kept_ids)
            fn = len(pos_ids - kept_ids)
            fp = len(kept_ids - pos_ids - neg_ids)
            tn = len(neg_ids - kept_ids) if neg_ids else 0
            
            # Update aggregates.
            self.total_tp += tp
            self.total_fp += fp
            self.total_tn += tn
            self.total_fn += fn
            self.total_pos += len(pos_ids)
            self.total_neg += len(neg_ids)
            self.total_kept += len(kept_ids)
            self.total_elements += stats.get('total', 0)
            
            # Compute metrics.
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            pos_preserved_rate = tp / len(pos_ids) if len(pos_ids) > 0 else 1.0
            neg_filtered_rate = tn / len(neg_ids) if len(neg_ids) > 0 else 1.0
            
            return EvaluationMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                prune_rate=stats.get('prune_rate', 0),
                pos_preserved_rate=pos_preserved_rate,
                neg_filtered_rate=neg_filtered_rate,
                total_elements=stats.get('total', 0),
                kept_elements=len(kept_ids),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn
            )
        
        # ========== Method 2: fallback text matching ==========
        # Extract positive-candidate text.
        pos_texts = set()
        for pos in pos_candidates:
            text = self.extract_candidate_text(pos)
            if text:
                pos_texts.add(text.lower().strip())
        
        # Extract negative-candidate text.
        neg_texts = set()
        for neg in neg_candidates:
            text = self.extract_candidate_text(neg)
            if text:
                neg_texts.add(text.lower().strip())
        
        # Extract kept-element text.
        kept_texts = set()
        for elem in kept_elements:
            text = elem.get('text', '').lower().strip()
            if text:
                kept_texts.add(text)
        
        # Compute true positives.
        tp = 0
        for pos_text in pos_texts:
            for kept_text in kept_texts:
                if self.text_match(pos_text, kept_text):
                    tp += 1
                    break
        
        fn = len(pos_texts) - tp
        fp = len(kept_texts) - tp
        
        # Compute true negatives.
        tn = 0
        for neg_text in neg_texts:
            matched = False
            for kept_text in kept_texts:
                if self.text_match(neg_text, kept_text):
                    matched = True
                    break
            if not matched:
                tn += 1
        
        # Update aggregates.
        self.total_tp += tp
        self.total_fp += fp
        self.total_tn += tn
        self.total_fn += fn
        self.total_pos += len(pos_texts)
        self.total_neg += len(neg_texts)
        self.total_kept += len(kept_texts)
        self.total_elements += stats.get('total', 0)
        
        # Compute metrics.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        pos_preserved_rate = tp / len(pos_texts) if len(pos_texts) > 0 else 1.0
        neg_filtered_rate = tn / len(neg_texts) if len(neg_texts) > 0 else 1.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            prune_rate=stats.get('prune_rate', 0),
            pos_preserved_rate=pos_preserved_rate,
            neg_filtered_rate=neg_filtered_rate,
            total_elements=stats.get('total', 0),
            kept_elements=len(kept_texts),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn
        )
    
    def get_overall_metrics(self) -> EvaluationMetrics:
        """Return aggregate metrics."""
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        pos_preserved_rate = self.total_tp / self.total_pos if self.total_pos > 0 else 1.0
        neg_filtered_rate = self.total_tn / self.total_neg if self.total_neg > 0 else 1.0
        prune_rate = 1 - self.total_kept / self.total_elements if self.total_elements > 0 else 0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            prune_rate=prune_rate,
            pos_preserved_rate=pos_preserved_rate,
            neg_filtered_rate=neg_filtered_rate,
            total_elements=self.total_elements,
            kept_elements=self.total_kept,
            tp=self.total_tp,
            fp=self.total_fp,
            tn=self.total_tn,
            fn=self.total_fn
        )
    
    def print_metrics(self, metrics: EvaluationMetrics, prefix: str = ""):
        """Print evaluation metrics."""
        print(f"\n{prefix}=== Evaluation Results ===")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1 Score: {metrics.f1:.4f}")
        print(f"Prune Rate: {metrics.prune_rate:.4f}")
        print(f"Positive Preserved Rate: {metrics.pos_preserved_rate:.4f}")
        print(f"Negative Filtered Rate: {metrics.neg_filtered_rate:.4f}")
        print(f"Total Elements: {metrics.total_elements}")
        print(f"Kept Elements: {metrics.kept_elements}")
        print(f"TP: {metrics.tp}, FP: {metrics.fp}, TN: {metrics.tn}, FN: {metrics.fn}")

# """Element-level evaluation for TIDP"""

# from typing import List, Dict, Any, Tuple, Set
# from dataclasses import dataclass
# import json


# @dataclass
# class EvaluationMetrics:
#     precision: float
#     recall: float
#     f1: float
#     prune_rate: float
#     pos_preserved_rate: float
#     neg_filtered_rate: float
#     total_elements: int
#     kept_elements: int
#     tp: int
#     fp: int
#     tn: int
#     fn: int


# class Evaluator:
# Implementation note.
    
#     def __init__(self):
#         self.reset()
    
#     def reset(self):
#         self.total_tp = 0
#         self.total_fp = 0
#         self.total_tn = 0
#         self.total_fn = 0
#         self.total_pos = 0
#         self.total_neg = 0
#         self.total_kept = 0
#         self.total_elements = 0
    
#     def extract_candidate_text(self, candidate: Dict) -> str:
# Implementation note.
# Implementation note.
#         attrs = candidate.get('attributes', {})
#         if isinstance(attrs, str):
#             try:
#                 attrs = json.loads(attrs)
#             except:
#                 attrs = {}
        
# Implementation note.
#         if 'aria_label' in attrs and attrs['aria_label']:
#             return attrs['aria_label']
        
# Implementation note.
#         text = candidate.get('text', '')
#         if text and text != 'None' and text.strip():
#             return text.strip()
        
# Implementation note.
#         if 'class' in attrs and attrs['class']:
# Implementation note.
#             classes = attrs['class'].split()
#             for cls in classes:
#                 if len(cls) > 3 and not cls.startswith('_'):
#                     return cls
        
# Implementation note.
#         return candidate.get('tag', '')
    
#     def text_match(self, text1: str, text2: str) -> bool:
# Implementation note.
#         if not text1 or not text2:
#             return False
#         t1 = text1.lower().strip()
#         t2 = text2.lower().strip()
        
# Implementation note.
#         if t1 == t2:
#             return True
        
# Implementation note.
#         import re
#         t1_clean = re.sub(r'[^\w\s]', '', t1)
#         t2_clean = re.sub(r'[^\w\s]', '', t2)
        
#         if t1_clean and t2_clean:
#             if t1_clean in t2_clean or t2_clean in t1_clean:
#                 return True
        
#         return False
    
#     def evaluate_step(self, pos_candidates: List[Dict], neg_candidates: List[Dict],
#                       kept_elements: List[Dict], stats: Dict) -> EvaluationMetrics:
# Implementation note.
# Implementation note.
#         pos_texts = set()
#         for pos in pos_candidates:
#             text = self.extract_candidate_text(pos)
#             if text:
#                 pos_texts.add(text.lower().strip())
        
# Implementation note.
#         neg_texts = set()
#         for neg in neg_candidates:
#             text = self.extract_candidate_text(neg)
#             if text:
#                 neg_texts.add(text.lower().strip())
        
# Implementation note.
#         kept_texts = set()
#         for elem in kept_elements:
#             text = elem.get('text', '').lower().strip()
#             if text:
#                 kept_texts.add(text)
        
# Implementation note.
#         tp = 0
#         for pos_text in pos_texts:
#             for kept_text in kept_texts:
#                 if self.text_match(pos_text, kept_text):
#                     tp += 1
#                     break
        
#         fn = len(pos_texts) - tp
#         fp = len(kept_texts) - tp
        
# Implementation note.
#         tn = 0
#         for neg_text in neg_texts:
#             matched = False
#             for kept_text in kept_texts:
#                 if self.text_match(neg_text, kept_text):
#                     matched = True
#                     break
#             if not matched:
#                 tn += 1
        
# Implementation note.
#         self.total_tp += tp
#         self.total_fp += fp
#         self.total_tn += tn
#         self.total_fn += fn
#         self.total_pos += len(pos_texts)
#         self.total_neg += len(neg_texts)
#         self.total_kept += len(kept_texts)
#         self.total_elements += stats.get('total', 0)
        
# Implementation note.
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         pos_preserved_rate = tp / len(pos_texts) if len(pos_texts) > 0 else 1.0
#         neg_filtered_rate = tn / len(neg_texts) if len(neg_texts) > 0 else 1.0
        
#         return EvaluationMetrics(
#             precision=precision,
#             recall=recall,
#             f1=f1,
#             prune_rate=stats.get('prune_rate', 0),
#             pos_preserved_rate=pos_preserved_rate,
#             neg_filtered_rate=neg_filtered_rate,
#             total_elements=stats.get('total', 0),
#             kept_elements=len(kept_texts),
#             tp=tp,
#             fp=fp,
#             tn=tn,
#             fn=fn
#         )
    
#     def get_overall_metrics(self) -> EvaluationMetrics:
# Implementation note.
#         precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0
#         recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         pos_preserved_rate = self.total_tp / self.total_pos if self.total_pos > 0 else 1.0
#         neg_filtered_rate = self.total_tn / self.total_neg if self.total_neg > 0 else 1.0
#         prune_rate = 1 - self.total_kept / self.total_elements if self.total_elements > 0 else 0
        
#         return EvaluationMetrics(
#             precision=precision,
#             recall=recall,
#             f1=f1,
#             prune_rate=prune_rate,
#             pos_preserved_rate=pos_preserved_rate,
#             neg_filtered_rate=neg_filtered_rate,
#             total_elements=self.total_elements,
#             kept_elements=self.total_kept,
#             tp=self.total_tp,
#             fp=self.total_fp,
#             tn=self.total_tn,
#             fn=self.total_fn
#         )
    
#     def print_metrics(self, metrics: EvaluationMetrics, prefix: str = ""):
#         print(f"\n{prefix}=== Evaluation Results ===")
#         print(f"Precision: {metrics.precision:.4f}")
#         print(f"Recall: {metrics.recall:.4f}")
#         print(f"F1 Score: {metrics.f1:.4f}")
#         print(f"Prune Rate: {metrics.prune_rate:.4f}")
#         print(f"Positive Preserved Rate: {metrics.pos_preserved_rate:.4f}")
#         print(f"Negative Filtered Rate: {metrics.neg_filtered_rate:.4f}")
#         print(f"Total Elements: {metrics.total_elements}")
#         print(f"Kept Elements: {metrics.kept_elements}")
#         print(f"TP: {metrics.tp}, FP: {metrics.fp}, TN: {metrics.tn}, FN: {metrics.fn}")
