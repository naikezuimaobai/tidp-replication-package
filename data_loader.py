"""Mind2Web Dataset Loader"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import glob

from config import DATASET_PATH


@dataclass
class StepData:
    """Data for one web-interaction step."""
    step_idx: int
    action_repr: str
    operation_type: str
    operation_value: str
    raw_html: str
    cleaned_html: str
    pos_candidates: List[Dict]
    neg_candidates: List[Dict]
    uid: str


@dataclass
class TaskData:
    """Data for one complete task."""
    task_id: str
    website: str
    domain: str
    subdomain: str
    confirmed_task: str
    action_reprs: List[str]
    steps: List[StepData]


class Mind2WebLoader:
    """Loader for the Mind2Web dataset."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DATASET_PATH
        self.data_dir = data_dir
    
    def load_json(self, file_path: str) -> List[Dict]:
        """Load one JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def parse_step(self, action: Dict, step_idx: int, action_repr: str) -> StepData:
        """Parse one step from an action dictionary."""
        operation = action.get('operation', {})
        
        # Both raw_html and cleaned_html may be available in the action.
        raw_html = action.get('raw_html', '')
        cleaned_html = action.get('cleaned_html', '')
        
        # Prefer raw_html and fall back to cleaned_html.
        html_content = raw_html if raw_html else cleaned_html
        
        return StepData(
            step_idx=step_idx,
            action_repr=action_repr,
            operation_type=operation.get('op', ''),
            operation_value=operation.get('value', ''),
            raw_html=html_content,
            cleaned_html=cleaned_html,
            pos_candidates=action.get('pos_candidates', []),
            neg_candidates=action.get('neg_candidates', []),
            uid=action.get('action_uid', f"step_{step_idx}")
        )
    
    def parse_task(self, task_data: Dict) -> TaskData:
        """Parse one complete task."""
        steps = []
        action_reprs = task_data.get('action_reprs', [])
        actions = task_data.get('actions', [])
        
        for i, action in enumerate(actions):
            action_repr = action_reprs[i] if i < len(action_reprs) else ""
            step = self.parse_step(action, i, action_repr)
            steps.append(step)
        
        return TaskData(
            task_id=task_data.get('annotation_id', ''),
            website=task_data.get('website', ''),
            domain=task_data.get('domain', ''),
            subdomain=task_data.get('subdomain', ''),
            confirmed_task=task_data.get('confirmed_task', ''),
            action_reprs=action_reprs,
            steps=steps
        )
    
    def load_split(self, split_name: str, max_files: int = None) -> List[TaskData]:
        """
        Load data from the requested split, including sharded files.
        
        Args:
            split_name: Dataset split name.
            max_files: Maximum number of files to load for small experiments.
        """
        # Determine the file-matching pattern.
        if split_name in ['test_cross_domain', 'test_domain']:
            pattern = "test_domain_*.json"
        elif split_name == 'test_cross_task':
            pattern = "test_task_*.json"
        elif split_name == 'test_cross_website':
            pattern = "test_website_*.json"
        elif split_name == 'train':
            pattern = "train_*.json"
        else:
            pattern = f"{split_name}_*.json"
        
        # Find all matching files.
        # Determine the subdirectory.
        if 'cross_task' in split_name or 'task' in split_name:
            sub_dir = 'test_task'
        elif 'cross_website' in split_name or 'website' in split_name:
            sub_dir = 'test_website'
        else:
            sub_dir = 'test_domain'

        search_pattern = os.path.join(self.data_dir, sub_dir, pattern)
        json_files = sorted(glob.glob(search_pattern))
        
        if not json_files:
            single_file = os.path.join(self.data_dir, f"{split_name}.json")
            if os.path.exists(single_file):
                json_files = [single_file]
            else:
                raise FileNotFoundError(
                    f"No files matching {pattern} were found in {self.data_dir}"
                )
        
        print(f"Found {len(json_files)} data files")
        
        if max_files and max_files > 0:
            json_files = json_files[:max_files]
            print(f"Limiting the load to the first {len(json_files)} files")
        
        # Load all selected files.
        all_tasks = []
        for file_path in json_files:
            print(f"Loading: {file_path}")
            data_list = self.load_json(file_path)
            tasks = [self.parse_task(item) for item in data_list]
            all_tasks.extend(tasks)
            print(f"  Loaded {len(tasks)} tasks; total loaded: {len(all_tasks)}")
        
        return all_tasks
    
    def load_samples(self, split_name: str, num_samples: int = 5) -> List[TaskData]:
        """
        Load a small sample for a quick experiment.
        """
        all_tasks = self.load_split(split_name)
        
        if len(all_tasks) > num_samples:
            import random
            random.seed(42)
            all_tasks = random.sample(all_tasks, num_samples)
            print(f"Randomly selected {len(all_tasks)} tasks")
        
        return all_tasks
