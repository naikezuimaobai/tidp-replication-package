"""Intent classification using an OpenAI-compatible chat model."""

import os
import re
from pathlib import Path
from openai import OpenAI

# ========== Local Ollama configuration ==========
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"


class IntentClassifier:
    """Classify web-operation intent with a local Ollama model."""
    
    INTENTS = ['content_browsing', 'form_selection', 'information_input', 'button_interaction']
    
    PROMPT_TEMPLATE = """You are a web operation intent classifier. Your classification will determine how the webpage elements are filtered to help the agent find the correct target element.

Classify the current subtask into ONE of these categories:

1. **content_browsing** (use when the goal is to FIND/READ/NAVIGATE):
   - Finding specific information on the page
   - Navigating to different sections or pages
   - Reading content, articles, or listings
   - Clicking links to browse (not to submit/confirm)
   - Examples: "View product details", "Browse next page", "Find contact information"

2. **form_selection** (use when the goal is to CHOOSE from OPTIONS):
   - Selecting from dropdowns, radio buttons, checkboxes
   - Choosing predefined options in forms
   - Examples: "Select province", "Check agree to terms", "Choose payment method"

3. **information_input** (use when the goal is to ENTER/INPUT text):
   - Typing text into input fields
   - Filling in text areas
   - Entering search keywords
   - Examples: "Enter email address", "Fill in username", "Search for products"

4. **button_interaction** (use when the goal is to CLICK to EXECUTE/SUBMIT):
   - Clicking buttons to submit forms
   - Clicking to confirm, save, delete, or perform actions
   - Interacting with UI controls (not navigation links)
   - Examples: "Click submit button", "Click confirm payment", "Click save settings"

**KEY DISTINCTION**: 
- If clicking a LINK to go somewhere → content_browsing
- If clicking a BUTTON to do something → button_interaction
- TYPE operation always → information_input
- SELECT operation always → form_selection

Full task: {confirmed_task}
Current subtask: {subtask_description}
Operation hint: {operation_hint}

Output format: intent: category_name
Intent:"""
    
    def __init__(
        self,
        api_base: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
    ):
        print("Initializing the intent classifier and connecting to Ollama...")
        self.client = OpenAI(
            base_url=OLLAMA_API_BASE,
            api_key="EMPTY",
            timeout=60.0
        )
        self.model = OLLAMA_MODEL
        self.prompt_template = self._load_prompt_template()
        if api_base or model or api_key or api_key_env:
            api_base = api_base or OLLAMA_API_BASE
            model = model or OLLAMA_MODEL
            api_key = api_key or (os.getenv(api_key_env) if api_key_env else None) or "EMPTY"
            self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
            self.model = model
        print(f"Using model: {self.model}")

    @classmethod
    def _load_prompt_template(cls) -> str:
        """Load the versioned prompt from the package, with a source-tree fallback."""
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "intent_classification.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError:
            return cls.PROMPT_TEMPLATE
    
    def classify(self, confirmed_task: str, subtask_description: str, 
             operation_hint: str = "") -> str:
        """Classify the current web operation."""
        # Resolve deterministic mappings before calling the model.
        if operation_hint == 'TYPE':
            return 'information_input'
        elif operation_hint == 'SELECT':
            return 'form_selection'
        
        # Only CLICK and unknown operations require an LLM call.
        prompt = self.prompt_template.format(
            confirmed_task=confirmed_task[:360],
            subtask_description=subtask_description[:220],
            operation_hint=operation_hint
        )
        
        try:
            print(f"  Calling Ollama for classification: {subtask_description[:50]}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web operation intent classifier. Output only the intent category."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=32,
                temperature=0
            )
            
            output = response.choices[0].message.content.strip()
            print(f"  Ollama response: {output}")
            intent = self._parse_output(output)
            
            if intent in self.INTENTS:
                return intent
            
            # fallback
            if operation_hint == 'CLICK':
                return 'button_interaction'
            return 'content_browsing'
        
        except Exception as e:
            print(f"  Intent classification error: {e}")
            # fallback
            if operation_hint == 'TYPE':
                return 'information_input'
            elif operation_hint == 'SELECT':
                return 'form_selection'
            elif operation_hint == 'CLICK':
                return 'button_interaction'
            return 'content_browsing'
    
    def _parse_output(self, output: str) -> str:
        """Parse the model output."""
        output_lower = output.lower()
        
        match = re.search(r'intent:\s*(\w+)', output_lower)
        if match:
            intent = match.group(1)
            if intent in self.INTENTS:
                return intent
        
        for intent in self.INTENTS:
            if intent in output_lower:
                return intent
        
        return ''
    
    def classify_with_fallback(self, confirmed_task: str, subtask_description: str,
                                operation_hint: str = "") -> str:
        """Classify with deterministic fallback rules."""
        intent = self.classify(confirmed_task, subtask_description, operation_hint)
        
        if operation_hint == 'TYPE' and intent != 'information_input':
            return 'information_input'
        if operation_hint == 'SELECT' and intent != 'form_selection':
            return 'form_selection'
        
        return intent
