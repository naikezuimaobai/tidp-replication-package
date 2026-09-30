# ===================== preprocess_simplified.py =====================
# Simplified preprocessing: retain only action + element_type.
# ====================================================================

import pandas as pd
import os
import re

BASE_PATH = os.environ.get("TIDP_CLUSTER_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_PATH, "mind2web_all_actions_features.csv")
output_file = os.path.join(BASE_PATH, "step_actions_simplified.xlsx")

df = pd.read_csv(file_path)

def extract_action_type(text):
    """Extract the operation type."""
    match = re.search(r'->\s*(\w+)', text)
    return match.group(1).lower() if match else None

def extract_element_type(text):
    """Extract the element type."""
    match = re.search(r'\[(\w+)\]', text)
    if match:
        elem = match.group(1).lower()
        mapping = {
            'combobox': 'dropdown',
            'searchbox': 'search_input',
            'button': 'btn',
            'span': 'text',
            'svg': 'icon',
            'div': 'container',
            'img': 'image',
            'input': 'input_field',
            'link': 'link',
            'checkbox': 'checkbox',
        }
        return mapping.get(elem, elem)
    return None

def format_simplified(action_repr):
    """Format as {action}_{element_type}."""
    action = extract_action_type(action_repr)
    element = extract_element_type(action_repr)
    
    if action and element:
        return f"{action}_{element}"
    elif action:
        return action
    else:
        return "unknown"

df['unified_simplified'] = df['action_repr'].apply(format_simplified)

# Summarize the normalized operations.
print("="*50)
print("Simplified preprocessing (action + element_type)")
print("="*50)
print(f"Unique operation count: {df['unified_simplified'].nunique()}")

print("\nOperation distribution:")
counts = df['unified_simplified'].value_counts()
for action, count in counts.head(30).items():
    bar = "█" * int(count / 50)
    print(f"   {count:5d}  {action:<25} {bar}")

# Save the result.
df.to_excel(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")
