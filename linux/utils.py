"""Utility functions for DOM processing"""

from bs4 import BeautifulSoup, Tag, Comment, NavigableString
from typing import List, Dict, Any, Optional, Tuple
import re

# ============ Constants ============
MAX_ELEMENT_TEXT_LEN = 200
AVG_TOKENS_PER_ELEMENT = 150


def parse_html(html_content: str) -> BeautifulSoup:
    """Parse HTML content to BeautifulSoup object"""
    return BeautifulSoup(html_content, 'html.parser')


def basic_node_filtering(soup: BeautifulSoup) -> BeautifulSoup:
    """Apply basic DOM filtering while protecting interactive elements."""
    
    # Remove script and style-related tags.
    for tag in soup(['script', 'style', 'link', 'meta', 'noscript']):
        tag.decompose()
    
    # Remove comment nodes.
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()
    
    # Remove invisible elements, except interactive elements.
    invisible_selectors = [
        '[hidden]', '[style*="display: none"]', '[style*="display:none"]',
        '[style*="visibility: hidden"]', '[style*="visibility:hidden"]',
        '[aria-hidden="true"]'
    ]
    
    # Interactive tag allowlist.
    INTERACTIVE_TAGS = {'input', 'textarea', 'select', 'button', 'a', 'option'}
    
    for selector in invisible_selectors:
        for elem in soup.select(selector):
            # Keep interactive elements even when marked invisible.
            tag = elem.name.lower() if elem.name else ''
            if tag in INTERACTIVE_TAGS:
                continue
            
            if is_element_effectively_invisible(elem):
                elem.decompose()
    
    # Remove empty nodes, except interactive elements.
    for elem in soup.find_all():
        tag = elem.name.lower() if elem.name else ''
        if tag in INTERACTIVE_TAGS:
            continue
        
        if is_empty_element(elem):
            elem.decompose()
    
    # Unwrap structural skeleton elements while preserving their contents.
    for skeleton in soup.find_all(['html', 'body', 'main']):
        skeleton.unwrap()
    
    return soup
# def basic_node_filtering(soup: BeautifulSoup) -> BeautifulSoup:
#     """
# Implementation note.
# Implementation note.
#     """
# Implementation note.
#     for tag in soup(['script', 'style', 'link', 'meta', 'noscript']):
#         tag.decompose()
    
# Implementation note.
#     for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
#         comment.extract()
    
# Implementation note.
#     invisible_selectors = [
#         '[hidden]', '[style*="display: none"]', '[style*="display:none"]',
#         '[style*="visibility: hidden"]', '[style*="visibility:hidden"]',
#         '[aria-hidden="true"]'
#     ]
#     for selector in invisible_selectors:
#         for elem in soup.select(selector):
# Implementation note.
#             if is_element_effectively_invisible(elem):
#                 elem.decompose()
    
# Implementation note.
#     for elem in soup.find_all():
#         if is_empty_element(elem):
#             elem.decompose()
    
# Implementation note.
#     for skeleton in soup.find_all(['html', 'body', 'main']):
#         skeleton.unwrap()
    
#     return soup


def is_element_effectively_invisible(elem: Tag) -> bool:
    """Check whether an element is effectively invisible without touching children."""
    # Check zero width or height.
    style = elem.get('style', '')
    width_match = re.search(r'width:\s*0', style)
    height_match = re.search(r'height:\s*0', style)
    if width_match or height_match:
        return True
    
    # Check the hidden attribute.
    if elem.has_attr('hidden'):
        return True
    
    return False


def is_empty_element(elem: Tag) -> bool:
    """Check whether an element has no text and no meaningful children."""
    text = elem.get_text(strip=True)
    if text:
        return False
    
    # Non-empty child tags make the element meaningful.
    children = [c for c in elem.children if isinstance(c, Tag)]
    if children:
        return False
    
    return True


def get_element_text(elem: Tag, max_len: int = 200) -> str:
    """Extract an element's text representation.

    Priority: aria-label > direct text > placeholder > value.
    """
    if not elem.name:
        return ""
    
    # Prefer aria-label.
    aria_label = elem.get('aria-label', '')
    if aria_label:
        return aria_label[:max_len]
    
    tag_name = elem.name.lower() if elem.name else ''
    
    # Use element-specific attributes for interactive elements.
    if tag_name in ['input', 'textarea', 'select', 'button']:
        # Use placeholder.
        placeholder = elem.get('placeholder', '')
        if placeholder:
            return placeholder[:max_len]
        
        # Use value.
        value = elem.get('value', '')
        if value:
            return value[:max_len]
    
    # Use direct child text without recursively expanding deep descendants.
    direct_text = ''.join([
        str(child).strip() for child in elem.children 
        if isinstance(child, NavigableString) and str(child).strip()
    ])
    if direct_text:
        return direct_text[:max_len]
    
    # Fall back to all text with a length limit.
    full_text = elem.get_text(separator=' ', strip=True)
    return full_text[:max_len] if full_text else ""


def get_element_type(elem: Tag) -> str:
    """Return the normalized element type identifier."""
    if not elem.name:
        return 'TEXT'
    
    tag = elem.name.lower()
    
    if tag in ['input']:
        input_type = elem.get('type', 'text').lower()
        if input_type in ['radio', 'checkbox']:
            return 'CHECKBOX' if input_type == 'checkbox' else 'RADIO'
        elif input_type in ['submit', 'button', 'reset']:
            return 'BTN'
        else:
            return 'INPUT'
    elif tag in ['textarea']:
        return 'TEXTAREA'
    elif tag in ['select']:
        return 'SELECT'
    elif tag in ['option']:
        return 'OPTION'
    elif tag in ['button', 'a']:
        return 'BTN'
    elif tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'li']:
        return 'TEXT'
    else:
        return tag.upper()


def get_relevant_attributes(elem: Tag) -> Dict[str, str]:
    """Return attributes relevant to element identification."""
    attrs = {}
    important_attrs = ['id', 'class', 'name', 'type', 'role', 'href', 'value', 'placeholder', 'aria-label']
    
    for attr in important_attrs:
        if elem.has_attr(attr):
            attrs[attr] = elem[attr]
    
    return attrs


def get_dom_depth(elem: Tag, root: BeautifulSoup = None) -> int:
    """Compute DOM depth, with the root at depth zero."""
    depth = 0
    current = elem
    while current.parent and current.parent != root:
        depth += 1
        current = current.parent
    return depth


def find_common_ancestor(elem1: Tag, elem2: Tag, max_depth: int = 3) -> Optional[Tag]:
    """Find a common ancestor within the specified upward search depth."""
    ancestors1 = set()
    e = elem1
    for _ in range(max_depth):
        if e is None or e.parent is None:
            break
        e = e.parent
        ancestors1.add(e)
    
    e = elem2
    for _ in range(max_depth):
        if e is None or e.parent is None:
            break
        e = e.parent
        if e in ancestors1:
            return e
    return None


def is_near_element(elem1: Tag, elem2: Tag, max_depth_diff: int = 3) -> bool:
    """Check whether two elements are near each other in the DOM tree."""
    depth1 = get_dom_depth(elem1)
    depth2 = get_dom_depth(elem2)
    
    if abs(depth1 - depth2) <= max_depth_diff:
        return True
    
    common = find_common_ancestor(elem1, elem2, max_depth_diff)
    return common is not None


def extract_all_elements(soup: BeautifulSoup) -> List[Tag]:
    """Extract all visible elements that have text or interaction semantics."""
    elements = []
    for elem in soup.find_all():
        # Skip decomposed elements.
        if elem.decomposed:
            continue
        # Keep visible elements with useful text or interaction semantics.
        text = get_element_text(elem, MAX_ELEMENT_TEXT_LEN)
        if text or elem.name in ['input', 'button', 'select', 'textarea', 'a']:
            elements.append(elem)
    return elements


def estimate_tokens(text_or_element) -> int:
    """Estimate token count using the lightweight package heuristic."""
    if isinstance(text_or_element, str):
        return len(text_or_element) // 3
    else:
        # Approximate an element by its tag, attributes, and text.
        return AVG_TOKENS_PER_ELEMENT


def compress_element(elem: Tag) -> str:
    """Serialize an element into a compact representation."""
    tag = elem.name.lower() if elem.name else 'text'
    text = get_element_text(elem, 50)
    if tag in ['input', 'button', 'select', 'a']:
        attrs = get_relevant_attributes(elem)
        attrs_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items() if v])
        return f"<{tag} {attrs_str}>{text}</{tag}>"[:100]
    else:
        return f"<{tag}>{text}</{tag}>"[:80]
