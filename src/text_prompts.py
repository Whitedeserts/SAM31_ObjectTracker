"""Parse category phrases without loading inference dependencies."""
import re


def parse_text_prompts(text):
    """Split comma/semicolon lists, preserving first spelling and phrase order."""
    if not isinstance(text, str):
        raise ValueError("Enter at least one object prompt, for example car; swimming pool.")
    prompts, seen = [], set()
    for part in re.split(r"[;,]", text):
        phrase = " ".join(part.split())
        key = phrase.casefold()
        if phrase and key not in seen:
            prompts.append(phrase)
            seen.add(key)
    if not prompts:
        raise ValueError("Enter at least one object prompt, for example car; swimming pool.")
    return prompts
