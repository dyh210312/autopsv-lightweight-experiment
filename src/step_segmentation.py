"""Versioned mechanical segmentation, not a claim of semantic error localization."""
import re
VERSION = 'lines_sentences_v2'


def split_steps(text):
    # Preserve multiline display math as one unit. Split prose on line boundaries
    # and sentence-ending punctuation followed by an uppercase letter or ####.
    parts = re.split(r'(\\\[.*?\\\]|\$\$.*?\$\$)', text.strip(), flags=re.S)
    result = []
    for part in parts:
        if not part.strip():
            continue
        if part.startswith(('\\[', '$$')):
            result.append(part.strip())
        else:
            for line in part.splitlines():
                result.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z]|####)',line) if s.strip())
    if not result:
        raise ValueError('Empty solution')
    # The only permitted transformation is whitespace at boundaries.
    if re.sub(r'\s+', '', ''.join(result)) != re.sub(r'\s+', '', text):
        raise ValueError('Segmentation changed solution content')
    return result


def split_steps_v3(text):
    """Attach colon-ended lead-ins and standalone numbering to following content."""
    raw = split_steps(text)
    result, pending = [], []
    for part in raw:
        if part.endswith((':', '：')) or re.fullmatch(r'\d+[.)]', part):
            pending.append(part)
            continue
        result.append('\n'.join(pending + [part]))
        pending = []
    if pending:
        # Keep dangling text for review; never discard it.
        result.append('\n'.join(pending))
    assert re.sub(r'\s+', '', ''.join(result)) == re.sub(r'\s+', '', text)
    return result
