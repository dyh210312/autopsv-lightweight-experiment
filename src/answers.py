"""Conservative extraction of a stated answer, never re-solving the question."""
import re
from decimal import Decimal, InvalidOperation, localcontext

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"


def extract(text):
    text = text.replace("−", "-").replace(r"\$", "$")
    text = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    lines = [x.strip() for x in text.splitlines() if x.strip() and x.strip() not in (r"\]", r"\)", "```")]
    if not lines:
        return None
    last = lines[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", last)
    explicit = bool(boxed or "####" in last)
    if boxed:
        last = boxed[-1]
    elif "####" in last:
        last = last.rsplit("####", 1)[1]
    elif "=" in last:
        # Read the last stated RHS, including an incorrect arithmetic result.
        last = last.rsplit("=", 1)[1]
        explicit = True
    last = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", last)
    last = last.replace(r"\]", "").replace(r"\)", "").strip().rstrip(".")
    # Reject malformed comma groupings instead of turning 1,2 into 12.
    if re.search(r"\d,\d", re.sub(r"\b\d{1,3}(?:,\d{3})+(?!\d)", "", last)):
        return None
    matches = list(re.finditer(r"(?<![\w.])(?:" + NUMBER + r")(?![\w.])", last))
    signal = explicit or re.search(r"\b(?:answer|total|therefore|thus|so|has|holds|spent|spends|cost|is|are|equals|save|saves|savings|remain|remaining|left)\b", last, re.I)
    bare = re.fullmatch(r"[\s$*]*(?:" + NUMBER + r")[\s*]*", last)
    if not matches or not (signal or bare):
        return None
    try:
        values = [Decimal(m.group().replace(",", "")) for m in matches]
        if len(values) == 1:
            answer = values[0]
        elif len(values) == 2 and last[matches[0].end():matches[1].start()].strip() == "/":
            if values[1] == 0:
                return None
            with localcontext() as ctx:
                ctx.prec = 50
                answer = values[0] / values[1]
        else:
            return None  # alternative answers, ranges and mixed explanations require review
        return str(answer.normalize()) if answer.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None
