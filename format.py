from typing import Any, Dict, Iterable, List, Optional, Tuple


def int_(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{int(n):,}"

def pct(x: float, digits: int = 2) -> str:
    """Format percentage."""
    return f"{x:.{digits}f}%"

def gb(bytes_: int, digits: int = 2) -> str:
    """Format bytes as GB."""
    return f"{bytes_ / (1024**3):.{digits}f} GB"

def dtype(dt: Any) -> str:
    """
    Format dtype string without abbreviations.
    Examples: 'float16', 'bfloat16', 'float32'
    """
    if dt is None:
        return "unknown"
    return str(dt).replace("torch.", "")

def none_as(v: Any, default: str = "None") -> Any:
    return default if v is None else v

def kv(items: Iterable[Tuple[str, Any]], key_w: int = 28) -> List[str]:
    lines = []
    for k, v in items:
        key = str(k).ljust(key_w)
        value = str(none_as(v))

        value_lines = value.splitlines() or [""]
        lines.append(f"{key} : {value_lines[0]}")

        cont_prefix = " " * (key_w + 3)
        for cont in value_lines[1:]:
            lines.append(f"{cont_prefix}{cont}")

    return lines

def box(title: str, lines: List[str], width: int = 92) -> str:
    max_len = max((len(l) for l in lines), default=0)
    w = max(width, len(title) + 8, max_len + 4)
    top_dashes = w - len(title) - 5
    top = f"┏━ {title} " + "━" * max(0, top_dashes) + "┓"
    mid = [f"┃ {l}{' ' * max(0, w - len(l) - 4)} ┃" for l in lines]
    bot = "┗" + "━" * (w - 2) + "┛"
    return "\n".join([top] + mid + [bot])

def render(
    sections: Dict[str, List[Tuple[str, Any]]],
    *,
    width: int = 92,
    key_w: int = 28,
    order: Optional[List[str]] = None,
) -> str:
    titles = order or list(sections.keys())
    blocks = []
    for t in titles:
        blocks.append(box(t, kv(sections.get(t, []), key_w=key_w), width=width))
    return "\n\n".join(blocks)
