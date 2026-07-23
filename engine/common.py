"""
Shared constants and small data structures used by every engine backend
(pure-Python fallback and the compiled Cython engine) and by the GUI.

Keeping these in one place guarantees the two engine implementations stay
bit-for-bit compatible with each other.
"""
from dataclasses import dataclass, field
from typing import Tuple

# --- Chunk geometry (mirrors the original C++ constants) ---
CHUNK_SHIFT = 5                       # 2^5 = 32
CHUNK_SIZE = 1 << CHUNK_SHIFT          # 32
CHUNK_MASK = CHUNK_SIZE - 1
CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE    # 1024

# Direction order: 0=up, 1=right, 2=down, 3=left (screen coords, +y is down)
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIRECTION_SYMBOLS = ("\u2191", "\u2192", "\u2193", "\u2190")  # ↑ → ↓ ←

# Branchless corner lookup table. Index = (entrySide << 2) | exitSide.
# Returns corner 0-3, or -1 if the entry/exit pair doesn't form a corner.
CORNER_LUT = (
    -1,  1, -1,  0,   # entry 0 (Top)
     1, -1,  2, -1,   # entry 1 (Right)
    -1,  2, -1,  3,   # entry 2 (Bottom)
     0, -1,  3, -1,   # entry 3 (Left)
)


def chunk_index(coord: int) -> int:
    """Floor-division chunk index. Python's // already floors correctly
    for negative numbers, so this is simpler than the C++ equivalent."""
    return coord // CHUNK_SIZE


@dataclass
class AntStatisticsSummary:
    total_cells_visited: int = 0     # actually "total steps taken" (matches original naming)
    max_visits_per_cell: int = 0
    most_visited_cell: Tuple[int, int] = (0, 0)
    average_visits: float = 0.0
    unique_cells_visited: int = 0
    simulation_time_ms: int = 0


def expand_rule_shorthand(text: str):
    """Port of MainWindow::updateRules()'s expansion/compression logic.

    'L2R3' -> expanded='LLRRR', compressed='L2R3'.
    Returns (expanded, compressed, error_message_or_None).
    """
    text = text.strip().upper()
    for ch in text:
        if ch.isalpha() and ch not in "LR":
            return None, None, "Rules can only contain L or R letters."

    expanded = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isalpha():
            cur = text[i]
            i += 1
            while i < n and text[i] == " ":
                i += 1
            num_str = ""
            while i < n and text[i].isdigit():
                num_str += text[i]
                i += 1
            count = 1
            if num_str:
                count = max(1, min(int(num_str), 1_000_000))
            expanded.append(cur * count)
        else:
            i += 1
    expanded_rules = "".join(expanded)

    if not expanded_rules:
        expanded_rules = "LR"
        return expanded_rules, "L1R1", None

    compressed = []
    i = 0
    n = len(expanded_rules)
    while i < n:
        cnt = 1
        ch = expanded_rules[i]
        compressed.append(ch)
        i += 1
        while i < n and expanded_rules[i] == ch:
            cnt += 1
            i += 1
        if cnt > 1:
            compressed.append(str(cnt))

    return expanded_rules, "".join(compressed), None


def compress_rules(rules: str) -> str:
    """Port of AntFieldWidget::getRules() - run-length compress a rule
    string of L/R characters (e.g. 'LLLRR' -> 'L3R2')."""
    if not rules:
        return ""
    out = []
    n = len(rules)
    i = 0
    while i < n:
        ch = rules[i]
        cnt = 1
        i += 1
        while i < n and rules[i] == ch:
            cnt += 1
            i += 1
        out.append(ch if cnt == 1 else f"{ch}{cnt}")
    return "".join(out)


PRESETS = {
    0: ("Classic LR", "LR"),
    1: ("Symmetric LLRR", "LLRR"),
    2: ("Highway", "LLRRRLRLRLLR"),
    3: ("Complex", "LRRRRRLLR"),
}
