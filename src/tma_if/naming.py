from __future__ import annotations
import re
from typing import Optional, Tuple

_CORE_RE = re.compile(r'(\d{1,2})\s*[_-]\s*(\d{1,2})')

def normalize_core(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts core row/col from strings like:
      10_1.ome, 10-1.ome, 10_1.ome_R_citoplasma8, ...
    Returns:
      core_norm: "10_1"
      core_dash: "10-1"
    """
    if s is None:
        return None, None
    ss = str(s)
    m = _CORE_RE.search(ss)
    if not m:
        return None, None
    r = int(m.group(1))
    c = int(m.group(2))
    return f"{r}_{c}", f"{r}-{c}"
