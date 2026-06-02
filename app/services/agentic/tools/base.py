from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ToolResult:
    answer: str
    sources: List[str]
    metadata: Optional[Dict[str, object]] = None
    context_texts: Optional[List[str]] = None
