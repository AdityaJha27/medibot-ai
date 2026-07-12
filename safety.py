import re

_CRISIS_PATTERNS = [
    r"\bkill\s*(myself|me)\b",
    r"\bsuicid(e|al)\b",
    r"\bend\s+(my|it all|this)\s*(life)?\b",
    r"\bwant\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+live\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bhurt\s+myself\b",
    r"\bself[\s-]?harm\b",
    r"\battempt\s+(death|suicide)\b",
    r"\btips?\s+(of|for|to)\s+death\b",
    r"\bstep(s)?\s+(of|to|towards)\s+death\b",
    r"\bhow\s+to\s+(die|commit\s+suicide|end\s+my\s+life)\b",
    r"\boverdose\s+(to\s+die|myself)\b",
    r"\bways?\s+to\s+die\b",
    r"\bpainless\s+(way\s+to\s+)?death\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]

CRISIS_RESPONSE = """## Summary
I'm concerned about what you shared, and I want to make sure you get real support right now.

If you're in India, please reach out to one of these free, confidential helplines:
- **iCall**: 9152987821
- **AASRA**: 9820466726
- **Tele-MANAS (Govt. of India)**: 14416

If you're outside India, please contact your local emergency number or a crisis helpline in your country.

I'm not able to help with this question through the medical knowledge base — please talk to a real person who can support you right now."""


def is_crisis_query(text: str) -> bool:
    return any(pattern.search(text) for pattern in _COMPILED)