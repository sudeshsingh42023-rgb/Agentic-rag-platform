"""Deterministic input/output guardrails. Regex, not an LLM: the guardrail must
be cheaper and more predictable than the thing it is guarding.

Input : PII detection, prompt-injection heuristics, length cap.
Output: PII leak scan, citation presence, refusal pass-through.
"""
import re
from dataclasses import dataclass, field

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_in": re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
}

INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the |your )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|above|previous)", re.I),
    re.compile(r"you are now (a|an|in) ", re.I),
    re.compile(r"reveal (your )?(system )?prompt", re.I),
    re.compile(r"\bDAN\b|\bdeveloper mode\b", re.I),
    re.compile(r"<\|im_start\|>|</?system>", re.I),
]

MAX_QUERY_CHARS = 2000


@dataclass
class GuardrailResult:
    allowed: bool = True
    text: str = ""
    violations: list[str] = field(default_factory=list)
    redactions: dict = field(default_factory=dict)


def scan_pii(text: str) -> dict[str, list[str]]:
    return {k: p.findall(text) for k, p in PII_PATTERNS.items() if p.search(text)}


def redact(text: str) -> tuple[str, dict]:
    found = {}
    for name, pat in PII_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            found[name] = len(matches)
            text = pat.sub(f"[REDACTED_{name.upper()}]", text)
    return text, found


def check_input(query: str) -> GuardrailResult:
    res = GuardrailResult(text=query)
    if len(query) > MAX_QUERY_CHARS:
        res.allowed = False
        res.violations.append("query_too_long")
        return res
    for pat in INJECTION_PATTERNS:
        if pat.search(query):
            res.allowed = False
            res.violations.append("prompt_injection_suspected")
            return res
    res.text, res.redactions = redact(query)
    if res.redactions:
        res.violations.append("pii_redacted_from_input")
    return res


def check_output(answer: str, citations: list[str], is_refusal: bool = False) -> GuardrailResult:
    res = GuardrailResult(text=answer)
    res.text, res.redactions = redact(answer)
    if res.redactions:
        res.violations.append("pii_redacted_from_output")
    if not is_refusal and not citations:
        res.violations.append("answer_without_citations")
        res.allowed = False
    return res
