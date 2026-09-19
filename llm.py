"""Thin provider abstraction so model swaps are a config change, not a refactor.
Returns text plus token usage so cost can be attributed per graph node."""
import json
import time
from dataclasses import dataclass

from .config import get_settings
from . import telemetry as tel

# INR per 1M tokens. Edit these to match current pricing before quoting numbers.
PRICING = {
    "claude-sonnet-4-5": {"in": 250.0, "out": 1250.0},
    "claude-haiku-4-5":  {"in": 70.0,  "out": 350.0},
    "gpt-4o":            {"in": 210.0, "out": 840.0},
    "gpt-4o-mini":       {"in": 13.0,  "out": 50.0},
}


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0

    @property
    def cost_inr(self) -> float:
        p = PRICING.get(self.model, {"in": 0.0, "out": 0.0})
        return (self.input_tokens * p["in"] + self.output_tokens * p["out"]) / 1_000_000


def complete(system: str, user: str, model: str | None = None,
             max_tokens: int = 1024, operation: str = "chat") -> LLMResult:
    s = get_settings()
    model = model or s.model_strong
    t0 = time.perf_counter()

    with tel.span(
        f"gen_ai.{operation}",
        **{tel.GEN_AI_SYSTEM: s.llm_provider,
           tel.GEN_AI_REQUEST_MODEL: model,
           tel.GEN_AI_OPERATION_NAME: operation},
    ) as sp:
        if s.llm_provider == "anthropic":
            res = _anthropic(system, user, model, max_tokens)
        elif s.llm_provider == "openai":
            res = _openai(system, user, model, max_tokens)
        else:
            res = _echo(system, user, model)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        sp.set_attribute(tel.GEN_AI_USAGE_INPUT_TOKENS, res.input_tokens)
        sp.set_attribute(tel.GEN_AI_USAGE_OUTPUT_TOKENS, res.output_tokens)
        sp.set_attribute("gen_ai.cost.inr", res.cost_inr)
        return res


def complete_json(system: str, user: str, schema_hint: str, **kw) -> dict:
    """Structured output. A free-form judge or critic drifts; a rubric with a
    fixed JSON shape is aggregable and far lower variance."""
    sys_prompt = (
        f"{system}\n\nRespond with ONLY valid JSON matching this shape. "
        f"No prose, no markdown fences.\n{schema_hint}"
    )
    raw = complete(sys_prompt, user, **kw).text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _anthropic(system, user, model, max_tokens) -> LLMResult:
    import anthropic
    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return LLMResult(
        text="".join(b.text for b in msg.content if b.type == "text"),
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        model=model,
    )


def _openai(system, user, model, max_tokens) -> LLMResult:
    from openai import OpenAI
    client = OpenAI(api_key=get_settings().openai_api_key)
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return LLMResult(
        text=r.choices[0].message.content or "",
        input_tokens=r.usage.prompt_tokens,
        output_tokens=r.usage.completion_tokens,
        model=model,
    )


def _echo(system, user, model) -> LLMResult:
    """Offline stub. Keeps the graph exercisable with zero credentials."""
    if "ONLY valid JSON" in system:
        if "grounded" in system.lower() or "critic" in system.lower():
            text = '{"grounded": true, "unsupported_claims": [], "verdict": "pass"}'
        elif "sub_queries" in system:
            text = '{"needs_decomposition": false, "sub_queries": []}'
        else:
            text = "{}"
    else:
        text = ("[echo provider] No LLM configured. Set LLM_PROVIDER in .env.\n\n"
                f"Context received:\n{user[:400]}")
    return LLMResult(text=text, input_tokens=len(user) // 4,
                     output_tokens=len(text) // 4, model=model)
