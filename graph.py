"""The agent graph.

    guard_in -> plan -> gather -> synthesize -> critic -> (revise loop) -> guard_out

Built on LangGraph because the loop needs durable state and a bounded, auditable
revise cycle. Falls back to an equivalent hand-rolled runner if langgraph is not
installed, so nothing here is un-runnable.
"""
import re
import time
from typing import Any, TypedDict

from . import guardrails, prompts
from . import telemetry as tel
from .config import get_settings
from .llm import complete, complete_json
from .store import HybridStore

DOMAIN = "enterprise policy and regulatory documents"


class RAGState(TypedDict, total=False):
    query: str
    clean_query: str
    sub_queries: list[str]
    hits: list[Any]
    answer: str
    citations: list[str]
    critic: dict
    revisions: int
    blocked: bool
    violations: list[str]
    trace: dict


class RAGPipeline:
    def __init__(self, store: HybridStore, strategy: str | None = None,
                 model: str | None = None):
        self.store = store
        self.settings = get_settings()
        self.strategy = strategy
        self.model = model or self.settings.model_strong
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def n_guard_in(self, state: RAGState) -> RAGState:
        with tel.span("guardrail.input"):
            res = guardrails.check_input(state["query"])
            if not res.allowed:
                return {**state, "blocked": True, "violations": res.violations,
                        "answer": "Request blocked by input guardrail.",
                        "citations": []}
            return {**state, "clean_query": res.text, "violations": res.violations,
                    "blocked": False}

    def n_plan(self, state: RAGState) -> RAGState:
        with tel.span("agent.plan"):
            try:
                plan = complete_json(
                    prompts.PLANNER_SYSTEM.format(domain=DOMAIN),
                    state["clean_query"],
                    prompts.PLANNER_SCHEMA,
                    model=self.settings.model_cheap,   # planning is a cheap-model job
                    operation="plan",
                )
            except Exception:  # noqa: BLE001 - planner failure must not kill the query
                plan = {"needs_decomposition": False, "sub_queries": []}
            subs = plan.get("sub_queries") or []
            if not plan.get("needs_decomposition"):
                subs = []
            return {**state, "sub_queries": subs[:3]}

    def n_gather(self, state: RAGState) -> RAGState:
        queries = state.get("sub_queries") or [state["clean_query"]]
        seen, merged = set(), []
        with tel.span("agent.gather", **{"rag.sub_query_count": len(queries)}):
            for q in queries:
                for hit in self.store.search(q, strategy=self.strategy):
                    if hit.chunk.id not in seen:
                        seen.add(hit.chunk.id)
                        merged.append(hit)
        merged.sort(key=lambda h: -h.score)
        return {**state, "hits": merged[: self.settings.top_k_final * 2]}

    def n_synthesize(self, state: RAGState) -> RAGState:
        hits = state["hits"]
        if not hits:
            return {**state, "answer": "NOT_IN_CONTEXT: retrieval returned no passages.",
                    "citations": []}
        ctx = prompts.format_context(hits)
        user = f"QUESTION: {state['clean_query']}\n\nCONTEXT:\n{ctx}"
        with tel.span("agent.synthesize"):
            res = complete(prompts.SYNTHESIZER_SYSTEM, user, model=self.model,
                           operation="synthesize")
        cites = _extract_citations(res.text, hits)
        return {**state, "answer": res.text.strip(), "citations": cites}

    def n_critic(self, state: RAGState) -> RAGState:
        if not self.settings.use_critic or is_refusal(state["answer"]):
            return {**state, "critic": {"grounded": True, "verdict": "pass"}}
        ctx = prompts.format_context(state["hits"])
        user = (f"QUESTION: {state['clean_query']}\n\nCONTEXT:\n{ctx}\n\n"
                f"ANSWER:\n{state['answer']}")
        with tel.span("agent.critic"):
            try:
                verdict = complete_json(prompts.CRITIC_SYSTEM, user, prompts.CRITIC_SCHEMA,
                                        model=self.model, operation="critic")
            except Exception:  # noqa: BLE001
                verdict = {"grounded": True, "verdict": "pass"}
        return {**state, "critic": verdict}

    def n_revise(self, state: RAGState) -> RAGState:
        ctx = prompts.format_context(state["hits"])
        flagged = state["critic"].get("unsupported_claims", [])
        user = (f"QUESTION: {state['clean_query']}\n\nCONTEXT:\n{ctx}\n\n"
                f"DRAFT ANSWER:\n{state['answer']}\n\n"
                f"UNSUPPORTED CLAIMS TO REMOVE OR FIX:\n" + "\n".join(f"- {c}" for c in flagged))
        with tel.span("agent.revise", **{"agent.revision": state.get("revisions", 0) + 1}):
            res = complete(prompts.REVISE_SYSTEM, user, model=self.model, operation="revise")
        return {**state, "answer": res.text.strip(),
                "citations": _extract_citations(res.text, state["hits"]),
                "revisions": state.get("revisions", 0) + 1}

    def n_guard_out(self, state: RAGState) -> RAGState:
        with tel.span("guardrail.output"):
            res = guardrails.check_output(state["answer"], state.get("citations", []),
                                          is_refusal(state["answer"]))
        return {**state, "answer": res.text,
                "violations": state.get("violations", []) + res.violations}

    # ------------------------------------------------------------- edge logic
    def route_after_critic(self, state: RAGState) -> str:
        crit = state.get("critic", {})
        if crit.get("verdict") == "revise" and \
           state.get("revisions", 0) < self.settings.max_critic_revisions:
            return "revise"
        return "guard_out"

    # ---------------------------------------------------------------- wiring
    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except Exception:  # noqa: BLE001
            return None
        g = StateGraph(RAGState)
        g.add_node("guard_in", self.n_guard_in)
        g.add_node("plan", self.n_plan)
        g.add_node("gather", self.n_gather)
        g.add_node("synthesize", self.n_synthesize)
        g.add_node("critic", self.n_critic)
        g.add_node("revise", self.n_revise)
        g.add_node("guard_out", self.n_guard_out)
        g.set_entry_point("guard_in")
        g.add_conditional_edges("guard_in",
                                lambda s: "end" if s.get("blocked") else "plan",
                                {"plan": "plan", "end": END})
        g.add_edge("plan", "gather")
        g.add_edge("gather", "synthesize")
        g.add_edge("synthesize", "critic")
        g.add_conditional_edges("critic", self.route_after_critic,
                                {"revise": "revise", "guard_out": "guard_out"})
        g.add_edge("revise", "critic")
        g.add_edge("guard_out", END)
        return g.compile()

    def _run_fallback(self, state: RAGState) -> RAGState:
        state = self.n_guard_in(state)
        if state.get("blocked"):
            return state
        state = self.n_gather(self.n_plan(state))
        state = self.n_critic(self.n_synthesize(state))
        while self.route_after_critic(state) == "revise":
            state = self.n_critic(self.n_revise(state))
        return self.n_guard_out(state)

    # ------------------------------------------------------------------- api
    def run(self, query: str) -> dict:
        t0 = time.perf_counter()
        state: RAGState = {"query": query, "revisions": 0}
        with tel.span("rag.request", **{tel.RAG_QUERY: query}):
            out = self._graph.invoke(state) if self._graph else self._run_fallback(state)
        return {
            "query": query,
            "answer": out.get("answer", ""),
            "citations": out.get("citations", []),
            "refused": is_refusal(out.get("answer", "")),
            "blocked": out.get("blocked", False),
            "violations": out.get("violations", []),
            "sub_queries": out.get("sub_queries", []),
            "revisions": out.get("revisions", 0),
            "critic": out.get("critic", {}),
            "retrieved": [
                {"id": h.chunk.id, "doc_id": h.chunk.doc_id,
                 "heading": h.chunk.heading, "score": round(h.score, 4)}
                for h in out.get("hits", [])
            ],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def is_refusal(answer: str) -> bool:
    return answer.strip().upper().startswith("NOT_IN_CONTEXT")


CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _extract_citations(answer: str, hits) -> list[str]:
    ids = []
    for group in CITE_RE.findall(answer):
        for n in group.split(","):
            i = int(n.strip()) - 1
            if 0 <= i < len(hits) and hits[i].chunk.id not in ids:
                ids.append(hits[i].chunk.id)
    return ids
