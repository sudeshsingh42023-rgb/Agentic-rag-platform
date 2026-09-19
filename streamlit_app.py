"""Streamlit demo UI.

Shows the answer next to the evidence, the critic verdict and the cost. The
evidence panel is not decoration -- being able to see which chunk grounded which
sentence is what turns a demo into a debugging session.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.graph import Pipeline, answer_question  # noqa: E402

st.set_page_config(page_title="Agentic RAG", layout="wide")


@st.cache_resource
def get_pipeline(mode: str) -> tuple[Pipeline, Settings]:
    settings = Settings.load(retrieval={"mode": mode})
    return Pipeline.build(settings), settings


st.title("Agentic RAG")
st.caption("Hybrid retrieval, planner-critic loop, cited answers or honest refusals.")

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid_rerank", "hybrid", "dense", "bm25"],
        help="Switch modes to feel the difference the ablation table measures.",
    )
    show_evidence = st.checkbox("Show retrieved evidence", value=True)
    show_trace = st.checkbox("Show stage timings", value=True)

question = st.text_input("Question", placeholder="Ask something about the indexed corpus")

if question:
    pipe, settings = get_pipeline(mode)
    with st.spinner("Retrieving and synthesising..."):
        result = answer_question(question, settings, pipe=pipe)

    if result.get("blocked"):
        st.error(result.get("answer"))
    elif result.get("refused"):
        st.warning(result.get("answer"))
        st.caption(
            "A refusal here is the system working. The alternative is a fluent "
            "answer with nothing behind it."
        )
    else:
        st.markdown(result.get("answer", ""))

    left, right = st.columns([1, 1])
    with left:
        critique = result.get("critique") or {}
        verdict = critique.get("verdict", "n/a")
        colour = {"pass": "green", "warn": "orange", "fail": "red"}.get(verdict, "gray")
        st.markdown(f"**Critic verdict:** :{colour}[{verdict}]")
        for issue in critique.get("issues", []):
            st.caption(f"- {issue}")
        for claim in critique.get("unsupported_claims", []):
            st.caption(f"- unsupported: {claim}")

    with right:
        usage = result.get("usage", {})
        timings = result.get("timings", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Latency", f"{timings.get('total_ms', 0):.0f} ms")
        c2.metric("Cost", f"${usage.get('cost_usd', 0):.5f}")
        c3.metric("LLM calls", usage.get("llm_calls", 0))

    if show_trace and result.get("timings"):
        st.subheader("Stage timings")
        st.bar_chart(
            {k: v for k, v in result["timings"].items() if k != "total_ms"}
        )

    findings = result.get("guardrail_findings") or []
    if findings:
        st.subheader("Guardrail findings")
        st.dataframe(findings, use_container_width=True)

    if show_evidence and result.get("evidence"):
        st.subheader("Retrieved evidence")
        for i, ev in enumerate(result["evidence"], start=1):
            with st.expander(f"[{i}] {ev['citation']}  (score {ev['score']:.3f})"):
                st.text(ev["text"])
