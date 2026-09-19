# How to finish this before it goes on your resume

This scaffold was generated without network access, so it has **not** been run
end-to-end. Every module is real, working code — not pseudocode — but "the code
looks right" and "I ran it and it works" are different claims. Do these steps,
in order, before you write a single number in your README.

## 1. Install and smoke-test (30 min)

```bash
pip install -r requirements.txt
make test           # unit tests — should pass with zero API calls
```

If a test fails, that's a real bug in the scaffold — fix it, don't skip it.
That debugging is itself worth a line in your README's commit history.

## 2. Pick a real corpus (1-2 hours)

The included sample doc exists only so `make ingest` has something to chunk.
Replace `data/corpus/` with your actual domain before building a golden set.
Good options that read well against this specific JD (enterprise/BFSI-flavoured):

- IRDAI insurance policy wordings (public PDFs)
- RBI master circulars / directions
- Your university's academic regulations + syllabi (you already have these)
- A public API's full documentation (e.g. Stripe, Razorpay)

Aim for 30-100 pages. Bigger isn't better for a portfolio project — a corpus
you can personally verify answers against is more valuable than a huge one you
can't.

## 3. Ingest and sanity-check retrieval (30 min)

```bash
export ANTHROPIC_API_KEY=...
make ingest
make ask Q="<something you know the answer to>"
```

Read the retrieved evidence. If it's obviously wrong, fix chunking before
building the golden set — a golden set built on top of broken retrieval will
just describe the brokenness, not catch it.

## 4. Build the golden set (2-4 hours — this is the actual project)

```bash
cp data/golden/golden_set.example.jsonl data/golden/golden_set.jsonl
```

Then, per `data/golden/README.md`:
- Write 60-100 questions by hand.
- Include ~15% genuinely unanswerable questions.
- Run `make label` and hand-pick the correct chunk ids for each answerable
  question — do not auto-label with the retriever you're about to evaluate.

This step is not optional busywork. It *is* the project. An interviewer will
ask how you built it, and "I wrote every question myself against documents I
read" is the answer that gets you the next question instead of a follow-up trap.

## 5. Run the ablations (varies — API cost, budget ~$5-15)

```bash
make eval-smoke      # first: 5 questions, no judge, confirm nothing crashes
make eval-all        # then: the real thing
```

Copy `results/summary.md` into the README tables. Fill in the stage-latency
breakdown from the JSON output. Run `Judge.self_consistency()` on ~20 items and
report the stdev — see `agentic_rag/eval/judge.py`.

## 6. Write the "what I found" paragraph

The table is not the deliverable — the sentence explaining it is. Before your
interview, be able to answer without looking:
- Which retrieval mode won, and by how much (with the CI)?
- What did the critic node cost, and what did it buy?
- Where did the latency actually go?
- What's the one thing you'd fix with another week?

## 7. Optional but strong: wire up real tracing

Run Langfuse or Phoenix locally (`docker-compose up` includes a Qdrant
service you can extend), point `OTEL_EXPORTER_OTLP_ENDPOINT` at it, and take a
screenshot of a real trace with retrieval and generation spans for the README.
This is a 20-minute addition that most fresher resumes don't have at all.
