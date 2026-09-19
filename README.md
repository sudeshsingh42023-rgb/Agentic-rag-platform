# Golden set

`golden_set.jsonl` is not tracked -- it is *your* dataset over *your* corpus, and
it is the actual deliverable of this project, not boilerplate to skip.

## Schema (one JSON object per line)

```json
{
  "id": "q014",
  "question": "...",
  "reference_answer": "short reference answer, or null if unanswerable",
  "relevant_chunk_ids": ["docid-00012", "docid-00013"],
  "unanswerable": false
}
```

## Build it in three passes

1. **Write 60-100 questions by hand** against your corpus. Mix simple lookups,
   multi-hop questions that need two chunks, and near-duplicates that probe
   whether the retriever confuses similar sections.
2. **Add ~15% genuinely unanswerable questions** -- plausible-sounding, but
   about something the corpus does not cover. `unanswerable: true`,
   `relevant_chunk_ids: []`. Skip this and you cannot measure hallucination,
   which is the whole point of the critic node.
3. **Label `relevant_chunk_ids` using `make label`**, which retrieves
   candidates and lets *you* pick -- never auto-label with the retriever you're
   about to evaluate, or your recall number measures nothing.

Run `python -m agentic_rag.eval.run_eval --golden ... ` to validate the schema
before spending API budget on a bad file.
