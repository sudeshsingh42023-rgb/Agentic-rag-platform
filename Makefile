.PHONY: install ingest ask serve ui eval eval-all label test lint

install:
	pip install -r requirements.txt

ingest:
	PYTHONPATH=src python -m agentic_rag.cli ingest --corpus data/corpus

ask:
	PYTHONPATH=src python -m agentic_rag.cli ask "$(Q)"

label:
	PYTHONPATH=src python -m agentic_rag.cli label

serve:
	PYTHONPATH=src uvicorn agentic_rag.api:app --reload --port 8000

ui:
	PYTHONPATH=src streamlit run app/streamlit_app.py

eval:
	PYTHONPATH=src python -m agentic_rag.eval.run_eval --golden data/golden/golden_set.jsonl --out results

eval-all:
	PYTHONPATH=src python -m agentic_rag.eval.run_eval --ablation all --golden data/golden/golden_set.jsonl --out results

eval-smoke:
	PYTHONPATH=src python -m agentic_rag.eval.run_eval --golden data/golden/golden_set.jsonl --out results --limit 5 --no-judge

test:
	PYTHONPATH=src pytest tests/ -v

lint:
	PYTHONPATH=src ruff check src/ tests/
