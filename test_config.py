from agentic_rag.config import Settings


def test_fingerprint_stable_for_identical_settings():
    a = Settings.load()
    b = Settings.load()
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_retrieval_mode():
    a = Settings.load()
    b = Settings.load(retrieval={"mode": "dense"})
    assert a.fingerprint() != b.fingerprint()


def test_overrides_apply_on_top_of_yaml():
    s = Settings.load(llm={"model": "claude-haiku-4-5"})
    assert s.llm.model == "claude-haiku-4-5"
