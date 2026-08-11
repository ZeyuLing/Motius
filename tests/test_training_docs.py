from tools.audit_training_docs import audit


def test_training_support_is_documented_end_to_end() -> None:
    assert audit() == []
