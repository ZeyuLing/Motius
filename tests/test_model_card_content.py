from tools.audit_model_card_content import audit_model_card_content


def test_model_card_content_contract() -> None:
    assert audit_model_card_content() == []
