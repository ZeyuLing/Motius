from tools.audit_model_card_format import audit_model_cards
from tools.normalize_model_cards import (
    REQUIRED_SECTIONS,
    _catalog_cards,
    normalize_all,
)


def test_all_catalog_cards_are_normalized() -> None:
    assert normalize_all(write=False) == []


def test_all_catalog_cards_follow_the_v2_contract() -> None:
    cards = _catalog_cards()
    assert len(cards) == 42
    assert REQUIRED_SECTIONS == (
        "Visual Results",
        "Model Overview",
        "Quick Start",
        "Evaluation Results",
        "Motion Representation",
        "Citation and License",
    )
    assert audit_model_cards() == []
