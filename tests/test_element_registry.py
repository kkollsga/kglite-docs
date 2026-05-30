"""0.0.12 Phase 1: the extensible element-discriminator registry + classify
markers. Core stays domain-opaque — it routes opaque element tokens; the values
arrive from a registered domain schema (legal pack lands in a later phase)."""

from __future__ import annotations

import pytest

from kglite_docs import schema


def test_register_and_resolve_element_discriminator() -> None:
    schema.register_element_discriminator("chunk.test_role", {"foo": "Foo", "bar": "Bar"})
    assert schema.label_for("chunk.test_role", "foo") == "Foo"
    assert set(schema.labels_for("chunk.test_role")) == {"Foo", "Bar"}
    assert schema.element_label("foo") == "Foo"
    assert schema.element_label("nope") is None        # unknown ⇒ None (caller raises)
    assert {"foo", "bar"} <= schema.valid_element_values()


def test_idempotent_and_conflict() -> None:
    schema.register_element_discriminator("chunk.test_dup", {"baz": "Baz"})
    schema.register_element_discriminator("chunk.test_dup", {"baz": "Baz"})  # identical ⇒ no-op
    with pytest.raises(ValueError, match="already registered differently"):
        schema.register_element_discriminator("chunk.test_dup", {"baz": "Other"})
    # An element value can't be claimed by two element discriminators.
    with pytest.raises(ValueError, match="must be unique"):
        schema.register_element_discriminator("chunk.test_other", {"baz": "Baz2"})
    # Built-in discriminators are not registrable.
    with pytest.raises(ValueError, match="built-in"):
        schema.register_element_discriminator("chunk.content_kind", {"x": "X"})


def test_builtins_unchanged() -> None:
    assert schema.label_for("study.stance", "supports") == "Supports"
    assert schema.label_for("chunk.content_kind", "table") == "Table"
    assert schema.label_for("assessment.provenance", "primary_text") == "PrimaryText"
    # Free-text fallback still PascalCases.
    assert schema.label_for("agent.role", "fact-checker") == "FactChecker"


def test_classify_markers() -> None:
    assert schema.label_for("chunk.classify", "classified") == schema.LABEL_CLASSIFIED == "Classified"
    assert schema.label_for("chunk.classify", "unclassified") == schema.LABEL_UNCLASSIFIED
    assert set(schema.labels_for("chunk.classify")) == {"Classified", "Unclassified"}
    assert schema.LABEL_CONTESTED == "Contested"
