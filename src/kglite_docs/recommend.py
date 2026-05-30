"""Follow-on study recommendations (R8).

A study answers one question; its findings often imply a *different* question
worth investigating next. This module detects that and **proposes** the
follow-on — pre-seeded with the findings that justify it — for a human to
approve. It never auto-runs a study: a proposal is a `StudyRecommendation` node;
approving it (`spawn_study`) creates the child and writes a `SPAWNED_FROM` edge.

The mapping from a finding signal to a study template is an **extensible trigger
registry** (generic seam, like the lens registry): core ships it empty; the legal
pack registers rules such as `disparate_treatment → "judicial bias"`. So a
concluded study never dead-ends a thread its own findings opened.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs import study as study_mod
from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError
from kglite_docs.schema import (
    REC_APPROVED,
    REC_PROPOSED,
    RECOMMENDS,
    SPAWNED_FROM,
    STUDY,
    STUDY_RECOMMENDATION,
    label_for,
    labels_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

#: finding_signal (a Finding's finding_type) → study template.
_TRIGGERS: dict[str, dict[str, str]] = {}


def register_recommendation_trigger(
    finding_signal: str, *, question_template: str, suggested_lens: str = "",
    rationale: str = "",
) -> None:
    """Register a rule: a finding of type `finding_signal` proposes a follow-on
    study asking `question_template`. Idempotent; raises on conflicting redefinition."""
    spec = {"question_template": question_template, "suggested_lens": suggested_lens, "rationale": rationale}
    existing = _TRIGGERS.get(finding_signal)
    if existing is not None and existing != spec:
        raise ValueError(f"recommendation trigger {finding_signal!r} already registered differently")
    _TRIGGERS[finding_signal] = spec


def available_triggers() -> tuple[str, ...]:
    return tuple(sorted(_TRIGGERS))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_recommendations(store: Store, *, study_id: str) -> list[dict[str, Any]]:
    """Recommendations proposed from a study (newest first)."""
    rows = _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:RECOMMENDS]->(r:StudyRecommendation) "
        "RETURN r.id AS recommendation_id, r.question AS question, r.rationale AS rationale, "
        "r.suggested_lens AS suggested_lens, r.trigger AS trigger, r.status AS status, "
        "r.seed_finding_ids_json AS seeds, r.child_study_id AS child_study_id "
        "ORDER BY r.created_at DESC",
        params={"id": study_id},
    ))
    for r in rows:
        raw = r.pop("seeds", None)
        try:
            r["seed_finding_ids"] = json.loads(raw) if raw else []
        except (TypeError, ValueError):
            r["seed_finding_ids"] = []
    return rows


def recommend_studies(store: Store, *, study_id: str) -> list[dict[str, Any]]:
    """Propose follow-on studies from this study's findings (proposals only —
    never auto-run). Groups findings by their `finding_type`; for each type with a
    registered trigger, proposes one study seeded with the matching findings. Idempotent:
    an already-proposed recommendation for the same trigger + seeds is reused."""
    if not study_mod._study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    findings = study_mod.list_findings(store, study_id=study_id)
    by_sig: dict[str, list[str]] = {}
    for f in findings:
        sig = f.get("finding_type") or ""
        if sig in _TRIGGERS:
            by_sig.setdefault(sig, []).append(f["finding_id"])

    existing = {(r["trigger"], tuple(sorted(r["seed_finding_ids"]))): r
                for r in list_recommendations(store, study_id=study_id)}
    out: list[dict[str, Any]] = []
    for sig, seeds in by_sig.items():
        key = (sig, tuple(sorted(seeds)))
        if key in existing:
            out.append(existing[key])
            continue
        trig = _TRIGGERS[sig]
        rid = "rec_" + uuid.uuid4().hex[:16]
        store.upsert_nodes(STUDY_RECOMMENDATION, [{
            "id": rid, "title": trig["question_template"][:120],
            "source_study_id": study_id, "question": trig["question_template"],
            "rationale": trig["rationale"], "suggested_lens": trig["suggested_lens"],
            "trigger": sig, "seed_finding_ids_json": json.dumps(sorted(seeds)),
            "status": REC_PROPOSED, "child_study_id": "", "created_at": _now(),
        }])
        store.upsert_edges(RECOMMENDS, [{"src": study_id, "dst": rid}],
                           source_type=STUDY, target_type=STUDY_RECOMMENDATION)
        store.add_label(STUDY_RECOMMENDATION, [rid], label_for("recommendation.status", REC_PROPOSED))
        out.append({
            "recommendation_id": rid, "question": trig["question_template"],
            "rationale": trig["rationale"], "suggested_lens": trig["suggested_lens"],
            "trigger": sig, "status": REC_PROPOSED, "seed_finding_ids": sorted(seeds),
            "child_study_id": "",
        })
    return out


def spawn_study(store: Store, *, recommendation_id: str, approved_by: str) -> dict[str, Any]:
    """Approve a recommendation: create the child study and write the
    `SPAWNED_FROM` edge (child → source) carrying the reason + seed findings.
    Human-approved; never automatic."""
    rows = _df_dicts(store.cypher(
        "MATCH (rec:StudyRecommendation {id: $id}) RETURN rec.source_study_id AS source, "
        "rec.question AS question, rec.rationale AS rationale, rec.trigger AS trigger, "
        "rec.seed_finding_ids_json AS seeds, rec.status AS status, rec.child_study_id AS child",
        params={"id": recommendation_id},
    ))
    if not rows:
        raise InvalidEnumError(f"recommendation not found: {recommendation_id}")
    rec = rows[0]
    if rec.get("status") == REC_APPROVED and rec.get("child"):
        return {"recommendation_id": recommendation_id, "child_study_id": rec["child"], "already_approved": True}
    register_agent(store, agent_id=approved_by)
    child_id = study_mod.define_study(store, question=rec["question"], created_by=approved_by)
    try:
        seeds = json.loads(rec.get("seeds") or "[]")
    except (TypeError, ValueError):
        seeds = []
    store.upsert_edges(
        SPAWNED_FROM, [{
            "src": child_id, "dst": rec["source"], "reason": rec.get("trigger") or "",
            "seed_finding_ids": json.dumps(seeds), "approved_by": approved_by, "created_at": _now(),
        }],
        source_type=STUDY, target_type=STUDY,
    )
    store.cypher(
        "MATCH (rec:StudyRecommendation {id: $id}) SET rec.status = $st, rec.child_study_id = $child",
        params={"id": recommendation_id, "st": REC_APPROVED, "child": child_id},
    )
    store.swap_label(
        STUDY_RECOMMENDATION, [recommendation_id],
        add=label_for("recommendation.status", REC_APPROVED),
        remove_any_of=labels_for("recommendation.status"),
    )
    return {
        "recommendation_id": recommendation_id, "child_study_id": child_id,
        "question": rec["question"], "seed_finding_ids": seeds, "approved_by": approved_by,
    }
