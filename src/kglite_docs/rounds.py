"""Leveled review — escalating, multi-reviewer passes you dial up on demand.

A study can spend more review effort *exactly where it moves the needle*: add
independent reviewers to raise confidence on contested findings (escalate
*accuracy*, R2), or run an analytical *lens* not yet run to surface a class
earlier passes couldn't even look for (escalate *detectability*, R3). Each
escalation is a `ReviewRound` whose `EXAMINED` edges are the coverage record
(R4), so an un-run lens is a *named* blind spot, not a silent gap.

This is a **control system, not new perception** — it reuses the existing
substrate per round: `study.verify_finding` (the vote → confidence), the
`claim_or_preview` punchcard (a claimable, non-overlapping worklist keyed on the
round), and the lens registry (`lenses.py`). Stateless functions over `store`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs import study as study_mod
from kglite_docs.activity import register_agent
from kglite_docs.checkout import claim_or_preview
from kglite_docs.errors import InvalidEnumError
from kglite_docs.lenses import available_lenses, is_registered_lens
from kglite_docs.schema import (
    AGENT,
    CHUNK,
    CLAIM_TTL_SECONDS,
    CONDUCTED_BY,
    EXAMINED,
    FINDING,
    HAS_ROUND,
    REVIEW_ROUND,
    ROUND_DONE,
    ROUND_OPEN,
    STUDY,
    VALID_ROUND_KINDS,
    VALID_ROUND_SCOPES,
    label_for,
    labels_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_meta(store: Store, round_id: str) -> dict[str, Any] | None:
    rows = _df_dicts(store.cypher(
        "MATCH (r:ReviewRound {id: $id}) RETURN r.id AS id, r.study_id AS study_id, "
        "r.level AS level, r.kind AS kind, r.lens AS lens, r.scope AS scope, "
        "r.status AS status, r.new_findings AS new_findings",
        params={"id": round_id},
    ))
    return rows[0] if rows else None


def _next_level(store: Store, study_id: str) -> int:
    rows = _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:HAS_ROUND]->(r:ReviewRound) "
        "RETURN max(r.level) AS lvl", params={"id": study_id},
    ))
    top = rows[0].get("lvl") if rows else None
    return int(top) + 1 if top is not None else 1


def escalate_study(
    store: Store,
    *,
    study_id: str,
    kind: str,
    created_by: str,
    level: int | None = None,
    lens: str | None = None,
    reviewers: int = 1,
    scope: str = "contested",
    limit: int = 50,
) -> dict[str, Any]:
    """Open a `ReviewRound` and hand back only the work it targets — never a
    blind re-run (R2/R3). `scope`:
    - `contested` / `low_depth` → the **findings** that need more reviewers
      (split, or fewer than 2 reviewers);
    - `uncovered` → study chunks **not yet examined by this `lens`** (a
      detectability sweep that may surface new findings);
    - `all` → every ready chunk in the study's material.
    An unknown `lens` raises (an un-run lens must be a *named* gap, never a
    silent zero)."""
    if not study_mod._study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    if kind not in VALID_ROUND_KINDS:
        raise InvalidEnumError(f"invalid round kind: {kind!r} (expected one of {sorted(VALID_ROUND_KINDS)})")
    if scope not in VALID_ROUND_SCOPES:
        raise InvalidEnumError(f"invalid scope: {scope!r} (expected one of {sorted(VALID_ROUND_SCOPES)})")
    if lens is not None and not is_registered_lens(lens):
        raise InvalidEnumError(
            f"unknown lens {lens!r} — not registered ({list(available_lenses())}). "
            "Load a schema pack that registers lenses (e.g. the legal pack)."
        )
    register_agent(store, agent_id=created_by)
    rid = "round_" + uuid.uuid4().hex[:16]
    lvl = int(level) if level is not None else _next_level(store, study_id)
    store.upsert_nodes(REVIEW_ROUND, [{
        "id": rid, "title": f"L{lvl} {kind}" + (f" [{lens}]" if lens else ""),
        "study_id": study_id, "level": lvl, "kind": kind, "lens": lens or "",
        "scope": scope, "reviewers": int(reviewers), "status": ROUND_OPEN,
        "created_by": created_by, "created_at": _now(),
        "new_findings": 0, "confidence_lift": 0.0,
    }])
    store.upsert_edges(HAS_ROUND, [{"src": study_id, "dst": rid}], source_type=STUDY, target_type=REVIEW_ROUND)
    store.upsert_edges(CONDUCTED_BY, [{"src": rid, "dst": created_by}], source_type=REVIEW_ROUND, target_type=AGENT)
    store.add_label(REVIEW_ROUND, [rid], label_for("round.status", ROUND_OPEN))

    worklist, target_kind = _build_worklist(store, study_id=study_id, scope=scope, lens=lens, limit=limit)
    return {
        "round_id": rid, "study_id": study_id, "level": lvl, "kind": kind,
        "lens": lens or "", "scope": scope, "reviewers": int(reviewers),
        "target_kind": target_kind, "worklist": worklist, "worklist_size": len(worklist),
    }


def _build_worklist(
    store: Store, *, study_id: str, scope: str, lens: str | None, limit: int,
) -> tuple[list[dict[str, Any]], str]:
    if scope in ("contested", "low_depth"):
        findings = study_mod.list_findings(store, study_id=study_id)
        if scope == "contested":
            sel = [f for f in findings if f.get("escalation_state") in ("contested", "needs_more")]
        else:
            sel = [f for f in findings if int(f.get("reviewer_count", 0)) < 2]
        wl = [{
            "id": f["finding_id"], "statement": f.get("statement", ""),
            "escalation_state": f.get("escalation_state"),
            "reviewer_count": f.get("reviewer_count", 0),
        } for f in sel[:limit]]
        return wl, "finding"
    # uncovered / all → chunks (preview; claim via next_review)
    return _uncovered_chunks(store, study_id=study_id, lens=lens, scope=scope, limit=limit, agent_id=None), "chunk"


def _uncovered_chunks(
    store: Store, *, study_id: str, lens: str | None, scope: str, limit: int,
    agent_id: str | None, ttl_seconds: int = CLAIM_TTL_SECONDS, checkout_key: str | None = None,
) -> list[dict[str, Any]]:
    base_params: dict[str, Any] = {"sid": study_id, "lim": int(limit)}
    where_sql = (
        "c.status = 'ready' AND EXISTS { "
        "MATCH (c)-[:ASSESSED_AS]->(:Assessment)-[:OF_STUDY]->(:Study {id: $sid}) }"
    )
    if scope == "uncovered" and lens:
        base_params["lens"] = lens
        not_done = "NOT EXISTS { MATCH (c)<-[:EXAMINED]-(:ReviewRound {lens: $lens}) }"
    else:
        not_done = "c.id IS NOT NULL"  # 'all' scope: every ready study chunk
    return claim_or_preview(
        store, where_sql=where_sql, not_done=not_done, base_params=base_params,
        checkout_key=checkout_key or study_id, agent_id=agent_id, ttl_seconds=ttl_seconds,
    )


def next_review(
    store: Store, *, round_id: str, agent_id: str | None = None,
    limit: int = 20, ttl_seconds: int = CLAIM_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """Uncovered chunks for THIS round's lens (a detectability round). With
    `agent_id`, atomically claims a batch (punchcard keyed on the round) so a
    fan-out of reviewers never overlaps; without it, a read-only preview. (A
    finding-scoped round's worklist is returned by `escalate_study` directly.)"""
    meta = _round_meta(store, round_id)
    if meta is None:
        raise InvalidEnumError(f"round not found: {round_id}")
    return _uncovered_chunks(
        store, study_id=meta["study_id"], lens=meta.get("lens") or None,
        scope=meta.get("scope") or "all", limit=limit, agent_id=agent_id,
        ttl_seconds=ttl_seconds, checkout_key=round_id,
    )


def record_review(
    store: Store,
    *,
    round_id: str,
    target_id: str,
    target_kind: str = "finding",
    verdict: str | None = None,
    agent_id: str,
    notes: str = "",
    provenance: str | None = None,
) -> dict[str, Any]:
    """Record that this round examined a unit (the `EXAMINED` coverage edge) and,
    for a finding with a `verdict`, cast the reviewer vote (delegates to
    `verify_finding`, updating confidence/escalation_state). For a chunk, records
    coverage only — any new pattern the lens surfaces is a `study("finding", …,
    origin_round_id=round_id)`."""
    meta = _round_meta(store, round_id)
    if meta is None:
        raise InvalidEnumError(f"round not found: {round_id}")
    if target_kind not in ("finding", "chunk"):
        raise InvalidEnumError(f"invalid target_kind: {target_kind!r} (expected 'finding' or 'chunk')")
    node_type = FINDING if target_kind == "finding" else CHUNK
    exists = _df_dicts(store.cypher(
        f"MATCH (n:{node_type} {{id: $id}}) RETURN n.id AS id", params={"id": target_id},
    ))
    if not exists:
        raise InvalidEnumError(f"{target_kind} not found: {target_id}")
    register_agent(store, agent_id=agent_id)
    store.upsert_edges(
        EXAMINED, [{"src": round_id, "dst": target_id}],
        source_type=REVIEW_ROUND, target_type=node_type,
    )
    result: dict[str, Any] = {"round_id": round_id, "target_id": target_id, "examined": True}
    if target_kind == "finding" and verdict:
        result["vote"] = study_mod.verify_finding(
            store, finding_id=target_id, verdict=verdict,
            verifier_agent_id=agent_id, notes=notes, provenance=provenance,
        )
    return result


def close_round(store: Store, *, round_id: str) -> dict[str, Any]:
    """Close a round: count the findings it produced (`origin_round_id`) and mark
    it done. (`confidence_lift` across rounds is reported by `study_confidence`.)"""
    meta = _round_meta(store, round_id)
    if meta is None:
        raise InvalidEnumError(f"round not found: {round_id}")
    new_findings = int(_df_dicts(store.cypher(
        "MATCH (f:Finding {origin_round_id: $rid}) RETURN count(f) AS n",
        params={"rid": round_id},
    ))[0]["n"])
    store.cypher(
        "MATCH (r:ReviewRound {id: $id}) SET r.status = $st, r.new_findings = $nf",
        params={"id": round_id, "st": ROUND_DONE, "nf": new_findings},
    )
    store.swap_label(
        REVIEW_ROUND, [round_id],
        add=label_for("round.status", ROUND_DONE),
        remove_any_of=labels_for("round.status"),
    )
    return {"round_id": round_id, "status": ROUND_DONE, "new_findings": new_findings}


def list_rounds(store: Store, *, study_id: str) -> list[dict[str, Any]]:
    """All review rounds for a study, oldest first (the escalation history)."""
    return _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:HAS_ROUND]->(r:ReviewRound) "
        "RETURN r.id AS round_id, r.level AS level, r.kind AS kind, r.lens AS lens, "
        "r.scope AS scope, r.status AS status, r.reviewers AS reviewers, "
        "r.new_findings AS new_findings, r.created_by AS created_by, r.created_at AS created_at "
        "ORDER BY r.level, r.created_at",
        params={"id": study_id},
    ))
