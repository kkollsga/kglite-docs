"""Versioned study reports — keep analysis on the graph, not in `.md` litter.

A `Study` is a node; its conclusions already persist as `Summary` nodes. This is
the richer sibling: a full **markdown report** attached to the study, **named**
(so several distinct reports — a client brief, a judge-conduct memo — coexist)
and **append-only versioned** (each save is a new version node; full history,
latest-wins on read), matching the house style (assessments/`supersede`,
`ReviewEvent`). Reports live inside the `.kgl` — portable, diffable, queryable —
and are written to disk only **on demand** (`export_report`), so agents stop
cluttering the working folder with one-off files.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kglite_docs import study as study_mod
from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError
from kglite_docs.schema import HAS_REPORT, REPORT, STUDY
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_report(
    store: Store, *, study_id: str, name: str, text: str, agent_id: str,
    cites: list[str] | None = None,
) -> dict[str, Any]:
    """Save a markdown report on a study. Append-only: re-saving the same `name`
    writes a **new version** (history preserved). `cites` records the
    finding/assessment ids the report rests on (traceable to evidence)."""
    if not study_mod._study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    name = (name or "").strip() or "report"
    if not (text or "").strip():
        raise InvalidEnumError("report text must be non-empty")
    register_agent(store, agent_id=agent_id)
    prev = _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:HAS_REPORT]->(r:Report {name: $name}) "
        "RETURN max(r.version) AS v",
        params={"id": study_id, "name": name},
    ))
    version = int((prev[0].get("v") if prev else None) or 0) + 1
    rid = "report_" + uuid.uuid4().hex[:16]
    store.upsert_nodes(REPORT, [{
        "id": rid, "title": f"{name} v{version}", "study_id": study_id,
        "name": name, "version": version, "text": text, "by_agent": agent_id,
        "cites_json": json.dumps(list(cites or [])), "created_at": _now(),
    }])
    store.upsert_edges(HAS_REPORT, [{"src": study_id, "dst": rid}],
                       source_type=STUDY, target_type=REPORT)
    return {"report_id": rid, "study_id": study_id, "name": name, "version": version}


def list_reports(store: Store, *, study_id: str) -> list[dict[str, Any]]:
    """Report names on a study, each with its latest version + version count."""
    return _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:HAS_REPORT]->(r:Report) "
        "RETURN r.name AS name, max(r.version) AS latest_version, count(r) AS versions "
        "ORDER BY r.name",
        params={"id": study_id},
    ))


def get_report(
    store: Store, *, study_id: str, name: str | None = None, version: int | None = None,
) -> dict[str, Any] | None:
    """Fetch a report's markdown. Latest version by default; pass `version` for a
    specific one, or omit `name` to get the study's most recent report overall."""
    params: dict[str, Any] = {"id": study_id}
    name_clause = ""
    if name:
        name_clause = " {name: $name}"
        params["name"] = name
    ver_clause = ""
    if version is not None:
        ver_clause = "WHERE r.version = $ver "
        params["ver"] = int(version)
    rows = _df_dicts(store.cypher(
        f"MATCH (:Study {{id: $id}})-[:HAS_REPORT]->(r:Report{name_clause}) {ver_clause}"
        "RETURN r.id AS report_id, r.name AS name, r.version AS version, r.text AS text, "
        "r.by_agent AS by_agent, r.cites_json AS cites_json, r.created_at AS created_at "
        "ORDER BY r.version DESC, r.created_at DESC LIMIT 1",
        params=params,
    ))
    if not rows:
        return None
    r = rows[0]
    raw = r.pop("cites_json", None)
    try:
        r["cites"] = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        r["cites"] = []
    return r


def export_report(
    store: Store, *, study_id: str, name: str | None = None, out_path: str,
    version: int | None = None,
) -> dict[str, Any]:
    """Write a report's markdown to disk **on demand** (the only time a report
    becomes a file). Raises if there's no matching report."""
    rep = get_report(store, study_id=study_id, name=name, version=version)
    if rep is None:
        raise InvalidEnumError(f"no report for study {study_id} (name={name!r}, version={version})")
    Path(out_path).write_text(rep["text"], encoding="utf-8")
    return {"out_path": out_path, "name": rep["name"], "version": rep["version"]}
