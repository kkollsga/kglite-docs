"""Shared punchcard: claim a batch of chunks for one agent so parallel agents
never overlap. Used by both the study assessment work-list (`study.next_unassessed`)
and the classification work-list (`classify.next_unclassified`), keyed on disjoint
checkout keys (a study id vs the classify sentinel) so the two never collide.

Without `agent_id` it's a read-only preview (no claim, no mutation). With it, the
returned chunks are atomically checked out; claims auto-expire after `ttl_seconds`.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.schema import (
    AGENT,
    CHECKED_OUT,
    CHECKOUT,
    CHUNK,
    CLAIM_TTL_SECONDS,
    HOLDS,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

#: Checkout key for the classification work-list — disjoint from any real study
#: id ("study_<hex>"), so classify claims and study claims never interfere.
CLASSIFY_CHECKOUT_KEY = "__classify__"

_lock = threading.Lock()
_COLS = (
    "c.id AS id, c.doc_id AS doc_id, c.page_number AS page, "
    "c.chunk_index AS chunk_index, c.text AS text, c.title AS title"
)
#: Default reading-order. Callers may pass a rank-prefixed order (e.g. element
#: scoping) — it must end with `LIMIT $lim` and reference only the chunk `c`.
DEFAULT_ORDER = "ORDER BY c.doc_id, c.page_number, c.chunk_index LIMIT $lim"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim_or_preview(
    store: Store,
    *,
    where_sql: str,
    not_done: str,
    base_params: dict[str, Any],
    checkout_key: str,
    agent_id: str | None,
    order_by: str = DEFAULT_ORDER,
    ttl_seconds: int = CLAIM_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """Select the work-list (`WHERE {where_sql} AND {not_done}`), ordered by
    `order_by` (reading order by default).

    Preview (no `agent_id`): just return it. Claim (`agent_id`): GC stale
    checkouts for `checkout_key`, select only unclaimed rows, and punch a
    Checkout for them — under a lock so the GC→select→punch is atomic.
    `base_params` must carry `$lim` (and any params referenced by the SQL).
    """
    if not agent_id:
        return _df_dicts(store.cypher(
            f"MATCH (c:Chunk) WHERE {where_sql} AND {not_done} RETURN {_COLS} {order_by}",
            params=base_params,
        ))
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)).isoformat()
    not_claimed = "NOT EXISTS { MATCH (c)<-[:CHECKED_OUT]-(:Checkout {study_id: $ckey}) }"
    params = {**base_params, "ckey": checkout_key}
    with _lock:
        store.cypher(
            "MATCH (co:Checkout) WHERE co.study_id = $ckey AND co.at < $cutoff DETACH DELETE co",
            params={"ckey": checkout_key, "cutoff": cutoff},
        )
        rows = _df_dicts(store.cypher(
            f"MATCH (c:Chunk) WHERE {where_sql} AND {not_done} AND {not_claimed} "
            f"RETURN {_COLS} {order_by}",
            params=params,
        ))
        if rows:
            register_agent(store, agent_id=agent_id)
            co_id = "co_" + uuid.uuid4().hex[:16]
            store.upsert_nodes(CHECKOUT, [{
                "id": co_id, "title": f"checkout {agent_id} {checkout_key}",
                "study_id": checkout_key, "by_agent": agent_id, "at": _now(),
            }])
            store.upsert_edges(
                HOLDS, [{"src": agent_id, "dst": co_id}],
                source_type=AGENT, target_type=CHECKOUT,
            )
            store.upsert_edges(
                CHECKED_OUT, [{"src": co_id, "dst": r["id"]} for r in rows],
                source_type=CHECKOUT, target_type=CHUNK,
            )
    return rows
