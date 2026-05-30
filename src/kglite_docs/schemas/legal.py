"""Legal element vocabulary — a bundled schema pack (DATA only).

Importing this module registers three element discriminators with the core
engine so a court-case corpus can be classified once and many studies route to
their relevant chunks (e.g. "rank judge remarks by strangeness" reads only
`:JudgeRemark` chunks). The engine never references any term below — these maps
flow through the generic `register_element_discriminator` seam.

Multi-label is the rule: a passage is often several of these at once (a ruling
chunk can be `holding` + `reasoning` + `case_citation`). Element *type* is what
kind of passage it is — stable, classify once; it is NOT stance/impact (+1/−1 for
a party), which stays per-study in the assess layer.

Jurisdiction-neutral (common + civil law). PT/PROJUDI mappings are illustrative:
`dispositivo`→`holding`, `fundamentação`→`reasoning`, `contestação`→`party_argument`,
`pedido`→`relief_sought`, `sentença`→`disposition_order`, `perícia`→`expert_opinion`.
"""

from __future__ import annotations

from kglite_docs.schema import register_element_discriminator

# ─── element families (value → secondary-label name) ───────────────────────

#: Structural / procedural role of the passage.
LEGAL_ROLE = {
    "holding": "Holding",
    "reasoning": "Reasoning",
    "judge_remark": "JudgeRemark",
    "fact_finding": "FactFinding",
    "procedural_posture": "ProceduralPosture",
    "party_argument": "PartyArgument",
    "relief_sought": "ReliefSought",
    "disposition_order": "DispositionOrder",
    "settlement": "Settlement",
}

#: Cited / relied-on authority (additive).
LEGAL_AUTHORITY = {
    "statute": "Statute",
    "case_citation": "CaseCitation",
    "regulation": "Regulation",
    "constitutional": "Constitutional",
    "doctrine": "Doctrine",
}

#: Evidentiary character (additive).
LEGAL_EVIDENCE = {
    "testimony": "Testimony",
    "documentary": "Documentary",
    "expert_opinion": "ExpertOpinion",
    "evidentiary_ruling": "EvidentiaryRuling",
}

# ─── detection rubric (what an agent reads to classify consistently) ────────

#: element id → (definition, recognition cues). Recall-critical elements
#: (`holding`, `disposition_order`) should be over-emitted, not missed.
RUBRIC: dict[str, tuple[str, list[str]]] = {
    "holding": ("The operative ruling / dispositivo — what the court actually decides.",
                ["'we hold', 'the court finds', 'it is ordered'", "the answer to the legal question", "dispositivo / decide"]),
    "reasoning": ("The court's analysis / ratio decidendi supporting the holding.",
                  ["'because', 'therefore', applying the law to facts", "fundamentação / considerando"]),
    "judge_remark": ("A bench comment, aside, or observation by the judge (often dicta or tone).",
                     ["first-person judicial remark not part of the holding", "criticism/praise of a party or counsel", "rhetorical or evaluative aside"]),
    "fact_finding": ("The court's findings of fact (what happened, as found).",
                     ["'the court finds that', established facts", "dated events, who-did-what"]),
    "procedural_posture": ("The stage / what is being decided (motion, appeal, etc.).",
                           ["'on motion to', 'this matter comes before'", "scheduling, jurisdiction, admissibility framing"]),
    "party_argument": ("A litigant's contention (not the court's view).",
                       ["'plaintiff argues', 'defendant contends'", "contestação / razões da parte"]),
    "relief_sought": ("What a party asks the court to do.",
                      ["'plaintiff requests', prayer for relief", "pedido / requer"]),
    "disposition_order": ("The executory order — sentence, injunction, costs, judgment entered.",
                          ["'judgment is entered', 'defendant is enjoined', costs", "sentença / condeno"]),
    "settlement": ("A settlement, consent decree, or agreement between parties.",
                   ["'the parties agree', settlement terms, release", "acordo / transação / homologação"]),
    "statute": ("Statutory text or a statutory citation.",
                ["§, 'Section', 'Art.', code citations", "lei nº / artigo"]),
    "case_citation": ("A citation to precedent / case law.",
                      ["v. (versus) case names, reporters, 'see'", "REsp / RE / acórdão"]),
    "regulation": ("An administrative / regulatory rule.",
                   ["C.F.R., agency rules, 'Regulation'", "resolução / portaria / instrução normativa"]),
    "constitutional": ("A constitutional provision.",
                       ["'Constitution', 'Amendment', constitutional clause", "CF/88 / art. da Constituição"]),
    "doctrine": ("Scholarly / treatise authority.",
                 ["treatise, law-review, named scholars", "doutrina"]),
    "testimony": ("Witness statement or deposition excerpt.",
                  ["Q/A transcript, 'witness testified', sworn statement", "depoimento / testemunha"]),
    "documentary": ("An exhibit or document of record.",
                    ["'Exhibit', attached document, contract/letter quoted", "documento / fls."]),
    "expert_opinion": ("An expert / perito finding or opinion.",
                       ["'expert opines', technical report", "perícia / laudo / parecer técnico"]),
    "evidentiary_ruling": ("An admissibility / weight ruling on evidence.",
                           ["hearsay, best-evidence, 'admitted/excluded', weight given", "valoração da prova"]),
}


def rubric_text() -> str:
    """A prompt block an agent reads to classify a chunk into element ids.
    Returns the controlled vocabulary with definitions + cues."""
    lines = [
        "Classify the passage into ZERO OR MORE of these legal element types "
        "(multi-label; return the element ids). Element type = what KIND of "
        "passage this is, NOT whether it helps a party. Over-include `holding` "
        "and `disposition_order` rather than miss them.",
        "",
    ]
    for eid, (definition, cues) in RUBRIC.items():
        lines.append(f"- {eid}: {definition}  cues: {'; '.join(cues)}")
    return "\n".join(lines)


# Register on import (idempotent). The order is irrelevant; values are globally
# unique across the three families.
register_element_discriminator("chunk.legal_role", LEGAL_ROLE)
register_element_discriminator("chunk.legal_authority", LEGAL_AUTHORITY)
register_element_discriminator("chunk.legal_evidence", LEGAL_EVIDENCE)
