"""
Temporal event alignment for NarrativeLens.

Selection order:
1. time filter
2. query relevance
3. same-event cohesion
4. source dedupe
5. best 4-6 comparable reports

This version tightens query relevance so a broad related article (for example,
a different country's tariff story) is less likely to survive merely because
it shares generic trade vocabulary.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

COHESION_THRESHOLD = 0.45
MIN_ARTICLES = 4
MAX_ARTICLES = 6

# Slightly stricter than the original 0.30 because real retrieval showed
# semantically related but event-incorrect stories surviving.
RELEVANCE_THRESHOLD = 0.36
TITLE_RELEVANCE_FLOOR = 0.24
MIN_LEXICAL_COVERAGE = 0.34

_QUERY_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "about", "latest", "coverage", "news", "update", "updates", "story"
}


def filter_by_time_window(
    articles: List[Dict[str, Any]],
    mode: str,
    specific_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> List[Dict[str, Any]]:
    def parse(a):
        try:
            return datetime.fromisoformat(a["published_at"])
        except Exception:
            return None

    now = datetime.utcnow()
    if mode == "Today":
        lo, hi = now.replace(hour=0, minute=0, second=0), now
    elif mode == "Past 7 Days":
        lo, hi = now - timedelta(days=7), now
    elif mode == "Specific Date" and specific_date:
        d = datetime.fromisoformat(specific_date)
        lo, hi = d.replace(hour=0, minute=0, second=0), d.replace(hour=23, minute=59, second=59)
    elif mode == "Custom Date Range" and start_date and end_date:
        lo = datetime.fromisoformat(start_date)
        hi = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
    else:
        lo, hi = now - timedelta(days=30), now

    out = []
    for a in articles:
        t = parse(a)
        if t is not None and lo <= t <= hi:
            out.append(a)
    return out


def _lead_text(a: Dict[str, Any]) -> str:
    from .text_analysis import split_sentences
    lead_sents = split_sentences(a.get("text", ""))[:2]
    return (a.get("title", "") + ". " + " ".join(lead_sents)).strip()


def _published_ts(a: Dict[str, Any]) -> float:
    try:
        return datetime.fromisoformat(a["published_at"]).timestamp()
    except Exception:
        return 0.0


def _query_tokens(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [
        t for t in tokens
        if t not in _QUERY_STOP and (len(t) >= 3 or t in {"us", "uk", "eu", "un"})
    ]


def _lexical_coverage(query: str, text: str) -> float:
    tokens = _query_tokens(query)
    if not tokens:
        return 1.0
    hay = set(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))
    return sum(1 for t in tokens if t in hay) / len(tokens)


def rank_by_relevance(articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Hybrid query relevance: semantic similarity + light lexical grounding.

    The semantic score keeps the system tolerant of paraphrases.
    The title/lexical checks reduce false positives that are broadly related
    to the same domain but concern a different event.
    """
    if not articles:
        return []

    from .text_analysis import embed

    query_vec = embed([query])[0]
    leads = [_lead_text(a) for a in articles]
    titles = [a.get("title", "") or "" for a in articles]

    lead_vecs = embed(leads)
    title_vecs = embed(titles)

    lead_sims = lead_vecs @ query_vec
    title_sims = title_vecs @ query_vec

    scored = []
    for a, lead_sim, title_sim, lead_text in zip(articles, lead_sims, title_sims, leads):
        coverage = _lexical_coverage(query, lead_text)

        # Weighted score: lead carries context; title helps preserve event identity.
        score = 0.70 * float(lead_sim) + 0.30 * float(title_sim)

        # A story must be semantically relevant AND show some evidence of the
        # actual query terms/event identity. Very high semantic title similarity
        # can rescue paraphrased wording.
        grounded = coverage >= MIN_LEXICAL_COVERAGE or float(title_sim) >= 0.42

        if score >= RELEVANCE_THRESHOLD and float(title_sim) >= TITLE_RELEVANCE_FLOOR and grounded:
            b = dict(a)
            b["_relevance"] = score
            b["_lead_relevance"] = float(lead_sim)
            b["_title_relevance"] = float(title_sim)
            b["_lexical_coverage"] = coverage
            scored.append(b)

    scored.sort(key=lambda a: (-a["_relevance"], -_published_ts(a)))
    return scored


def check_event_cohesion(articles: List[Dict[str, Any]]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    if len(articles) < MIN_ARTICLES:
        return False, (
            f"Only {len(articles)} sufficiently relevant report(s) remained after event alignment. "
            f"NarrativeLens needs at least {MIN_ARTICLES} independent sources. "
            f"Try a slightly broader date range or a more specific event wording."
        ), []

    from .text_analysis import embed

    leads = [_lead_text(a) for a in articles]
    vecs = embed(leads)
    sims = vecs @ vecs.T

    n = len(articles)
    visited = [False] * n
    components: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not visited[v] and sims[u, v] >= COHESION_THRESHOLD:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    components.sort(key=len, reverse=True)
    largest = components[0]

    if len(components) == 1 or len(largest) >= max(MIN_ARTICLES, int(0.7 * n)):
        group = [articles[i] for i in largest]

        # Preserve relevance ordering after selecting the connected component.
        group.sort(key=lambda a: (-a.get("_relevance", 0.0), -_published_ts(a)))

        seen_sources = set()
        deduped = []
        for a in group:
            src_key = (a.get("source") or "").strip().lower()
            if src_key and src_key not in seen_sources:
                seen_sources.add(src_key)
                deduped.append(a)

        selected = deduped[:MAX_ARTICLES]
        if len(selected) < MIN_ARTICLES:
            return False, (
                "Fewer than four independent sources describe one comparable development "
                "after source deduplication. Try a broader date range or a more specific query."
            ), []

        return True, "", selected

    return False, (
        "The selected period contains multiple distinct developments rather than one comparable event cluster. "
        "Use a more specific query or a narrower date range."
    ), []
