"""
NarrativeLens text analysis.

THE CORE
- sentence embeddings
- semantic clustering
- unique-source support
- controlled/extractive reader-facing evidence paragraphs

THE LENSES
- fixed theme descriptions
- sentence-to-theme assignment
- per-source thematic share
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

_MODEL = None
MODEL_NAME = "all-MiniLM-L6-v2"

SIM_THRESHOLD = 0.60
MIN_SOURCE_SUPPORT = 3
CLAIM_EVIDENCE_THRESHOLD = 0.63
THEME_THRESHOLD = 0.28
MAX_CLUSTERS = 5
MIN_SENTENCE_LEN = 40


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed(sentences: List[str]) -> np.ndarray:
    model = get_model()
    if not sentences:
        return np.zeros((0, 384))
    return np.asarray(
        model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


# ---------------------------------------------------------------------------
# Sentence cleanup / splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])')

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)

_BOILERPLATE_PATTERNS = [
    # "Published Date -", "Published:", "Updated:", "Last Updated:"
    re.compile(
        r"^\s*(?:Published(?:\s+Date)?|Updated|Last\s+Updated|Posted|Edited)"
        r"\s*[:\-–—]\s*",
        re.I,
    ),

    # "5 min read", including no whitespace before the next token:
    # "5 min readAug 4, 2026 ..."
    re.compile(r"^\s*\d{1,2}\s*min(?:ute)?s?\s*read\s*[:\-–—]?\s*", re.I),

    # "23 July 2026, 10:30 AM IST ..."
    re.compile(
        rf"^\s*\d{{1,2}}\s+{_MONTH}\s+\d{{4}},?\s*"
        r"(?:\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?\s*"
        r"(?:IST|GMT|UTC|EST|PST)?\s*)?[:\-–—]?\s*",
        re.I,
    ),

    # "Aug 4, 2026 05:40 PM IST ..."
    re.compile(
        rf"^\s*{_MONTH}\s+\d{{1,2}},?\s+\d{{4}},?\s*"
        r"(?:\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?\s*"
        r"(?:IST|GMT|UTC|EST|PST)?\s*)?[:\-–—]?\s*",
        re.I,
    ),

    # "2026-08-29 09:20 IST ..."
    re.compile(
        r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*"
        r"(?:\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*"
        r"(?:IST|GMT|UTC|EST|PST)?)?\s*[:\-–—]?\s*"
    ),

    # time-only chrome
    re.compile(
        r"^\s*\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?\s*"
        r"(?:IST|GMT|UTC|EST|PST)\s*[:\-–—]?\s*",
        re.I,
    ),

    # all-caps wire datelines: "NEW DELHI (Reuters) -", "WASHINGTON:"
    re.compile(r"^\s*[A-Z][A-Z .'\-]{2,35}\s*\([A-Za-z .]+\)\s*[-–—:]\s*"),
    re.compile(r"^\s*[A-Z][A-Z .'\-]{2,35}:\s*"),
]

# After a date/time prefix has been removed, a title-case location can remain:
# "New Delhi: Several police personnel ..."
_TITLECASE_DATELINE = re.compile(
    r"^\s*(?:[A-Z][a-z.'\-]+(?:\s+[A-Z][a-z.'\-]+){0,3})\s*:\s*"
)


def _strip_boilerplate(text: str, max_passes: int = 6) -> str:
    """Conservatively remove leading article chrome; never paraphrase text."""
    cleaned = text or ""
    removed_date_like = False

    for _ in range(max_passes):
        before = cleaned

        for idx, pattern in enumerate(_BOILERPLATE_PATTERNS):
            new = pattern.sub("", cleaned, count=1)
            if new != cleaned:
                if idx in {0, 1, 2, 3, 4, 5}:
                    removed_date_like = True
                cleaned = new

        # Only strip a title-case "New Delhi:" style dateline when some
        # publication/read-time chrome was just removed. This avoids deleting
        # ordinary content labels such as "Prime Minister:".
        if removed_date_like:
            cleaned = _TITLECASE_DATELINE.sub("", cleaned, count=1)

        if cleaned == before:
            break

    return cleaned.strip(" \t-–—|")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = _strip_boilerplate(text)

    if not text:
        return []

    raw = _SENT_SPLIT_RE.split(text)
    cleaned: List[str] = []

    for sentence in raw:
        sentence = _strip_boilerplate(sentence.strip()).strip()
        if len(sentence) >= MIN_SENTENCE_LEN:
            cleaned.append(sentence)

    return cleaned


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_STOPWORDS = set("""
the a an and or but of to in on for with as is are was were be been being
this that these those it its their his her they he she which who whom at
by from into over under again further then once here there all any both
each few more most other some such no nor not only own same so than too
very can will just should now did does do has have had said says say told
after before during while about into out up down off per cent percent
""".split())


def extract_keyphrases(sentences: List[str], top_k: int = 4) -> List[str]:
    from collections import Counter

    words = []
    for sentence in sentences:
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", sentence):
            lowered = word.lower()
            if lowered not in _STOPWORDS:
                words.append(lowered)

    return [word for word, _ in Counter(words).most_common(top_k)]


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def _remove_leading_article_title(text: str, title: str) -> str:
    """Remove a repeated scraped headline from the beginning of article text.

    Google News titles frequently end in a publisher suffix:
      "Headline - BBC"
      "Headline | DW.com"

    Extraction can then return:
      "Headline First real body sentence..."

    We first try an exact case-insensitive prefix match, then a conservative
    word-level fuzzy prefix match. Only the repeated headline prefix is
    removed; body text is never paraphrased.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    title = re.sub(r"\s+", " ", title or "").strip()

    if not text or not title:
        return text

    title_candidates = [title]

    for sep in (" - ", " – ", " — ", " | "):
        if sep in title:
            left = title.rsplit(sep, 1)[0].strip()
            if len(left) >= 18:
                title_candidates.append(left)

    title_candidates = sorted(set(title_candidates), key=len, reverse=True)

    # Exact prefix.
    for candidate in title_candidates:
        if text.casefold().startswith(candidate.casefold()):
            remainder = text[len(candidate):].lstrip(" \t:-–—|")
            if len(remainder) >= 20:
                return remainder

    # Conservative fuzzy word-prefix match. This catches punctuation changes
    # such as curly quotes or a source suffix removed by the article page.
    text_words = text.split()

    for candidate in title_candidates:
        title_words = candidate.split()
        if len(title_words) < 4 or len(text_words) < len(title_words):
            continue

        lhs = [_normalize_word(w) for w in title_words]
        rhs = [_normalize_word(w) for w in text_words[:len(title_words)]]

        pairs = [
            (a, b)
            for a, b in zip(lhs, rhs)
            if a and b
        ]
        if not pairs:
            continue

        matches = sum(1 for a, b in pairs if a == b)
        ratio = matches / len(pairs)

        if ratio >= 0.82:
            remainder = " ".join(text_words[len(title_words):]).lstrip(
                " \t:-–—|"
            )
            if len(remainder) >= 20:
                return remainder

    return text


def collect_sentences(articles: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    out = []

    for article in articles:
        article_text = _remove_leading_article_title(
            article.get("text", ""),
            article.get("title", ""),
        )

        for sentence in split_sentences(article_text):
            if _NAVIGATION_NOISE.search(sentence):
                continue
            out.append((sentence, article["source"], article["id"]))

    return out


_ATTRIBUTION_START = re.compile(
    r"^(?:he|she|they)\s+(?:said|added|stated|told|claimed|noted|wrote)\b",
    re.I,
)
_ATTRIBUTION_END = re.compile(
    r",?\s*(?:he|she|they)\s+(?:said|added|stated|told|claimed|noted|wrote)"
    r"\.?\s*$",
    re.I,
)
_METADATA_WORDS = re.compile(
    r"\b(?:published|updated|last updated|min read|IST|GMT|UTC)\b",
    re.I,
)

_NAVIGATION_NOISE = re.compile(
    r"^\s*(?:read\s+more|also\s+read|related(?:\s+stories?)?|recommended|"
    r"more\s+from|see\s+also|subscribe|sign\s+up|click\s+here|follow\s+us|"
    r"advertisement|watch\s*:|listen\s*:|newsletter)\b",
    re.I,
)


def _readability_score(sentence: str, centrality: float) -> float:
    """Prefer a clean, self-contained sentence without changing evidence."""
    length = len(sentence)
    score = 2.0 * float(centrality)

    if 70 <= length <= 260:
        score += 1.0
    elif 45 <= length <= 330:
        score += 0.35
    else:
        score -= 0.5

    if _METADATA_WORDS.search(sentence):
        score -= 2.0

    if re.search(
        r"\b(seems?|clearly|obviously|apparently|arguably|misjudged)\b",
        sentence,
        re.I,
    ):
        score -= 1.25

    if _ATTRIBUTION_START.search(sentence):
        score -= 1.0

    if _ATTRIBUTION_END.search(sentence):
        score -= 1.25

    # Penalize visibly incomplete quote fragments.
    if sentence.count("“") != sentence.count("”"):
        score -= 0.55

    return score


def _choose_display_representative(
    members: List[int],
    medoid_idx: int,
    sims: np.ndarray,
    sentences: List[str],
) -> int:
    """Keep medoid semantics, but prefer a nearby sentence that reads better."""
    candidates = []

    for idx in members:
        similarity_to_medoid = float(sims[idx, medoid_idx])

        # Do not swap in a prettier sentence that drifts away from the
        # cluster's semantic centre.
        if idx != medoid_idx and similarity_to_medoid < 0.55:
            continue

        centrality = (
            float(np.mean([sims[idx, j] for j in members if j != idx]))
            if len(members) > 1
            else 1.0
        )

        candidates.append(
            (
                _readability_score(sentences[idx], centrality),
                centrality,
                idx,
            )
        )

    if not candidates:
        return medoid_idx

    candidates.sort(reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# THE CORE
# ---------------------------------------------------------------------------

def find_consensus_clusters(
    articles: List[Dict[str, Any]],
    total_sources: int,
) -> List[Dict[str, Any]]:
    """Find recurring claims with strict source-level evidence validation.

    Greedy clustering is still used for recall, but a source is counted as
    supporting a claim only when its best sentence is directly similar to the
    cluster medoid at CLAIM_EVIDENCE_THRESHOLD.

    This prevents a sentence about the same broad event from being displayed as
    evidence for a different factual claim merely because both were placed in
    the same loose semantic cluster.
    """
    triples = collect_sentences(articles)

    if not triples:
        return []

    sentences = [item[0] for item in triples]
    sources = [item[1] for item in triples]

    vecs = embed(sentences)
    sims = cosine_sim_matrix(vecs, vecs)

    n = len(sentences)
    assigned = [-1] * n
    clusters: List[List[int]] = []

    # First-stage clustering: broad enough to collect possible paraphrases.
    for i in range(n):
        if assigned[i] != -1:
            continue

        best_cluster, best_score = -1, 0.0

        for ci, members in enumerate(clusters):
            avg_sim = float(
                np.mean([sims[i, member] for member in members])
            )

            if avg_sim >= SIM_THRESHOLD and avg_sim > best_score:
                best_cluster, best_score = ci, avg_sim

        if best_cluster >= 0:
            clusters[best_cluster].append(i)
            assigned[i] = best_cluster
        else:
            clusters.append([i])
            assigned[i] = len(clusters) - 1

    results = []

    for members in clusters:
        raw_sources = sorted(set(sources[i] for i in members))

        if len(raw_sources) < MIN_SOURCE_SUPPORT:
            continue

        # Initial medoid for the loose cluster.
        if len(members) == 1:
            initial_medoid = members[0]
        else:
            initial_avgs = [
                float(
                    np.mean(
                        [sims[i, j] for j in members if j != i]
                    )
                )
                for i in members
            ]
            initial_medoid = members[int(np.argmax(initial_avgs))]

        # One best directly matching sentence per source.
        best_by_source: Dict[str, Tuple[int, float]] = {}

        for source in raw_sources:
            source_members = [
                idx for idx in members
                if sources[idx] == source
            ]

            best_idx = max(
                source_members,
                key=lambda idx: float(sims[initial_medoid, idx]),
            )
            direct_sim = float(sims[initial_medoid, best_idx])

            if direct_sim >= CLAIM_EVIDENCE_THRESHOLD:
                best_by_source[source] = (
                    best_idx,
                    direct_sim,
                )

        if len(best_by_source) < MIN_SOURCE_SUPPORT:
            continue

        strict_indices = [
            idx for idx, _ in best_by_source.values()
        ]

        # Recompute medoid using only validated source-level evidence.
        if len(strict_indices) == 1:
            medoid_idx = strict_indices[0]
        else:
            strict_avgs = [
                float(
                    np.mean(
                        [
                            sims[i, j]
                            for j in strict_indices
                            if j != i
                        ]
                    )
                )
                for i in strict_indices
            ]
            medoid_idx = strict_indices[int(np.argmax(strict_avgs))]

        # Validate once more against the strict medoid.
        final_by_source: Dict[str, Tuple[int, float]] = {}

        for source in raw_sources:
            source_members = [
                idx for idx in members
                if sources[idx] == source
            ]

            best_idx = max(
                source_members,
                key=lambda idx: float(sims[medoid_idx, idx]),
            )
            direct_sim = float(sims[medoid_idx, best_idx])

            if direct_sim >= CLAIM_EVIDENCE_THRESHOLD:
                final_by_source[source] = (
                    best_idx,
                    direct_sim,
                )

        if len(final_by_source) < MIN_SOURCE_SUPPORT:
            continue

        support_indices = [
            idx for idx, _ in final_by_source.values()
        ]
        member_sources = sorted(final_by_source.keys())

        display_idx = _choose_display_representative(
            support_indices,
            medoid_idx,
            sims,
            sentences,
        )

        representative_source = sources[display_idx]

        snippets = {
            source: sentences[idx]
            for source, (idx, _) in final_by_source.items()
        }

        secondary_candidates = [
            (source, idx, score)
            for source, (idx, score) in final_by_source.items()
            if source != representative_source
        ]

        secondary_snippet = None
        if secondary_candidates:
            source, idx, _ = max(
                secondary_candidates,
                key=lambda item: float(sims[display_idx, item[1]]),
            )
            secondary_snippet = (
                source,
                sentences[idx],
            )

        # Confidence = average pairwise similarity between validated source
        # representatives, not the loose cluster average.
        pair_values = [
            float(sims[support_indices[i], support_indices[j]])
            for i in range(len(support_indices))
            for j in range(i + 1, len(support_indices))
        ]
        avg_conf = (
            float(np.mean(pair_values))
            if pair_values
            else 1.0
        )

        member_sentences = [
            sentences[idx]
            for idx in support_indices
        ]

        results.append(
            {
                "representative_sentence": sentences[display_idx],
                "representative_source": representative_source,
                "medoid_sentence": sentences[medoid_idx],
                "medoid_source": sources[medoid_idx],
                "supporting_sources": member_sources,
                "support_count": len(member_sources),
                "total_sources": total_sources,
                "confidence": round(avg_conf, 3),
                "snippets": snippets,
                "secondary_snippet": secondary_snippet,
                "keyphrases": extract_keyphrases(member_sentences),
            }
        )

    results.sort(
        key=lambda cluster: (
            cluster["support_count"],
            cluster["confidence"],
        ),
        reverse=True,
    )

    return results[:MAX_CLUSTERS]


def _human_source_list(sources: List[str]) -> str:
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0]
    if len(sources) == 2:
        return f"{sources[0]} and {sources[1]}"
    return ", ".join(sources[:-1]) + f", and {sources[-1]}"


def _short_evidence_quote(text: str, max_chars: int = 260) -> str:
    """Trim a long extractive quote for readability without paraphrasing it."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return clipped + "…"


def build_cluster_description(cluster: Dict[str, Any]) -> str:
    """Student-facing evidence paragraph with no runtime LLM.

    The representative statement already tells the reader WHAT happened.
    This paragraph answers WHO ELSE reported it and gives one additional
    piece of evidence. Technical similarity details stay in UI metadata.
    """
    sources = cluster["supporting_sources"]
    source_text = _human_source_list(sources)

    parts = [
        f"This same development is also present in coverage from {source_text}."
    ]

    secondary = cluster.get("secondary_snippet")
    if secondary:
        source, snippet = secondary
        parts.append(
            f"{source} adds this supporting detail: "
            f"“{_short_evidence_quote(snippet)}”"
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# EVENT BRIEF — "What is going on?"
# ---------------------------------------------------------------------------

BRIEF_QUERY_FLOOR = 0.12

# The Incident Brief is intentionally generic.  These roles describe the
# information a reader needs for almost any news event (disaster, election,
# court ruling, protest, business announcement, conflict, science story, etc.).
# They are NOT topic labels and do not assume a particular event type.
BRIEF_SECTIONS: List[Dict[str, Any]] = [
    {
        "name": "Event overview",
        "prompt": (
            "the central current event: what happened, who or what was involved, "
            "where or when it happened, and the immediate result"
        ),
        "min_anchor": 0.28,
        "lead_bonus": 0.18,
        "max_sentences": 2,
        "positive": re.compile(
            r"\b(happened|occurred|announced|declared|won|lost|elected|result|"
            r"collapsed|struck|hit|launched|signed|approved|rejected|ruled|"
            r"ordered|attack|flood|earthquake|storm|election|vote|protest|deal|"
            r"agreement|merger|release|outbreak|fire)\b",
            re.I,
        ),
    },
    {
        "name": "How it unfolded",
        "prompt": (
            "how the current event developed, including its direct cause, trigger, "
            "sequence of actions, decision, dispute, or mechanism"
        ),
        "min_anchor": 0.23,
        "lead_bonus": 0.08,
        "max_sentences": 2,
        "positive": re.compile(
            r"\b(because|caused|triggered|led to|prompted|began|"
            r"started|collapsed|failed|dispute|challeng|alleg|decision|ruling|"
            r"counted|ballot|investigation found|report found|sequence)\b",
            re.I,
        ),
    },
    {
        "name": "Impact / consequences",
        "prompt": (
            "the immediate consequences of the current event: casualties, damage, "
            "results, disruption, financial or social effects, winners and losers, "
            "or other concrete outcomes"
        ),
        "min_anchor": 0.21,
        "lead_bonus": 0.04,
        "max_sentences": 2,
        "positive": re.compile(
            r"\b(killed|died|injur|missing|damage|destroy|affected|disrupt|casualt|"
            r"evacuat|closed|halted|suspend|result|won|lost|percent|vote|cost|"
            r"rose|fell|increase|decrease|outage|shortage|impact|consequence)\b",
            re.I,
        ),
    },
    {
        "name": "Response / actions",
        "prompt": (
            "what authorities, organisations, courts, emergency teams, companies, "
            "opposition groups, or other actors did in response to the current event"
        ),
        "min_anchor": 0.19,
        "lead_bonus": 0.02,
        "max_sentences": 2,
        "positive": re.compile(
            r"\b(rescue|search|relief|aid|deploy|evacuat|investigat|respond|announc|"
            r"ordered|appeal|challeng|filed|lawsuit|court|authorit|government|"
            r"police|commission|regulator|agency|officials?|statement|rebuild|"
            r"restore|reopen|negotiat)\b",
            re.I,
        ),
    },
    {
        "name": "Aftermath / wider context",
        "prompt": (
            "what happened next, what remains unresolved, and only the most useful "
            "broader historical, scientific, political, economic, or social context"
        ),
        "min_anchor": 0.13,
        "lead_bonus": 0.0,
        "max_sentences": 2,
        "positive": re.compile(
            r"\b(later|afterward|afterwards|since|ongoing|remain|continues?|expected|"
            r"next|future|previous|earlier|history|historical|years|decades|context|"
            r"long-term|wider|background|trend)\b",
            re.I,
        ),
    },
]


_BRIEF_NOISE = re.compile(
    r"^\s*(?:read\s+more|also\s+read|related(?:\s+stories?)?|recommended|"
    r"more\s+from|see\s+also|subscribe|sign\s+up|click\s+here|follow\s+us|"
    r"advertisement|watch\s*:|listen\s*:|newsletter)\b",
    re.I,
)
_BRIEF_CHROME = re.compile(
    r"\b(?:cookie policy|privacy policy|terms of use|all rights reserved|"
    r"download our app|subscribe to our newsletter)\b",
    re.I,
)
_BRIEF_QUOTE_START = re.compile(r'^\s*[“\"\']')
_BRIEF_FIRST_PERSON = re.compile(r"\b(?:I|we|our|us)\b", re.I)


def _brief_records(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Article sentences for the Incident Brief only.

    This intentionally does not change the sentence pool used by consensus or
    The Lenses.  It simply removes obvious navigation/related-link text and
    retains sentence position so the event overview can prefer article leads.
    """
    records: List[Dict[str, Any]] = []

    for article in articles:
        body = _remove_leading_article_title(
            article.get("text", ""),
            article.get("title", ""),
        )
        sentences = split_sentences(body)

        for pos, sentence in enumerate(sentences):
            cleaned = re.sub(r"\s+", " ", sentence or "").strip()
            if not cleaned:
                continue
            if _BRIEF_NOISE.search(cleaned) or _BRIEF_CHROME.search(cleaned):
                continue
            # Navigation headings and scraped link titles are often short and
            # sentence-like.  Keep short text only when it contains a strong
            # event verb/number signal.
            if len(cleaned) < 58 and not re.search(
                r"\b(?:killed|died|injured|won|lost|announced|ruled|collapsed|"
                r"struck|voted|arrested|detained|\d{1,4})\b",
                cleaned,
                re.I,
            ):
                continue

            records.append(
                {
                    "sentence": cleaned,
                    "source": article["source"],
                    "article_id": article["id"],
                    "title": article.get("title", ""),
                    "position": pos,
                    "published_at": article.get("published_at", ""),
                }
            )

    return records


def _brief_event_year(articles: List[Dict[str, Any]]) -> int | None:
    years = []
    for article in articles:
        value = article.get("published_at", "")
        try:
            years.append(int(str(value)[:4]))
        except Exception:
            continue
    return max(years) if years else None


def _historical_penalty(sentence: str, event_year: int | None, section_name: str) -> float:
    """Keep older-event examples out of current-event sections.

    Older examples are still allowed in the final context section.  This is
    what prevents a sentence about (say) a 2021 disaster from being presented
    as the impact of a 2026 disaster.
    """
    if section_name == "Aftermath / wider context" or event_year is None:
        return 0.0

    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", sentence)]
    if any(year <= event_year - 2 for year in years):
        return 0.60

    if re.search(
        r"\b(?:years? ago|decades? ago|previously|historically|in an earlier|"
        r"during the previous|past election|prior event)\b",
        sentence,
        re.I,
    ):
        return 0.35

    return 0.0


def _brief_quality_score(sentence: str) -> float:
    """Small readability preference; it never changes/paraphrases evidence."""
    n = len(sentence)
    score = 0.0

    if 80 <= n <= 300:
        score += 0.85
    elif 55 <= n <= 380:
        score += 0.35
    else:
        score -= 0.35

    if re.search(r"\b\d+(?:\.\d+)?%?\b", sentence):
        score += 0.12

    if _BRIEF_QUOTE_START.search(sentence):
        score -= 0.35
    if _BRIEF_FIRST_PERSON.search(sentence):
        score -= 0.20
    if _ATTRIBUTION_START.search(sentence):
        score -= 0.20
    if _SUBJECTIVE_CUES.search(sentence):
        score -= 0.45
    if _METADATA_WORDS.search(sentence):
        score -= 0.80

    return score


def _brief_anchor_vector(
    query: str,
    articles: List[Dict[str, Any]],
    anchor_text: str = "",
) -> np.ndarray:
    """Average query/title/consensus evidence into one event-identity vector."""
    pieces = [query.strip()]
    if anchor_text:
        pieces.append(anchor_text.strip())
    pieces.extend(
        (article.get("title", "") or "").strip()
        for article in articles[:6]
        if (article.get("title", "") or "").strip()
    )
    pieces = [p for p in pieces if p]

    vecs = embed(pieces)
    if len(vecs) == 0:
        return np.zeros((384,))

    # Give the corroborated anchor (when present) and the user's query extra
    # weight without letting any one publisher title dominate.
    weights = []
    for i, _ in enumerate(pieces):
        if i == 0:
            weights.append(1.6)
        elif anchor_text and i == 1:
            weights.append(1.8)
        else:
            weights.append(1.0)

    mean = np.average(vecs, axis=0, weights=np.asarray(weights))
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def build_event_brief(
    query: str,
    articles: List[Dict[str, Any]],
    max_points: int = 5,
    anchor_text: str = "",
) -> Dict[str, Any]:
    """Build a general-purpose, event-anchored extractive Incident Brief.

    The algorithm deliberately keeps generation out of the loop:
      1. derive event identity from query + aligned titles + (when available)
         the strongest corroborated claim;
      2. score clean article sentences against that event identity;
      3. separately ask for overview, unfolding/cause, impact, response and
         aftermath/context;
      4. reject old-event examples from current-event sections;
      5. avoid repeated sentences and overusing one source.

    Every displayed sentence remains verbatim article evidence.
    """
    records = _brief_records(articles)
    if not records:
        return {"points": [], "num_sources": 0, "anchor_text": anchor_text}

    sentences = [record["sentence"] for record in records]
    vecs = embed(sentences)
    anchor_vec = _brief_anchor_vector(query, articles, anchor_text)
    anchor_sims = vecs @ anchor_vec

    role_prompts = [
        f"For the current news event '{query}', article evidence about {section['prompt']}."
        for section in BRIEF_SECTIONS[:max_points]
    ]
    role_vecs = embed(role_prompts)
    role_sims = vecs @ role_vecs.T

    event_year = _brief_event_year(articles)
    selected_points: List[Dict[str, Any]] = []
    selected_indices: List[int] = []
    source_use: Dict[str, int] = {}

    for section_idx, section in enumerate(BRIEF_SECTIONS[:max_points]):
        ranked = []

        # Generic role cues are used only as a guardrail.  When the selected
        # articles contain clear evidence for a role (rescue/action, damage,
        # cause, etc.), a generic but merely event-related sentence should not
        # steal that slot.
        cue_count = sum(
            1 for record in records
            if section["positive"].search(record["sentence"])
        )
        prefer_role_cues = section["name"] != "Event overview" and cue_count >= 2

        for i, record in enumerate(records):
            if i in selected_indices:
                continue

            sentence = record["sentence"]
            anchor_sim = float(anchor_sims[i])
            role_sim = float(role_sims[i, section_idx])
            has_role_cue = bool(section["positive"].search(sentence))

            # The sentence must still be about this particular event.  A strong
            # generic role cue can rescue a moderately similar sentence because
            # the article itself has already passed same-event alignment.
            if (
                anchor_sim < section["min_anchor"]
                and role_sim < 0.38
                and not (has_role_cue and anchor_sim >= 0.08)
            ):
                continue

            quality = _brief_quality_score(sentence)
            lexical = 0.16 if has_role_cue else 0.0
            if prefer_role_cues and not has_role_cue:
                lexical -= 0.38
            lead = section["lead_bonus"] if record["position"] <= 2 else 0.0
            history = _historical_penalty(sentence, event_year, section["name"])

            redundancy = 0.0
            if selected_indices:
                redundancy = max(float(vecs[i] @ vecs[j]) for j in selected_indices)
            redundancy_penalty = max(0.0, redundancy - 0.58) * 1.35
            source_penalty = 0.08 * source_use.get(record["source"], 0)

            score = (
                0.49 * anchor_sim
                + 0.34 * role_sim
                + 0.10 * quality
                + lexical
                + lead
                - history
                - redundancy_penalty
                - source_penalty
            )
            ranked.append((score, anchor_sim, role_sim, i))

        ranked.sort(reverse=True)
        if not ranked:
            continue

        picks = []
        best_score = ranked[0][0]

        for score, anchor_sim, role_sim, idx in ranked:
            if picks and score < best_score - 0.16:
                break

            # A second sentence should add information, not repeat the first.
            if picks and max(float(vecs[idx] @ vecs[j]) for j in picks) >= 0.76:
                continue

            # Prefer a second independent source when possible.
            if picks and records[idx]["source"] == records[picks[0]]["source"]:
                alternatives = [
                    item for item in ranked
                    if item[3] not in picks
                    and records[item[3]]["source"] != records[picks[0]]["source"]
                    and item[0] >= score - 0.05
                ]
                if alternatives:
                    continue

            picks.append(idx)
            if len(picks) >= int(section["max_sentences"]):
                break

        if not picks:
            continue

        sources = []
        for idx in picks:
            source = records[idx]["source"]
            if source not in sources:
                sources.append(source)
            selected_indices.append(idx)
            source_use[source] = source_use.get(source, 0) + 1

        joined = " ".join(records[idx]["sentence"] for idx in picks)

        selected_points.append(
            {
                "role": section["name"],
                "sentence": joined,
                "sources": sources,
                "source": sources[0] if sources else "Selected report",
                "article_id": records[picks[0]]["article_id"],
                "anchor_similarity": round(
                    float(np.mean([anchor_sims[idx] for idx in picks])), 3
                ),
                "role_similarity": round(
                    float(np.mean([role_sims[idx, section_idx] for idx in picks])), 3
                ),
            }
        )

    return {
        "points": selected_points,
        "num_sources": len({record["source"] for record in records}),
        "anchor_text": anchor_text,
    }




# Helpers retained unchanged for The Lenses.
def _article_sentences(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for article in articles:
        body = _remove_leading_article_title(
            article.get("text", ""),
            article.get("title", ""),
        )

        for sentence in split_sentences(body):
            records.append(
                {
                    "sentence": sentence,
                    "source": article["source"],
                    "article_id": article["id"],
                    "title": article.get("title", ""),
                }
            )

    return records


_SUBJECTIVE_CUES = re.compile(
    r"\b(seems?|clearly|obviously|apparently|arguably|brutally|"
    r"intrigue|misjudged|overwhelming response)\b",
    re.I,
)


def _sentence_informativeness(sentence: str) -> float:
    """Readability/context bonus used by the existing Lenses logic."""
    length = len(sentence)
    score = 0.0

    if 75 <= length <= 300:
        score += 1.0
    elif 45 <= length <= 380:
        score += 0.45
    else:
        score -= 0.4

    proper_phrases = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b",
        sentence,
    )
    score += min(0.7, 0.16 * len(proper_phrases))

    if re.search(r"\b\d+\b", sentence):
        score += 0.15

    if _ATTRIBUTION_START.search(sentence):
        score -= 0.30

    if _SUBJECTIVE_CUES.search(sentence):
        score -= 0.75

    if _METADATA_WORDS.search(sentence):
        score -= 1.0

    return score


# ---------------------------------------------------------------------------
# EVENT-SPECIFIC LENSES — "How is each article telling this event?"
# ---------------------------------------------------------------------------

FACET_CLUSTER_SIM = 0.49
FACET_QUERY_FLOOR = 0.15
MAX_EVENT_FACETS = 7
MAX_SENTENCES_PER_SOURCE = 12


def _select_relevant_records(
    query: str,
    articles: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Keep enough event-relevant sentences from EVERY selected article.

    Unlike the old fixed-theme threshold, this never turns a relevant selected
    article into an empty Lens merely because it failed a generic category.
    """
    all_records = _article_sentences(articles)

    if not all_records:
        return [], np.zeros((0, 384)), np.zeros((0,))

    all_sentences = [r["sentence"] for r in all_records]
    all_vecs = embed(all_sentences)
    query_vec = embed([query])[0]
    query_sims = all_vecs @ query_vec

    selected_indices = []

    by_source: Dict[str, List[int]] = {}
    for i, record in enumerate(all_records):
        by_source.setdefault(record["source"], []).append(i)

    for source, indices in by_source.items():
        ranked = sorted(
            indices,
            key=lambda i: float(query_sims[i]),
            reverse=True,
        )

        # Always retain a useful body of text from a source that already
        # passed event alignment. Prefer query-relevant sentences, then fill
        # from the source's strongest remaining sentences.
        kept = [
            i for i in ranked
            if query_sims[i] >= FACET_QUERY_FLOOR
        ][:MAX_SENTENCES_PER_SOURCE]

        minimum = min(6, len(ranked))
        if len(kept) < minimum:
            for i in ranked:
                if i not in kept:
                    kept.append(i)
                if len(kept) >= minimum:
                    break

        selected_indices.extend(kept[:MAX_SENTENCES_PER_SOURCE])

    selected_indices = sorted(set(selected_indices))

    records = [all_records[i] for i in selected_indices]
    vecs = all_vecs[selected_indices]
    sims = query_sims[selected_indices]

    return records, vecs, sims


def _cluster_event_sentences(
    records: List[Dict[str, Any]],
    vecs: np.ndarray,
) -> List[List[int]]:
    clusters: List[List[int]] = []

    for i in range(len(records)):
        best_cluster = -1
        best_score = 0.0

        for ci, members in enumerate(clusters):
            avg_sim = float(np.mean([vecs[i] @ vecs[m] for m in members]))

            if avg_sim >= FACET_CLUSTER_SIM and avg_sim > best_score:
                best_cluster = ci
                best_score = avg_sim

        if best_cluster >= 0:
            clusters[best_cluster].append(i)
        else:
            clusters.append([i])

    return clusters


def _facet_entity(sentences: List[str]) -> Optional[str]:
    from collections import Counter

    phrases = []
    blacklist = {
        "New Delhi",
        "Delhi Police",
        "The Indian",
        "The New",
        "Prime Minister",
    }

    for sentence in sentences:
        for phrase in re.findall(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
            sentence,
        ):
            if phrase not in blacklist:
                phrases.append(phrase)

    if not phrases:
        return None

    counts = Counter(phrases)
    phrase, count = counts.most_common(1)[0]

    # A named person/organisation is useful even if present once inside a
    # small distinctive facet (e.g. Sonam Wangchuk).
    return phrase if len(phrase) >= 5 else None


def _label_event_facets(
    cluster_sentences: List[List[str]],
) -> List[str]:
    """Create short event-specific labels using TF-IDF n-grams.

    No LLM and no fixed generic news taxonomy.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [" ".join(sentences) for sentences in cluster_sentences]

    if not docs:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=1800,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z'-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(docs)
        features = vectorizer.get_feature_names_out()
    except Exception:
        return [
            " / ".join(extract_keyphrases(sents, top_k=3)).title()
            or f"Event facet {i + 1}"
            for i, sents in enumerate(cluster_sentences)
        ]

    labels = []

    for row_i, sentences in enumerate(cluster_sentences):
        row = matrix.getrow(row_i)
        scored = []

        for idx, value in zip(row.indices, row.data):
            phrase = features[idx]
            words = phrase.split()

            # Prefer informative bigrams/trigrams over isolated generic words.
            bonus = 1.0 + 0.22 * (len(words) - 1)
            generic_penalty = 0.55 if phrase in {
                "protest", "protesters", "police", "people", "india",
                "delhi", "movement", "government", "students"
            } else 1.0
            scored.append((float(value) * bonus * generic_penalty, phrase))

        scored.sort(reverse=True)

        best_phrase = scored[0][1] if scored else ""
        entity = _facet_entity(sentences)

        # Avoid duplicated labels such as
        # "Prime Minister Narendra Modi · Narendra Modi's..."
        if entity and best_phrase:
            entity_tokens = set(re.findall(r"[a-z]+", entity.lower()))
            phrase_tokens = set(re.findall(r"[a-z]+", best_phrase.lower()))
            overlap = (
                len(entity_tokens & phrase_tokens) / max(1, len(entity_tokens))
            )

            noisy_phrase = re.search(
                r"\b(weeks? ago|days? ago|months? ago|monsoon session|"
                r"published|updated|said|says|according to|news conference|"
                r"tuesday|monday|wednesday|thursday|friday|saturday|sunday)\b",
                best_phrase,
                re.I,
            )

            if overlap >= 0.50 or noisy_phrase:
                label = entity
            else:
                label = f"{entity} · {best_phrase.title()}"
        elif entity:
            label = entity
        elif best_phrase:
            label = best_phrase.title()
        else:
            keys = extract_keyphrases(sentences, top_k=3)
            label = " / ".join(keys).title() if keys else f"Event facet {row_i + 1}"

        # Clean possessive fragments and keep the UI compact.
        label = re.sub(r"\b([A-Za-z]+)'s\b", r"\1", label)
        label = re.sub(
            r"\b(Published|Updated|Weeks? Ago|Days? Ago|Months? Ago)\b",
            "",
            label,
            flags=re.I,
        )
        label = re.sub(r"\s+", " ", label).strip(" ·-/")

        if len(label) > 50:
            label = label[:47].rsplit(" ", 1)[0] + "…"

        labels.append(label)

    return labels


def analyze_coverage_lenses(
    query: str,
    articles: List[Dict[str, Any]],
    max_facets: int = MAX_EVENT_FACETS,
) -> Dict[str, Any]:
    """Evidence-driven, event-specific cross-source coverage analysis.

    Output answers:
      - What event-specific facets appear in this selected coverage?
      - Which facets receive the most attention in each article?
      - What exact article sentences support that reading?
      - Which facets visible in other selected reports are comparatively less
        visible in this article?

    "Less visible" is a comparison within the selected articles only. It does
    not claim suppression, intent, or absence from the outlet's wider coverage.
    """
    records, vecs, query_sims = _select_relevant_records(query, articles)

    sources = [a["source"] for a in articles]

    if not records:
        return {
            "facets": [],
            "sources": sources,
            "matrix": {},
            "source_angles": {},
            "differences": [],
        }

    raw_clusters = _cluster_event_sentences(records, vecs)

    # Describe and score every cluster first.
    cluster_meta = []

    for cid, members in enumerate(raw_clusters):
        member_sources = sorted(set(records[i]["source"] for i in members))
        sentences = [records[i]["sentence"] for i in members]

        avg_q = float(np.mean([query_sims[i] for i in members]))
        support_ratio = len(member_sources) / max(1, len(set(sources)))
        size_score = min(1.0, len(members) / 5.0)

        score = (
            0.45 * max(0.0, avg_q)
            + 0.37 * support_ratio
            + 0.18 * size_score
        )

        cluster_meta.append(
            {
                "raw_id": cid,
                "members": members,
                "sentences": sentences,
                "supporting_sources": member_sources,
                "source_count": len(member_sources),
                "avg_query_similarity": avg_q,
                "score": score,
            }
        )

    labels = _label_event_facets(
        [c["sentences"] for c in cluster_meta]
    )

    for cluster, label in zip(cluster_meta, labels):
        cluster["label"] = label

    # Sentence counts per source for the selected relevant article material.
    source_totals = {
        source: sum(1 for r in records if r["source"] == source)
        for source in sources
    }

    for cluster in cluster_meta:
        counts = {
            source: sum(
                1
                for i in cluster["members"]
                if records[i]["source"] == source
            )
            for source in sources
        }

        cluster["source_counts"] = counts
        cluster["source_shares"] = {
            source: (
                counts[source] / source_totals[source]
                if source_totals.get(source, 0)
                else 0.0
            )
            for source in sources
        }

    # Ensure the final facet set includes globally important facets AND each
    # source's strongest local focus. This prevents any article from going blank.
    chosen_raw_ids = []

    for cluster in sorted(
        cluster_meta,
        key=lambda c: c["score"],
        reverse=True,
    ):
        if cluster["raw_id"] not in chosen_raw_ids:
            chosen_raw_ids.append(cluster["raw_id"])
        if len(chosen_raw_ids) >= max_facets:
            break

    for source in sources:
        local = sorted(
            cluster_meta,
            key=lambda c: (
                c["source_shares"].get(source, 0.0),
                c["score"],
            ),
            reverse=True,
        )
        if local and local[0]["source_shares"].get(source, 0.0) > 0:
            rid = local[0]["raw_id"]
            if rid not in chosen_raw_ids:
                chosen_raw_ids.append(rid)

    # Keep the display manageable; global + local coverage usually lands 5–8.
    chosen_clusters = [
        c for c in cluster_meta
        if c["raw_id"] in chosen_raw_ids
    ]
    chosen_clusters.sort(key=lambda c: c["score"], reverse=True)
    chosen_clusters = chosen_clusters[:max(max_facets, len(sources))]

    # Give stable compact IDs.
    raw_to_display = {}
    facets = []

    for display_id, cluster in enumerate(chosen_clusters):
        raw_to_display[cluster["raw_id"]] = display_id

        # Representative sentence = highest query relevance with a small
        # readability preference.
        best_member = max(
            cluster["members"],
            key=lambda i: (
                float(query_sims[i])
                + 0.05 * _sentence_informativeness(records[i]["sentence"])
            ),
        )

        evidence_by_source = {}

        for source in sources:
            source_members = [
                i for i in cluster["members"]
                if records[i]["source"] == source
            ]

            ranked = sorted(
                source_members,
                key=lambda i: (
                    float(query_sims[i])
                    + 0.04 * _sentence_informativeness(records[i]["sentence"])
                ),
                reverse=True,
            )

            evidence_by_source[source] = [
                {
                    "sentence": records[i]["sentence"],
                    "query_similarity": round(float(query_sims[i]), 3),
                }
                for i in ranked[:2]
            ]

        facets.append(
            {
                "id": display_id,
                "label": cluster["label"],
                "representative_sentence": records[best_member]["sentence"],
                "supporting_sources": cluster["supporting_sources"],
                "source_count": cluster["source_count"],
                "source_shares": cluster["source_shares"],
                "evidence_by_source": evidence_by_source,
            }
        )

    # Matrix is now event-specific facet share, not generic theme classification.
    matrix = {
        source: {
            facet["label"]: round(facet["source_shares"].get(source, 0.0), 4)
            for facet in facets
        }
        for source in sources
    }

    source_angles = {}

    for source in sources:
        ranked_facets = sorted(
            facets,
            key=lambda f: f["source_shares"].get(source, 0.0),
            reverse=True,
        )

        nonzero = [
            f for f in ranked_facets
            if f["source_shares"].get(source, 0.0) > 0
        ]

        # Absolute focus within the article.
        main = nonzero[0] if nonzero else None
        secondary = nonzero[1] if len(nonzero) > 1 else None

        # Comparative standout: what is more prominent here than elsewhere.
        standout = None
        standout_diff = 0.0

        for facet in facets:
            own = facet["source_shares"].get(source, 0.0)
            other_vals = [
                facet["source_shares"].get(other, 0.0)
                for other in sources
                if other != source
            ]
            other_mean = float(np.mean(other_vals)) if other_vals else 0.0
            diff = own - other_mean

            if own > 0 and diff > standout_diff:
                standout = facet
                standout_diff = diff

        less_visible = []

        for facet in facets:
            own = facet["source_shares"].get(source, 0.0)
            other_sources = [
                other
                for other in sources
                if other != source
                and facet["source_shares"].get(other, 0.0) >= 0.08
            ]

            if own <= 0.02 and len(other_sources) >= 2:
                less_visible.append(
                    {
                        "label": facet["label"],
                        "sources": other_sources,
                        "representative_sentence": facet["representative_sentence"],
                    }
                )

        less_visible = less_visible[:2]

        main_evidence = (
            main["evidence_by_source"].get(source, [])[:2]
            if main else []
        )
        secondary_evidence = (
            secondary["evidence_by_source"].get(source, [])[:1]
            if secondary else []
        )

        # Fallback evidence from the article's strongest event-relevant
        # sentences, so no selected source ever produces an empty Lens.
        if not main_evidence:
            source_indices = [
                i for i, record in enumerate(records)
                if record["source"] == source
            ]
            ranked_indices = sorted(
                source_indices,
                key=lambda i: float(query_sims[i]),
                reverse=True,
            )
            main_evidence = [
                {
                    "sentence": records[i]["sentence"],
                    "query_similarity": round(float(query_sims[i]), 3),
                }
                for i in ranked_indices[:2]
            ]

        source_angles[source] = {
            "main_facet": main["label"] if main else "Event coverage",
            "main_share": (
                round(main["source_shares"].get(source, 0.0), 4)
                if main else 0.0
            ),
            "secondary_facet": secondary["label"] if secondary else None,
            "secondary_share": (
                round(secondary["source_shares"].get(source, 0.0), 4)
                if secondary else 0.0
            ),
            "standout_facet": (
                standout["label"]
                if standout and standout_diff >= 0.05
                else None
            ),
            "standout_difference": round(standout_diff, 4),
            "evidence": main_evidence,
            "secondary_evidence": secondary_evidence,
            "less_visible": less_visible,
            "selected_sentence_count": source_totals.get(source, 0),
        }

    # Cross-source differences worth surfacing above the cards.
    differences = []

    for facet in facets:
        shares = facet["source_shares"]
        values = list(shares.values())

        if not values:
            continue

        spread = max(values) - min(values)
        if spread < 0.08:
            continue

        strongest = sorted(
            shares.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        high = [
            source for source, value in strongest
            if value >= max(0.08, strongest[0][1] * 0.60)
        ]
        absent = [
            source for source, value in shares.items()
            if value <= 0.02
        ]

        differences.append(
            {
                "label": facet["label"],
                "most_visible_in": high[:3],
                "less_visible_in": absent[:3],
                "spread": round(spread, 4),
            }
        )

    differences.sort(key=lambda d: d["spread"], reverse=True)

    return {
        "facets": facets,
        "sources": sources,
        "matrix": matrix,
        "source_angles": source_angles,
        "differences": differences[:5],
        "selected_sentence_counts": source_totals,
    }


# ---------------------------------------------------------------------------
# THE LENSES
# ---------------------------------------------------------------------------

THEMES: Dict[str, str] = {
    "Economic":
        "Economic impact, trade, tariffs, markets, industries, jobs, prices, and financial consequences.",
    "Political":
        "Domestic political reaction, party positions, government leadership, and policy debate.",
    "Diplomatic / International":
        "Diplomacy, foreign relations, negotiations between governments, treaties, and international meetings.",
    "Security":
        "National security, military action, defense, conflict, threats, and law enforcement response.",
    "Public / Social Impact":
        "Effects on ordinary people, communities, public opinion, protests, and daily life.",
    "Historical / Context":
        "Historical background, prior events, and long-term context leading up to this development.",
}

_THEME_EMB = None


def get_theme_embeddings() -> Dict[str, np.ndarray]:
    global _THEME_EMB

    if _THEME_EMB is None:
        names = list(THEMES.keys())
        vecs = embed(list(THEMES.values()))
        _THEME_EMB = {
            name: vecs[i]
            for i, name in enumerate(names)
        }

    return _THEME_EMB


def compute_theme_matrix(
    articles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return theme shares PLUS extractive evidence for each source/theme.

    The evidence sentences are not generated or paraphrased. They are the
    article sentences that received the strongest MiniLM similarity to a
    theme after clearing THEME_THRESHOLD.
    """
    theme_emb = get_theme_embeddings()
    theme_names = list(theme_emb.keys())
    theme_matrix_vecs = np.stack(
        [theme_emb[theme] for theme in theme_names]
    )

    per_source_counts: Dict[str, Dict[str, int]] = {}

    # source -> theme -> [(similarity, sentence), ...]
    evidence_candidates: Dict[
        str, Dict[str, List[Tuple[float, str]]]
    ] = {}

    for article in articles:
        source = article["source"]

        # Avoid letting a duplicated scraped headline become "evidence".
        article_text = _remove_leading_article_title(
            article.get("text", ""),
            article.get("title", ""),
        )
        sentences = split_sentences(article_text)

        if not sentences:
            # Keep the source visible in the output even when extraction
            # produced no usable sentences.
            per_source_counts.setdefault(
                source,
                {theme: 0 for theme in theme_names},
            )
            evidence_candidates.setdefault(
                source,
                {theme: [] for theme in theme_names},
            )
            continue

        vecs = embed(sentences)
        sims = cosine_sim_matrix(vecs, theme_matrix_vecs)

        best_theme_idx = np.argmax(sims, axis=1)
        best_theme_sim = np.max(sims, axis=1)

        counts = per_source_counts.setdefault(
            source,
            {theme: 0 for theme in theme_names},
        )
        source_evidence = evidence_candidates.setdefault(
            source,
            {theme: [] for theme in theme_names},
        )

        for sentence, idx, sim in zip(
            sentences,
            best_theme_idx,
            best_theme_sim,
        ):
            if sim >= THEME_THRESHOLD:
                theme = theme_names[int(idx)]
                score = float(sim)

                counts[theme] += 1
                source_evidence[theme].append(
                    (score, sentence)
                )

    matrix: Dict[str, Dict[str, float]] = {}
    evidence: Dict[
        str, Dict[str, List[Dict[str, Any]]]
    ] = {}

    for source, counts in per_source_counts.items():
        total = sum(counts.values())

        if total == 0:
            matrix[source] = {
                theme: 0.0 for theme in theme_names
            }
        else:
            matrix[source] = {
                theme: counts[theme] / total
                for theme in theme_names
            }

        evidence[source] = {}

        for theme in theme_names:
            ranked = sorted(
                evidence_candidates
                .get(source, {})
                .get(theme, []),
                key=lambda item: item[0],
                reverse=True,
            )

            # Keep up to two strong, non-duplicate article sentences.
            picked: List[Dict[str, Any]] = []
            seen = set()

            for score, sentence in ranked:
                normalized = re.sub(
                    r"\W+",
                    " ",
                    sentence.lower(),
                ).strip()

                # Basic near-duplicate guard for repeated page text.
                key = normalized[:180]
                if not key or key in seen:
                    continue

                seen.add(key)
                picked.append(
                    {
                        "sentence": sentence,
                        "similarity": round(score, 3),
                    }
                )

                if len(picked) >= 2:
                    break

            evidence[source][theme] = picked

    return {
        "themes": theme_names,
        "matrix": matrix,
        "counts": per_source_counts,

        # New additive field. Existing pipeline/app consumers remain valid.
        "evidence": evidence,
    }


THEME_PLAIN_LANGUAGE: Dict[str, str] = {
    "Economic": "trade, prices, jobs, business and money",
    "Political": "politicians, parties and government decisions",
    "Diplomatic / International": "relations and negotiations between countries",
    "Security": "police, law enforcement, conflict and public safety",
    "Public / Social Impact": "protesters, communities and effects on ordinary people",
    "Historical / Context": "background information and earlier events",
}


def _theme_explanation(theme: str) -> str:
    return THEME_PLAIN_LANGUAGE.get(theme, theme.lower())


def build_theme_paragraph(
    source: str,
    shares: Dict[str, float],
) -> str:
    """Explain a source's theme profile in normal current-affairs language."""
    ranked = sorted(
        shares.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    nonzero = [
        (theme, share)
        for theme, share in ranked
        if share > 0
    ]

    if not nonzero:
        return (
            f"NarrativeLens could not confidently map enough of {source}'s "
            "sentences to the six fixed themes. Treat this as an unclear "
            "theme result, not as evidence that the article had no focus."
        )

    top_theme, top_share = nonzero[0]
    top_pct = round(top_share * 100)
    top_plain = _theme_explanation(top_theme)

    if len(nonzero) == 1:
        return (
            f"Among the sentences NarrativeLens could confidently classify, "
            f"{source} focuses almost entirely on {top_theme} ({top_pct}%). "
            f"In plain terms, the classifiable part of the article is mainly "
            f"about {top_plain}."
        )

    second_theme, second_share = nonzero[1]
    second_pct = round(second_share * 100)
    second_plain = _theme_explanation(second_theme)

    if abs(top_share - second_share) <= 0.05:
        opening = (
            f"{source} gives almost equal attention to {top_theme} "
            f"({top_pct}%) and {second_theme} ({second_pct}%). "
            f"In plain terms, the article discusses both {top_plain} and "
            f"{second_plain} in roughly equal measure."
        )
    else:
        opening = (
            f"{source} focuses most on {top_theme} ({top_pct}%), especially "
            f"{top_plain}. Its second strongest theme is {second_theme} "
            f"({second_pct}%), covering {second_plain}."
        )

    rest_share = max(0.0, 1.0 - top_share - second_share)

    if rest_share > 0.01:
        closing = (
            f"The remaining {round(rest_share * 100)}% is spread across the "
            "other tracked themes."
        )
    else:
        closing = (
            "Almost all of the classifiable thematic content falls into "
            "these two themes."
        )

    return opening + " " + closing
