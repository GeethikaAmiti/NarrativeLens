"""
NarrativeLens pipeline.

Selection:
retrieval -> temporal filtering -> query relevance -> same-event cohesion
-> source diversity -> analysis.

Analysis:
- Event brief: 5–6 evidence-grounded current-affairs points
- The Core: repeated cross-source claim clusters
- The Lenses: event-specific coverage facets + per-source evidence
- Visual Lens: image similarity + visible-content descriptors
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import alignment, image_analysis, news_retrieval, text_analysis, utils


class PipelineError(Exception):
    """Raised for graceful user-facing pipeline failures."""


def run_pipeline(
    query: str,
    time_mode: str,
    specific_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        candidates = news_retrieval.retrieve_candidates(query)
    except Exception as e:
        raise PipelineError(
            "Live news retrieval failed due to a network or parsing error. "
            "Please try again."
        ) from e

    if not candidates:
        raise PipelineError(
            "No usable news coverage could be retrieved for this query. "
            "Try a different phrasing or a broader time window."
        )

    windowed = alignment.filter_by_time_window(
        candidates,
        time_mode,
        specific_date,
        start_date,
        end_date,
    )

    relevant = alignment.rank_by_relevance(windowed, query)

    if not relevant:
        raise PipelineError(
            "No retrieved coverage was sufficiently relevant to this query "
            "inside the selected time window. Try a broader date range or "
            "a more specific event description."
        )

    ok, message, selected = alignment.check_event_cohesion(relevant)

    if not ok:
        raise PipelineError(message)

    if image_dir:
        utils.ensure_dir(image_dir)

        for article in selected:
            if article.get("image_url"):
                dest = os.path.join(
                    image_dir,
                    f"{article['id']}.jpg",
                )

                if news_retrieval.download_image(
                    article["image_url"],
                    dest,
                ):
                    article["image_path"] = dest

    return _analyze(query, selected)


def run_pipeline_on_articles(
    query: str,
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return _analyze(query, articles)


def _analyze(
    query: str,
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_sources = len(
        utils.unique_sources(articles)
    )

    # 1) Reader-facing situation brief.
    event_brief = text_analysis.build_event_brief(
        query,
        articles,
        max_points=6,
    )

    # 2) Repeated claims still power the strict consensus part of The Core.
    core_clusters = text_analysis.find_consensus_clusters(
        articles,
        total_sources,
    )

    for cluster in core_clusters:
        cluster["description"] = (
            text_analysis.build_cluster_description(
                cluster
            )
        )

    # 3) Primary Lenses analysis is now event-specific and evidence-driven.
    lens_result = text_analysis.analyze_coverage_lenses(
        query,
        articles,
    )

    # The old broad-theme functions remain in text_analysis.py as a simple
    # secondary/experimental method, but are no longer the primary user Lens.
    # This avoids generic-category thresholds blanking out a relevant article.

    visual_result = image_analysis.analyze_visual_framing(
        articles
    )
    visual_summary = image_analysis.build_visual_summary(
        visual_result,
        total_sources,
    )

    scope_paragraph = (
        f"NarrativeLens identified {len(articles)} reports from "
        f"{total_sources} distinct sources, published "
        f"{utils.fmt_time_range(articles)}, that describe a comparable "
        f"development related to “{query}”. Clearly separate developments "
        f"were excluded during temporal and semantic event alignment."
    )

    return {
        "query": query,
        "generated_at": utils.now_iso(),
        "num_sources": total_sources,
        "num_articles": len(articles),
        "time_range": utils.fmt_time_range(articles),
        "scope_paragraph": scope_paragraph,

        "event_brief": event_brief,
        "core_clusters": core_clusters,
        "lens_result": lens_result,

        "visual_result": visual_result,
        "visual_summary": visual_summary,
        "articles": articles,
    }
