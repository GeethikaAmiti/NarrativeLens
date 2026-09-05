"""
NarrativeLens — polished presentation UI v4.

This file changes presentation and fixes one live-path issue:
live analyses now provide an image_dir to pipeline.run_pipeline(), so article
images are downloaded locally and MobileNetV2 can actually compute the Visual
Lens instead of showing "insufficient images" while remote images are visible.
"""

import hashlib
import html
import sys
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import pipeline  # noqa: E402

st.set_page_config(
    page_title="NarrativeLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@500&family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

:root{
  --ink:#101A32;
  --muted:#68758D;
  --line:#DCE5F2;
  --paper:#F4F7FC;
  --card:#FFFFFF;
  --blue:#315CF4;
  --violet:#7457F6;
  --cyan:#0E7490;
  --cyan-soft:#E8F8FC;
  --orange:#C96B08;
  --orange-soft:#FFF5E8;
  --navy:#0B1731;
}

html, body, [class*="css"], .stApp {
    font-family:'Manrope',system-ui,sans-serif !important;
}

.stApp{
    color:var(--ink) !important;
    background:
      radial-gradient(circle at 12% 8%, rgba(84,115,255,.13), transparent 21rem),
      radial-gradient(circle at 90% 13%, rgba(119,78,246,.11), transparent 23rem),
      radial-gradient(circle at 50% 92%, rgba(14,116,144,.07), transparent 26rem),
      linear-gradient(180deg,#F9FBFF 0%,#F3F7FC 100%) !important;
}

[data-testid="stHeader"]{
    background:rgba(249,251,255,.84) !important;
    border-bottom:1px solid rgba(220,229,242,.72);
    backdrop-filter:blur(10px);
}
[data-testid="stToolbar"], #MainMenu, footer {visibility:hidden !important;}

.block-container{
    max-width:1180px;
    padding-top:4.9rem !important;
    padding-bottom:4.5rem !important;
}

h1,h2,h3,h4,h5,h6,p,label,.stMarkdown,.stCaption{
    color:var(--ink) !important;
}

/* ---------- Hero ---------- */
.nl-hero{
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:2.7rem 2.8rem 2.45rem;
    background:
      radial-gradient(circle at 82% 20%, rgba(98,112,255,.36), transparent 19rem),
      radial-gradient(circle at 100% 100%, rgba(27,160,178,.16), transparent 18rem),
      linear-gradient(135deg,#09152E 0%,#142A52 54%,#193B79 100%);
    border:1px solid rgba(97,124,190,.48);
    box-shadow:0 28px 70px rgba(18,39,82,.22);
}
.nl-hero:after{
    content:"";
    position:absolute;
    right:-95px;
    top:-105px;
    width:330px;
    height:330px;
    border:1px solid rgba(255,255,255,.14);
    border-radius:50%;
    box-shadow:
      0 0 0 42px rgba(255,255,255,.035),
      0 0 0 84px rgba(255,255,255,.02);
}
.nl-kicker{
    font-family:'DM Mono',monospace;
    color:#9DB6FF;
    text-transform:uppercase;
    letter-spacing:.14em;
    font-size:.72rem;
    position:relative;
    z-index:2;
}
.nl-title{
    font-family:'Playfair Display',serif;
    font-size:4rem;
    line-height:.98;
    letter-spacing:-.05em;
    color:#FFFFFF;
    margin:.45rem 0 0;
    position:relative;
    z-index:2;
}
.nl-tagline{
    color:#BFD0F7;
    font-size:1.12rem;
    font-weight:700;
    margin:.75rem 0 .85rem;
    position:relative;
    z-index:2;
}
.nl-hero-copy{
    max-width:760px;
    color:#E3EAFF;
    font-size:1rem;
    line-height:1.72;
    position:relative;
    z-index:2;
}
.nl-lens-strip{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.8rem;
    margin-top:1.6rem;
    position:relative;
    z-index:2;
}
.nl-lens{
    background:rgba(255,255,255,.075);
    border:1px solid rgba(255,255,255,.13);
    border-radius:16px;
    padding:.9rem .95rem;
    backdrop-filter:blur(6px);
}
.nl-lens-num{
    font-family:'DM Mono',monospace;
    color:#92ADFF;
    font-size:.7rem;
    letter-spacing:.1em;
}
.nl-lens-title{
    color:#FFFFFF;
    font-weight:800;
    font-size:.89rem;
    margin:.23rem 0 .2rem;
}
.nl-lens-copy{
    color:#C7D5F5;
    font-size:.77rem;
    line-height:1.45;
}

/* ---------- Typography hierarchy ---------- */
.nl-eyebrow{
    font-family:'DM Mono',monospace;
    color:var(--blue);
    letter-spacing:.11em;
    text-transform:uppercase;
    font-size:.69rem;
    font-weight:500;
    margin-bottom:.35rem;
}
.nl-section-title{
    font-family:'Playfair Display',serif;
    color:var(--ink);
    font-size:2.05rem;
    line-height:1.1;
    letter-spacing:-.035em;
    margin:0 0 .25rem;
}
.nl-section-sub{
    color:var(--muted);
    font-size:.95rem;
    line-height:1.65;
    margin-bottom:1.05rem;
}
.nl-analysis-title{
    font-family:'Playfair Display',serif;
    font-size:2.7rem;
    line-height:1.05;
    letter-spacing:-.04em;
    margin:0;
    color:var(--ink);
}

/* ---------- Cards ---------- */
.nl-card{
    background:rgba(255,255,255,.96);
    border:1px solid var(--line);
    border-radius:18px;
    padding:1.15rem 1.25rem;
    box-shadow:0 10px 30px rgba(31,58,112,.055);
    margin-bottom:.95rem;
}
.nl-metric{
    background:rgba(255,255,255,.96);
    border:1px solid var(--line);
    border-radius:18px;
    padding:1rem 1.05rem;
    min-height:108px;
    box-shadow:0 9px 24px rgba(31,58,112,.05);
    position:relative;
    overflow:hidden;
}
.nl-metric:before{
    content:"";
    position:absolute;
    left:0; right:0; top:0;
    height:3px;
    background:linear-gradient(90deg,var(--blue),var(--violet));
}
.nl-metric-value{
    font-family:'Playfair Display',serif;
    font-size:2rem;
    font-weight:800;
    color:var(--ink);
}
.nl-metric-label{
    color:var(--muted);
    font-size:.69rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.07em;
    margin-top:.28rem;
}
.nl-scope{
    background:linear-gradient(135deg,#EEF4FF,#F4F1FF);
    border:1px solid #CFDCF7;
    border-left:4px solid var(--blue);
    border-radius:14px;
    padding:1rem 1.1rem;
    color:#34445E;
    line-height:1.65;
    margin:1.2rem 0 1.45rem;
}
.nl-badge{
    display:inline-block;
    border-radius:999px;
    background:#EAF0FF;
    color:#2451C3;
    padding:.26rem .62rem;
    font-size:.72rem;
    font-weight:800;
    margin-right:.35rem;
}
.nl-badge-neutral{background:#F0F3F8;color:#536174;}
.nl-source{
    display:inline-block;
    border-radius:999px;
    background:#EFF3F8;
    color:#46556A;
    padding:.25rem .58rem;
    font-size:.72rem;
    font-weight:700;
    margin:.15rem .2rem .05rem 0;
}
.nl-label{
    font-family:'DM Mono',monospace;
    color:#75829A;
    text-transform:uppercase;
    letter-spacing:.09em;
    font-size:.67rem;
    margin-top:.85rem;
}
.nl-claim{
    margin-top:.65rem;
    background:#F7F9FC;
    border:1px solid #DDE5EF;
    border-radius:13px;
    padding:1rem 1.05rem;
    color:#1E293B;
    font-size:.97rem;
    line-height:1.68;
}
.nl-info{
    background:var(--cyan-soft);
    border:1px solid #BDE7F0;
    border-left:4px solid #1591A7;
    border-radius:13px;
    padding:.95rem 1.05rem;
    color:#285E68;
    line-height:1.65;
}
.nl-note{
    background:var(--orange-soft);
    border:1px solid #F0D09A;
    border-left:4px solid var(--orange);
    border-radius:13px;
    padding:.95rem 1.05rem;
    color:#744706;
    line-height:1.65;
}

/* ---------- Native Streamlit ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:rgba(255,255,255,.92);
    border:1px solid var(--line) !important;
    border-radius:20px !important;
    box-shadow:0 12px 35px rgba(31,58,112,.055);
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div{
    background:#FFFFFF !important;
    border-color:#D4DEEB !important;
    color:var(--ink) !important;
}
input{
    color:var(--ink) !important;
    -webkit-text-fill-color:var(--ink) !important;
}
.stButton > button{
    border-radius:11px !important;
    min-height:42px;
    font-weight:800 !important;
}
.stButton > button[kind="primary"]{
    background:linear-gradient(135deg,var(--blue),var(--violet)) !important;
    border:none !important;
    color:#FFFFFF !important;
    box-shadow:0 9px 22px rgba(49,92,244,.22);
}

/* FULL-WIDTH tabs */
.stTabs [data-baseweb="tab-list"]{
    display:flex !important;
    width:100% !important;
    gap:0 !important;
    border:1px solid var(--line);
    border-radius:15px;
    background:rgba(255,255,255,.78);
    padding:.3rem;
    margin-bottom:1.2rem;
}
.stTabs [data-baseweb="tab"]{
    flex:1 1 0 !important;
    justify-content:center !important;
    font-family:'DM Mono',monospace !important;
    font-size:.73rem !important;
    letter-spacing:.04em;
    color:#66758C !important;
    border-radius:11px !important;
    min-height:46px;
}
.stTabs [aria-selected="true"]{
    color:#FFFFFF !important;
    background:linear-gradient(135deg,#294FCF,#6847E8) !important;
}
.stTabs [data-baseweb="tab-highlight"]{display:none !important;}

@media (max-width:850px){
    .nl-title{font-size:3.2rem;}
    .nl-lens-strip{grid-template-columns:1fr;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "view" not in st.session_state:
    st.session_state.view = "landing"
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None


def go_to_analysis(result: dict):
    st.session_state.result = result
    st.session_state.error = None
    st.session_state.view = "analysis"


def go_home():
    st.session_state.view = "landing"
    st.session_state.result = None
    st.session_state.error = None


def _live_image_dir(query, time_mode, specific_date, start_date, end_date):
    raw = f"{query}|{time_mode}|{specific_date}|{start_date}|{end_date}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    p = Path("data") / "live_images" / key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def render_landing():
    st.markdown(
        """
        <div class="nl-hero">
          <div class="nl-kicker">CROSS-SOURCE NEWS INTELLIGENCE</div>
          <div class="nl-title">NarrativeLens</div>
          <div class="nl-tagline">One event. Many lenses.</div>
          <div class="nl-hero-copy">
            News can describe the same development while emphasizing very different things.
            NarrativeLens aligns independent reports from a comparable time window, surfaces
            recurring claims, maps thematic emphasis, and compares the visual choices used
            to represent the story.
          </div>

          <div class="nl-lens-strip">
            <div class="nl-lens">
              <div class="nl-lens-num">01</div>
              <div class="nl-lens-title">THE CORE</div>
              <div class="nl-lens-copy">What multiple independent sources repeatedly report.</div>
            </div>
            <div class="nl-lens">
              <div class="nl-lens-num">02</div>
              <div class="nl-lens-title">THE LENSES</div>
              <div class="nl-lens-copy">How attention shifts across economic, diplomatic and other themes.</div>
            </div>
            <div class="nl-lens">
              <div class="nl-lens-num">03</div>
              <div class="nl-lens-title">VISUAL LENS</div>
              <div class="nl-lens-copy">How similar or distinct the article imagery is across outlets.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="nl-eyebrow">START AN ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="nl-section-title">Compare a specific development</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nl-section-sub">Use a concrete event or development rather than an extremely broad topic. '
        'The system will retrieve independent coverage and refuse weak comparisons when too few comparable sources survive.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        query = st.text_input(
            "Event or development",
            placeholder="e.g. India–US trade negotiations after the latest tariff announcement",
            help="Specific wording improves same-event alignment.",
        )

        c1, c2 = st.columns([1, 1.15])
        with c1:
            display_mode = st.selectbox(
                "Time window",
                [
                    "Past 30 days from today",
                    "Today",
                    "Past 7 days",
                    "Specific date",
                    "Custom date range",
                ],
            )

        mode_map = {
            "Past 30 days from today": "Latest Coverage",
            "Today": "Today",
            "Past 7 days": "Past 7 Days",
            "Specific date": "Specific Date",
            "Custom date range": "Custom Date Range",
        }
        time_mode = mode_map[display_mode]

        specific_date = start_date = end_date = None
        with c2:
            if time_mode == "Specific Date":
                specific_date = st.date_input("Date", value=date.today()).isoformat()
            elif time_mode == "Custom Date Range":
                d1, d2 = st.columns(2)
                start_date = d1.date_input("Start", value=date.today()).isoformat()
                end_date = d2.date_input("End", value=date.today()).isoformat()
            else:
                st.caption("Only reports inside this publication window are eligible for alignment.")

        if st.button("Analyze coverage →", type="primary"):
            if not query.strip():
                st.warning("Enter a specific event or development first.")
            else:
                image_dir = _live_image_dir(
                    query.strip(), time_mode, specific_date, start_date, end_date
                )
                with st.spinner("Retrieving coverage, aligning the event, and running multimodal analysis..."):
                    try:
                        result = pipeline.run_pipeline(
                            query=query.strip(),
                            time_mode=time_mode,
                            specific_date=specific_date,
                            start_date=start_date,
                            end_date=end_date,
                            image_dir=image_dir,   # critical live Visual Lens fix
                        )
                        result["title"] = query.strip()
                        go_to_analysis(result)
                        st.rerun()
                    except pipeline.PipelineError as e:
                        st.session_state.error = str(e)
                    except Exception:
                        st.session_state.error = (
                            "The live analysis could not be completed. Try a more specific event wording "
                            "or a slightly broader time window."
                        )

    if st.session_state.error:
        st.write("")
        st.error(st.session_state.error)

    st.markdown(
        '<div style="height:1.2rem;"></div>'
        '<div class="nl-card" style="background:linear-gradient(135deg,#F0F5FF,#F6F2FF);">'
        '<div class="nl-eyebrow">DESIGNED FOR COMPARISON, NOT VERDICTS</div>'
        '<div style="font-weight:800;margin-bottom:.35rem;">NarrativeLens does not decide which outlet is “right.”</div>'
        '<div style="color:#5E6C84;line-height:1.65;">It makes cross-source differences visible: what recurs, '
        'what receives more attention, and how imagery converges or diverges. Truth scoring, ideology labels '
        'and editorial-intent claims are deliberately outside the MVP.</div></div>',
        unsafe_allow_html=True,
    )


def visual_metric(result):
    clusters = result.get("visual_result", {}).get("clusters", [])
    return len(clusters) if clusters else "—"


def render_header(result):
    if st.button("← New search"):
        go_home()
        st.rerun()

    st.markdown('<div class="nl-eyebrow">ALIGNED EVENT COMPARISON</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="nl-analysis-title">{result.get("title","Event")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="nl-section-sub" style="margin-top:.55rem;">'
        f'{result["num_sources"]} independent sources · {result["time_range"]}</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    metrics = [
        (result["num_sources"], "Sources compared"),
        (len(result.get("core_clusters", [])), "Shared claim clusters"),
        (len(result.get("lens_result", {}).get("facets", [])), "Coverage facets"),
        (visual_metric(result), "Visual groups"),
    ]
    for c, (v, label) in zip(cols, metrics):
        c.markdown(
            f'<div class="nl-metric"><div class="nl-metric-value">{v}</div>'
            f'<div class="nl-metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="nl-scope"><b>Comparison scope.</b> '
        + result.get("scope_paragraph", "")
        + '</div>',
        unsafe_allow_html=True,
    )


def _short_core_quote(text, max_chars=320):
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:-") + "…"


def _display_source_name(source):
    aliases = {
        "global.chinadaily.com.cn": "China Daily Global Edition",
        "chinadaily.com.cn": "China Daily Global Edition",
        "aljazeera.com": "Al Jazeera",
    }
    return aliases.get(source, source)


def _source_list_plain(sources):
    sources = [_display_source_name(s) for s in (sources or [])]
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0]
    if len(sources) == 2:
        return f"{sources[0]} and {sources[1]}"
    return ", ".join(sources[:-1]) + f", and {sources[-1]}"


def render_core(result):
    st.markdown(
        '<div class="nl-eyebrow">01 · UNDERSTAND THE EVENT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nl-section-title">The Core</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nl-section-sub">'
        'A concise incident overview, followed by developments independently '
        'reported across the selected sources.'
        '</div>',
        unsafe_allow_html=True,
    )

    brief = result.get("event_brief", {})
    points = brief.get("points", [])

    if points:
        paragraphs = []

        for point in points:
            heading = point.get("role", "Event detail")
            paragraph_text = _short_core_quote(
                point.get("sentence", ""),
                max_chars=620,
            ).strip()

            raw_sources = point.get("sources") or [
                point.get("source", "Selected report")
            ]
            sources = []
            for source in raw_sources:
                label = _display_source_name(source)
                if label and label not in sources:
                    sources.append(label)
            source_text = " · ".join(sources)

            paragraphs.append(
                '<div style="margin-top:1rem;">'
                f'<div class="nl-label" style="margin-top:0;color:#315CF4;">'
                f'{html.escape(heading)}</div>'
                f'<div style="margin-top:.32rem;color:#25324A;line-height:1.72;'
                f'font-size:1rem;">{html.escape(paragraph_text, quote=False)}</div>'
                f'<div style="margin-top:.38rem;color:#8A95A6;font-size:.72rem;'
                f'font-weight:700;">Evidence: {html.escape(source_text)}</div>'
                '</div>'
            )

        st.markdown(
            '<div class="nl-card" style="padding:1.15rem 1.35rem 1.35rem;">'
            '<div style="font-family:Playfair Display,serif;font-size:1.52rem;'
            'font-weight:800;color:#14213D;">Incident brief</div>'
            '<div style="margin-top:.25rem;color:#6B778C;font-size:.88rem;line-height:1.55;">'
            'An event-anchored extractive brief built only from the selected reports. '
            'Older examples are kept out of current-event sections and may appear only as context.'
            '</div>'
            f'<div style="margin-top:.25rem;">{"".join(paragraphs)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "The selected articles did not contain enough clean text to "
            "build an incident brief."
        )

    clusters = result.get("core_clusters", [])

    st.markdown(
        '<div class="nl-label" style="margin-top:1.25rem;">CORROBORATED ACROSS SOURCES</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#6B778C;font-size:.86rem;line-height:1.5;margin:.2rem 0 .75rem;">'
        'Only claim clusters supported by at least three independent selected sources are shown.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not clusters:
        st.info(
            "No single statement cleared the strict three-source consensus threshold."
        )
        return

    for cluster in clusters[:3]:
        sources = cluster.get("supporting_sources", [])
        tags = "".join(
            f'<span class="nl-source">{html.escape(_display_source_name(source))}</span>'
            for source in sources
        )

        shared_statement = html.escape(
            _short_core_quote(cluster.get("representative_sentence", ""), max_chars=330),
            quote=False,
        )

        st.markdown(
            '<div class="nl-card" style="padding:.95rem 1.05rem;">'
            f'<span class="nl-badge">reported by '
            f'{cluster["support_count"]} of {cluster["total_sources"]} sources</span>'
            f'<span class="nl-badge nl-badge-neutral">'
            f'similarity {cluster["confidence"]:.2f}</span>'
            f'<div style="margin-top:.62rem;color:#25324A;line-height:1.58;'
            f'font-size:.94rem;font-weight:600;">{shared_statement}</div>'
            f'<div style="margin-top:.62rem;">{tags}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Cosine similarity reflects semantic agreement between statements; "
        "it is not a truth or reliability score."
    )


def _evidence_excerpt(sentence, max_chars=350):
    sentence = " ".join((sentence or "").split())
    if len(sentence) <= max_chars:
        return sentence
    clipped = sentence[:max_chars].rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:-") + "…"


def _lens_source_list(sources):
    sources = [_display_source_name(s) for s in (sources or [])]
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0]
    if len(sources) == 2:
        return f"{sources[0]} and {sources[1]}"
    return ", ".join(sources[:-1]) + f", and {sources[-1]}"


def render_lenses(result: dict):
    st.markdown(
        '<div class="nl-eyebrow">02 · COMPARE THE COVERAGE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nl-section-title">The Lenses</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nl-section-sub">'
        'See which parts of this specific event each article foregrounds, '
        'the exact article evidence behind that reading, and which facets '
        'are comparatively less visible in that report.'
        '</div>',
        unsafe_allow_html=True,
    )

    lens = result.get(
        "lens_result",
        {},
    )
    facets = lens.get(
        "facets",
        [],
    )
    sources = lens.get(
        "sources",
        [],
    )
    matrix = lens.get(
        "matrix",
        {},
    )
    angles = lens.get(
        "source_angles",
        {},
    )
    differences = lens.get(
        "differences",
        [],
    )

    if not sources:
        st.info(
            "Coverage angles could not be extracted from the selected articles."
        )
        return

    st.caption(
        "Coverage angle describes emphasis within the selected article. "
        "“Less visible” is comparative within this set, not a claim about "
        "editorial intent or the outlet’s wider coverage."
    )

    # --------------------------------------------------------------
    # Cross-source differences
    # --------------------------------------------------------------
    if differences:
        difference_rows = []

        for item in differences[:4]:
            visible = _lens_source_list(
                item.get(
                    "most_visible_in",
                    [],
                )
            )
            less = _lens_source_list(
                item.get(
                    "less_visible_in",
                    [],
                )
            )

            sentence = (
                f'<b>{html.escape(item["label"])}</b> is most visible in '
                f'{html.escape(visible)}'
            )

            if less:
                sentence += (
                    f', while it is much less visible in '
                    f'{html.escape(less)}'
                )

            difference_rows.append(
                f'<div style="padding:.45rem 0;line-height:1.65;'
                f'color:#334155;">• {sentence}.</div>'
            )

        st.markdown(
            '<div class="nl-card" '
            'style="background:linear-gradient(135deg,#F3F7FF,#F8F5FF);">'
            '<div class="nl-label" style="margin-top:0;">WHERE THE COVERAGE DIFFERS</div>'
            f'<div style="margin-top:.35rem;">{"".join(difference_rows)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------
    # Event-specific facet heatmap
    # --------------------------------------------------------------
    if facets and matrix:
        facet_labels = [
            facet["label"]
            for facet in facets
        ]

        z = [
            [
                matrix.get(source, {}).get(label, 0.0) * 100
                for label in facet_labels
            ]
            for source in sources
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=facet_labels,
                y=[_display_source_name(s) for s in sources],
                zmin=0,
                zmax=max(
                    35,
                    max(
                        max(row)
                        for row in z
                    ),
                ),
                colorscale=[
                    [0.0, "#F8FAFD"],
                    [0.25, "#E4EBFF"],
                    [0.5, "#BCCBFF"],
                    [0.75, "#809BFF"],
                    [1.0, "#365CCB"],
                ],
                text=[
                    [
                        f"{value:.0f}%"
                        if value > 0
                        else "—"
                        for value in row
                    ]
                    for row in z
                ],
                texttemplate="%{text}",
                hovertemplate=(
                    "%{y}<br>%{x}: %{z:.0f}% of selected event-relevant "
                    "sentences<extra></extra>"
                ),
                colorbar=dict(
                    title="Attention share",
                    ticksuffix="%",
                ),
            )
        )

        fig.update_layout(
            height=max(
                390,
                145 + 58 * len(sources),
            ),
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=50,
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                color="#334155",
                family="Manrope",
            ),
            xaxis=dict(
                tickangle=-20,
                title="Event-specific coverage facets",
            ),
            yaxis=dict(title=""),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "Percentages show how the event-relevant sentences retained from "
            "each article are distributed across event-specific facets. "
            "They are not sentiment or bias scores."
        )

    # --------------------------------------------------------------
    # Source-by-source evidence cards
    # --------------------------------------------------------------
    st.write("")
    st.markdown(
        '<div class="nl-label">SOURCE-BY-SOURCE LENSES</div>',
        unsafe_allow_html=True,
    )

    for source in sources:
        angle = angles.get(
            source,
            {},
        )

        main = angle.get(
            "main_facet",
            "Event coverage",
        )
        main_share = angle.get(
            "main_share",
            0.0,
        )
        secondary = angle.get(
            "secondary_facet"
        )
        secondary_share = angle.get(
            "secondary_share",
            0.0,
        )
        standout = angle.get(
            "standout_facet"
        )
        selected_count = angle.get(
            "selected_sentence_count",
            0,
        )

        evidence = angle.get(
            "evidence",
            [],
        )
        secondary_evidence = angle.get(
            "secondary_evidence",
            [],
        )
        less_visible = angle.get(
            "less_visible",
            [],
        )

        summary_parts = [
            f'<b>{html.escape(_display_source_name(source))}</b> gives its strongest attention to '
            f'<b>{html.escape(main)}</b>.'
        ]

        if main_share > 0:
            summary_parts.append(
                f' This facet accounts for about '
                f'<b>{round(main_share * 100)}%</b> of the event-relevant '
                f'sentences retained from this article.'
            )

        if secondary:
            summary_parts.append(
                f' It also gives noticeable attention to '
                f'<b>{html.escape(secondary)}</b>'
                + (
                    f' ({round(secondary_share * 100)}%).'
                    if secondary_share > 0
                    else '.'
                )
            )

        if standout and standout != main:
            summary_parts.append(
                f' Compared with the other selected reports, '
                f'<b>{html.escape(standout)}</b> is especially prominent here.'
            )
        elif standout:
            summary_parts.append(
                f' This is also the facet that stands out most here compared '
                f'with the other selected reports.'
            )

        evidence_html = ""

        if evidence:
            evidence_rows = []

            for item in evidence[:2]:
                evidence_rows.append(
                    '<div style="margin-top:.5rem;padding:.75rem .85rem;'
                    'background:#F8FAFD;border:1px solid #E1E8F2;'
                    'border-radius:11px;line-height:1.66;color:#334155;">'
                    f'“{html.escape(_evidence_excerpt(item.get("sentence","")), quote=False)}”'
                    '</div>'
                )

            evidence_html = (
                '<div class="nl-label">EVIDENCE FROM THIS ARTICLE</div>'
                + "".join(evidence_rows)
            )

        secondary_html = ""

        if secondary and secondary_evidence:
            item = secondary_evidence[0]
            secondary_html = (
                f'<div class="nl-label">ALSO COVERED — '
                f'{html.escape(secondary.upper())}</div>'
                '<div style="margin-top:.5rem;padding:.75rem .85rem;'
                'background:#FBFCFE;border:1px solid #E6EBF3;'
                'border-radius:11px;line-height:1.66;color:#475569;">'
                f'“{html.escape(_evidence_excerpt(item.get("sentence","")), quote=False)}”'
                '</div>'
            )

        less_html = ""

        if less_visible:
            rows = []

            for item in less_visible:
                others = _lens_source_list(
                    item.get(
                        "sources",
                        [],
                    )
                )

                rows.append(
                    '<div style="margin-top:.42rem;line-height:1.62;color:#475569;">'
                    f'• <b>{html.escape(item["label"])}</b> appears more clearly '
                    f'in {html.escape(others)}.'
                    '</div>'
                )

            less_html = (
                '<div class="nl-label">COMPARATIVELY LESS VISIBLE HERE</div>'
                + "".join(rows)
            )

        st.markdown(
            '<div class="nl-card">'
            f'<div style="font-family:Playfair Display,serif;font-size:1.3rem;'
            f'font-weight:800;color:#14213D;">{html.escape(_display_source_name(source))}</div>'
            f'<div style="margin-top:.4rem;">'
            f'<span class="nl-badge nl-badge-neutral">'
            f'{selected_count} event-relevant sentences examined'
            f'</span></div>'
            '<div class="nl-label">OBSERVED COVERAGE ANGLE</div>'
            f'<div style="margin-top:.4rem;line-height:1.75;color:#26364F;">'
            f'{"".join(summary_parts)}</div>'
            f'{evidence_html}'
            f'{secondary_html}'
            f'{less_html}'
            '</div>',
            unsafe_allow_html=True,
        )


def _image_ref(article):
    local = article.get("image_path")
    if local:
        try:
            p = Path(local)
            if p.exists():
                return str(p)
        except Exception:
            pass
    return article.get("image_url")


def _closest_pair(visual):
    matrix = visual.get("sim_matrix")
    sources = visual.get("sources_with_images", [])
    if not matrix or len(sources) < 2:
        return None
    arr = np.asarray(matrix, dtype=float)
    best = None
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            score = float(arr[i, j])
            if best is None or score > best[2]:
                best = (sources[i], sources[j], score)
    return best


def _semantic_info(visual, source):
    return visual.get("source_semantics", {}).get(source, {})


def _semantic_badges(visual, source):
    info = _semantic_info(visual, source)
    primary = info.get("primary_label")
    secondary = info.get("secondary_label")
    score = info.get("primary_score")

    parts = []
    if primary:
        parts.append(
            f'<span class="nl-badge nl-badge-neutral">{primary}</span>'
        )
    if secondary:
        parts.append(
            f'<span class="nl-badge nl-badge-neutral">{secondary}</span>'
        )
    if score is not None:
        parts.append(
            f'<span class="nl-badge nl-badge-neutral">'
            f'relative content match {float(score):.0%}</span>'
        )
    return "".join(parts)


def render_visual(result):
    st.markdown('<div class="nl-eyebrow">03 · IMAGERY</div>', unsafe_allow_html=True)
    st.markdown('<div class="nl-section-title">The Visual Lens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nl-section-sub">'
        'Compare the main images used across reports by visual similarity and '
        'observable scene content.'
        '</div>',
        unsafe_allow_html=True,
    )

    articles = result.get("articles", [])
    visual = result.get("visual_result", {})


    # ------------------------------------------------------------------
    # Image gallery + plain-language semantic description
    # ------------------------------------------------------------------
    if articles:
        cols = st.columns(min(3, len(articles)))
        for i, article in enumerate(articles):
            source = article.get("source", "Source")
            source_label = _display_source_name(source)
            with cols[i % len(cols)]:
                ref = _image_ref(article)
                if ref:
                    try:
                        st.image(ref, caption=source_label, width="stretch")
                    except Exception:
                        st.markdown(
                            f'<div class="nl-card"><b>{source_label}</b>'
                            '<div style="color:#64748B;margin-top:.3rem;">Image could not be displayed.</div></div>',
                            unsafe_allow_html=True,
                        )

                info = _semantic_info(visual, source)
                if info:
                    primary = info.get("primary_label", "General news scene")
                    secondary = info.get("secondary_label")

                    extra = (
                        f'<div style="margin-top:.28rem;color:#64748B;font-size:.84rem;">'
                        f'Also visible: <b>{secondary}</b></div>'
                        if secondary else ""
                    )
                    st.markdown(
                        f'<div class="nl-card" style="padding:.8rem .9rem;">'
                        f'<div class="nl-label" style="margin-top:0;">MAIN IMAGE FOCUS</div>'
                        f'<div style="margin-top:.35rem;line-height:1.55;color:#334155;'
                        f'font-weight:800;">{primary}</div>'
                        f'{extra}</div>',
                        unsafe_allow_html=True,
                    )

    clusters = visual.get("clusters", [])
    repeated = [c for c in clusters if c.get("size", 0) >= 2]
    singles = [c for c in clusters if c.get("size", 0) == 1]

    # ------------------------------------------------------------------
    # Similarity groups + what those images show
    # ------------------------------------------------------------------
    st.write("")
    if repeated:
        st.markdown(
            '<div class="nl-label">REPEATED VISUAL PATTERNS — WHAT IS SIMILAR?</div>',
            unsafe_allow_html=True,
        )

        for cluster in repeated:
            source_lines = []
            for source in cluster["sources"]:
                info = _semantic_info(visual, source)
                primary = info.get("primary_label")
                secondary = info.get("secondary_label")

                if primary and secondary:
                    source_lines.append(
                        f'<b>{_display_source_name(source)}</b> → {primary} + {secondary}'
                    )
                elif primary:
                    source_lines.append(
                        f'<b>{_display_source_name(source)}</b> → {primary}'
                    )
                else:
                    source_lines.append(
                        f'<b>{_display_source_name(source)}</b> → no clear scene descriptor'
                    )

            st.markdown(
                f'<div class="nl-card">'
                f'<span class="nl-badge">{cluster["label"]}</span>'
                f'<span class="nl-badge nl-badge-neutral">'
                f'mean image similarity {cluster["mean_similarity"]:.2f}</span>'
                f'<div style="margin-top:.65rem;color:#334155;line-height:1.72;">'
                f'<b>Sources:</b> {", ".join(_display_source_name(s) for s in cluster["sources"])}<br>'
                f'<b>What the images show:</b><br>'
                + "<br>".join(source_lines)
                + '</div></div>',
                unsafe_allow_html=True,
            )
    else:
        pair = _closest_pair(visual)
        if pair:
            pair_lines = []
            for source in (pair[0], pair[1]):
                info = _semantic_info(visual, source)
                phrase = info.get("primary_label")
                second = info.get("secondary_label")
                if phrase:
                    if second:
                        phrase += f" + {second}"
                    pair_lines.append(f"<b>{source}</b> → {phrase}")

            semantic_text = (
                "<br><b>Visible content in that closest pair:</b><br>"
                + "<br>".join(pair_lines)
                if pair_lines else ""
            )

            st.markdown(
                f'<div class="nl-info"><b>No pair crossed the grouping threshold.</b> '
                f'The closest visual pair is <b>{pair[0]}</b> and <b>{pair[1]}</b> '
                f'with cosine similarity <b>{pair[2]:.2f}</b>. '
                'The images therefore remain separate rather than being forced into a group.'
                f'{semantic_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="nl-note"><b>Visual comparison unavailable.</b> '
                'Fewer than two article images were successfully processed for visual comparison.</div>',
                unsafe_allow_html=True,
            )

    if singles:
        st.write("")
        st.markdown(
            '<div class="nl-label">DISTINCT VISUAL CHOICES</div>',
            unsafe_allow_html=True,
        )

        rows = []
        for cluster in singles:
            source = cluster["sources"][0]
            info = _semantic_info(visual, source)
            primary = info.get("primary_label")
            secondary = info.get("secondary_label")

            if primary and secondary:
                content = f"{primary} + {secondary}"
            else:
                content = primary or "No clear scene descriptor"

            rows.append(
                f'<div style="padding:.35rem 0;border-bottom:1px solid #EEF2F7;">'
                f'<span class="nl-source">{_display_source_name(source)}</span> '
                f'<span style="color:#334155;">{content}</span></div>'
            )

        st.markdown(
            f'<div class="nl-card">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Quantitative matrix
    # ------------------------------------------------------------------
    matrix = visual.get("sim_matrix")
    sources = visual.get("sources_with_images", [])

    if matrix and len(sources) >= 2:
        st.write("")
        st.markdown(
            '<div class="nl-label">PAIRWISE IMAGE SIMILARITY</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[_display_source_name(s) for s in sources],
            y=[_display_source_name(s) for s in sources],
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#F8FAFD"],
                [0.5, "#A6BAFF"],
                [1.0, "#344EB0"],
            ],
            text=[[f"{float(v):.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Cosine similarity"),
        ))
        fig.update_layout(
            height=max(390, 130 + 55 * len(sources)),
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="Manrope"),
        )
        st.plotly_chart(fig, width="stretch")

        vals = []
        arr = np.asarray(matrix, dtype=float)
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                vals.append(float(arr[i, j]))

        if vals:
            st.markdown(
                f'<div class="nl-card"><div class="nl-label" style="margin-top:0;">VISUAL DIVERSITY SNAPSHOT</div>'
                f'<div style="margin-top:.45rem;line-height:1.7;color:#334155;">'
                f'Across {len(vals)} cross-source image pairs, mean cosine similarity is '
                f'<b>{np.mean(vals):.2f}</b>; the most similar pair reaches <b>{max(vals):.2f}</b> '
                f'and the least similar pair is <b>{min(vals):.2f}</b>. '
                'These values quantify visual convergence/divergence only.</div></div>',
                unsafe_allow_html=True,
            )



def render_analysis():
    result = st.session_state.result
    render_header(result)

    t1,t2,t3 = st.tabs(["THE CORE","THE LENSES","THE VISUAL LENS"])
    with t1:
        render_core(result)
    with t2:
        render_lenses(result)
    with t3:
        render_visual(result)


if st.session_state.view == "landing":
    render_landing()
else:
    render_analysis()
