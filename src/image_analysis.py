"""
NarrativeLens Visual Lens.

Two deliberately separate ML layers are used:

1) MobileNetV2 deep image embeddings
   image -> 1280-dim feature vector -> cosine similarity -> neutral visual groups.
   This is the Product-Image-Search-style component and measures visual
   convergence/divergence only.

2) CLIP zero-shot semantic descriptors
   image -> comparison against prompt ensembles for broad, observable news
   scenes. This adds one clean primary descriptor and, only when useful, one
   secondary descriptor. It does NOT infer editorial intent, bias,
   manipulation, motive, sentiment, or truth.

Raw CLIP similarity values are used internally for ranking only and are not
presented to users as probabilities or confidence scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# MobileNetV2 similarity layer
# ---------------------------------------------------------------------------

_MOBILENET = None
IMAGE_SIZE = (224, 224)
IMAGE_SIM_THRESHOLD = 0.55


def get_mobilenet():
    global _MOBILENET
    if _MOBILENET is None:
        from tensorflow.keras.applications import MobileNetV2

        # Explicit input shape avoids the harmless "undefined input_shape"
        # warning and documents exactly what the pipeline expects.
        _MOBILENET = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
    return _MOBILENET


def _load_image_array(path: str):
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(path, target_size=IMAGE_SIZE)
    return keras_image.img_to_array(img)


def extract_image_embeddings(image_paths: List[str]):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = get_mobilenet()
    arrays, valid_idx = [], []

    for i, path in enumerate(image_paths):
        try:
            arrays.append(_load_image_array(path))
            valid_idx.append(i)
        except Exception:
            continue

    if not arrays:
        return np.zeros((0, 1280)), []

    batch = preprocess_input(np.stack(arrays))
    embeddings = model.predict(batch, verbose=0)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms, valid_idx


# ---------------------------------------------------------------------------
# CLIP semantic descriptor layer
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# User-facing label -> natural-language prompt sent to CLIP.
VISUAL_PROMPTS: Dict[str, List[str]] = {
    # Natural disasters / environment
    "Flood / high water": [
        "a news photograph showing floodwater, high water, an overflowing river, or inundated streets",
        "a news image of buildings, roads, fields, or communities surrounded by flood water",
        "a disaster photograph showing widespread flooding or a swollen river",
    ],
    "Landslide / mudslide": [
        "a news photograph showing a landslide, mudslide, collapsed slope, or moving earth",
        "a disaster image showing mud, rocks, or debris flowing down a hillside or valley",
        "a news photograph of a slope failure or debris-covered road",
    ],
    "Glacier / snow": [
        "a news photograph dominated by a glacier, ice field, snow, or snow-covered mountain terrain",
        "a news image showing glacier ice, alpine snow, or a high mountain valley",
        "a photograph of icy or snowy mountain terrain connected to a news event",
    ],
    "Wildfire / smoke": [
        "a news photograph showing wildfire, forest fire, flames, or heavy outdoor smoke",
        "a disaster image of burning vegetation, wildfire smoke, or fire crews near a blaze",
        "a news image dominated by flames or smoke from a large fire",
    ],
    "Storm / severe weather": [
        "a news photograph showing a cyclone, hurricane, severe storm, heavy rain, strong winds, or storm damage",
        "a weather disaster image showing intense rain, wind, waves, or storm conditions",
        "a news photograph of severe weather affecting buildings, roads, or communities",
    ],
    "Earthquake damage": [
        "a news photograph showing earthquake damage, collapsed masonry, cracked buildings, or seismic destruction",
        "a disaster image of buildings or structures damaged by an earthquake",
        "a news photograph of rubble and structural collapse after seismic activity",
    ],
    "Damaged infrastructure": [
        "a news photograph showing damaged homes, buildings, roads, bridges, utilities, or public infrastructure",
        "a disaster image focused on destroyed or heavily damaged structures",
        "a news photograph showing broken roads, collapsed buildings, damaged bridges, or ruined property",
    ],
    "Rescue operation": [
        "a news photograph showing rescuers, emergency responders, search teams, rescue boats, or disaster relief workers",
        "an emergency response image showing people searching for, assisting, or evacuating victims",
        "a news image focused on rescue or recovery activity after an emergency",
    ],
    "Evacuation / displacement": [
        "a news photograph showing people evacuating, leaving homes, entering shelters, or being displaced",
        "a news image of displaced families, evacuation centres, or people carrying belongings after an emergency",
        "a disaster photograph focused on civilians moving to safety",
    ],
    "Drought / dry conditions": [
        "a news photograph showing drought, cracked dry ground, empty reservoirs, dry fields, or severe water shortage",
        "an environmental news image showing parched land or drought-affected agriculture",
        "a news photograph focused on unusually dry conditions or water scarcity",
    ],
    "Environmental pollution": [
        "a news photograph showing polluted water, oil spill, industrial pollution, waste, smog, or environmental contamination",
        "an environmental news image showing visible pollution or contamination",
        "a news photograph focused on environmental damage caused by waste or pollutants",
    ],

    # Politics / public affairs
    "Election / voting": [
        "a news photograph showing voting, ballots, ballot boxes, polling stations, election officials, or vote counting",
        "an election news image of voters casting ballots or election materials being counted",
        "a photograph focused on the voting process or ballot administration",
    ],
    "Campaign rally": [
        "a news photograph showing an election campaign rally, political supporters, campaign signs, or a candidate addressing a crowd",
        "a political campaign image with a stage, supporters, banners, or election gathering",
        "a news photograph focused on a candidate campaign event",
    ],
    "Political leader": [
        "a news photograph primarily showing a president, prime minister, minister, elected representative, or political leader",
        "a political news portrait or appearance by a senior government or party leader",
        "a news image focused on a politician speaking, meeting, or appearing publicly",
    ],
    "Protest / demonstration": [
        "a news photograph showing protesters, placards, a march, rally, sit-in, strike, or public demonstration",
        "a news image of people demonstrating in public with signs, banners, or slogans",
        "a photograph focused on organized public protest activity",
    ],
    "Police / security": [
        "a news photograph showing police officers, riot police, law enforcement, security personnel, barricades, or crowd control",
        "a news image focused on police or security forces at an event",
        "a photograph showing law enforcement presence or public-security operations",
    ],
    "Diplomatic meeting": [
        "a news photograph showing leaders or officials in a diplomatic meeting, summit, bilateral talks, or international negotiation",
        "a foreign affairs image of officials meeting, shaking hands, or seated for formal talks",
        "a news photograph focused on diplomacy between governments",
    ],
    "Government briefing": [
        "a news photograph showing an official government briefing, podium, microphones, press conference, or formal statement",
        "a public affairs image of officials speaking to journalists at a briefing",
        "a news photograph focused on an official announcement or government press event",
    ],
    "Court / legal proceedings": [
        "a news photograph showing a courtroom, judge, lawyers, courthouse, legal hearing, or judicial proceeding",
        "a legal news image focused on a court building, hearing, or trial",
        "a photograph connected to litigation, judges, attorneys, or formal legal proceedings",
    ],
    "Military / conflict": [
        "a news photograph showing soldiers, military vehicles, weapons, combat, armed conflict, or a war zone",
        "a conflict image focused on troops, military equipment, battlefield activity, or armed forces",
        "a news photograph showing military operations or active conflict",
    ],

    # People / society
    "Crowd / public gathering": [
        "a news photograph showing a large public crowd or gathering without a clearly dominant protest or campaign activity",
        "a news image of many people gathered at a public place or event",
        "a photograph focused on a crowd, audience, or public gathering",
    ],
    "Medical response": [
        "a news photograph showing medical workers, ambulances, hospital treatment, emergency care, or injured people receiving help",
        "a health or emergency image focused on doctors, paramedics, patients, or medical response",
        "a news photograph showing treatment or emergency medical activity",
    ],
    "Mourning / memorial": [
        "a news photograph showing a memorial, funeral, candles, flowers, mourning relatives, or public remembrance",
        "a news image focused on grief, remembrance, or a funeral gathering",
        "a photograph showing people mourning victims or attending a memorial",
    ],
    "Refugees / migration": [
        "a news photograph showing refugees, migrants, border crossings, displacement camps, or people travelling with belongings",
        "a migration news image focused on displaced people or border movement",
        "a photograph showing refugees or migrants in transit or temporary shelter",
    ],
    "Community scene": [
        "a news photograph focused on ordinary residents, neighbourhood life, families, or a local community",
        "a human-interest news image showing people in a community setting",
        "a photograph of residents or civilians in everyday local surroundings",
    ],
    "Education / classroom": [
        "a news photograph showing students, teachers, classrooms, schools, universities, exams, or educational activity",
        "an education news image focused on a classroom, campus, students, or learning",
        "a photograph showing school or university activity",
    ],

    # Economy / infrastructure / industry
    "Market / finance": [
        "a news photograph showing financial markets, trading screens, banks, currency, business executives, or economic activity",
        "a finance news image focused on markets, money, banking, investment, or corporate economics",
        "a photograph connected to financial or business activity",
    ],
    "Factory / industry": [
        "a news photograph showing a factory, industrial plant, machinery, manufacturing line, mine, or industrial workers",
        "an industry news image focused on manufacturing, production, heavy machinery, or an industrial facility",
        "a photograph showing industrial production or a factory setting",
    ],
    "Port / shipping": [
        "a news photograph showing a port, cargo ship, shipping containers, cranes, freight terminal, or maritime trade",
        "a trade news image focused on cargo, shipping, docks, or container transport",
        "a photograph of commercial shipping or port activity",
    ],
    "Transport / vehicles": [
        "a news photograph showing cars, buses, trains, aircraft, airports, roads, railways, or public transport",
        "a transport news image focused on vehicles or transportation infrastructure",
        "a photograph showing travel, traffic, trains, planes, or road transport",
    ],
    "Construction / development": [
        "a news photograph showing construction work, cranes, building sites, new roads, bridges, or infrastructure development",
        "a development news image focused on active construction or major infrastructure projects",
        "a photograph showing a construction site or infrastructure being built",
    ],
    "Agriculture / farming": [
        "a news photograph showing farms, crops, farmers, livestock, agricultural fields, or harvesting",
        "an agriculture news image focused on farming, crops, animals, or rural production",
        "a photograph showing agricultural work or farmland",
    ],

    # Technology / science / health / sport
    "Technology / computing": [
        "a news photograph showing computers, smartphones, servers, semiconductors, robots, software, data centres, or technology products",
        "a technology news image focused on digital devices, computing hardware, chips, or software systems",
        "a photograph showing technology products or computing infrastructure",
    ],
    "Science / research": [
        "a news photograph showing scientists, laboratory equipment, experiments, research facilities, microscopes, or scientific fieldwork",
        "a science news image focused on researchers, experiments, or scientific equipment",
        "a photograph showing laboratory or research activity",
    ],
    "Space / astronomy": [
        "a news image showing a rocket, spacecraft, satellite, astronaut, planet, telescope image, or space mission",
        "a space news image focused on astronomy, spacecraft, launch activity, or celestial objects",
        "a photograph or scientific image connected to space exploration",
    ],
    "Health / medicine": [
        "a news photograph showing doctors, hospitals, medicines, vaccines, medical research, clinics, or public health activity",
        "a health news image focused on healthcare, medicine, disease response, or clinical work",
        "a photograph connected to hospitals, healthcare workers, or medical treatment",
    ],
    "Sports / competition": [
        "a news photograph showing athletes, a sports match, race, stadium, court, field, podium, or sporting competition",
        "a sports news image focused on players, teams, competition, or an athletic event",
        "a photograph showing active sport or competition",
    ],
    "Crime / investigation": [
        "a news photograph showing a crime scene, investigators, forensic work, police tape, evidence collection, or criminal investigation",
        "a crime news image focused on investigators or a secured crime scene",
        "a photograph connected to an active criminal investigation",
    ],

    # Information-style visuals
    "Map / infographic": [
        "a news image mainly showing a map, chart, diagram, infographic, data visualization, or explanatory graphic",
        "a non-photographic news visual used to explain locations, numbers, trends, or an event",
        "an informational graphic containing maps, charts, diagrams, or labelled data",
    ],
    "Logo / branding": [
        "a news image mainly showing a company, organization, institution, campaign, or media logo",
        "a branding image dominated by a logo, emblem, wordmark, or organization name",
        "a non-photographic visual focused primarily on branding or a logo",
    ],
    "Document / text": [
        "a news image mainly showing a document, legal paper, report page, letter, printed text, or official notice",
        "a photograph or screenshot focused on written documents or official text",
        "a news visual dominated by a page, document, statement, or textual material",
    ],
    "Interview / media appearance": [
        "a news photograph showing a person being interviewed, speaking to media, appearing on television, or seated for a formal interview",
        "a media news image focused on an interview, television appearance, or journalist speaking with a guest",
        "a photograph of a person in a media interview or broadcast setting",
    ],
    "General news scene": [
        "a general news photograph of a place, street, building, landscape, or people without one clearly dominant activity",
        "a general-purpose news image where no specific event category is visually dominant",
        "a neutral news scene that does not strongly match another specific visual category",
    ],
}


CLIP_PRIMARY_MIN = 0.18
CLIP_SECONDARY_MIN = 0.17
CLIP_SECONDARY_MAX_GAP = 0.025


def classify_images_zero_shot(image_paths: List[str]) -> Dict[str, Any]:
    """Run CLIP in a SEPARATE Python process.

    Why a subprocess?
    On some Apple-Silicon/macOS combinations, PyTorch/Transformers vision
    inference can terminate the interpreter with a native segmentation fault
    instead of raising a Python exception. Running the optional CLIP layer in
    a worker process prevents that native failure from taking down Streamlit.

    If the worker crashes or times out, NarrativeLens simply skips semantic
    labels and keeps the already-working MobileNetV2 similarity analysis.
    """
    if not image_paths:
        return {}

    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    worker = Path(__file__).with_name("clip_worker.py")
    if not worker.exists():
        print("[NarrativeLens][visual] CLIP worker missing; semantic labels skipped.")
        return {}

    payload = {
        "image_paths": image_paths,
        "model_name": CLIP_MODEL_NAME,
        "prompts": VISUAL_PROMPTS,
        "primary_min": CLIP_PRIMARY_MIN,
        "secondary_min": CLIP_SECONDARY_MIN,
        "secondary_max_gap": CLIP_SECONDARY_MAX_GAP,
    }

    input_file = None
    output_file = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            input_file = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            output_file = f.name

        env = os.environ.copy()
        # Reduce native-thread/OpenMP contention on Apple Silicon.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        completed = subprocess.run(
            [sys.executable, str(worker), input_file, output_file],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )

        if completed.returncode != 0:
            # -11 is SIGSEGV on Unix/macOS. Whatever the native error is,
            # keep the main Streamlit process alive and degrade gracefully.
            print(
                "[NarrativeLens][visual] CLIP worker exited abnormally "
                f"(return code {completed.returncode}); semantic labels skipped."
            )
            if completed.stdout:
                tail = completed.stdout[-1200:]
                print("[NarrativeLens][visual] CLIP worker log tail:\n" + tail)
            return {}

        try:
            result = json.loads(Path(output_file).read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                "[NarrativeLens][visual] Could not read CLIP worker output; "
                f"semantic labels skipped: {exc}"
            )
            return {}

        if not result.get("ok"):
            print(
                "[NarrativeLens][visual] CLIP worker reported failure; "
                f"semantic labels skipped: {result.get('error', 'unknown error')}"
            )
            return {}

        return result.get("results", {})

    except subprocess.TimeoutExpired:
        print(
            "[NarrativeLens][visual] CLIP worker timed out; "
            "semantic labels skipped while MobileNet analysis is preserved."
        )
        return {}
    except Exception as exc:
        print(
            "[NarrativeLens][visual] CLIP worker could not run; "
            f"semantic labels skipped: {exc}"
        )
        return {}
    finally:
        for p in (input_file, output_file):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


def _semantic_phrase(info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not info:
        return None

    primary = info.get("primary_label")
    secondary = info.get("secondary_label")

    if not primary:
        return None
    if secondary:
        return f"{primary} + {secondary}"
    return primary


# ---------------------------------------------------------------------------
# Combined Visual Lens analysis
# ---------------------------------------------------------------------------

def analyze_visual_framing(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run MobileNet similarity and the independent CLIP descriptor layer."""
    paths, sources = [], []

    for article in articles:
        if article.get("image_path"):
            paths.append(article["image_path"])
            sources.append(article["source"])

    if len(paths) < 2:
        return {
            "clusters": [],
            "sim_matrix": None,
            "sources_with_images": sources,
            "source_labels": {},
            "source_semantics": {},
        }

    embeddings, valid_idx = extract_image_embeddings(paths)
    valid_paths = [paths[i] for i in valid_idx]
    sources = [sources[i] for i in valid_idx]

    if len(sources) < 2:
        return {
            "clusters": [],
            "sim_matrix": None,
            "sources_with_images": sources,
            "source_labels": {},
            "source_semantics": {},
        }

    # Add semantic descriptors independently. They never affect grouping.
    clip_by_path = classify_images_zero_shot(valid_paths)
    source_semantics = {
        sources[i]: clip_by_path.get(valid_paths[i], {})
        for i in range(len(sources))
    }
    source_labels = {
        source: _semantic_phrase(info)
        for source, info in source_semantics.items()
    }

    sims = embeddings @ embeddings.T

    n = len(sources)
    assigned = [-1] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if assigned[i] != -1:
            continue

        best_c, best_score = -1, 0.0

        for ci, members in enumerate(clusters):
            avg_sim = float(np.mean([sims[i, m] for m in members]))
            if avg_sim >= IMAGE_SIM_THRESHOLD and avg_sim > best_score:
                best_c, best_score = ci, avg_sim

        if best_c >= 0:
            clusters[best_c].append(i)
            assigned[i] = best_c
        else:
            clusters.append([i])
            assigned[i] = len(clusters) - 1

    cluster_dicts = []

    for members in clusters:
        if len(members) > 1:
            pair_sims = [
                float(sims[members[i], members[j]])
                for i in range(len(members))
                for j in range(i + 1, len(members))
            ]
            mean_sim = float(np.mean(pair_sims))
        else:
            mean_sim = None

        cluster_dicts.append(
            {
                "sources": [sources[i] for i in members],
                "size": len(members),
                "mean_similarity": (
                    round(mean_sim, 2) if mean_sim is not None else None
                ),
            }
        )

    cluster_dicts.sort(key=lambda c: c["size"], reverse=True)

    letter_idx = 0
    for cluster in cluster_dicts:
        if cluster["size"] >= 2:
            cluster["label"] = f"Visual Group {chr(ord('A') + letter_idx)}"
            letter_idx += 1
        else:
            cluster["label"] = "Distinct Image"

    return {
        "clusters": cluster_dicts,
        "sim_matrix": sims.tolist(),
        "sources_with_images": sources,
        "source_labels": source_labels,       # backwards-compatible simple text
        "source_semantics": source_semantics, # richer UI data
    }


def build_visual_summary(result: Dict[str, Any], total_sources: int) -> str:
    """Reader-facing visual comparison with cautious semantic descriptors."""
    clusters = result.get("clusters", [])
    sources = result.get("sources_with_images", [])
    semantics = result.get("source_semantics", {})

    if not clusters:
        return (
            "Insufficient locally processed article images were available "
            "to perform a cross-source visual comparison."
        )

    groups = [c for c in clusters if c["size"] >= 2]
    singles = [c for c in clusters if c["size"] == 1]

    parts: List[str] = []

    for group in groups:
        member_sources = group["sources"]
        parts.append(
            f"{', '.join(member_sources)} form {group['label']} with a mean "
            f"MobileNetV2 cosine similarity of {group['mean_similarity']}."
        )

        descriptions = []
        for source in member_sources:
            phrase = _semantic_phrase(semantics.get(source))
            if phrase:
                descriptions.append(f"{source}: {phrase}")

        if descriptions:
            parts.append(
                "Within this visually similar group, the strongest CLIP "
                "content matches are " + "; ".join(descriptions) + "."
            )

    if singles:
        descriptions = []
        for cluster in singles:
            source = cluster["sources"][0]
            phrase = _semantic_phrase(semantics.get(source))
            if phrase:
                descriptions.append(f"{source}: {phrase}")
            else:
                descriptions.append(f"{source}: no clear semantic descriptor")

        parts.append(
            "The remaining distinct images differ from the repeated visual "
            "group(s). Their strongest content matches are "
            + "; ".join(descriptions) + "."
        )

    if not groups:
        descriptions = []
        for source in sources:
            phrase = _semantic_phrase(semantics.get(source))
            if phrase:
                descriptions.append(f"{source}: {phrase}")

        parts.insert(
            0,
            f"All {len(sources)} available article images remain distinct at "
            f"the current visual-similarity grouping threshold of {IMAGE_SIM_THRESHOLD:.2f}."
        )
        if descriptions:
            parts.append(
                "Their strongest visible-content matches are "
                + "; ".join(descriptions) + "."
            )

    parts.append(
        "MobileNetV2 measures visual similarity; CLIP adds broad observable "
        "scene descriptors. Neither layer infers editorial intent, bias, motive, "
        "or truth."
    )

    return " ".join(parts)
