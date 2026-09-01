"""
NarrativeLens Visual Lens.

Two deliberately separate ML layers are used:

1) MobileNetV2 deep image embeddings
   image -> 1280-dim feature vector -> cosine similarity -> neutral visual groups.
   This is the Product-Image-Search-style component and measures visual
   convergence/divergence only.

2) CLIP zero-shot semantic descriptors
   image -> comparison against a fixed list of natural-language prompts.
   This adds an observable-content description (e.g. police/security presence,
   protest crowd, injury aftermath). It does NOT infer editorial intent, bias,
   manipulation, motive, sentiment, or truth.

CLIP match scores shown to users are RELATIVE matches within the fixed prompt
set. They are not calibrated probabilities that a label is objectively true.
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
VISUAL_PROMPTS: Dict[str, str] = {
    "Protest crowd / demonstration":
        "a news photograph showing a protest crowd, march, rally, or public demonstration",
    "Police / security presence":
        "a news photograph showing police officers, riot police, law enforcement, or security personnel",
    "Street confrontation / clash":
        "a news photograph showing an active street confrontation, clash, scuffle, or people physically confronting each other",
    "Injury / medical aftermath":
        "a news photograph showing an injured person, medical treatment, an ambulance, or people tending to injuries",
    "Political leader / official":
        "a news photograph showing a political leader, minister, government official, or elected representative",
    "Press conference / statement":
        "a news photograph showing a press conference, podium, microphones, media briefing, or official statement",
    "Economic / trade imagery":
        "a news photograph about trade or the economy showing ports, shipping containers, factories, markets, money, tariffs, or commerce",
    "Flags / diplomatic imagery":
        "a news photograph showing national flags, a diplomatic meeting, officials shaking hands, or international diplomacy",
    "Property damage / destruction":
        "a news photograph showing damaged property, broken objects, rubble, fire damage, or destruction",
    "General / unclear news scene":
        "a general news photograph or street scene without one clearly dominant activity",
}

CLIP_PRIMARY_MIN = 0.16
CLIP_SECONDARY_MIN = 0.12
CLIP_SECONDARY_RATIO = 0.62


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
        "secondary_ratio": CLIP_SECONDARY_RATIO,
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
        "MobileNetV2 measures visual similarity; CLIP adds observable-content "
        "descriptors. CLIP match scores are relative to the fixed prompt set, "
        "and neither layer infers editorial intent, bias, or motive."
    )

    return " ".join(parts)
