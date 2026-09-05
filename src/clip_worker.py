"""Crash-isolated CLIP worker for NarrativeLens.

This script is launched by src/image_analysis.py in a subprocess.
It writes structured JSON to a file so any Hugging Face progress/logging on
stdout cannot corrupt the result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Keep native thread counts small before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(input_path: str, output_path: str) -> int:
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        image_paths = payload["image_paths"]
        prompts_map = payload["prompts"]
        model_name = payload["model_name"]
        primary_min = float(payload["primary_min"])
        secondary_min = float(payload["secondary_min"])
        secondary_max_gap = float(payload["secondary_max_gap"])

        import numpy as np
        import torch
        from PIL import Image
        from transformers import (
            CLIPImageProcessorPil,
            CLIPModel,
            CLIPTokenizerFast,
        )

        labels = list(prompts_map.keys())
        flat_prompts = []
        prompt_label_indices = []
        for label_idx, label in enumerate(labels):
            variants = prompts_map[label]
            if isinstance(variants, str):
                variants = [variants]
            for prompt in variants:
                flat_prompts.append(prompt)
                prompt_label_indices.append(label_idx)

        model = CLIPModel.from_pretrained(model_name)
        model.to("cpu")
        model.eval()

        image_processor = CLIPImageProcessorPil.from_pretrained(model_name)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

        images = []
        valid_paths = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    images.append(img.convert("RGB").copy())
                valid_paths.append(p)
            except Exception:
                continue

        if not images:
            Path(output_path).write_text(
                json.dumps({"ok": True, "results": {}}),
                encoding="utf-8",
            )
            return 0

        image_inputs = image_processor(images=images, return_tensors="pt")
        text_inputs = tokenizer(
            flat_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(
                pixel_values=image_inputs["pixel_values"],
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True).clamp_min(1e-12)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Build one robust semantic prototype per user-facing descriptor by
        # averaging several prompt variants for that class. This is less
        # brittle than forcing one wording to represent an entire concept.
        prototypes = []
        for label_idx in range(len(labels)):
            indices = [
                i for i, mapped_idx in enumerate(prompt_label_indices)
                if mapped_idx == label_idx
            ]
            proto = text_embeds[indices].mean(dim=0)
            proto = proto / proto.norm().clamp_min(1e-12)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)

        similarities = (image_embeds @ prototypes.T).cpu().numpy()
        results = {}

        for path, row in zip(valid_paths, similarities):
            order = np.argsort(row)[::-1]
            top_i = int(order[0])
            second_i = int(order[1]) if len(order) > 1 else top_i

            top_score = float(row[top_i])
            second_score = float(row[second_i])

            primary_label = labels[top_i]
            low_confidence = top_score < primary_min

            if low_confidence:
                primary_label = "General news scene"

            secondary_label = None
            secondary_score = None

            # A second descriptor is shown only when it is independently
            # plausible and very close to the primary match. The UI never
            # displays these raw CLIP values as probabilities.
            if (
                not low_confidence
                and second_score >= secondary_min
                and (top_score - second_score) <= secondary_max_gap
                and labels[second_i] != primary_label
                and labels[second_i] != "General news scene"
            ):
                secondary_label = labels[second_i]
                secondary_score = round(second_score, 3)

            results[path] = {
                "primary_label": primary_label,
                "primary_score": round(top_score, 3),
                "secondary_label": secondary_label,
                "secondary_score": secondary_score,
                "top_matches": [
                    {
                        "label": labels[int(i)],
                        "score": round(float(row[int(i)]), 3),
                    }
                    for i in order[:3]
                ],
                "low_separation": low_confidence,
            }

        Path(output_path).write_text(
            json.dumps({"ok": True, "results": results}),
            encoding="utf-8",
        )
        return 0

    except Exception as exc:
        try:
            Path(output_path).write_text(
                json.dumps({"ok": False, "error": repr(exc)}),
                encoding="utf-8",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
