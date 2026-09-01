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
        secondary_ratio = float(payload["secondary_ratio"])

        import numpy as np
        import torch
        from PIL import Image
        from transformers import (
            CLIPImageProcessorPil,
            CLIPModel,
            CLIPTokenizerFast,
        )

        labels = list(prompts_map.keys())
        prompts = list(prompts_map.values())

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
            prompts,
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

        relative = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
        results = {}

        for path, row in zip(valid_paths, relative):
            order = np.argsort(row)[::-1]
            top_i = int(order[0])
            second_i = int(order[1]) if len(order) > 1 else top_i

            top_score = float(row[top_i])
            second_score = float(row[second_i])

            primary_label = labels[top_i]
            low_separation = top_score < primary_min

            if low_separation:
                primary_label = "General / unclear news scene"

            secondary_label = None
            secondary_score = None

            if (
                not low_separation
                and second_score >= secondary_min
                and second_score >= top_score * secondary_ratio
                and labels[second_i] != primary_label
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
                "low_separation": low_separation,
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
