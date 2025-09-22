#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, base64, glob
from pathlib import Path
from openai import OpenAI

# ===== Path configuration =====
script_dir = os.path.dirname(__file__)
key_path = os.path.join(script_dir, "keys.txt")
snapshots_dir = os.path.join(script_dir, "..", "onvif_image_capture", "snapshots")
# ==============================


def read_api_key(path: str) -> str:
    """Read the first valid API key (starting with sk-) from keys.txt"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sk-"):
                return line
    raise RuntimeError("No API key found in keys.txt (must start with sk-)")


def pick_latest_image(dir_path: str) -> Path:
    """Pick the latest image file in the snapshots folder"""
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dir_path, pat)))
    if not files:
        raise FileNotFoundError(f"No image files found in {dir_path}")

    paths = [Path(p) for p in files]
    # Sort by modification time
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return latest


def image_to_data_url(img_path: Path) -> str:
    """Read an image file and convert it to base64 data URL"""
    suffix = img_path.suffix.lower()
    mime = "image/jpeg"
    if suffix == ".png":
        mime = "image/png"

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main():
    api_key = read_api_key(key_path)
    latest_img = pick_latest_image(snapshots_dir)
    print(f"Latest image selected: {latest_img.name}")

    data_url = image_to_data_url(latest_img)

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # You can switch to gpt-4o or gpt-4-turbo
        messages=[
            {"role": "system", "content": "You are an image recognition assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "What car(including model) in the image, simply answer:"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        max_tokens=500
    )

    print("\n=== GPT Response ===")
    print(resp.choices[0].message.content)

    if hasattr(resp, "usage") and resp.usage is not None:
        used = resp.usage.total_tokens
        print("\n=== Token Usage ===")
        print("Prompt tokens    :", resp.usage.prompt_tokens)
        print("Completion tokens:", resp.usage.completion_tokens)
        print("Total tokens     :", used)


if __name__ == "__main__":
    main()
