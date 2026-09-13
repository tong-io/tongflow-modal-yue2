"""Modal download entry for yue2 (YuE2 + SheetSage2).

Run:
  modal run download.py::download

Fetches pinned Hugging Face snapshots into a HF cache on the shared `models`
volume (/models/yue2-hf). deploy.py loads them by repo id + revision with
HF_HUB_CACHE pointed there and HF_HUB_OFFLINE=1:
  - m-a-p/YuE2-3B           generation model
  - m-a-p/YuE2-Vae          listening decoder
  - m-a-p/SheetSage2        audio -> score transcription (music-cover)
  - m-a-p/MERT-v2-FullSong  SheetSage2's encoder parent (revision from its config)

All repos are public. Weights are licensed CC BY-NC 4.0.

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

HF_CACHE_DIR = "/models/yue2-hf"

# (repo_id, revision, ignore_patterns). Demo audio, figures, bundled wheels and
# rendering soundfonts are not needed for inference.
_REPOS: list[tuple[str, str, list[str]]] = [
    ("m-a-p/YuE2-3B", "29b3558dd46954a0cd9021dc76d5c91864a0f1c7", ["assets/*", "*.whl"]),
    ("m-a-p/YuE2-Vae", "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a", ["assets/*"]),
    ("m-a-p/SheetSage2", "eab522a8168e8b8b8c4856bf8609cd86198f01fe", ["assets/*", "render_assets/soundfonts/*"]),
    ("m-a-p/MERT-v2-FullSong", "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42", ["assets/*"]),
]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub==0.36.0"
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None

    for repo_id, revision, ignore in _REPOS:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=HF_CACHE_DIR,
            ignore_patterns=ignore,
            token=token,
        )
        print(f"{repo_id}@{revision[:7]} -> {path}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
