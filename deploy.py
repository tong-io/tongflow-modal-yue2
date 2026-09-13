"""Modal deploy entry for yue2 (YuE2 song generation + zero-shot covers).

Wraps the YuE2 open-source release (github.com/multimodal-art-projection/YuE,
weights m-a-p/YuE2-3B + m-a-p/YuE2-Vae) as TongFlow music nodes:
  - gen-music: style + lyrics -> symbolic plan (ABC) -> full song
  - music-cover: source audio -> SheetSage2 melody score -> YuE2 re-performs it
    in the requested style with the supplied lyrics

SheetSage2 pins torch 2.8 / transformers 4.45 while YuE2 pins torch 2.10 /
transformers 4.57, so transcription runs in its own class and image; the YuE2
handler calls it remotely and exchanges only audio bytes and ABC text.

Weights are licensed CC BY-NC 4.0 (non-commercial use only).

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
from tongflow import deploy

_cfg: dict[str, Any] = {}

# Cover defaults: a score-conditioned render follows the source arrangement
# unless text guidance is on, so covers default to CFG 2.0 (1.0 disables it).
COVER_CFG_SCALE = 2.0
COVER_TRANSPOSE = -3

# Pinned source + weight revisions (see download.py, which fetches the same).
REPO_URL = "https://github.com/multimodal-art-projection/YuE.git"
REPO_REV = "88da114a67df892af0329472073b96a5ef700b93"
YUE2_MODEL = ("m-a-p/YuE2-3B", "29b3558dd46954a0cd9021dc76d5c91864a0f1c7")
YUE2_VAE = ("m-a-p/YuE2-Vae", "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a")
SHEETSAGE2 = ("m-a-p/SheetSage2", "eab522a8168e8b8b8c4856bf8609cd86198f01fe")

# Hugging Face cache on the shared `models` volume, populated by download.py.
HF_CACHE_DIR = "/models/yue2-hf"

SAMPLE_RATE = 48000

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.gen_music import GenMusicInput, GenMusicOutput
from tongflow.models.music_cover import MusicCoverInput, MusicCoverOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "gen-music": {
        "cot": {
            "type": "select",
            "options": ["full", "melody", "off"],
            "default": "full",
            "label": "Score planning",
            "description": "full: plan melody + chords; melody: plan melody only; off: no score",
        },
        "cfg_scale": {"type": "number", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "label": "Text guidance (CFG)"},
        "temperature": {"type": "number", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Temperature"},
        "top_p": {"type": "number", "default": 0.95, "min": 0.05, "max": 1.0, "step": 0.01, "label": "Top-p"},
    },
    "music-cover": {
        "cot": {
            "type": "select",
            "options": ["melody", "full"],
            "default": "melody",
            "label": "Source score",
            "description": "melody: keep the melody, re-arrange freely; full: also keep the source chords",
        },
        "transpose": {
            "type": "integer",
            "default": -3,
            "min": -12,
            "max": 12,
            "step": 1,
            "label": "Transpose (semitones)",
            "description": "Shift the source key before transcription; negative lowers it (e.g. -3 for a male voice)",
        },
        "cfg_scale": {"type": "number", "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.05, "label": "Text guidance (CFG)"},
        "temperature": {"type": "number", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Temperature"},
        "top_p": {"type": "number", "default": 0.95, "min": 0.05, "max": 1.0, "step": 0.01, "label": "Top-p"},
    },
}


app = modal.App(Path(__file__).resolve().parent.name)

_hf_env = {"HF_HUB_CACHE": HF_CACHE_DIR, "HF_HUB_OFFLINE": "1"}

# YuE2 runtime: the repo's own package pins torch/transformers/numpy.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1")
    .pip_install(f"yue2-infer @ git+{REPO_URL}@{REPO_REV}")
    .pip_install("tongflow==0.3.3", "fastapi[standard]")
    .env(_hf_env)
)

# SheetSage2 runtime: CUDA 12.6 torch 2.8 + the model repo's requirements.
# Audio is decoded by the ffmpeg CLI (SheetSage2's default preset).
transcribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "transformers==4.45.2",
        "huggingface-hub==0.36.0",
        "safetensors==0.5.3",
        "numpy==1.24.3",
        "scipy==1.13.1",
        "mir_eval==0.8.2",
        "pretty_midi==0.2.10",
        "mido==1.3.3",
        "setuptools==78.1.1",
    )
    .pip_install("tongflow==0.3.3")
    .env(_hf_env)
)


def _seed(seed: int | None) -> int:
    if seed is not None and 0 <= int(seed) < 2**63:
        return int(seed)
    import secrets

    return secrets.randbelow(2**31)


def _sampling() -> dict:
    return {"temperature": _adv("temperature", 1.0), "top_p": _adv("top_p", 0.95)}


def _cfg_scale() -> float | None:
    # None keeps YuE2's per-mode default guidance.
    v = current_params().get("cfg_scale")
    return None if v is None else float(v)


def _pitch_shift(audio: bytes, semitones: int) -> bytes:
    """Shift pitch by resampling, then restore the original tempo with atempo.

    The shifted audio only feeds transcription, so resampling artifacts are
    irrelevant; what matters is that the score lands in the new key at the
    original tempo.
    """
    import subprocess

    ratio = 2 ** (semitones / 12)
    rate = 44100
    af = f"aresample={rate},asetrate={rate * ratio:.4f},aresample={rate},atempo={1 / ratio:.6f}"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", "pipe:0", "-vn", "-af", af, "-f", "wav", "pipe:1"],
        input=audio,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if r.returncode:
        raise RuntimeError("Pitch shift failed: " + r.stderr.decode(errors="replace")[-600:])
    return r.stdout


@app.cls(
    scaledown_window=2,
    image=transcribe_image,
    gpu="L4",
    volumes={"/models": volume},
    timeout=1800,
)
class Transcriber:
    """SheetSage2 audio -> ABC score (loads its MERT-v2 parent from the cache)."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel

        repo, rev = SHEETSAGE2
        self.model = AutoModel.from_pretrained(
            repo,
            revision=rev,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).eval().to("cuda")

    @modal.method()
    def transcribe(self, audio: bytes, melody_only: bool, semitones: int = 0) -> str:
        if semitones:
            audio = _pitch_shift(audio, semitones)
        result = self.model.transcribe(audio, melody_only=melody_only)
        abc = result.get("abc")
        if not abc or result.get("abc_error"):
            raise RuntimeError(
                f"SheetSage2 could not transcribe a usable score: {result.get('abc_error') or 'empty ABC'}"
            )
        return abc


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    timeout=1800,
)
class Inference:
    @modal.enter()
    def load(self):
        from yue2 import YuE2Pipeline

        (model, model_rev), (vae, vae_rev) = YUE2_MODEL, YUE2_VAE
        self.pipe = YuE2Pipeline.from_pretrained(
            model,
            vae=vae,
            revision=model_rev,
            vae_revision=vae_rev,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
            device="cuda",
            progress=False,
        )

    def _render(
        self, *, style: str, lyrics: str, cot: str, seed: int, cfg_scale: float | None, abc: str | None = None
    ) -> bytes:
        import io

        import soundfile as sf

        sampling = _sampling()
        song = self.pipe(
            style=style,
            lyrics=lyrics,
            cot=cot,
            seed=seed,
            abc=abc,
            cfg_scale=cfg_scale,
            semantic_sampling=sampling,
        )
        buf = io.BytesIO()
        sf.write(buf, song.audio, song.sample_rate, format="FLAC", subtype="PCM_24")
        return buf.getvalue()

    @modal.method()
    @node_slot(NodeSlots.GEN_MUSIC)
    def gen_music(self, input: GenMusicInput) -> GenMusicOutput:
        lyrics = (input.lyrics or "").strip()
        if not lyrics:
            return GenMusicOutput(success=False, error="Missing lyrics")
        # YuE2 has no separate bpm/key/language fields: they belong in the style
        # prompt (its examples lead with the language and end with the tempo).
        parts = [input.language, input.text, input.tags, input.keyscale]
        if input.bpm:
            parts.append(f"{int(round(input.bpm))} BPM")
        style = ", ".join(p.strip() for p in parts if p and p.strip())
        if not style:
            return GenMusicOutput(success=False, error="Missing style prompt")
        try:
            raw = self._render(
                style=style,
                lyrics=lyrics,
                cot=_adv("cot", "full"),
                seed=_seed(input.seed),
                cfg_scale=_cfg_scale(),
            )
        except Exception as e:
            return GenMusicOutput(success=False, error=str(e))
        return GenMusicOutput(success=True, audio=asset(raw, mime="audio/flac"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_COVER)
    def music_cover(self, input: MusicCoverInput) -> MusicCoverOutput:
        style = (input.text or "").strip()
        if not style:
            return MusicCoverOutput(success=False, error="Missing target style prompt")
        # SheetSage2 transcribes notes, not words: the cover sings these lyrics.
        lyrics = (input.lyrics or "").strip()
        if not lyrics:
            return MusicCoverOutput(success=False, error="Missing lyrics (YuE2 covers sing the supplied lyrics)")
        cot = _adv("cot", "melody")
        try:
            with asset_as_path(input.audio) as src:
                data = Path(src).read_bytes()
            abc = Transcriber().transcribe.remote(data, cot == "melody", _adv("transpose", COVER_TRANSPOSE))
            raw = self._render(
                style=style,
                lyrics=lyrics,
                cot=cot,
                seed=_seed(input.seed),
                cfg_scale=_adv("cfg_scale", COVER_CFG_SCALE),
                abc=abc,
            )
        except Exception as e:
            return MusicCoverOutput(success=False, error=str(e))
        return MusicCoverOutput(success=True, audio=asset(raw, mime="audio/flac"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
