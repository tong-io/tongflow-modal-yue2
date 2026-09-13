# tongflow-modal-yue2

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Song generation and zero-shot covers with **YuE2** (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`), running on a GPU via [Modal](https://modal.com). Covers transcribe the source recording with **SheetSage2** (`m-a-p/SheetSage2`).

YuE2 first writes an editable melody-and-chord score (ABC), then renders it as a full song with vocals and accompaniment — 48 kHz stereo FLAC.

> **License:** YuE2 and SheetSage2 weights are **CC BY-NC 4.0 — non-commercial use only**. The upstream code is Apache 2.0.

## Capabilities

- **Music generation** (`gen-music`) — style prompt + lyrics → full song.
- **Music cover** (`music-cover`) — source audio → SheetSage2 melody score → YuE2 re-performs it in a new style with the lyrics you supply.

### Input tips

- **Style** (prompt / tags): genre, instruments, vocal character, language and tempo in one line, e.g. `English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM`. The node's language / key / BPM fields are appended to the style.
- **Lyrics** are required. Use section tags such as `[Verse]` and `[Chorus]`.
- **Covers** don't transcribe words: supply the lyrics to sing, aligned with the source's sections. The cover node's *strength* and *reference audio* inputs are not used.
- Duration is not controllable; song length follows the lyrics and the score.
- **Advanced:** score planning mode (`full` / `melody` / `off`), text guidance, temperature, top-p. Covers also offer **Transpose (semitones)**: the source is pitch-shifted before transcription, so the cover is sung in the new key at the original tempo (e.g. `-3` to suit a lower voice).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the weights (~10 GB, public — no Hugging Face token required) to your Modal `models` volume, deploys the app to your Modal account, and caches the build. Generation runs on an L40S GPU; cover transcription on an L4.
