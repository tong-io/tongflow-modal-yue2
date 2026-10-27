# tongflow-modal-yue2

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Song generation and zero-shot covers with **YuE2** (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`), running on a GPU via [Modal](https://modal.com). Covers transcribe the source recording with **SheetSage2** (`m-a-p/SheetSage2`).

YuE2 first writes an editable melody-and-chord score (ABC), then renders it as a full song with vocals and accompaniment — 48 kHz stereo FLAC. Both nodes can also make **instrumentals**.

> **License:** YuE2 and SheetSage2 weights are **CC BY-NC 4.0**. Since 2026-09-16 the YuE2 licensors also let **individual creators, musicians and personal users** publish, sell and otherwise monetize the music they generate, free of charge; **companies** need a commercial license for commercial use of the weights. Details: upstream [MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE). The upstream code is Apache 2.0; the instrumental score tools are MIT.

## Capabilities

- **Music generation** (`gen-music`) — style prompt + lyrics → full song. Without lyrics → instrumental.
- **Music cover** (`music-cover`) — source audio → SheetSage2 melody score → YuE2 re-performs it in a new style with the lyrics you supply. Without lyrics → instrumental cover (an instrument plays the melody).

### Input tips

- **Style** (prompt / tags): genre, instruments, vocal character, language and tempo in one line, e.g. `English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM`. The node's language / key / BPM fields are appended to the style.
- **Lyrics:** use section tags such as `[Verse]` and `[Chorus]`. Leave them empty for an instrumental.
- **Instrumentals** follow upstream's recipe: YuE2 writes (or SheetSage2 transcribes) the score, every vocal note moves to the instrument voice with pitch and timing unchanged, and the revised score is rendered with `Instrumental, …, no vocals` added to the style. The **Instrumental** advanced switch does the same while keeping your lyrics as a structure guide for the planned score (gen-music); on covers it ignores the lyrics. This rewrites the score rather than removing vocals from audio, so a faint vocal-like timbre can still slip in.
- **Covers** don't transcribe words: supply the lyrics to sing, aligned with the source's sections. The cover node's *strength* and *reference audio* inputs are not used.
- **Chords (covers, `full` source score):** SheetSage2 spells chord names in the local key, so key changes in the source keep correct chord labels.
- Duration is not controllable; song length follows the lyrics and the score.
- **Advanced:** score planning mode (`full` / `melody` / `off`), **Instrumental**, text guidance, temperature, top-p. Covers also offer **Transpose (semitones)**: the source is pitch-shifted before transcription, so the cover is sung in the new key at the original tempo (e.g. `-3` to suit a lower voice; instrumental covers keep the source key unless you set it).
- **Tempo (covers):** the score tempo and the prompt's BPM are kept in sync, as YuE2 expects. Precedence: the **Target BPM** advanced param → a `NNN BPM` written in the prompt → the transcribed source tempo. The chosen tempo is written into both the score (`Q:`) and the prompt, so `Heavy metal, …, 160 BPM` re-tempos a slow source.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the weights (~10 GB, public — no Hugging Face token required) to your Modal `models` volume, deploys the app to your Modal account, and caches the build. Generation runs on an L40S GPU; cover transcription on an L4.
