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

## Community notes and ideas

Collected from the YuE2 community on 2026-10-27. These are **not implemented and not verified by us**, and most numbers are the authors' own. They are kept here as candidates for future plugin features.

**Worth trying first**

- **Instrumental LoRA** — [Mothersuperior/YuE2-instrumental-cot-full-loras](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras) is an AR LoRA (rank 64), trained on about 2.7k instrumentals.
  - A third party ([YuE2-ComfyUI](https://github.com/pytraveler/YuE2-ComfyUI)) measured 17/17 songs with no vocals. Without it, 6/6 had singing.
  - Our score-rewrite instrumentals can still leak vocal-like sounds. Upstream [#208](https://github.com/multimodal-art-projection/YuE/issues/208) reports 3 out of 10.
  - Requirements: `cot=full`, and lyrics that are only section tags. Merge it into the AR weights with `W += B@A`. License: CC BY-NC.
- **Cover CFG default** — upstream defaults to CFG 1.0, and we use 2.0 for covers.
  - We chose 2.0 because at 1.0 our test covers kept the source arrangement.
  - However, any CFG other than 1 doubles the AR compute and disables [YuE2-Turbo](https://github.com/NoizAI/YuE2-Turbo) acceleration.
  - Re-check this by listening to the same cover at 1.0 and 2.0.

**Cheap new parameters**

- **NAR steps** (`ode_steps`, default 32). Turbo and the ComfyUI nodes expose it, and some community examples use 8. The effect of fewer steps on quality is unmeasured.
- **Decoder choice** — options are `m-a-p/YuE2-Vae` (default), `m-a-p/YuE2-Vae-legacy` (scores higher on the Musicality benchmark) and the community [YuE2-Vae-merge-0.666](https://huggingface.co/Mothersuperior/YuE2-Vae-merge-0.666). The encoder is the same in all three, so the decoders are drop-in replacements.
- **Score-level transpose for gen-music.**
  - A key written in the style prompt is ignored: 0 out of 18 attempts changed key.
  - Editing only `K:` doesn't work either, because notes are read relative to the key signature.
  - Shifting every note, chord and key together, then rendering, landed in the target key 14 out of 14 times.
- **Short songs** — max_tokens ≈ seconds × 25 + 250 (pending upstream PR [#218](https://github.com/multimodal-art-projection/YuE/issues/218)). With min_tokens = 200 the minimum is about 8 s. Changing the limit changes the song even with the same seed.

**Faster inference**

- **[YuE2-Turbo](https://github.com/NoizAI/YuE2-Turbo)** runs the AR stage on vLLM.
  - Reported speedup: 1.68× for a single song, 3.31× with 4 in parallel.
  - Quality is reported unchanged.
  - It falls back to plain PyTorch when CFG ≠ 1 or `cot=off`.
- **vllm-omni** has an official [YuE2 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/m-a-p/YuE2-3B.md). Its VRAM use goes up (8.7 GB → 17–20 GB).
- **Upstream FP8 AR** (`--quantization fp8`) works on an L40S. In 0.1.5 and 0.1.6 it turns off CUDA graphs; [#262](https://github.com/multimodal-art-projection/YuE/issues/262) has a patch.
- **EAGLE-3 speculative decoding** — reported only 5–10% faster, so it's not worth it.

**New modes**

- **Continue from a melody** — put a melody's ABC after `[ABC_START]` and leave out `[ABC_END]`. The model then writes the full song, usually repeating that melody as the hook. See [YuE2-hum-to-song](https://huggingface.co/Mothersuperior/YuE2-hum-to-song). This could become a "first N seconds → full song" cover mode.
- **Covers that keep the original singer's voice** — [Ornn8/yue2-cover](https://github.com/Ornn8/yue2-cover) ([#261](https://github.com/multimodal-art-projection/YuE/issues/261)) uses a 30 s vocal clip as a forced prefix.
  - It relies on a community audio→semantic-token encoder ([realaudio-tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)), whose top-1 accuracy is about 16–19%.
  - It's experimental. Upstream has no official reference-audio input.
- **Local edits** — the YuE2-ComfyUI Edit Track node can re-sing a few bars, swap words, move sections or extend the song. Audio outside the edited span is kept sample for sample. Supporting this would need the score, tokens and latents to be saved between runs.

**LoRA ecosystem**

- Upstream ships no training code. For training, [Ostris ai-toolkit](https://github.com/ostris/ai-toolkit) supports YuE2 (PR #1042).
- Hugging Face has about 35 repos based on `m-a-p/YuE2-3B`, 11 of them adapters: genre and language LoRAs, plus [ntc-ai sliders](https://huggingface.co/ntc-ai/yue2-particle-sliders).
- Several LoRAs are trained on named commercial artists, so don't ship any of them as a default.

**Known pitfalls**

- **Lyrics layout** — give one `[Verse]` / `[Chorus]` block per score section. One giant block drifts after the first section.
- **SheetSage2 transcription**
  - It can drift around 50 s, and the ABC uses a single averaged tempo ([SheetSage2 #3](https://huggingface.co/m-a-p/SheetSage2/discussions/3), [#4](https://huggingface.co/m-a-p/SheetSage2/discussions/4)).
  - It has no triplets within a beat and no free rhythm.
  - A long phrase-final note is often split into two equal notes, so the last word is sung twice. Tie the pair on lines with more notes than syllables.
- **Weak controls**
  - BPM is a soft target and often comes out at half or double speed.
  - Keys written in the style are ignored.
  - "A cappella" prompts still get accompaniment.
  - Rock prompts drift toward country.
  - There is no per-section male/female voice.
- **Languages** — officially zh / en / ja / ko / ru / es. Cantonese is weak, and Arabic breaks once an ABC score is attached.
