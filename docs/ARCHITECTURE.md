# Dubbing Platform — Architecture & Build Plan

Status: design draft, 2026-07-16
Scope: video ingest → transcode/proxy/thumbnails → diarize → transcribe → translate → voice-clone → time-fit → optional lip-sync → mix → publish.

---

## 0. Read this first: four findings that change the plan

### 0.1 There is no GPU on this machine

```
lspci      → Intel AlderLake-S GT1 (iGPU only)
nvidia-smi → not installed (exit 127)
/dev/nvidia* → absent
CPU        → i9-12900K · 16 PHYSICAL cores (8 P + 8 E, hybrid) · 24 threads
AVX-512    → ABSENT (fused off on Alder Lake)
152 GB RAM · 125 GB free disk
```

⚠️ **"24 cores" is misleading and the CPU is close to the worst case for this workload.** Three separate problems, all verified on the box:
- It is **16 physical cores, not 24** — 8 performance + 8 efficiency, plus hyperthreading.
- **No AVX-512.** Benchmarks show its absence costs roughly **2×** on Whisper inference. A Xeon 6369P without it manages 3.39× RTF where a Zen5 chip with it hits 7.10×.
- **Hybrid P/E is actively hostile to whisper.cpp**, which splits work statically across threads — so the slowest E-core gates every layer. Measured elsewhere: a hybrid Core Ultra 7 155H ran **3.3× slower than a Ryzen 7 7840U** with half the cores.

**Mitigation, and it's the same fix for all three:** shard the audio and run independent chunk-parallel workers. Chunk parallelism is immune to the hybrid-gating problem (each chunk is independent, so an E-core just finishes its own chunk later) *and* it sidesteps the core-scaling ceiling below. Pin workers to P-cores where latency matters. Prefer **faster-whisper/CTranslate2 over whisper.cpp** here.

Every model in the proposed pipeline is GPU-bound. Concretely, on CPU:

| Stage | CPU-only verdict |
|---|---|
| Separation (Bandit v2) | ⚠️ **Unmeasured — the highest-uncertainty number here.** No CPU RTF is published for Bandit at all. Its paper's 27.7× GPU-over-CPU ratio implies a **multi-hour** CPU job for a 37M-param 48 kHz model with overlapped chunking. Tens of minutes on one GPU. Benchmark before assuming CPU-only works |
| ASR (faster-whisper turbo, int8, chunk-parallel) | Workable — **~40–80 min** per 2-hr film on *this* box (no AVX-512 doubles the ~20–40 min a Zen5 box would see). Low confidence — measure, don't trust it |
| Diarization (pyannote) | Workable — roughly real-time on CPU (senko, MIT: 42s per hour of audio on a 9950X) |
| **Voice cloning TTS** | **Not viable.** XTTS-class models run 30–50× slower than realtime on CPU |
| **Lip-sync** | **Not viable.** MuseTalk needs ~5 min *on a GPU* for 8 seconds of video |

**Implication:** this is not "add a GPU later" — it is the difference between a working product and a demo. The architecture below is therefore built as **queue + pluggable workers**, so CPU stages run here and GPU stages run on a rented GPU host (or Modal/RunPod/Lambda) with no code change. Decide the GPU target before writing worker code, not after.

Minimum viable GPU: 24 GB VRAM (LatentSync 1.6 alone wants 18 GB). A single RTX 4090/A5000-class box covers everything.

### 0.2 Half the models in the proposed pipeline are not commercially licensed

This matters more than usual because the business model is *paying actors for voice rights and selling the output*. Shipping on non-commercial weights while paying for voice rights is an incoherent risk posture — you'd be buying the cheap permission and stealing the expensive one.

**Must be removed from the plan:**

| Proposed | Problem |
|---|---|
| **XTTS-v2** | Code MPL-2.0, **weights are Coqui Public Model License = non-commercial**. Coqui dissolved in Jan 2024, so **no entity exists to sell you a commercial license**. Unfixable. |
| **F5-TTS** | Code MIT, **checkpoints CC-BY-NC-4.0** (trained on Emilia). Fine-tuning does not launder it. |
| **NLLB-200** | **Weights CC-BY-NC-4.0.** |
| **Wav2Lip** | README: *"any form of commercial use is strictly prohibited"* (LRS2 dataset). All HD forks inherit the taint. |
| **Meta Seamless / SeamlessExpressive** | CC-BY-NC / proprietary gated license. |
| **Demucs (all checkpoints)** | Code MIT, **weights are not**. Author adefossez, [issue #327](https://github.com/facebookresearch/demucs/issues/327): *"The model weights are not covered by the MIT license, and are provided only for scientific purposes."* Corroborated by training data — **I verified MUSDB18-HQ's Zenodo license is `other-nc`** ([record 3338373](https://zenodo.org/records/3338373)), which states the material *"should not be used for any commercial purpose."* Applies to htdemucs, htdemucs_ft, hdemucs_mmi, mdx, mdx_extra. |
| **Meta denoiser** (facebookresearch/denoiser) | LICENSE line 1 is `Attribution-NonCommercial 4.0 International`. |

**Also excluded** (transitive taints that catch people): InsightFace pretrained packs (non-commercial — taints LatentSync's default inference path, Hallo/2/3, LivePortrait), **CodeFormer** (S-Lab License 1.0, non-commercial — and it's the *default* tool people reach for to clean up lip-sync output), GPEN (no license file at all → all rights reserved; its author took it down "due to commercial issues"), unlicensed Roformer community checkpoints.

⚠️ **GFPGAN is not safe either, despite the Apache-2.0 badge.** Its own LICENSE reads: *"GFPGAN is licensed under the Apache License Version 2.0 **except for the third-party components listed below**"* — those being **StyleGAN2** (NVIDIA: *"only may be used or intended for use non-commercially"*) and **DFDNet** (CC BY-NC-SA 4.0). This is the single most-missed taint in the whole stack, because every lip-sync README tells you to reach for it — MuseTalk's own docs recommend it by name. Use **RestoreFormer++** (Apache-2.0, verified clean) instead, with **Real-ESRGAN** (BSD-3) for general upscaling.

**A HuggingFace `license:` tag is an uploader's claim, not a verified fact.** The three Hallo repos demonstrably ship non-commercial InsightFace weights under an MIT tag. Verify every checkpoint against its actual LICENSE file and its training data.

### 0.3 "Train a voice model per speaker" is the wrong framing

The plan assumes per-speaker training. In 2026 the state of the art is **zero-shot cloning from a ~10-second reference clip** — no training run, no per-speaker GPU job, no model registry.

This is good news: it deletes an entire subsystem. "Manage Voices" is not a training pipeline, it is a **curated reference-clip library**. The work that actually determines quality is clip *curation* (clean, dry, no music, no reverb, emotionally representative, correct language), not gradient descent.

Fine-tuning stays on the roadmap as an escape hatch for the small number of voices where zero-shot underperforms — not as the default path.

### 0.4 DTW is not how you fix duration

> **Measured on real film, 2026-07-19.** Before tuning any of the four levers, check for
> **runaway generations**. Chatterbox failed to stop on 7 of 58 takes — a 1-syllable line producing
> 5.4s, "Can't see." producing 11s on one run and 2.3s on the next — accounting for **24% of all dub
> audio**. That is not a duration problem to compress away; it is a sampling failure to detect and
> retry. Rejecting them moved the average from 1.22x over to 0.83x, more than every other lever
> combined, and it had been hiding the effect of all of them.
>
> **Measured on real film, 2026-07-17.** Speech duration is **affine, not a rate**:
> `duration_ms ≈ 510 + 421 × syllables`, fitted over 58 real takes. Every utterance has a fixed
> ~510ms cost and only the remainder scales. A rate model is 1.4x less accurate overall and badly
> wrong below ~4 syllables, where most film dialogue lives — which means some slots are physically
> unfittable and no amount of rewriting helps. Say so rather than asking for another edit.
> Also: only OVERRUN is a defect. Under-length is free (lever 4 spends the surrounding silence).
>
> **Measured on real film, 2026-07-17.** Lever 2 (TTS speaking rate) is not a free parameter —
> it is mostly decided by the REFERENCE CLIP, whose pacing transfers to the clone. And the engine
> is slower than the actor it replaces: Chatterbox ~4.6 syl/s vs Hindi actors 7.26. A dub is
> therefore ~1.6x its slot *before* anyone chooses a word, which loads almost all the work onto
> lever 1 (translation length). Budget target syllables from the ENGINE's rate, never the source's.

Dynamic Time Warping *aligns* two sequences; it does not make cloned audio fit a slot. Time-fitting is a four-lever budget, applied in this order:

1. **Constrain the translation length** — the cheapest lever by far, and the only one with no audio cost.
2. **Control TTS speaking rate** at generation time.
3. **Time-stretch with WSOLA/rubberband**, hard-capped at ±10–12%. Past ~15% it audibly chipmunks.
4. **Absorb the remaining slack into inter-phrase pauses** (the original silence is yours to spend).

If a line still doesn't fit after all four, it goes to a human queue. That's a product feature, not a failure.

---

## 1. Corrected pipeline

The proposed order has two structural problems: it runs separation and diarization in parallel, and it diarizes *then* transcribes per-segment. Both cost quality.

```
[ Ingest ]
    │
    ├─► Probe (ffprobe) ─► Transcode: proxy H.264 720p + thumbnails + sprite sheet
    ├─► Shot detection (PySceneDetect) ────────────► shot list
    │
    ▼
[ 1. Separation FIRST ]
    │   Bandit-v2 (cinematic D/M/E) — or Demucs fallback
    │   ⚡ If source is 5.1: try centre-channel dialogue extraction first (free, better)
    ├─► DIALOGUE stem ──────────► downstream
    └─► MUSIC + FX stem ────────► held for final mix (never touched again)
    │
    ▼
[ 2. Diarization on the CLEAN dialogue stem ]     ◄── why: pyannote on music-bed audio is
    │   pyannote community-1                           markedly worse. Separation first is free accuracy.
    └─► speaker turns (overlap-aware, unlimited speakers)
    │
    ▼
[ 3. ASR over the WHOLE dialogue stem, not per segment ]
    │   IndicConformer-600M (Hindi) or faster-whisper turbo int8 — chunk-parallel
    │   + MFA or NeMo NFA forced alignment → word-level timestamps
    │   ⚠️ do NOT use WhisperX's default Hindi aligner — see §2
    └─► words with timestamps ──► INTERSECT with diarization → speaker-attributed lines
    │      ⚡ why: Whisper's quality comes from long context. Feeding it 3-second
    │            diarized chunks throws that away and fragments sentences.
    ▼
[ 4. Speaker identification & merge ]  ── human-in-the-loop UI
    │   voice embeddings cluster SPEAKER_00… across the film
    │   user names them, merges duplicates, assigns a voice from the Voice Library
    │   + Active-speaker detection ties each speaker to an on-screen FACE  ◄── missing from
    ▼                                                                          original plan;
[ 5. Transcript review ]  ── human-in-the-loop UI                              required for lip-sync
    │   fix ASR errors before they propagate. Confidence-scored, worst-first.
    ▼
[ 6. Translation — length-constrained, context-aware ]
    │   IndicTrans2 (MIT) backbone  +  LLM pass for context/register/character voice
    │   Target: DURATION budget per line, not character count
    ▼
[ 7. Translation review ]  ── human-in-the-loop UI
    ▼
[ 8. TTS / voice cloning ]
    │   Chatterbox Multilingual (MIT weights) ← primary
    │   reference clip = the assigned voice from the Voice Library
    ▼
[ 9. Time-fit ]  (the 4-lever budget above — NOT DTW)
    ▼
[ 10. Lip-sync — OPTIONAL, PER SHOT ]  ◄── see §4. Most of a film does not need this.
    │   LatentSync 1.6 (Apache-2.0) on close-up on-screen dialogue only
    ▼
[ 11. Mix ]  ffmpeg: cloned dialogue + untouched M&E stem + video
    ▼
[ 12. QC & publish ]
```

### Why separation before diarization
Diarization models degrade on music beds. Running Bandit first hands pyannote a clean dialogue-only signal. This is a free accuracy win and costs nothing — you were running separation anyway. It also protects the aligner: forced alignment is *less* robust in noise than Whisper's native DTW (§2).

### ⚠️ Set your expectations from real film, not the DnR test set
Bandit's headline **15.7 dB dialogue is on the synthetic DnR test set. It will not reproduce on your footage.** The SDX23 cinematic challenge is the only one ever run on real film (CDXDB23 = 11 real Sony Pictures movies), and the gap is the whole story:

| Training data | Best dialogue SDR on real movies |
|---|---|
| DnR-only (Leaderboard A) | **7.98 dB** |
| Any data, incl. real film (Leaderboard B) | **14.62 dB** |

**~6.6 dB.** That gap is where the quality actually lives, and it's why AudioShake is ahead commercially. Budget for fine-tuning on real film audio — and note that doing so reopens the ShareAlike question on *your* weights (§7).

### The sung-vocals problem has no clean answer yet
Your instinct about background preservation is right, and it's worse than you framed it. A character who sings, a diegetic radio, a musical number — a music-trained separator rips all of them into the dialogue stem. DnR v3's abstract names *"vocal content in non-dialogue stems"* as a defect it exists to fix, and "Facing the Music" (ISMIR 2024) extends the architecture to a 4-stem split with singing voice as its own class. But **standard 3-stem CASS has no answer for a character who sings** — plan a manual override path for musical scenes.

### Why one long ASR pass, not per-segment
Whisper is a context-hungry sequence model. Its accuracy on a 3-second isolated chunk is dramatically worse than the same 3 seconds inside a paragraph. Transcribe the whole dialogue stem, get word-level timestamps via WhisperX forced alignment, then assign each word to a speaker by overlapping it with the diarization timeline. This also fixes sentences that straddle a speaker change.

---

## 2. The model stack (all commercially licensed)

| Stage | Model | License (code / **weights**) | Notes |
|---|---|---|---|
| Probe/transcode/mix | **ffmpeg** | LGPL/GPL build-dependent | Use an LGPL build if you ever link it; CLI invocation is fine |
| Shot detection | **PySceneDetect** | BSD-3 | Drives per-shot lip-sync decisions |
| Separation (**only** viable option) | **Bandit-v2**, `checkpoint-multi.ckpt` | Apache-2.0 / **CC BY-SA 4.0** ([Zenodo 12701995](https://zenodo.org/records/12701995)) | Purpose-built *cinematic* D/M/E separation (Netflix-affiliated). Beats Hybrid Demucs by **+2.1 dB dialogue at half the parameters**; its DX (15.7) even exceeds the ideal ratio mask (14.4). **`checkpoint-multi` is not optional for dubbing** — monolingual checkpoints collapse cross-lingually (ENG-trained on Faroese: 2.5 dB). fp32 only; mixed precision was unstable. ⚠️ ShareAlike — §7 |
| Separation (fallback) | **— none —** | | 🔴 **The Demucs fallback is gone** (§0.2). Bandit v1 weights are CC BY-NC; Banquet is CC BY-NC-SA; every ZFTurbo/UVR/viperx checkpoint is unlicensed or worse. If Bandit v2 fails legal review, the honest answer is **license AudioShake's D/M/E API**, not find another checkpoint |
| Dataset | **DnR v3** | **CC BY-SA 4.0** | 48 kHz/24-bit, 30+ langs incl. Indo-Aryan/Dravidian. **DnR v1/v2 are NC-contaminated** — their `cc-by-4.0` tag is an aggregator's claim over FSD50K (6,041 NC clips) and FMA (~88% NC). v3 exists to fix exactly this; its license wiki has zero NC entries. ⚠️ The Zenodo **training** set is corrupt — its own title says so. Use the HF or Netflix mirrors |
| Speaker embeddings | **WeSpeaker / ReDimNet — vox2-only weights** | Apache/MIT | ⚠️ **Use vox2 variants only.** The best-EER `vb2` checkpoints are trained on VoxBlink2, which states *"no commercial application is allowed"* (CC BY-NC-SA) — this silently taints them despite MIT/Apache repo files. Alternative: TitaNet (CC-BY-4.0, 0.66% EER) |
| ASR (Hindi) | **IndicConformer-600M** | **MIT** | 22 Indian languages, Hindi WER 13.2. **CTC decoding → natively alignment-friendly**, unlike Whisper. On HF, 82k downloads. The cleanest Indic pick |
| ASR (Hindi, best WER) | **IndicWhisper** | **MIT** | Best published Hindi WER (13.6 avg Vistaar; wins 39/59 benchmarks, −4.1 WER). ⚠️ **`ai4bharat/indicwhisper` does not exist on HF** — it ships from the [vistaar](https://github.com/AI4Bharat/vistaar) object store. HF hits are third-party mirrors; vet provenance individually |
| ASR (multilingual) | **faster-whisper large-v3-turbo int8** | **MIT** | Turbo = 4 decoder layers vs 32. ⚠️ **Its ~3× speedup does NOT transfer to CPU** — the encoder is unchanged from large-v3 and the encoder dominates on CPU. ⚠️ Never use its `task=translate` path: turbo was fine-tuned *excluding* translation data. The flag still runs; the output is degraded |
| Alignment | **MFA** (Montreal Forced Aligner) | MIT | **21.9 ms mean word-boundary error vs WhisperX's 34.3 ms.** At 24fps a frame is ~42ms — sub-50ms is exactly where lip-sync lives. Heavy Kaldi dep, needs pronunciation dictionaries |
| Alignment (alt) | **NeMo NFA** | Apache-2.0 / CC-BY-4.0 | CTC-native → **pairs directly with IndicConformer.** The cleanest pairing if you go Indic-first |
| Diarization (default) | **DIY: Silero VAD + WeSpeaker + clustering** | MIT / **CC-BY-4.0** / BSD | **Ungated — no account, no token.** Verified: 2/2 speakers correct on ground-truth audio, ~6× realtime CPU, runs offline. ⚠️ Clustering → **no overlap handling**, which films need. ~20-25 DER vs pyannote's 17. Built because independence was worth more than the DER here |
| Diarization (upgrade) | **pyannote community-1** | **CC-BY-4.0** (gated) | Overlap-aware, **unlimited speakers**, beats 3.1 by 1–4 DER. The gate is lead-capture, auto-approved — *not* a commercial license gate; no paid plan required. ⚠️ CC-BY carries an **attribution obligation** that MIT doesn't — if you want zero obligations, **3.1 is MIT** and only modestly worse |
| Translation | **IndicTrans2** | **MIT** (models + code) | 22 Indian languages. Has IN22-Conv, a *conversational* benchmark |
| Translation (context) | **LLM** (Gemini/Groq per your plan) | API | Context, register, character voice, length budget |
| TTS / cloning | **Chatterbox Multilingual** | **MIT / MIT** | 23 langs incl. Hindi. ~10s reference. Dedicated MIT Hindi pack exists |
| TTS (Indic-only alt) | **IndicF5** | **MIT** (gated) | 11 Indic langs, excellent — but **no English**. ⚠️ provenance caveat, §7 |
| Lip-sync (lead) | **X-Dub** (Kling/Kuaishou + Tsinghua, ICML 2026) | **Apache-2.0 / Apache-2.0** | Weights released 2026-03-19. **Mask-free v2v visual dubbing** — auto-crops the face, dubs at 512×512, maps back. The only model that is both genuinely permissive *and* built to edit existing footage. ~21 GB. ⚠️ **single-person only**; authors are candid that the public Wan2.2-TI2V-5B release is weaker than their internal one — flickering, identity/colour drift, *"severe noisy frames in ~2% of cases"*, ~2× slower. Verify that 2% against your QC bar |
| Lip-sync (new) | **Lip Forcing** (KAIST) | **Apache-2.0 / Apache-2.0** | **Released 2026-07-07 — nine days old.** First autoregressive-diffusion v2v lip-sync; 2-step, no CFG, real-time streaming, 512×512. ⚠️ **~37 GB VRAM** (14B) and far too new to have a track record. Watch it; don't bet Phase 5 on it yet |
| Lip-sync (alt) | **InfiniteTalk** | **Apache-2.0 / Apache-2.0** | Purpose-built for *sparse-frame video dubbing*, native 720p, syncs head/body/expression too. ⚠️ **Regenerates every pixel** — README concedes it *"mimics the original video's camera movement, though not identically"* and SDEdit *"introduces color shift."* For a hero plate, that's a bigger risk than a mouth-only inpaint |
| Lip-sync (alt) | **LatentSync 1.6** | Apache-2.0 / **openrail++** | 512×512, 18 GB. 🔴 Inference auto-downloads InsightFace `buffalo_l` (NC). Removable — v1.0/1.5 used face-alignment+mediapipe; InsightFace only added 2025-04-11 (`354310f`) |
| Lip-sync (light) | **MuseTalk** | **MIT / "any purpose, even commercially"** | 256×256, 4 GB, real-time. No InsightFace. ⚠️ HF card **contradicts itself** — YAML says `creativeml-openrail-m`, body says commercial-OK. Get written clarification from Tencent |
| Face restore | **RestoreFormer++** | **Apache-2.0** | The clean path. **Never GFPGAN/CodeFormer/GPEN** — see §0.2 |
| Upscale | **Real-ESRGAN** | BSD-3 | General upscaler, not a face restorer |
| Face detect | **MediaPipe** | Apache-2.0 | Clean replacement for InsightFace. Avoid YOLOv8 (AGPL-3.0) |

### The second place where buying may beat building
**AudioShake's Dialogue/Music/Effects API** is purpose-built for exactly this and explicitly targets localization/dubbing. It's the commercial state of the art, it sidesteps the entire ShareAlike question, and per SDX23 the real-data-trained systems are ~6.6 dB ahead of anything you can train on open data. Negotiate **output ownership and redistribution rights explicitly** — that's a contract question, not a license-file one. Same for mvsep.com, whose terms do grant commercial use of outputs.

⚠️ **The catch is confidentiality, not licensing:** both route pre-release film audio through a third party. For a studio client that may be disqualifying on its own — which is an argument for self-hosting Bandit even at a quality cost.

### One place where buying beats building outright
**pyannoteAI's hosted precision-2 costs ~€0.22 for a 2-hour film** and cuts DER by ~40% vs self-hosted community-1 (DIHARD3: 14.7 vs 20.2). That is far cheaper than the engineering to close the gap, and diarization errors are the ones that propagate worst — a wrong speaker boundary means the wrong actor's voice says the line. Self-host community-1 as the default; offer precision-2 as a paid quality tier. It's an API call, so keep it behind the same worker interface.

### Nothing here is benchmarked on film
AMI/AliMeeting are meetings; DIHARD-3 and VoxConverse are the closest proxies you have. Cinematic music beds, reverb, and whispered/shouted delivery **will break VAD before they break the embeddings** — which is exactly why separation runs first (§1). Expect to tune, and expect the published DER numbers to be optimistic for your content.

### 🔴 The biggest hidden trap in the obvious stack: WhisperX's Hindi aligner
Everyone reaches for WhisperX. Its code is clean (BSD-2). Its **default Hindi alignment model is not.**

`DEFAULT_ALIGN_MODELS_HF['hi']` → [`theainerd/Wav2Vec2-large-xlsr-hindi`](https://huggingface.co/theainerd/Wav2Vec2-large-xlsr-hindi), and it fails three ways at once:
- **No license field at all** on HF — not permissive, *undetermined*. 1.13M downloads and legally unclear.
- **72.62% WER** on Common Voice Hindi.
- It is a **character-based CTC model, not a phoneme model** — contrary to WhisperX's own design assumption.

Only en/fr/de/es/it get proper torchaudio models; the other 31 languages are community uploads of wildly varying quality and licensing. **Tamil, Bengali and Marathi have no entry at all.**

Two more WhisperX limits that specifically bite dubbing:
- **Numbers and currency get no timestamp, silently.** Per its own README, words without characters in the alignment dictionary — `"2014."`, `"£13.60"` — *"cannot be aligned and therefore are not given a timing."* Whisper emits non-normalized text. Those are exactly the tokens a dub must time correctly.
- **Forced alignment is *less* robust in noise than Whisper's native DTW.** Under SNR 1:5, WhisperX collapses to 59.0 F1 — *below* vanilla DTW's 68.3 — while CrisperWhisper holds 79.5. For film audio full of music and effects, that's the wrong direction. (Another reason separation runs first, §1.)

**Use MFA or NeMo NFA.** Also avoid `torchaudio`'s `MMS_FA` and `ctc-forced-aligner`'s defaults — both ship **CC-BY-NC** models; ctc-forced-aligner's own docs say to *"use a different model for commercial usage."*

### ⚠️ Chunk-parallel ASR: cut in SILENCE or don't cut

Implemented — and the first attempt made accuracy *worse*, which is worth recording.

Sharding at **shot boundaries** seemed obvious (a picture cut is probably a pause). It isn't. Measured
on real audio, cutting at shot/arbitrary times truncated a line mid-word — `"She sells seashells by
the seashore every morning"` came back as `"She's held sea-shelled by the seashore every m-"` — and
corrupted another (`"The quick"` → `"A quick"`). A picture cut is not guaranteed to be a pause in the
dialogue, and Whisper hallucinates around abrupt starts.

The fix: detect silences (`ffmpeg silencedetect`) and cut only at their midpoints, preferring one
near the target length. Shot boundaries are the fallback; if no safe cut exists nearby, the chunk is
left long. **A slow transcript is recoverable; a corrupted one is not.**

Residual, honestly: the first word of each chunk is still slightly degraded (abrupt start). At the
300s default that's ~1 word per 5 minutes. Do not lower `ASR_CHUNK_SECONDS` far without re-measuring.

### Threading: the config that actually matters
CTranslate2's own guidance, verified in source: *"Avoid the total number of threads `inter_threads * intra_threads` to be larger than the number of physical cores"* and *"if you are processing a large volume of data, prefer increasing `inter_threads` over `intra_threads`."*

On this box that means **`num_workers=4, cpu_threads=4`** (16 physical cores), **not `cpu_threads=24`**.

⚠️ **The catch that makes it a no-op:** `num_workers` only helps when `transcribe()` is called from multiple Python threads. **A single `transcribe()` call on a 2-hour film gets zero benefit.** You must chunk the audio yourself and dispatch from a thread pool. A film is embarrassingly parallel at scene boundaries — and you already have the shot list from Phase 1. **This is likely worth more than the model choice.**

### On the Hindi ASR number to plan around
Vistaar benchmark, Hindi WER: IndicWhisper 13.6 avg — but **26.8 on Gramvaani** (noisy, spontaneous telephone speech). Gramvaani is the closest proxy you have to real film dialogue. Plan your human-review budget around ~25% WER, not ~13%.

**Nobody benchmarks Hindi.** The Open ASR Leaderboard is English-only in its main track and de/fr/it/es/pt in its multilingual track — **Hindi appears nowhere.** No rigorous public comparison exists between large-v3, Voxtral, Qwen3-ASR and the Indic models. The Vistaar numbers above are from 2023. **If Hindi accuracy is decision-critical — and it is — a small in-house Hindi eval set is the only way to rank these.** Build it in Phase 2.

### Excluded on language, not licence
NVIDIA **Parakeet** and **Canary** are genuinely clean (CC-BY-4.0, NeMo is Apache-2.0) and top the English leaderboards — and **none of them support Hindi**. Parakeet v3's "multilingual" is 25 *European* languages. Their license quality is a red herring for you. Same for Granite Speech, Kyutai STT, Moonshine. **Qwen3-ASR** is Apache-2.0 and does Hindi, but its timestamps come from `Qwen3-ForcedAligner-0.6B`, which covers 11 languages — **Hindi is not one.** Transcription yes, alignment no.

### On the watermark
Every Chatterbox generation is stamped with Resemble's **Perth watermark** before audio leaves the model — imperceptible, survives MP3 re-encode. MIT technically permits stripping it. **Don't.** For a product whose entire pitch is *licensed, consented voice*, a durable provenance mark is an asset, not a tax. Make it a selling point in the actor contract.

---

## 3. Voice Library (replaces "train voice model")

### Data model
```
Voice
 ├─ id, display_name, owner/actor identity
 ├─ consent_record  ── signed agreement, scope, expiry, territories, permitted use
 ├─ status: draft | pending_consent | active | suspended | expired
 ├─ languages[]     ── which languages this voice is validated for
 └─ ReferenceClip[]
      ├─ audio (WAV 24k mono), transcript, duration
      ├─ emotion tag: neutral | angry | soft | shouting | laughing
      ├─ QC: SNR, reverb estimate, clipping, music-bleed flag
      └─ is_default_for(language, emotion)
```

### Ingest flow for a voice
1. Upload raw audio/video of the actor.
2. **Auto-run separation** to strip any music/FX bleed.
3. Auto-segment into candidate clips; run VAD; score each for SNR/reverb/clipping.
4. Present a **clip picker**: user auditions and keeps 3–10 good clips per emotion.
5. **Bake-off**: synthesize the same test sentence from each candidate clip, let the user A/B them. Pick the default.
6. Mark active only when a consent record is attached.

Step 5 is the highest-leverage thing in this whole subsystem. Zero-shot clone quality varies wildly by reference clip in ways nobody can predict by looking at a waveform. Let the user hear it.

### Consent & rights (do not skip this)
Your differentiator is *"pay them for their voice"* — so the consent record is the product, not paperwork.

Note the specific risk in the examples used (Anil Kapoor, Shah Rukh Khan): **Anil Kapoor won a personality-rights injunction in the Delhi High Court in 2023** covering his voice specifically, and Amitabh Bachchan won a comparable order. Indian courts have actively enforced voice as a protectable personality right. Those are the two names least available to clone without a signed agreement.

Required per voice: signed scope (which titles/territories/languages/duration), expiry, revocation path, per-use logging (which project, how many minutes, when), and a kill switch that suspends a voice across all in-flight projects.

---

## 4. Lip-sync: make it optional and per-shot

The plan treats lip-sync as a mandatory stage. Reconsider — this is where cost, risk, and quality all concentrate:

- **Netflix-grade human dubbing does not lip-sync.** Hundreds of millions of viewers accept dubbed content with no mouth manipulation. The bar is *isochrony and performance*, not visemes.
- **Most of the 2025–26 "talking head" SOTA cannot dub at all.** This is the trap that wastes the most time, and two independent research passes reached it separately. Two axes decide a model, and conflating them is fatal: **(a) can it edit existing footage?** and **(b) is it license-clean?** Dubbing needs **video-to-video** editing of your hero plate. But EchoMimicV3, LongCat-Video-Avatar, SadTalker, Hallo/2/3, LivePortrait, MultiTalk, Ditto, Wan2.2-S2V, FantasyTalking, HunyuanVideo-Avatar, SkyReels, Sonic, FLOAT, MEMO and JoyVASA are all **image-to-video** — they generate a new performance from a still and *cannot touch your film*, no matter how permissive the license. **Only five v2v models exist**: X-Dub, InfiniteTalk, LatentSync, MuseTalk, and VideoRetalking (the last is out — it hard-wires GPEN + GFPGAN into `main()` with no flag to disable).
- **Two marketing traps in that list**, and both were caught only by reading the argparse rather than the README. SkyReels-V3 advertises "video-to-video," but that's a *separate* model that **continues** a video with new scenes; its Talking Avatar takes `--input_image`. LongCat-Video-Avatar 1.5 has genuinely appealing MIT weights (code *and* weights) but is `choices=['ai2v','at2v']` — **no v2v option** — and its "Video Continuation" autoregressively extends *the model's own generated output*, not your footage. Appealing license, wrong capability.
- **Don't plan around the API-only names.** OmniHuman is service-only (`github.com/bytedance/OmniHuman` is a 404). EMO/EMO2 has no code or weights — the repo is a README and two media files, last touched Aug 2024. SyncAnyone is a strong v2v paper with no repo. SkyReels-A3 was never released.
- **Resolution and licensing are the same problem, not two.** Everything renders below 1080p (96 / 256 / 512 / 720 px), so a cinematic close-up — where the face is 600–900 px tall — needs restoration to composite back cleanly. And the standard restorers are exactly the tainted layer (§0.2). InfiniteTalk's native 720p matters mostly because it *minimizes how much restoration you need*.
- **It is the most expensive stage by an order of magnitude.**
- Market pricing confirms the split: audio-only dubbing runs ~$0.33–0.60/min (ElevenLabs), lip-sync-inclusive runs ~$1.50–3.00/min (Rask/HeyGen). ElevenLabs — the volume leader — **doesn't offer lip-sync at all.**

**Recommendation:** ship audio-only dubbing first. Then add lip-sync as a **per-shot opt-in**, driven by shot detection + active-speaker detection + face size:

```
for each shot:
    if speaker is off-screen           → skip (also relaxes isochrony — see Amazon's off-screen PA paper)
    elif face bbox < ~8% of frame      → skip (nobody can tell)
    elif shot is a close-up            → lip-sync candidate → queue → human A/B approve
```

This turns lip-sync from a 100%-of-runtime cost into roughly a 10–20%-of-runtime cost applied only where it's perceptible, and it gives the operator a veto on every regenerated face.

**Missing prerequisite:** you need **active-speaker detection** to know which on-screen face belongs to which diarized speaker. Without it you cannot lip-sync a two-shot correctly. (TalkNet-ASD / Light-ASD are the usual choices — *verify their licenses before adopting; I have not.*)

**Evaluation order when you get here:**
1. **X-Dub** — the only v2v dubber that is Apache-2.0 on code *and* weights with no transitive taint. Mask-free editing preserves the plate better than full regeneration. Gate it on the ~2% noisy-frame rate and the single-person limit (a two-shot needs a different path).
2. **LatentSync 1.6** — the proven fallback, best face-crop fidelity. Budget the InsightFace removal and the OpenRAIL++ flow-down.
3. **MuseTalk 1.5** — where 4 GB and real-time matter more than resolution.
4. **InfiniteTalk** — only if you want head/body sync *and* can tolerate camera drift and colour shift.

**No model dubs at native 1080p in mid-2026.** The ceiling is a 512×512 face crop composited back into full-res footage (InfiniteTalk's 720p is full-frame regeneration, which is a different trade, not a better one). LatentSync 1.6 (June 2025) is still the latest — there is no LatentSync 2. Note the VRAM cliff: **1.5 runs in 8 GB, 1.6 needs 18 GB** — if budget forces one GPU, 1.5 is the pragmatic pick. Both MuseTalk and LatentSync are stale, so budget for maintaining a fork.

**This section has been rewritten three times in one day and will age fastest of anything here.** The 2026 literature pivoted hard to mask-free v2v dubbing; X-Dub shipped in March, **Lip Forcing shipped nine days ago**, and JUST-DUB-IT shipped code and weights but gated behind the restrictive LTX-2 Community License (not OSI-open — don't let the public repo fool you). Still paper-only: UniSync, SyncAnyone, StableDub. HighSync is v2v but has **no LICENSE file anywhere** → all rights reserved. Recheck before Phase 5 starts, not now.

Useful reading: [Talking Head Generation survey](https://arxiv.org/abs/2507.02900) (still the reference — no 2026 successor exists) and [Multimodal LLM-Enabled Video Translation: A Role-Oriented Survey](https://arxiv.org/abs/2604.11283) (Apr 2026 — the closest thing to a survey of your exact pipeline shape).

---

## 5. Translation that fits the mouth

Research consensus, and it contradicts two choices in the plan:

**Google Translate API is the wrong tool.** It has no document context, no length control, no register control. Dubbing needs all three.

**Character count is the wrong length target.** VideoDubber (AAAI 2023) is unambiguous: the same character count maps to different durations across languages — and Devanagari character counts are a *particularly* poor duration proxy against Latin source text. Target **estimated duration**, not chars.

**Hybrid design:**
```
LLM pass          → document-level context, character register, prior lines,
                    honors an explicit duration budget in-prompt
IndicTrans2 pass  → En→Hi quality backbone + license-clean self-hosting
Length pass       → constrained rewrite targeting predicted duration
```

The direction of travel matters: IndicGenBench (Google Research, 29 Indic langs) shows LLMs are far better **Indic→English** than **English→Indic** (GPT-4: 32.1 vs 54.5 char-F1). If the product dubs English→Hindi, that is the LLM's *weak* direction and IndicTrans2's *strong* one. Don't let the LLM be the only translator on the En→Hi path.

Also worth knowing before you tune tolerances: Amazon's *Dubbing in Practice* study found **human dubbers are less strict about isochrony than engineers assume**. Don't over-constrain and wreck the writing.

Key papers: [Isometric MT](https://arxiv.org/abs/2112.08682) · [Jointly Optimizing Translations and Speech Timing](https://arxiv.org/abs/2302.12979) · [VideoDubber](https://arxiv.org/html/2211.16934v2) · [Dubbing in Practice](https://arxiv.org/pdf/2212.12137) · [Cross-Lingual Dubbing of Lecture Videos into Indian Languages](https://arxiv.org/abs/2211.01338) (IIT-Madras; the only Indic dubbing pipeline paper — reports 75% human-effort reduction, MOS 4.09)

---

## 6. System architecture

### The core product insight

**This is not a fire-and-forget pipeline. It is a review workstation.**

Every automated stage is ~85% right. On a 2-hour film that leaves hundreds of errors, and the compounding is brutal: an ASR error becomes a translation error becomes a wrong performance in a cloned voice. The IIT-Madras paper's headline result is *75% human effort reduction* — not elimination. Every credible player in this space is a human-in-the-loop tool.

So the requirement "usable without a developer" resolves to: **each stage emits a reviewable, editable artifact, and the operator approves it before the next stage spends money.** Gates go where errors are cheapest to fix — after ASR, after speaker ID, after translation, after TTS.

### Services

```
┌──────────────┐     ┌──────────────┐
│  web (Next)  │────►│  api (FastAPI)│
└──────────────┘     └───────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌─────────┐   ┌───────────┐  ┌──────────┐
        │ postgres│   │   redis   │  │  minio   │
        │ (state) │   │  (queue)  │  │ (objects)│
        └─────────┘   └─────┬─────┘  └──────────┘
                            │
      ┌──────────┬──────────┼──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼
 ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
 │ media  │ │separate│ │  asr   │ │  tts   │ │lipsync │
 │(ffmpeg)│ │(bandit)│ │(whisper│ │(chatter│ │(latent │
 │  CPU   │ │  CPU   │ │ +pyann)│ │  box)  │ │ sync)  │
 │        │ │        │ │  CPU   │ │  GPU   │ │  GPU   │
 └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**One container per model family — this is not optional.** Whisper, pyannote, Bandit, Chatterbox, and LatentSync want mutually incompatible torch/CUDA/numpy versions. A single "ai-worker" image will consume weeks in dependency hell. Separate images, shared object store, talk over the queue.

**Verified over Tailscale, 2026-07-19.** A worker needs only the queue, the object store and the
database — no public IP, no inbound port, no reverse proxy. A VM with only a `100.x` address ran a
full transcode remotely. Backing services bind to the Tailscale address alone, never `0.0.0.0`.
⚠️ Exposing the broker means authenticating it: an open Celery broker is remote code execution, not
just a data leak.

**GPU workers are the same containers with a different runtime.** Tag jobs by capability (`cpu` / `gpu`) and let GPU workers subscribe from wherever they live. That's what makes the no-GPU-here problem survivable.

### Storage layout
```
s3://media/{project_id}/
  source/master.mkv
  proxy/proxy_720p.mp4          # H.264 CRF 23 +faststart — the UI never touches the master
  thumbs/{poster.jpg, sprite.jpg, sprite.vtt}
  stems/{dialogue.wav, music_fx.wav}
  segments/{segment_id}/{original.wav, tts_take_01.wav, ...}
  renders/{version}/final.mp4
```

Model weights go in a **named volume**, not baked into images (multi-GB layers, and gated HF repos need runtime tokens anyway).

### Data model (sketch)
```
Project(id, title, source_lang, target_langs[], status)
Asset(id, project_id, kind, uri, probe_json)
Job(id, project_id, stage, state, capability, attempts, error, worker_id)
Speaker(id, project_id, label, display_name, voice_id→Voice, merged_into→Speaker)
Segment(id, project_id, speaker_id, t_start, t_end,
        text_src, text_src_edited, asr_confidence,
        text_tgt, text_tgt_edited, duration_budget_ms,
        state: draft|approved|needs_review)
Take(id, segment_id, uri, tts_params, stretch_ratio, chosen: bool)
Voice(id, ...)  ReferenceClip(id, voice_id, ...)  ConsentRecord(id, voice_id, ...)
Shot(id, project_id, t_start, t_end, is_closeup, speaker_face_id, lipsync_state)
```

`text_src` vs `text_src_edited` matters: **never destroy the machine output.** You need it to measure model quality and to re-run a stage without losing human edits.

### Queue
Celery+Redis is fine and you clearly know it. But note the workload: jobs are **long** (30+ min), **expensive to retry**, and **must survive a worker restart**. Whatever you pick, make jobs resumable and idempotent, keyed by content hash — re-running TTS on 900 segments because a worker OOM'd on segment 847 is the failure mode that will actually hurt.

---

## 7. Open legal questions (get counsel before shipping)

1. **Bandit-v2 weights are CC BY-SA 4.0, and this is now load-bearing** — because the Demucs fallback is gone (§0.2), Bandit v2 is the *only* commercially-usable cinematic separator that exists. ShareAlike's reach into (a) fine-tuned derivatives and (b) *output audio* is unsettled. Output is very likely fine — ShareAlike binds adaptations of the work, and separated audio is arguably not an adaptation of the model. Fine-tuned weights are the real question, **and the author's own conduct is evidence for the conservative read: Watcharasupat released Bandit v2's weights as CC BY-SA precisely because they were trained on CC BY-SA data.** That is him treating ShareAlike as propagating through training. Since §1 says you *will* fine-tune on real film, this is not hypothetical. If it fails review, the fallback is **AudioShake's API**, not another checkpoint.
   🚩 **Pin your copy of Bandit v2 now.** The author's successor project [`banda`](https://github.com/kwatcharasupat/banda) — into which he states all model repos are being reworked — is **AGPL-3.0 / commercial dual-license**. Bandit v2 may be the most permissive snapshot this lineage ever produces.
2. **IndicF5** claims MIT and trains on CC-BY-4.0 data (indicvoices_r, Rasa) — but it uses the F5-TTS architecture and **neither the model card nor the repo states whether it was trained from scratch or initialized from the CC-BY-NC Emilia base.** If the latter, the MIT grant is questionable. Ask AI4Bharat in writing. Cheap to ask, expensive to get wrong.
3. **Demucs htdemucs** was trained on MUSDB18-HQ (academic/NC). The weights are MIT; the dataset restriction binds the dataset, not the weights. Unsettled residual, low risk.
4. **Personality rights (India).** Anil Kapoor and Amitabh Bachchan both hold Delhi HC orders protecting voice. Your consent record must be real, signed, scoped, and revocable.
5. **Reference clip provenance.** If an actor uploads a clip from a film they acted in, the *producer* may own that recording even though the *voice* is theirs. Consent from the actor may not be sufficient. This is a live question for the "upload some videos of them" flow.
6. **MuseTalk's HuggingFace card contradicts itself** — YAML front-matter says `creativeml-openrail-m`; the body of the same file says the weights are "available for any purpose, even commercially." OpenRAIL-M is the stricter read and is safe under either, but get it in writing from Tencent (benbinwu@tencent.com) before shipping. Evidence it's a copy-paste artifact: MuseTalk isn't an SD fine-tune — the architecture borrows the SD-v1-4 UNet but the weights were trained from scratch, so there's no CreativeML lineage to inherit.
7. **Does a training-set license bind downstream weight recipients?** This is the load-bearing question under half of §0.2 (Wav2Lip/LRS2, F5-TTS/Emilia, Demucs/MUSDB18, GFPGAN/StyleGAN2) and it is **unsettled law, not settled fact**. The conservative reading is what this document assumes. A lawyer may tell you some of these are over-cautious — but "the dataset says non-commercial and we sold it anyway" is not a position you want to defend while also running a voice-rights marketplace.
8. **Attribution obligations** (CC-BY-4.0: pyannote community-1, TitaNet) and **openrail++ use-restriction flow-down** (LatentSync) are real, cheap compliance steps — but they must land in the product's legal notices, not in a TODO.

---

## 8. Build plan

Sequenced so each phase is independently useful and the riskiest assumptions get tested first.

### Phase 0 — Decide (before any code)
- [ ] **Measure, don't derive.** One `whisper-cli -m ggml-large-v3-turbo.bin` run on 10 minutes of *your* audio on *this* box beats every estimate in this document. All the CPU numbers here are extrapolated from other people's hardware and carry low confidence — and this box (no AVX-512, hybrid cores) is unusual enough that the published figures may not transfer at all. ~1.5 GB model download; 15 minutes of work.
- [ ] **Choose the GPU target.** Rented box vs Modal/RunPod serverless vs buy. Blocks everything downstream.
- [ ] Confirm target language pairs. En→Hi? Hi→En? Both changes the model choice.
- [ ] Legal review of §7.
- [ ] **Disk:** 125 GB free is not enough for film work — a single 2-hour project with stems, segments, and takes will run 40–80 GB. Plan storage.

### Phase 1 — Ingest & media (all CPU) — ✅ **BUILT & RUNNING**
Upload → ffprobe → proxy transcode → thumbnails + sprite sheet → shot detection → player UI with filmstrip and shot ruler.
Deployed on **port 8621** (the only published port; Postgres/Redis/MinIO are internal-only). See [README](../README.md).
Verified end-to-end on a 4-shot test clip: all stages green, shot boundaries exact, Range streaming returns 206.

### Phase 2 — Analysis (CPU) — ✅ **BUILT & RUNNING**
Separation → diarization → ASR → **speaker naming/merging UI** → **transcript review UI**.
Verified on ground-truth speech: ~0% WER on clean audio (3/4 lines exact, 4th differs only by compounding).
Two stages degrade honestly and say so in the UI: `separate` falls back to passthrough without a Bandit
checkpoint (5.1 sources get real dialogue free via centre-channel), and `diarize` **skips** without an
`HF_TOKEN` rather than guessing. Word-level alignment (MFA/NFA) and chunk-parallel ASR are **not yet done** —
currently one `transcribe()` call using Whisper's native DTW timestamps (20ms floor).

### Phase 3 — Translation — ✅ **BUILT & RUNNING**
IndicTrans2 → **LLM refine (Gemini)** → syllable-based fit scoring → review UI with per-line duration
warnings. Verified En→Hi. The hybrid works exactly as §5 predicted: IndicTrans2 translates, the LLM
only *edits its draft* for context/register/length. Measured: the LLM moved lines from +109%/+44%
over budget to −9%/+11%, choosing `आर्द्र`→`नम` and `परीक्षण`→`टेस्ट` — formal→colloquial shifts a
dubbing editor would make. Three tiers kept (`text_mt` → `text_llm` → `text_edited`), so the LLM's
contribution stays measurable.

⚠️ **Licence ≠ distribution.** Official `ai4bharat` IndicTrans2 repos are MIT *and* `gated=auto` —
they need a logged-in HF account. Default is `prajdabre/rotary-indictrans2-*` (ungated, MIT, by an
IndicTrans2 co-author) — but it self-describes as an *independent reproduction*, so **the published
IndicTrans2 benchmarks do not apply to it**. `HF_TOKEN` switches to the official checkpoints and
also unlocks diarization.

Fit estimation counts **syllables, not characters** — VideoDubber's finding made concrete: Devanagari
encodes vowels as combining marks so `len()` overcounts badly. It calibrates against each line's own
delivery rate rather than a per-language constant. Still over-counts conjuncts ~20%; the real answer
remains a duration predictor over forced-aligned phonemes.

*End of Phase 3 you have a working subtitle/localization tool that could stand alone as a product.*

### Phase 4 — Voice (first GPU dependency)
Voice Library + consent records + clip QC + bake-off → Chatterbox TTS → time-fit → **mix with the preserved M&E stem** → publish.
*This is the first end-to-end dub. **Audio-only. Ship it here.***

### Phase 5 — Lip-sync (optional, per-shot)
Active-speaker detection → face tracking → per-shot opt-in → X-Dub (fall back to LatentSync 1.6) → RestoreFormer++ if needed → A/B approve → composite.

### Phase 6 — Scale
Fine-tuning escape hatch for weak voices; batch/multi-language; cost accounting per voice for actor royalties.

**Test the whole chain on a 3-minute clip with 2 speakers before you point it at a feature film.** Every parameter you'll fight over — stretch tolerance, clip choice, translation length budget — is discoverable in 3 minutes and agonizing at 2 hours.

---

## 9. Prior art worth reading (do not link — read)

| Project | Stars | License | Note |
|---|---|---|---|
| [pyvideotrans](https://github.com/jianchang512/pyvideotrans) | 18.3k | **GPL-3.0** | Most active. **Copyleft + declares itself non-commercial.** Study only. |
| [SoniTranslate](https://github.com/R3gm/SoniTranslate) | 1.4k | Apache-2.0 | Best-architected reference for your shape |
| [Softcatalà/open-dubbing](https://github.com/Softcatala/open-dubbing) | 415 | Apache-2.0 | Cleanest pipeline to read. Its **JSON-metadata-for-post-hoc-editing** design is worth stealing outright. Defaults to NLLB (NC) — the exact trap. |
| [Linly-Dubbing](https://github.com/Kedreamix/Linly-Dubbing) | 3.3k | Apache-2.0 | |
| [ViDubb](https://github.com/medahmedkrichen/ViDubb) | 113 | Apache-2.0 | Only one with a full lip-sync path (Wav2Lip → tainted) |

The pattern across all of them: **Apache-2.0 code that defaults to non-commercial weights.** Apache code does not save you. Check every checkpoint.

### Build-vs-buy sanity check
ElevenLabs ~$0.33–0.60/min (audio-only, no lip-sync). Rask/HeyGen ~$1.50–3.00/min (with lip-sync). All bill **per target language** — 10 min × 5 languages = 50 billable minutes.

Self-hosting the MIT stack removes per-minute translation cost almost entirely. **The build case rests on exactly one thing: Hindi/Indic voice-cloning quality that the incumbents don't offer.** If Chatterbox Hindi output doesn't beat ElevenLabs Hindi on your own ears, the honest move is to buy and spend your effort on the review workstation and the voice-rights marketplace — which is the actually differentiated part of this idea.
