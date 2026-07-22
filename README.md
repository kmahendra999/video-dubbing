# Dubbing Platform

AI dubbing pipeline: ingest → transcode → diarize → transcribe → translate → voice-clone → mix.

**Design and model selection: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).** Read §0 before adding
any model — several obvious choices (XTTS-v2, F5-TTS, NLLB-200, Wav2Lip, Demucs, GFPGAN) are
**not commercially licensed**, which matters because this product pays actors for voice rights.

## Status

| Phase | Scope | State |
|---|---|---|
| **1** | Ingest, probe, 720p proxy, poster, filmstrip, shot detection | ✅ **running, verified** |
| **2** | Separation → diarization → ASR → speaker + transcript review | ✅ **running, verified** — diarization needs `HF_TOKEN`; separation needs a Bandit checkpoint |
| **3** | Translation (IndicTrans2) + LLM refine + fit scoring + review | ✅ **running, verified** — Gemini wired; `HF_TOKEN` switches to the official MT checkpoint |
| **4** | Voice library, consent, TTS, time-fit, mix | ✅ **running, verified** — CPU-only at ~7.2× realtime |
| 5 | Lip-sync, per-shot | **not built — needs a GPU (~21 GB VRAM)** |

### What Phase 2 actually does today

Verified end-to-end on synthesized speech with known ground truth: **3 of 4 lines transcribed
exactly**, the 4th differing only by `seashells` vs `sea shells` — effectively 0% WER on clean audio.

Two stages degrade honestly rather than pretending, and say so in the UI:

- **`separate` → passthrough.** No Bandit v2 checkpoint configured, and the source wasn't 5.1, so
  the full mix is used as the dialogue stem and there's no music+FX stem to remix against.
  5.1 sources get real dialogue for free via centre-channel extraction.
  *Demucs is deliberately absent — its weights are non-commercial despite the MIT code badge.*
- **`diarize` → runs ungated by default.** See below.

## Diarization: no account, no token, no gate

`DIAR_BACKEND=diy` (the default) assembles real multi-speaker diarization from parts that need
nothing from anyone:

| Part | Licence | Gate |
|---|---|---|
| **Silero VAD** | MIT | none — the ONNX model ships **inside the pip package**, so there is no download |
| **pyannote WeSpeaker embeddings** | CC-BY-4.0 | **none** (`gated=False`, verified anonymous HTTP 200) |
| **sklearn agglomerative clustering** | BSD-3 | no model at all |

The useful irony: pyannote's *embedding* model is open — only its assembled pipeline is gated. So
you get pyannote's own speaker embeddings with no account.

**Verified twice.** On synthetic two-speaker audio: exactly 2 speakers, all 5 lines correct.
Then on a **real 3-minute Hindi Netflix clip** (Jaadugar — a doctor/patient two-hander): 3 speakers
with the two leads correctly dominant (58.6s / 45.6s). Runs ~6× realtime on CPU and **fully
offline** with the cache deleted and `HF_HUB_OFFLINE=1`.

### What the DIY backend costs

It is **clustering**: VAD → window → embed → cluster. It assigns one speaker per instant, so it
**cannot represent overlapping speech** — two people talking at once become one of them. Films are
full of that. Published DER for scale: pyannote ≈ 17.0 on AMI-IHM; clustering pipelines typically
land ~20–25, and worse on overlap-heavy material.

`DIAR_THRESHOLD` (0.70) is the cosine distance at which two windows stop being the same person.
**Tune it on real footage** — higher merges speakers, lower splits one person into several. If you
know the cast size, `DIAR_NUM_SPEAKERS=N` beats any threshold.

### Optional upgrade

`DIAR_BACKEND=pyannote` + `HF_TOKEN` once → overlap-aware, more accurate. After the first run it's
mirrored to MinIO and the token can be deleted. Accept terms at
<https://huggingface.co/pyannote/speaker-diarization-community-1> and
<https://huggingface.co/pyannote/segmentation-3.0>, token from
<https://huggingface.co/settings/tokens>. Without a token it silently falls back to DIY.

### What Phase 3 actually does today

**IndicTrans2** (MIT) translates the reviewed transcript, then **scores whether each line will fit**
the original's slot, and flags the ones that won't. IndicTrans2 has no length control of its own —
so the machine proposes and measures; the human decides.

Verified En→Hi on the transcript above. Sample: *"The weather in Mumbai is very humid during the
monsoon season."* → *"मानसून के मौसम में मुंबई का मौसम बहुत आर्द्र होता है।"* — correct and natural,
flagged **+44%** over budget. Shortening it to *"मुंबई में मानसून बहुत आर्द्र है।"* live re-scored to
**−11%** and cleared the flag, with the machine's original preserved in `text_mt`.

**Model provenance — read this.** All official `ai4bharat` IndicTrans2 repos are **`gated=auto`**:
MIT licence, but gated distribution requiring a logged-in HF account. Licence and distribution are
separate things. So:

| | Model | Note |
|---|---|---|
| **no token** (default) | `prajdabre/rotary-indictrans2-*` | Ungated, MIT, by an IndicTrans2 **co-author** — but the card calls it *"my independent reproduction"* with rotary embeddings. **AI4Bharat's published benchmarks do not describe these weights.** |
| **with `HF_TOKEN`** | `ai4bharat/indictrans2-*-dist-200M` | The official, benchmarked checkpoints. |

The same `HF_TOKEN` unlocks diarization *and* the official translation models.

**Fit estimation is an approximation, and the UI says so.** It counts syllables (not characters —
Devanagari encodes vowels as combining marks, so `len()` badly overcounts) and calibrates against
each line's *own* delivery rate, so a shouted argument and a slow monologue aren't judged by one
constant. It still over-counts Devanagari conjuncts by ~20%. The real answer is a duration predictor
over forced-aligned phonemes. Trust your ear over the badge.

### The LLM refine pass

`translate` → `refine` runs automatically. **IndicTrans2 does the translation; the LLM only edits
its draft** — it never translates from scratch, because English→Indic is the LLM's *weak* direction
(IndicGenBench: GPT-4 scores 32.1 en→xx vs 54.5 xx→en char-F1) and IndicTrans2's strong one. The LLM
supplies what IndicTrans2 structurally cannot: document context, character register, and length.

Three tiers are kept, none overwriting another — `text_mt` → `text_llm` → `text_edited`, resolving
human > llm > mt. That's how you tell whether the LLM pass is actually earning its keep. The UI shows
the MT draft under a disclosure toggle next to what the LLM changed and why.

Measured effect on the verified transcript (En→Hi, `gemini-2.5-flash`):

| Line | IndicTrans2 draft | After refine |
|---|---|---|
| the fox | +109% | **−9%** ✅ |
| sea shells | +92% | +54% |
| ASR pipeline | +74% | +47% |
| Mumbai weather | +44% | **+11%** ✅ |

The editorial choices are real dubbing-editor moves: `आर्द्र` → `नम` (formal → colloquial),
`परीक्षण` → `टेस्ट` (natural spoken Hindi), `समुद्री खोल` → `सीप`, and dropping "brown" from the
fox line to hit budget. Note the remaining overflow is partly an artifact of synthetic test audio —
espeak's timing is tight and has no natural pauses, whereas real dialogue segments carry silence
that widens the budget.

**xAI/Grok is wired but inert** — the key authenticates, but the team has no credits (403
`permission-denied`). Buy credits, then set `LLM_PROVIDER=xai`.

## Run

```bash
docker compose up -d          # build + start
docker compose logs -f worker # watch the pipeline
docker compose down           # stop
docker compose down -v        # stop and wipe all data
```

**UI: http://localhost:8621**

The interface is a **five-step wizard**, not a control panel. The twelve technical stages collapse
into five human decisions — Add video → Speakers → Script → Translation → Voices & dub — and exactly
one step is on screen at a time. Steps you cannot start yet are locked, and the wizard opens on
wherever the work actually is.

Stage-level detail (which of `probe`/`proxy`/`thumbnails`/`shots` is running) lives behind a
"Technical stages" disclosure on each step. Warnings never hide: a skipped or degraded stage still
prints its note in full, because a stage that quietly did less than you think is worse than one that
failed.

Casting is a searchable, auditionable grid of all 122 voices — filter by male/female/kids/performer,
play the reference clip, click to cast. A performer's voice with no consent on file shows *why* it
can't be used and offers to record the consent instead.

## Ports

This stack publishes **exactly one host port: 8621** (`UI_PORT` in `.env`). Postgres, Redis and
MinIO have no host ports at all — they're reachable only on the internal `dubbing_default`
network. Nothing here can collide with another application.

To move the UI, change `UI_PORT` in `.env` and `docker compose up -d`.

## Authentication

The UI is **password protected** (`APP_PASSWORD` in `.env`). A password was generated for you at
install; change it to anything you like and `docker compose up -d api`.

Sessions are HMAC-signed cookies keyed off the password, so **changing the password signs everyone
out**. Verified: unauthenticated requests to `/api/*` and `/api/media/*` return 401, forged cookies
are rejected, `/api/health` and `/static/*` stay open by design.

**Forgot the password?** It is printed in the startup banner:

```bash
docker compose logs api | grep -A2 "Sign in at"
```

That is always safe — reading container logs already implies access to the box.

**Copy-paste on the login page** (`DEV_SHOW_PASSWORD=1`) shows the password with Copy and Fill
buttons — but **only when `UI_BIND` is loopback as well**. Two gates, not one: printing a password on
a page anyone can load removes the authentication entirely, so setting the flag while bound to
`0.0.0.0` is refused and logged:

```
DEV_SHOW_PASSWORD is set but UI_BIND=0.0.0.0 — refusing to print the password
on a page your LAN can load. Set UI_BIND=127.0.0.1 to make it genuinely local first.
```

The page never decides this for itself; it asks `/api/login-hint`, and the server answers `null`
unless the instance is genuinely local-only.

> ⚠️ Leaving `APP_PASSWORD` empty disables auth entirely and logs a loud warning at startup.
> Set `secure=True` on the cookie once you put this behind TLS.
> Turn `DEV_SHOW_PASSWORD` off before this instance is ever reachable by anyone else.

## Services

| Service | Role | Host port |
|---|---|---|
| `api` | FastAPI + the UI; streams media out of MinIO with Range support | **8621** |
| `worker` | Celery; runs ffmpeg/ffprobe/PySceneDetect | — |
| `postgres` | Project/asset/job/shot state | — |
| `redis` | Job queue | — |
| `minio` | Object store for all media | — |

## What Phase 1 does

On upload the API stores the master in MinIO and chains four jobs:

1. **probe** — `ffprobe` → container/codec/duration/fps metadata
2. **proxy** — H.264 720p CRF 23 `+faststart`, never upscaled. **The master is never modified.**
3. **thumbnails** — poster at 10% in, sprite sheet (≤100 tiles), WebVTT
4. **shots** — PySceneDetect `ContentDetector` → shot boundaries

Shots are load-bearing later: Phase 5 gates lip-sync per shot, and Phase 2 uses shot boundaries
as **chunk boundaries for parallel ASR** — which is the single most important CPU optimization on
this box (see below).

## Hardware note

This machine is an **i9-12900K: 16 physical cores (8P + 8E), no AVX-512, no NVIDIA GPU.**

- Phases 1–3 run here fine.
- **Phases 4–5 need a GPU** (~24 GB VRAM). Workers are capability-tagged (`cpu`/`gpu`) so GPU
  stages can run on another host without code changes.
- Prefer **more workers with fewer threads each** over one wide job — Whisper thread-scaling
  saturates early, and hybrid P/E cores make static thread-splitting worse. `WORKER_CONCURRENCY`
  and `FFMPEG_THREADS` in `.env` control this.
- Disk: **125 GB free is not enough for feature-film work.** One 2-hour project with stems,
  segments and takes will run 40–80 GB.

## API

```
GET    /api/health
GET    /api/projects
POST   /api/projects                  {title, source_lang, target_langs}
GET    /api/projects/{id}
DELETE /api/projects/{id}             also deletes its objects
POST   /api/projects/{id}/source      multipart upload → triggers ingest
POST   /api/projects/{id}/reingest
POST   /api/projects/{id}/analyze     phase 2: separate → diarize → asr
GET    /api/projects/{id}/shots
GET    /api/projects/{id}/speakers
PATCH  /api/speakers/{id}             {display_name}
POST   /api/speakers/{id}/merge       {into}  — fold one speaker into another
GET    /api/projects/{id}/segments
PATCH  /api/segments/{id}             {text_src_edited, speaker_id, state}
GET    /api/media/{key}               Range-capable stream from MinIO
```

## Layout

```
app/
  main.py      FastAPI routes + media streaming + additive schema sync
  tasks.py     celery app, queue routing, phase 1 (probe/proxy/thumbnails/shots)
  tasks_ai.py  phase 2 (separate/diarize/asr) — all model imports are LAZY
  models.py    Project · Asset · Job · Shot · Speaker · Segment
  storage.py   MinIO/S3
  static/      UI (no build step)
docs/
  ARCHITECTURE.md
Dockerfile     media worker + api (ffmpeg, scenedetect)
Dockerfile.ai  ai worker (faster-whisper, pyannote) — separate ON PURPOSE
```

## Two things to know before extending this

**Machine output is never destroyed.** `text_src` holds the ASR result; `text_src_edited` holds the
human fix. Same pattern for translation. This is what lets you measure model quality and re-run a
stage without losing edits.

**Schema changes need Alembic before this holds real data.** `app/main.py:_sync_new_columns()` is an
additive-only dev convenience that ALTERs in missing columns at startup — `create_all` only creates
missing *tables*, so adding a model field otherwise 500s at query time. It cannot do renames, type
changes, backfills or rollbacks.

---

## Phase 4 — voice, consent, and the dub

Verified end-to-end on CPU: 4 lines synthesized → time-fit → mixed → **a playable dubbed MP4**
(H.264 + AAC, real audio at −22.7 dB mean).

### CPU is more viable than expected

Measured on this i9-12900K: Chatterbox synthesizes Hindi at **~7.2× slower than realtime**, not the
30–50× the design assumed. Roughly: a 3-minute clip ≈ 20 min; a 2-hour film (~40% of runtime is
speech) ≈ **6 hours** — an overnight job, not an impossible one. A GPU still makes this minutes.

`synth` is tagged `capability="gpu"` and routed by `SYNTH_QUEUE` (default `tts`). Point it at a
`gpu` queue and a GPU host takes over that stage alone — no code change.

### Consent is enforced, not documented

`Voice.usable` gates on a signed, unexpired, unrevoked `ConsentRecord`. Verified:

```
POST /api/speakers/{id}/cast   → 400 "Test Performer cannot be used — no signed consent record"
POST /api/voices/{id}/consent  → status=active, usable=true
POST /api/speakers/{id}/cast   → 200 ok
```

`synth` re-checks before loading a model and **refuses the whole job** if any cast voice lost
consent meanwhile. `POST /api/voices/{id}/revoke` is the kill switch — it suspends the voice across
all in-flight work. `VoiceUsage` logs segments/seconds per voice per project: what the actor is paid
on, and the evidence that scope was honoured.

Reference clips get QC on upload (duration, peak dB, RMS, clipping) because a bad clip poisons every
line cloned from it. **This is a clip library, not a training pipeline** — cloning is zero-shot from
~10s, so curation is the quality lever.

### Time-fit does what §0.4 says

Not DTW. Stretch is capped at ±15%, and anything that still doesn't fit is **flagged `over_budget`**
rather than squashed into chipmunk audio. Observed: 1 line stretched 1.064× to land exactly on
budget; 3 hit the cap and were flagged for the operator.

⚠️ The syllable estimator is directionally right but not precise — it predicted +11% on a line the
TTS actually rendered at +58%. Now that `takes.raw_ms` records real synthesized durations, the
honest upgrade is to calibrate the estimator against that ground truth.

## The three-image rule, learned the hard way

`Dockerfile` (ffmpeg) · `Dockerfile.ai` (whisper + pyannote + IndicTrans2) · `Dockerfile.tts`
(Chatterbox). Not preference — **necessity**. Installing Chatterbox into the ai image pinned
`torch==2.6`/`numpy<2` and pulled `transformers` 5.x, which broke pyannote (needs torch ≥2.8,
numpy ≥2.0) *and* IndicTrans2 (needs transformers <5), and dragged ~3 GB of CUDA wheels onto a
CPU-only box. They share nothing but MinIO and the queue — which is exactly why a GPU host can run
one image and leave the rest here.

## Chunk-parallel ASR

`ASR_WORKERS` × `ASR_THREADS` should stay ≤ **16 physical cores**. Defaults: 4 × 4.

Audio is split at **silences** (not shot boundaries — see docs §2 for why that failed) into
`ASR_CHUNK_SECONDS` chunks (default 300s) and transcribed concurrently. CTranslate2's `num_workers`
only helps when `transcribe()` is called from multiple Python threads, so the sharding is done here
rather than left to the library.

Chunks stay long on purpose: Whisper's accuracy comes from long context. This parallelises; it does
not fragment.

## Independence: what this box needs from the outside world

**Nothing, for the default stack.** Every model it uses is mirrored into your MinIO:

```
Systran/faster-whisper-small                      486 MB   ASR
prajdabre/rotary-indictrans2-en-indic-dist-200M   856 MB   translation
pyannote/wespeaker-voxceleb-resnet34-LM            27 MB   speaker embeddings
ResembleAI/chatterbox                            3213 MB   voice cloning
                                                 ─────────
                                                 ~4.6 GB
```

Set `HF_HUB_OFFLINE=1` in `.env` and the stack runs with no HuggingFace contact at all — verified by
deleting the local cache and transcribing/diarizing from MinIO alone.

**A note on what "HuggingFace dependency" actually meant.** `transformers` and `huggingface_hub` are
Apache-2.0 libraries running inside your containers — cloning `github.com/huggingface` gives you
exactly what `pip install` already did, and **no model weights**, because those repos contain none.
The token was never HuggingFace charging you; it was pyannote and AI4Bharat gating *their own*
downloads. The mirror removes that permanently.

The only outside services still in play are **Gemini** (optional — translation polish; the pipeline
runs without it, IndicTrans2 does the actual translating) and the **gated model upgrades**, which
are strictly optional.

---

## What a real film taught us (and the synthetic tests hid)

Everything was green on espeak robots. A real 3-minute Hindi clip broke two things immediately.
Both are now fixed **from measurement**, not guesswork.

### 1. `ASR_MODEL=small` is unusable for Hindi

Same audio, same settings, only the model changed:

| | `small` | `large-v3` |
|---|---|---|
| avg confidence | 0.50 | **0.87** |
| "wait please" | `लेज की जी?` ❌ | `वेट कीजिए` ✅ |
| "patient hours are from 9" | `आप प्ष्टन देखने का ताईम नोगबजे` ❌ | `पेशेंट देखने का टाइम नौ बजे से है` ✅ |
| "a lot of pain in the back" | `पर कमर में बहुज आदे है दरद हैं` ❌ | `पर कमर में बहुत जादा ही दर्द है` ✅ |
| speed | 1.4× realtime | 0.6× realtime |

`small` produced non-words at high confidence — the worst failure mode, because it *looks* like a
transcript. **`large-v3` is now the default.** It costs ~2.3× the CPU; chunk-parallelism absorbs it
(181s of film analyzed in ~190s end-to-end, including separation and diarization).

### 2. `DIAR_THRESHOLD=0.70` over-clustered badly

Swept on the real two-hander:

| threshold | speakers | seconds per cluster |
|---|---|---|
| 0.70 (old default) | **10** | 54, 51, 18, 9, 3, 2, 1, 1, 1 |
| 0.80 | 6 | 79, 57, 3, 1, 1, 0 |
| 0.90 | 2 | 84, 57 |

The old 0.70 came from general knowledge of WeSpeaker thresholds and had never met real audio.
Default is now **0.85**, plus `_absorb_specks`: any cluster holding under
`DIAR_MIN_SPEAKER_SECONDS` (2.0s) of speech is a breath or a door, not a character, and gets folded
into its nearest real speaker. Result on the same clip: **10 → 3**, leads correctly dominant.

Filtering by *how much a cluster actually speaks* is more robust than tuning the threshold alone —
the threshold trades specks against merging your real cast; the duration filter doesn't.

### Still open, seen on real footage

- **Stereo source → no music/FX bed.** The clip is stereo, so `separate` fell back to passthrough.
  A dub of it would be dialogue-only, with the score gone. This is the Bandit v2 gap.
- Whisper still fumbles proper nouns and code-switched English (`डूणा`, `सिर्वे`) — normal, and
  exactly what the transcript review UI is for.

---

## The synth crash, and the four bugs behind it

**Reported:** `Synthesize failed: max(): Expected reduction dim 1 to have non-zero size.`

**Fixed and verified: 58/58 lines synthesize on the real Hindi clip.** Chasing it found more than
the crash.

### 1. Chatterbox dies on short input — and hallucinates on empty

Tested against inputs that really occur in film transcripts:

| input | before |
|---|---|
| `''` (empty) | ✅ **2.28s of invented speech** ← worse than a crash |
| `'""'` | ✅ 2.08s of invented speech |
| `'I'`, `'U'`, `'9'` | ❌ IndexError |
| `'...'`, `'।'`, `'—'`, `'?!'`, `'♪'` | ❌ IndexError |

The trigger in your clip was the **eye-chart scene** — the patient reads `"I"`, `"L U V"`, `"U"`,
spelling out "I LUV U". Empty input was the dangerous one: it silently drops 2.3s of fabricated
speech into the mix at a real timecode.

Now: text with no letters or digits **never reaches the model** (nothing to say → silent, which is
correct); short text is padded only if the model actually refuses (`'I'` → `'I..'`, found by
testing — `'I.'` fails, `'I ...'` fails). **16/16 handled: 8 synthesized, 8 correctly skipped.**

### 2. One bad line killed all 58

`synth` had no per-line error handling, so a single unsynthesizable line threw away the whole job.
The architecture doc warned about exactly this. Lines now fail independently and are reported.

### 3. The fit estimator predicted the wrong speaker

It calibrated against the **original actor's** delivery rate. But the dub is spoken by Chatterbox:

| | syllables/sec |
|---|---|
| Hindi actors in the clip | **7.26** |
| Chatterbox | **3.21** |

So it under-predicted TTS duration by ~2.5×, told the operator lines would land 17–36% **short**,
then produced audio 2.22× **over**. It now predicts the engine (`TTS_SYLL_PER_SEC`), and
`observed_tts_rate()` overrides that with the rate measured from the project's own takes.

### 4. `cfg_weight` is backwards from the docs, and reference pacing dominates

Chatterbox's docs imply lowering `cfg_weight` improves pacing. Measured, it does the opposite:

| cfg_weight | rate | | Kokoro reference speed | resulting dub |
|---|---|---|---|---|
| **0.5** | **5.00 syl/s** | | 1.0 | 3.87 syl/s |
| 0.3 | 3.95 syl/s | | **1.3** | **4.62 syl/s** |
| 0.2 | 3.09 syl/s | | 1.6 | 2.79 syl/s ← degrades |

**The reference clip's own pacing transfers to the clone** — a slow reference makes a slow dub, and
pushing the reference too fast makes it *worse*. The voice bank is now generated at
`VOICEBANK_SPEED=1.3`.

### Result, and what is still wrong

| | before | after |
|---|---|---|
| over budget | 53 lines @ **2.22×** | 41 lines @ **1.55×** |
| fitting | 5/58 | 17/58 |

Better, **not solved**. The residue is structural: Chatterbox speaks ~4.6 syl/s where the actors run
7.26, so an English dub is inherently ~1.6× its slot unless the translation is compressed ~35%
harder. That is lever 1 (translation length) — the cheapest lever and the one still not pulled far
enough. Chasing it with time-stretch instead would chipmunk the audio, which is why `fit` refuses.

---

## Fit and background audio — what was wrong, and what is still wrong

### Background: the score is no longer thrown away

Three paths, best first. The stage says which one it took.

| source | what happens |
|---|---|
| **5.1** | centre channel is the dialogue, the rest is a real M&E bed. Free and exact. |
| **true stereo** | **centre-cancellation (L−R)** removes the centre-panned dialogue and leaves off-centre music/FX. A real bed, no model, no licence question. |
| **dual mono / mono** | **ducking** — the ORIGINAL audio is kept and pulled down under the dub. |

Ducking is what a dubbing suite does when handed no M&E track. It is **not separation**: the original
dialogue is still faintly under the new one. But the alternative — what this used to do — was a film
whose music and atmosphere simply vanished.

Measured on the real Hindi clip (dual mono, so the ducking path):

```
silent gap 13-22s   original -39.2 dB → dub -44.5 dB   (background preserved, -5.3)
dialogue   3-10s    original -33.5 dB → dub -30.6 dB   (dub on top,          +2.9)
```

Tune with `DUCK_LEVEL` (0.55), `DUCK_THRESHOLD`, `DUCK_RATIO`. A separated bed always wins when one
is available — this is the honest fallback, not the goal.

**Why no separation model:** Demucs, Open-Unmix and torchaudio's bundled separator are all trained on
MUSDB18 and non-commercially licensed (§0.2). Bandit v2 is the only commercially-clean cinematic
separator and is pending your legal review of its CC-BY-SA weights.

### Fit: speech duration is affine, not a rate

The estimator divided syllables by a speaking rate. That is wrong, and wrong worst exactly where film
dialogue lives. Least-squares over 58 real takes:

```
duration_ms ≈ 510 + 421 × syllables
```

Every utterance carries a **fixed ~510ms cost** — onset, offset, the breath around it — and only the
remainder scales with length. A rate model predicts ~220ms for a one-syllable line that really takes
820ms, and **38 of those 58 lines were ≤3 syllables**. Mean error: affine 847ms vs rate 1149ms.

Each project refits the model on its own takes; the constants above are only the cold-start fallback.

**Consequences worth knowing:**

- Some slots are **physically impossible**. A 479ms gap cannot hold any spoken line when the floor is
  510ms. The refine pass now counts these and says so, instead of asking you to shorten a line that
  no wording can rescue.
- **Under-length is no longer flagged.** Silence is free; overrun is not. Flagging short lines sent
  the operator to rewrite dialogue that was already fine and buried the real overruns.
- Budgeting at one rate while scoring at another made correctly-compressed lines read as *too short* —
  a 4-syllable line hitting a 4-syllable budget scored 0.54. One model is now used for both.

### The real cause: 24% of the audio was hallucination

Chasing a "faster voice" found something else. **Chatterbox is autoregressive and sometimes fails to
stop.** On the real script, 7 of 58 takes ran far past their text:

| line | syllables | slot | generated |
|---|---|---|---|
| `Blur.` | 1 | 860ms | **5,440ms** |
| `Problem?` | 2 | 1,439ms | 5,120ms |
| `Run like house fire.` | 4 | 2,019ms | 6,320ms |

`"Can't see."` produced **11 seconds** on one run and 2.3s on the next — from identical input. Those
7 takes were **24% of all dub audio**. No amount of translation compression fixes that: the text was
already two words.

It is *sampled*, though, so a retry usually lands fine. `synth` now checks each take against the
affine model and regenerates when it overruns by more than `TTS_RUNAWAY_FACTOR` (1.8×), keeping the
shortest of `TTS_RETRIES` (3) attempts. Measured after: **0 runaways, 0% hallucinated audio**, and
total dub audio fell 95.8s → 70.6s for the same script.

### Where the fit landed

| run | fitting | avg overrun |
|---|---|---|
| original | 5/58 | **2.22×** |
| pacing fix | 17/58 | 1.55× |
| compression | 21/58 | 1.47× |
| affine model | — | 1.22× |
| **runaway rejection** | **44/58** | **0.83×** |

A last labelling bug fell out of this: takes coming in *under* budget were being called
`over_budget` (the check was `abs(ratio-1) > tolerance`) and then **slowed down to fill the slot**,
which makes delivery sound drugged. Only overrun is a defect now — short lines are left alone,
because the original has silence around them and lever 4 spends exactly that.

**Still open:** 11 lines remain over, averaging 1.40×. Those are genuinely too long for their slots
and need either a shorter line or lever 4 (letting audio run into the following pause), which is
still unimplemented. Note also that a faster reference clip was *not* the answer — the measured
difference between reference styles was swamped by the runaway noise it was hiding.

---

## Adding a remote transcode machine (Tailscale)

A worker needs **no public IP and no inbound port**. It reaches *in* to the coordinator, so a VM
behind NAT with only a `100.x` Tailscale address works. This is what the queue-and-capability design
was for.

**Verified:** with the local media worker stopped, a worker connecting only over Tailscale ran a real
transcode — probe, proxy, thumbnails, shots — in 10 seconds.

### On the coordinator (this box) — already done

Redis, Postgres and MinIO are published on the **Tailscale address only** (`EXPOSE_BIND`), never
`0.0.0.0`, on unique ports so they cannot collide with neighbouring stacks:

| service | port | bound to |
|---|---|---|
| Redis | 6479 | 100.123.66.19 |
| Postgres | 5532 | 100.123.66.19 |
| MinIO | 9100 | 100.123.66.19 |
| UI | 8621 | 127.0.0.1 |

> ⚠️ **Redis had no password.** That was survivable while it was container-internal, but a Celery
> broker anyone can write to is a **remote-code-execution** vector — a task payload is code a worker
> will run. All three services now have generated credentials. Set `EXPOSE_BIND=` (empty) to pull
> them back inside if you stop using remote workers.

### On the remote VM — one command

```bash
scp -r docker-compose.remote.yml remote.env Dockerfile requirements.txt app \
       add-transcode-vm.sh  user@vm:~/dubbing/
ssh user@vm
cd ~/dubbing && ./add-transcode-vm.sh --tailscale-key tskey-auth-XXXX
```

`add-transcode-vm.sh` joins the tailnet, **verifies it can actually reach all three services before
touching anything else**, installs Docker if needed, sizes concurrency to the VM's *physical* cores,
and starts the worker. `--queues ai` if the VM has ~6GB+ RAM; `--skip-tailscale` if already joined.

There is **no agent, no System ID, no registration server**. A worker is a Celery process that
reaches in over Tailscale; it shows up in `celery inspect` seconds after it starts. If you were given
an "agent secret" and a `System ID`, that belongs to some other tool — this system has no such thing.

The key is passed as an argument or prompted for with hidden input, and unset once spent. Never bake
it into a file.

`remote.env` is generated for you and contains real credentials — it is chmod 600 and gitignored.
Regenerate it if you rotate the coordinator's passwords.

Tune per machine: `WORKER_CONCURRENCY` (match **physical** cores), `WORKER_NAME`, and `QUEUES`.

### What to run where

| queue | work | good remote candidate? |
|---|---|---|
| `media` | ffprobe, proxy transcode, thumbnails, shots, separate, fit, mix | **yes** — CPU-bound and embarrassingly parallel |
| `ai` | Whisper, diarization, IndicTrans2 | yes, if the VM has the RAM; it will mirror ~4 GB of models on first use |
| `tts` | Chatterbox | this is the GPU-shaped stage — point `SYNTH_QUEUE` at a GPU host when you have one |

Check who is connected:

```bash
docker compose exec worker-ai celery -A app.tasks.celery inspect active_queues
```
