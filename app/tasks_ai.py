"""Phase 2 analysis: separate → diarize → asr.

Every model import in this module is LAZY. The media worker imports this file
so Celery can register the task names, but it never has faster-whisper or
pyannote installed — those live in the `ai` image and the tasks are routed to
the `ai` queue.

Order matters and is not the obvious one (docs/ARCHITECTURE.md §1):
  · separation runs FIRST, so diarization sees a clean dialogue-only signal
  · ASR runs over the WHOLE dialogue stem, then words are intersected with the
    diarization timeline — NOT per-diarized-chunk, which throws away the long
    context Whisper's accuracy depends on
"""

import json
import logging
import os

from .config import settings
from .db import session
from .models import SPEAKER_COLORS, Asset, Project, Segment, Shot, Speaker
from .tasks import (
    _asset, _duration_of, _probe_meta, _put_asset, _run, _run_stderr, _stage,
    _workdir, celery,
)

log = logging.getLogger(__name__)

DIALOGUE_SR = 16000  # what both pyannote and Whisper want


# ── helpers ───────────────────────────────────────────────────────────────


def _fetch(project_id: str, kind: str, dest: str) -> str:
    from . import storage

    with session() as db:
        a = _asset(db, project_id, kind)
        if a is None:
            raise RuntimeError(f"missing '{kind}' asset — run the earlier stage first")
        key = a.key
    storage.download_file(key, dest)
    return dest


def _source_channels(project_id: str) -> int:
    with session() as db:
        a = _asset(db, project_id, "source")
        meta = (a.meta or {}) if a else {}
    for s in meta.get("streams", []):
        if s.get("codec_type") == "audio":
            return int(s.get("channels") or 0)
    return 0


# ── separate ──────────────────────────────────────────────────────────────


@celery.task(name="analyze.separate")
def separate(project_id: str) -> str:
    """Produce a dialogue stem and (where possible) a music+FX stem.

    Three paths, best first:
      1. 5.1 source  → the centre channel IS the dialogue stem. Free, exact,
         no model, no licence question. Always try this first.
      2. Bandit v2   → the only commercially-licensed cinematic D/M/E model
         (§0.2). Runs only if BANDIT_CHECKPOINT points at real weights.
      3. passthrough → full mix as "dialogue", music_fx unavailable. Honest
         degradation: flagged in the UI, NOT silently pretended.

    Deliberately absent: Demucs. Its weights are non-commercial despite the
    MIT code badge — see docs/ARCHITECTURE.md §0.2.
    """
    with _stage(project_id, "separate") as job, _workdir() as wd:
        src = _fetch(project_id, "source", os.path.join(wd, "source"))
        dialogue = os.path.join(wd, "dialogue.wav")
        channels = _source_channels(project_id)
        method = "passthrough"
        note = None

        if channels >= 6:
            # FC (front-centre) is the dialogue channel in a 5.1 mix.
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", src,
                "-filter_complex", "[0:a]pan=mono|c0=FC[d]", "-map", "[d]",
                "-ar", str(DIALOGUE_SR), "-ac", "1", "-c:a", "pcm_s16le", dialogue,
            ])
            # Everything except centre = the M&E bed.
            music_fx = os.path.join(wd, "music_fx.wav")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", src,
                "-filter_complex",
                "[0:a]pan=stereo|c0=0.5*FL+0.35*SL+0.35*LFE|c1=0.5*FR+0.35*SR+0.35*LFE[m]",
                "-map", "[m]", "-ar", "48000", "-c:a", "pcm_s16le", music_fx,
            ])
            _put_asset(project_id, "music_fx", f"{project_id}/stems/music_fx.wav",
                       music_fx, "audio/wav", meta={"method": "centre_channel"})
            method = "centre_channel"
            note = f"5.1 source ({channels}ch): centre channel used as dialogue. Best case — no model needed."
        else:
            ckpt = settings.bandit_checkpoint
            if ckpt and os.path.exists(ckpt):
                method = "bandit_v2"
                note = "Bandit v2 cinematic separation."
                raise RuntimeError(
                    "Bandit v2 adapter is not implemented yet — weights were found at "
                    f"{ckpt} but the inference path is a Phase 2b task."
                )

            # Dialogue stem: the full mix. Nothing to remove without a model.
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", src,
                "-vn", "-ar", str(DIALOGUE_SR), "-ac", "1", "-c:a", "pcm_s16le", dialogue,
            ])

            # Background: try mid/side before giving up. Film dialogue is almost
            # always centre-panned, so (L-R) cancels it and leaves the off-centre
            # music and effects — a real M&E bed, no model, no licence question.
            # It fails on dual-mono sources (L==R cancels EVERYTHING), so we
            # measure the result rather than assume it worked.
            side_ok = False
            if channels == 2:
                side = os.path.join(wd, "music_fx.wav")
                _run([
                    "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", src,
                    "-af", "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c0-0.5*c1",
                    "-ar", "48000", "-c:a", "pcm_s16le", side,
                ])
                side_db = _mean_db(side)
                full_db = _mean_db(dialogue)
                # If cancelling the centre leaves nothing, the source was dual
                # mono and this bed is silence, not music.
                side_ok = side_db is not None and side_db > -60 and (
                    full_db is None or side_db > full_db - 25)
                if side_ok:
                    _put_asset(project_id, "music_fx", f"{project_id}/stems/music_fx.wav",
                               side, "audio/wav", meta={"method": "mid_side", "mean_db": side_db})
                    method = "mid_side"
                    note = (
                        f"Stereo source: centre-cancellation (L−R) extracted a music/FX bed at "
                        f"{side_db:.1f} dB. Dialogue is centre-panned so it largely cancels — this "
                        "is a real background bed, though not as clean as a trained separator. "
                        "ASR still runs on the full mix."
                    )
                else:
                    note = (
                        f"Stereo source is DUAL MONO (L−R = silence, {side_db:.0f} dB), so centre "
                        "cancellation cannot recover a background bed."
                    )

            if not side_ok:
                note = (note or f"{channels}ch source.") + (
                    " No music/FX stem could be extracted, so the mix will DUCK the original "
                    "audio under the dub instead of replacing it — the score and atmosphere "
                    "survive between lines. See docs/ARCHITECTURE.md §0.2 for why no separation "
                    "model is enabled (Demucs/Open-Unmix weights are non-commercial; Bandit v2 "
                    "is pending legal review)."
                )

        _put_asset(project_id, "dialogue", f"{project_id}/stems/dialogue.wav",
                   dialogue, "audio/wav", meta={"method": method, "sr": DIALOGUE_SR})
        job["note"] = note
    return project_id


# ── diarize ───────────────────────────────────────────────────────────────


@celery.task(name="analyze.diarize")
def diarize(project_id: str) -> str:
    """Speaker diarization. Two backends, both commercially licensed.

    DIAR_BACKEND=diy (default)
        Silero VAD + WeSpeaker embeddings + clustering. Zero gates, zero
        accounts, fully self-hosted. Cannot represent overlapping speech.

    DIAR_BACKEND=pyannote
        pyannote community-1 — overlap-aware and more accurate, but its repo is
        gated, so it needs HF_TOKEN once (then it's mirrored and the token can
        go). Falls back to DIY rather than failing if no token is present.
    """
    token = settings.hf_token or None
    backend = (settings.diar_backend or "diy").lower()

    if backend == "pyannote" and not token:
        log.warning("DIAR_BACKEND=pyannote but no HF_TOKEN — falling back to the DIY backend")
        backend = "diy"

    if backend == "diy":
        return _diarize_diy(project_id)

    with _stage(project_id, "diarize") as job, _workdir() as wd:
        from pyannote.audio import Pipeline  # lazy: only exists in the ai image

        from . import mirror

        # Mirror the pipeline AND its dependencies — pyannote's config.yaml
        # references the segmentation and embedding repos, so caching only the
        # top-level pipeline still hits the network for the rest.
        for repo in (settings.diar_model, "pyannote/segmentation-3.0",
                     "pyannote/wespeaker-voxceleb-resnet34-LM"):
            try:
                mirror.ensure(repo, token)
            except Exception as e:
                log.warning("mirror unavailable for %s (%s)", repo, e)

        wav = _fetch(project_id, "dialogue", os.path.join(wd, "dialogue.wav"))
        pipe = Pipeline.from_pretrained(settings.diar_model, use_auth_token=token)

        # pyannote 4.x decodes audio via torchcodec, whose shared libs don't load
        # against this torch build. Its own documented workaround is to hand it a
        # preloaded waveform instead of a path — which also skips a redundant
        # decode, since the dialogue stem is already 16k mono PCM.
        ann = pipe(_load_waveform(wav))

        turns = [
            {"speaker": spk, "start": float(seg.start), "end": float(seg.end)}
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]
        if not turns:
            job["note"] = "No speech detected."
            _ensure_single_speaker(project_id)
            return project_id

        _store_turns(project_id, turns, wd)
        job["note"] = (
            f"{len(set(t['speaker'] for t in turns))} speakers over {len(turns)} turns · "
            "pyannote community-1 (overlap-aware, CC-BY-4.0). Mirrored to MinIO — the token "
            "is no longer needed on this host."
        )
    return project_id


def _diarize_diy(project_id: str) -> str:
    """Ungated backend — see app/diarize_diy.py for what it does and costs."""
    with _stage(project_id, "diarize") as job, _workdir() as wd:
        from . import diarize_diy

        wav = _fetch(project_id, "dialogue", os.path.join(wd, "dialogue.wav"))
        turns, info = diarize_diy.diarize(wav, num_speakers=settings.diar_num_speakers or None)

        if not turns:
            job["note"] = "No speech detected in the dialogue stem."
            _ensure_single_speaker(project_id)
            return project_id

        _store_turns(project_id, turns, wd)
        job["note"] = (
            f"{info['speakers']} speaker(s) over {len(turns)} turns · ungated backend "
            f"(Silero VAD + WeSpeaker + clustering — no HuggingFace account, no token). "
            f"{info['regions']} speech regions, {info['windows']} windows, "
            f"{info['speech_seconds']}s of speech. "
            "⚠️ Clustering assigns one speaker per instant, so OVERLAPPING SPEECH is not "
            "represented — two people talking at once become one. Set DIAR_BACKEND=pyannote "
            "(+ HF_TOKEN once) for overlap-aware diarization."
        )
    return project_id


def _store_turns(project_id: str, turns: list[dict], wd: str) -> None:
    """Persist diarization turns + the Speaker rows the review UI edits."""
    tpath = os.path.join(wd, "diarization.json")
    with open(tpath, "w") as f:
        json.dump({"turns": turns}, f)
    _put_asset(project_id, "diarization", f"{project_id}/analysis/diarization.json",
               tpath, "application/json", meta={"turns": turns, "count": len(turns)})

    totals: dict[str, float] = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + (t["end"] - t["start"])

    with session() as db:
        db.query(Speaker).filter(Speaker.project_id == project_id).delete()
        db.flush()
        for i, (label, secs) in enumerate(sorted(totals.items(), key=lambda kv: -kv[1])):
            db.add(Speaker(
                project_id=project_id, label=label,
                color=SPEAKER_COLORS[i % len(SPEAKER_COLORS)],
                speech_seconds=round(secs, 2),
            ))


def _ensure_single_speaker(project_id: str) -> None:
    with session() as db:
        existing = db.query(Speaker).filter(Speaker.project_id == project_id).count()
        if existing == 0:
            db.add(Speaker(
                project_id=project_id, label="SPEAKER_00",
                display_name=None, color=SPEAKER_COLORS[0],
            ))


# ── asr ───────────────────────────────────────────────────────────────────


def _silences(path: str, noise_db: int = -35, min_dur: float = 0.35) -> list[float]:
    """Midpoints of silent stretches — the only safe places to cut.

    Measured the hard way: cutting on shot boundaries alone truncated a line
    mid-word ("every m-") and corrupted another ("The quick"→"A quick"), because
    a picture cut is not guaranteed to be a pause in the dialogue. Whisper also
    hallucinates around abrupt starts. Cut in silence or don't cut.
    """
    out = _run_stderr([
        "ffmpeg", "-hide_banner", "-nostdin", "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-",
    ])
    starts, mids = [], []
    for line in out.splitlines():
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:")[1].strip().split()[0]))
            except (ValueError, IndexError):
                pass
        elif "silence_end:" in line and starts:
            try:
                end = float(line.split("silence_end:")[1].strip().split()[0])
                mids.append((starts.pop() + end) / 2)
            except (ValueError, IndexError):
                pass
    return sorted(mids)


def _chunk_plan(
    duration: float,
    target: float,
    silences: list[float] | None = None,
    shots: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Split audio into ~`target`-second chunks, cutting ONLY in silence.

    Two constraints in tension (§1, §2):
      · Whisper's accuracy comes from LONG context — so chunks stay long. This
        is nothing like transcribing 3-second diarized fragments.
      · This CPU saturates at ~4-8 threads per job and has no AVX-512, so the
        only real speedup is running several long chunks CONCURRENTLY.

    Preference order for a cut point: a silence near the target → a shot
    boundary near it → nothing (leave the chunk long). We would rather run one
    oversized chunk than cut through a word: a slow transcript is recoverable,
    a corrupted one is not.
    """
    if duration <= target * 1.5:
        return [(0.0, duration)]

    candidates = sorted(silences or [])
    if not candidates and shots:
        candidates = sorted(s for s, _ in shots if 0 < s < duration)

    cuts = [0.0]
    tolerance = target * 0.5   # how far from the ideal we'll accept a cut
    while True:
        want = cuts[-1] + target
        if want >= duration - target * 0.5:
            break
        near = [c for c in candidates
                if c > cuts[-1] + target * 0.4 and abs(c - want) <= tolerance]
        if not near:
            # No safe cut anywhere near — extend this chunk rather than slice a word.
            nxt = [c for c in candidates if c > cuts[-1] + target * 0.4]
            if not nxt or nxt[0] >= duration - target * 0.4:
                break
            cuts.append(nxt[0])
            continue
        cuts.append(min(near, key=lambda c: abs(c - want)))
    cuts.append(duration)

    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 0.5]


@celery.task(name="analyze.asr")
def asr(project_id: str) -> str:
    """faster-whisper over the dialogue stem, then attribute speakers.

    Chunk-parallel: the audio is split at shot boundaries and transcribed
    concurrently. CTranslate2's own guidance is that `inter_threads`
    (num_workers) beats `intra_threads` for bulk work — but num_workers only
    does anything when transcribe() is called from multiple Python threads, so
    we must do the sharding ourselves. On this box (16 physical cores, no
    AVX-512, hybrid P/E) this is worth more than the model choice.
    """
    from faster_whisper import WhisperModel  # lazy

    with _stage(project_id, "asr") as job, _workdir() as wd:
        wav = _fetch(project_id, "dialogue", os.path.join(wd, "dialogue.wav"))

        with session() as db:
            proj = db.get(Project, project_id)
            lang = (proj.source_lang or "").strip() or None
            a = _asset(db, project_id, "diarization")
            turns = ((a.meta or {}).get("turns") or []) if a else []

        # Resolve through the mirror: MinIO first, HuggingFace only on a cold
        # cache, and never again after that (see app/mirror.py).
        from . import mirror

        name = settings.asr_model
        repo = name if "/" in name else f"Systran/faster-whisper-{name}"
        try:
            local = mirror.ensure(repo, settings.hf_token or None)
        except Exception as e:
            log.warning("mirror unavailable for %s (%s) — falling back to HF", repo, e)
            local = None

        workers = max(1, settings.asr_workers)
        model = WhisperModel(
            _snapshot(local) or name,
            device="cpu",
            compute_type=settings.asr_compute_type,
            cpu_threads=settings.asr_threads,
            num_workers=workers,
            download_root=settings.model_dir,
        )

        duration = _duration_of(_probe_meta(project_id))
        with session() as db:
            shots = [
                (s.t_start, s.t_end)
                for s in db.query(Shot).filter(Shot.project_id == project_id)
                .order_by(Shot.idx).all()
            ]
        sil = _silences(wav) if duration > settings.asr_chunk_seconds * 1.5 else []
        plan = _chunk_plan(duration, settings.asr_chunk_seconds, silences=sil, shots=shots)
        log.info("asr: %d chunk(s) over %.1fs (%d silences found), %d workers x %d threads",
                 len(plan), duration, len(sil), workers, settings.asr_threads)

        # Cut the chunks up front — ffmpeg is cheap next to inference.
        parts = []
        for i, (a, b) in enumerate(plan):
            if len(plan) == 1:
                parts.append((0.0, wav))
                continue
            p = os.path.join(wd, f"chunk-{i:03d}.wav")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin",
                "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", wav,
                "-c:a", "pcm_s16le", "-ar", str(DIALOGUE_SR), "-ac", "1", p,
            ])
            parts.append((a, p))

        def run_chunk(item):
            offset, path = item
            segs, inf = model.transcribe(
                path, language=lang, beam_size=5,
                word_timestamps=True, vad_filter=True,
            )
            return offset, list(segs), inf

        info = None
        collected: list[tuple[float, object]] = []
        if len(parts) == 1:
            off, segs, info = run_chunk(parts[0])
            collected = [(off, s) for s in segs]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for off, segs, inf in pool.map(run_chunk, parts):
                    info = info or inf
                    collected.extend((off, s) for s in segs)

        speakers = _speaker_map(project_id)
        rows = []
        for offset, s in sorted(collected, key=lambda x: x[0] + x[1].start):
            text = (s.text or "").strip()
            if not text:
                continue
            conf = None
            if s.avg_logprob is not None:
                # avg_logprob → rough 0..1 confidence for review triage.
                conf = max(0.0, min(1.0, 2.718281828 ** s.avg_logprob))
            start, end = float(s.start) + offset, float(s.end) + offset
            rows.append({
                "idx": len(rows),
                "t_start": start,
                "t_end": end,
                "text": text,
                "conf": conf,
                "speaker_id": _dominant_speaker(start, end, turns, speakers),
            })

        with session() as db:
            db.query(Segment).filter(Segment.project_id == project_id).delete()
            db.flush()
            for r in rows:
                db.add(Segment(
                    project_id=project_id, idx=r["idx"],
                    t_start=r["t_start"], t_end=r["t_end"],
                    text_src=r["text"], asr_confidence=r["conf"],
                    speaker_id=r["speaker_id"],
                    duration_budget_ms=int((r["t_end"] - r["t_start"]) * 1000),
                    state="draft",
                ))
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "analyzed"

        job["note"] = (
            f"{len(rows)} segments · model={settings.asr_model} · "
            f"detected={info.language} ({info.language_probability:.2f})"
        )
    return project_id


def _mean_db(path: str) -> float | None:
    """Mean volume in dB, or None. Used to tell a real bed from silence."""
    out = _run_stderr([
        "ffmpeg", "-hide_banner", "-nostdin", "-i", path,
        "-af", "volumedetect", "-f", "null", "-",
    ])
    for line in out.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _load_waveform(path: str) -> dict:
    """Read a wav into the {waveform, sample_rate} dict pyannote accepts.

    Avoids pyannote 4.x's torchcodec dependency, which fails to load its shared
    libraries against this torch build.
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile gives (samples, channels); pyannote wants (channels, samples).
    return {"waveform": torch.from_numpy(data.T), "sample_rate": sr}


def _snapshot(cache_dir: str | None) -> str | None:
    """The HF cache stores the usable tree under snapshots/<revision>/.

    faster-whisper wants a directory of real model files, not the cache root.
    """
    if not cache_dir:
        return None
    snaps = os.path.join(cache_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    revs = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    revs = [r for r in revs if os.path.isdir(r)]
    if not revs:
        return None
    return max(revs, key=os.path.getmtime)


def _speaker_map(project_id: str) -> dict[str, str]:
    with session() as db:
        return {
            s.label: s.id
            for s in db.query(Speaker).filter(Speaker.project_id == project_id).all()
        }


def _dominant_speaker(start: float, end: float, turns: list, speakers: dict) -> str | None:
    """Assign the speaker whose diarization turns overlap this segment most."""
    if not turns or not speakers:
        return next(iter(speakers.values()), None)
    overlap: dict[str, float] = {}
    for t in turns:
        o = min(end, t["end"]) - max(start, t["start"])
        if o > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + o
    if not overlap:
        return None
    best = max(overlap.items(), key=lambda kv: kv[1])[0]
    return speakers.get(best)


# ── orchestration ─────────────────────────────────────────────────────────


def enqueue_analysis(project_id: str) -> None:
    from celery import chain

    from .models import ANALYSIS_STAGES, Job

    with session() as db:
        db.query(Job).filter(
            Job.project_id == project_id, Job.stage.in_(ANALYSIS_STAGES)
        ).delete(synchronize_session=False)
        db.flush()
        for stage in ANALYSIS_STAGES:
            db.add(Job(project_id=project_id, stage=stage, state="queued", capability="cpu"))
        proj = db.get(Project, project_id)
        if proj:
            proj.status = "analyzing"

    chain(
        separate.si(project_id),
        diarize.si(project_id),
        asr.si(project_id),
    ).apply_async()
