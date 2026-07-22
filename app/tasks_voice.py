"""Phase 4 · voice — synth → fit → mix.

TTS engine: Chatterbox (Resemble AI). MIT on code AND weights, 23 languages
including Hindi, zero-shot from ~10s of reference audio — the only model found
that is simultaneously commercially licensed, Hindi-capable and cloning-capable
(docs/ARCHITECTURE.md §2). XTTS-v2 and F5-TTS are excluded: non-commercial
weights, and Coqui no longer exists to sell a licence.

⚠️ CPU REALITY: Chatterbox is ~30–50× slower than realtime on CPU. A 3-minute
clip is workable; a feature film is not, until there is a GPU. The stage is
tagged capability="gpu" so a GPU worker can take it over with no code change.

Time-fitting is a budget of four levers, in cost order (§0.4) — NOT DTW, which
aligns sequences and cannot make audio fit a slot:
  1. constrain the translation (done in Phase 3 — cheapest, no audio cost)
  2. TTS speaking rate
  3. time-stretch, hard-capped at ±15% before it audibly chipmunks
  4. absorb the remainder into the surrounding silence
Anything still over budget is flagged for a human, not silently squashed.
"""

import logging
import os
import subprocess

from .config import settings
from .db import session
from .models import (
    Asset, Project, Segment, Speaker, Take, Translation, Voice, VoiceUsage,
)
from .tasks import _asset, _put_asset, _run, _stage, _workdir, celery

log = logging.getLogger(__name__)

TTS_SR = 24000
MAX_STRETCH = 0.15  # ±15% — past this it chipmunks


def _dur(path: str) -> float:
    out = _run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ])
    import json
    return float(json.loads(out)["format"]["duration"])


def _atempo_chain(ratio: float) -> str:
    """ffmpeg's atempo accepts 0.5–2.0 per instance; chain for anything wider.
    We never exceed ±15% here, so one filter is always enough — but the chain
    keeps it honest if MAX_STRETCH is ever raised."""
    parts = []
    r = ratio
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.6f}")
    return ",".join(parts)


# ── synth ─────────────────────────────────────────────────────────────────


@celery.task(name="voice.synth")
def synth(project_id: str, lang: str) -> str:
    with _stage(project_id, "synth") as job, _workdir() as wd:
        from . import storage

        with session() as db:
            rows = (
                db.query(Segment, Translation, Speaker)
                .join(Translation, (Translation.segment_id == Segment.id)
                      & (Translation.lang == lang))
                .outerjoin(Speaker, Segment.speaker_id == Speaker.id)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.idx)
                .all()
            )
            if not rows:
                raise RuntimeError(f"nothing translated into {lang} — run Phase 3 first")

            # Cast check BEFORE loading a model or spending a second of CPU.
            uncast, blocked, items = set(), {}, []
            for seg, tr, spk in rows:
                if not spk:
                    uncast.add("(unassigned)")
                    continue
                v = db.get(Voice, spk.voice_id) if spk.voice_id else None
                if not v:
                    uncast.add(spk.display_name or spk.label)
                    continue
                if not v.usable:
                    blocked[v.display_name] = v.block_reason
                    continue
                clip = next((c for c in v.clips if c.is_default), None) or (v.clips[0] if v.clips else None)
                items.append({
                    "seg_id": seg.id, "idx": seg.idx,
                    "text": tr.text, "voice_id": v.id, "voice": v.display_name,
                    "clip_key": clip.key, "clip_text": clip.transcript,
                    "budget_ms": seg.duration_budget_ms or 0,
                    "est_ms": tr.est_duration_ms or 0,
                })

        if blocked:
            # Refuse rather than synthesize an unlicensed voice. This is the
            # whole point of the consent record.
            raise RuntimeError(
                "Blocked — these voices have no live consent: "
                + "; ".join(f"{k} ({v})" for k, v in blocked.items())
            )
        if not items:
            raise RuntimeError(
                "No speaker has a voice cast. Assign voices in the Speakers panel first"
                + (f" (uncast: {', '.join(sorted(uncast))})" if uncast else "")
            )

        engine, model = _load_tts()
        clips: dict[str, str] = {}
        made = 0

        with session() as db:
            db.query(Take).filter(Take.project_id == project_id, Take.lang == lang).delete(
                synchronize_session=False)

        # One bad line must NOT kill the job. A feature film is ~1500 lines and a
        # single unsynthesizable one — an eye-chart letter, a grunt, a stray
        # bracket — would otherwise throw away hours of finished work. Failures
        # are collected and reported; the operator fixes those lines and re-runs.
        failures: list[dict] = []
        nothing_to_say: list[dict] = []

        for it in items:
            speak = _speakable(it["text"])
            if speak is None:
                # No letters or digits — a music cue, a danda, an empty string.
                # Must not reach the model: empty input makes it hallucinate.
                nothing_to_say.append({"idx": it["idx"], "text": it["text"]})
                continue
            if it["clip_key"] not in clips:
                local = os.path.join(wd, f"ref-{len(clips)}.wav")
                storage.download_file(it["clip_key"], local)
                clips[it["clip_key"]] = local
            ref = clips[it["clip_key"]]

            out = os.path.join(wd, f"take-{it['idx']:05d}.wav")
            try:
                _synthesize_line(model, speak, ref, lang, out,
                                 expect_ms=it["est_ms"] or None)
                raw_ms = int(_dur(out) * 1000)
            except Exception as e:
                log.warning("synth failed on line %d (%r): %s", it["idx"], it["text"], e)
                failures.append({"idx": it["idx"], "text": it["text"], "why": str(e)[:120]})
                continue

            key = f"{project_id}/takes/{lang}/{it['idx']:05d}.wav"
            _upload(out, key)
            with session() as db:
                db.add(Take(
                    project_id=project_id, segment_id=it["seg_id"], lang=lang,
                    voice_id=it["voice_id"], key=key, engine=engine,
                    raw_ms=raw_ms, final_ms=raw_ms, stretch=1.0, chosen=1,
                ))
            made += 1
            if made % 5 == 0:
                log.info("synth %d/%d", made, len(items))

        if not made:
            raise RuntimeError(
                f"every line failed to synthesize. First: line {failures[0]['idx']} "
                f"({failures[0]['text']!r}) — {failures[0]['why']}" if failures else
                "nothing was synthesized"
            )

        # Usage log — what the actor is paid on, and proof scope was honoured.
        with session() as db:
            db.query(VoiceUsage).filter(
                VoiceUsage.project_id == project_id, VoiceUsage.lang == lang
            ).delete(synchronize_session=False)
            per: dict[str, list] = {}
            for it in items:
                per.setdefault(it["voice_id"], []).append(it)
            for vid, group in per.items():
                secs = sum(g["budget_ms"] for g in group) / 1000
                db.add(VoiceUsage(
                    voice_id=vid, project_id=project_id, lang=lang,
                    segments=len(group), seconds=round(secs, 2), engine=engine,
                ))

        note = f"{made}/{len(items)} lines synthesized · {engine} · voices: " + ", ".join(
            sorted({it["voice"] for it in items})
        )
        if nothing_to_say:
            note += (f" · {len(nothing_to_say)} line(s) had nothing speakable "
                     f"(music cues/punctuation) and were left silent — correct, not an error")
        if failures:
            shown = "; ".join(f"line {f['idx']} ({f['text']!r})" for f in failures[:4])
            note += (f" · ⚠️ {len(failures)} line(s) could not be synthesized and are "
                     f"SILENT in the mix: {shown}"
                     + (" …" if len(failures) > 4 else "")
                     + ". Edit those lines in the translation review and re-dub.")
        job["note"] = note
    return project_id


def _load_tts():
    """Chatterbox, CPU. Kept in one place so a GPU worker only changes `device`."""
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    torch.set_num_threads(settings.tts_threads)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return "chatterbox-multilingual", model


def _speakable(text: str) -> str | None:
    """The text to synthesize, or None if there is nothing to say.

    Returning None matters more than it looks. Measured against Chatterbox:

        ''      → 2.28s of audio     ← HALLUCINATED SPEECH FROM NOTHING
        '""'    → 2.08s of audio     ← same
        '...'   → IndexError
        '।'     → IndexError         (Devanagari danda)
        '♪'     → IndexError         (music cue)
        '?!'    → IndexError

    An exception is loud and gets handled. Empty input is worse: it silently
    invents 2.3 seconds of speech and drops it into the mix at a real timecode.
    A line with no letters or digits has nothing to dub, so it must never reach
    the model at all.
    """
    t = (text or "").strip()
    if not t:
        return None
    # isalnum() is true for Devanagari letters, so real Hindi passes; danda,
    # dashes, ellipses and music cues do not.
    if not any(c.isalnum() for c in t):
        return None
    return t


# Progressively more padded forms, tried in order. Chatterbox's tokenizer fails
# on inputs too short to reduce over — measured: 'I' ✗, 'I.' ✗, 'I..' ✓,
# 'I ...' ✗, '"I".' ✓. The rules are not guessable, so rather than encode a
# theory we try the real text first and only pad if the model actually refuses.
def _variants(text: str):
    yield text
    yield text.rstrip(".…") + ".."
    yield f'"{text.rstrip(".…")}".'


def _synthesize_line(model, text: str, ref_wav: str, lang: str, out_path: str,
                     expect_ms: float | None = None) -> str:
    """Synthesize, padding as needed, and REJECT runaway generations.

    Chatterbox is autoregressive and sometimes fails to stop: measured on a real
    script, 7 of 58 takes ran past their expected length and accounted for 24%
    of all audio. "Blur." — one syllable, an 860ms slot — generated 5.44
    SECONDS. "Can't see." produced 11s on one run and 2.3s on the next from
    identical input.

    No amount of translation compression fixes that, because the text was
    already short. But it is sampled, so a retry usually lands fine. Generate,
    check the duration against the affine model, and keep the shortest of a few
    attempts when the first one rambles.
    """
    last = None
    for candidate in _variants(text):
        best_ms = None
        attempts = max(1, settings.tts_retries)
        for attempt in range(attempts):
            try:
                _synthesize(model, candidate, ref_wav, lang, out_path)
            except Exception as e:  # noqa: BLE001
                last = e
                break
            if expect_ms is None:
                return candidate
            got = _dur(out_path) * 1000
            if got <= expect_ms * settings.tts_runaway_factor:
                return candidate
            # Runaway. Keep it only if nothing better turns up.
            if best_ms is None or got < best_ms:
                best_ms = got
                _keep(out_path)
            log.info("runaway take (%.0fms vs %.0fms expected) — retry %d/%d for %r",
                     got, expect_ms, attempt + 1, attempts, candidate[:40])
        if best_ms is not None:
            _restore(out_path)
            log.warning("kept shortest of %d takes (%.0fms) for %r", attempts, best_ms, candidate[:40])
            return candidate
    raise RuntimeError(f"unsynthesizable after padding: {last}")


def _keep(path: str) -> None:
    import shutil
    shutil.copy2(path, path + ".best")


def _restore(path: str) -> None:
    import os as _os
    import shutil
    if _os.path.exists(path + ".best"):
        shutil.move(path + ".best", path)


def _synthesize(model, text: str, ref_wav: str, lang: str, out_path: str) -> None:
    import torchaudio

    # cfg_weight is lever 2 of the time-fit budget (§0.4) — TTS speaking rate —
    # and it was never wired up. Chatterbox's own guidance is that lower values
    # improve pacing; the default 0.5 speaks slowly enough that dubs overran
    # their slots by 2.2x.
    wav = model.generate(
        text,
        language_id=lang,
        audio_prompt_path=ref_wav,
        cfg_weight=settings.tts_cfg_weight,
        exaggeration=settings.tts_exaggeration,
    )
    torchaudio.save(out_path, wav.cpu(), model.sr)


def _upload(local: str, key: str) -> None:
    from . import storage

    storage.upload_file(local, key, "audio/wav")


# ── fit ───────────────────────────────────────────────────────────────────


@celery.task(name="voice.fit")
def fit(project_id: str, lang: str) -> str:
    """Lever 3 + 4: stretch within ±15%, then let the rest ride into silence."""
    with _stage(project_id, "fit") as job, _workdir() as wd:
        from . import storage

        with session() as db:
            rows = (
                db.query(Take, Segment)
                .join(Segment, Take.segment_id == Segment.id)
                .filter(Take.project_id == project_id, Take.lang == lang, Take.chosen == 1)
                .order_by(Segment.idx)
                .all()
            )
            items = [{
                "take_id": t.id, "key": t.key, "idx": s.idx,
                "raw_ms": t.raw_ms or 0, "budget_ms": s.duration_budget_ms or 0,
            } for t, s in rows]

        if not items:
            raise RuntimeError("no takes to fit — run synth first")

        ok = stretched = over = 0
        for it in items:
            budget, raw = it["budget_ms"], it["raw_ms"]
            if not budget or not raw:
                continue
            ratio = raw / budget  # >1 means too long

            # Only OVERRUN is a defect. A short line leaves silence, and the
            # original has silence around it — lever 4 spends exactly that
            # (docs §0.4). Slowing a short line down to fill its slot makes the
            # delivery sound drugged, and labelling it "over_budget" sends the
            # operator to rewrite dialogue that is already correct.
            if ratio <= 1.0 + 0.02:
                state, applied, final = "ok", 1.0, raw
                ok += 1
            else:
                capped = min(1 + MAX_STRETCH, ratio)
                local = os.path.join(wd, f"f-{it['idx']:05d}.wav")
                storage.download_file(it["key"], local)
                out = os.path.join(wd, f"o-{it['idx']:05d}.wav")
                _run([
                    "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", local,
                    "-filter:a", _atempo_chain(capped), "-c:a", "pcm_s16le", out,
                ])
                _upload(out, it["key"])
                final = int(_dur(out) * 1000)
                applied = capped
                if ratio - 1.0 > MAX_STRETCH:
                    # Couldn't be squashed enough without artefacts — say so.
                    state, over = "over_budget", over + 1
                else:
                    state, stretched = "stretched", stretched + 1

            with session() as db:
                t = db.get(Take, it["take_id"])
                t.final_ms, t.stretch, t.fit_state = final, round(applied, 4), state

        job["note"] = (
            f"{ok} fit as-is · {stretched} compressed within {int(MAX_STRETCH*100)}% · "
            f"{over} still over after the cap — those run into the following pause, or need "
            "the line shortened (step 4). Lines that come in SHORT are left alone: silence is free."
        )
    return project_id


# ── mix ───────────────────────────────────────────────────────────────────


@celery.task(name="voice.mix")
def mix(project_id: str, lang: str) -> str:
    """Cloned dialogue + the preserved music/FX bed + original video."""
    with _stage(project_id, "mix") as job, _workdir() as wd:
        from . import storage

        with session() as db:
            rows = (
                db.query(Take, Segment)
                .join(Segment, Take.segment_id == Segment.id)
                .filter(Take.project_id == project_id, Take.lang == lang, Take.chosen == 1)
                .order_by(Segment.idx)
                .all()
            )
            takes = [{"key": t.key, "start": s.t_start, "idx": s.idx} for t, s in rows]
            bed = _asset(db, project_id, "music_fx")
            bed_key = bed.key if bed else None
            proxy = _asset(db, project_id, "proxy")
            proxy_key = proxy.key if proxy else None
            srca = _asset(db, project_id, "source")
            source_key = srca.key if srca else None
            dur = 0.0
            src = _asset(db, project_id, "source")
            if src and src.meta:
                dur = float((src.meta.get("format") or {}).get("duration") or 0)

        if not takes:
            raise RuntimeError("no takes to mix")
        if not proxy_key:
            raise RuntimeError("no proxy video")

        # Ducking needs the original audio. The proxy carries it too, and is far
        # smaller than the master — use it unless there is no proxy audio.
        pxy_src = os.path.join(wd, "proxy_for_audio.mp4")
        storage.download_file(proxy_key, pxy_src)

        # Lay each take at its timecode on a silent bed of the right length.
        inputs, filters = [], []
        for i, t in enumerate(takes):
            local = os.path.join(wd, f"t-{t['idx']:05d}.wav")
            storage.download_file(t["key"], local)
            inputs += ["-i", local]
            filters.append(f"[{i}:a]adelay={int(t['start']*1000)}|{int(t['start']*1000)},"
                           f"aresample=48000[d{i}]")

        n = len(takes)
        dialogue = os.path.join(wd, "dub.wav")
        mixstr = "".join(f"[d{i}]" for i in range(n))
        _run([
            "ffmpeg", "-y", "-hide_banner", "-nostdin", *inputs,
            "-filter_complex",
            ";".join(filters) + f";{mixstr}amix=inputs={n}:duration=longest:normalize=0[out]",
            "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            "-t", f"{dur:.3f}" if dur else "0", dialogue,
        ])

        # The background is the reason a dub still sounds like the film. Two
        # ways to keep it, best first.
        final_audio = dialogue
        bed_note = ""

        if bed_key:
            # A real separated bed: replace the original audio entirely.
            bedlocal = os.path.join(wd, "bed.wav")
            storage.download_file(bed_key, bedlocal)
            mixed = os.path.join(wd, "mixed.wav")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin",
                "-i", dialogue, "-i", bedlocal,
                "-filter_complex",
                "[0:a]volume=1.0[a];[1:a]volume=1.0[b];[a][b]amix=inputs=2:duration=first:normalize=0[out]",
                "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", mixed,
            ])
            final_audio = mixed
            bed_note = "mixed over the separated music/FX bed"
        else:
            # No bed to mix against. Rather than throw the score away, DUCK the
            # original audio under the dub — full level between lines, pulled
            # down while someone is speaking.
            #
            # This is what a dubbing suite does when handed no M&E track. It is
            # NOT separation: the original dialogue is still there, quietly,
            # under the new one. But the alternative is a film whose music and
            # atmosphere vanish, which is worse and sounds broken.
            #
            # sidechaincompress ducks input 0 whenever input 1 has energy, so the
            # duck follows the dub itself — no timing table to drift out of sync.
            orig = os.path.join(wd, "orig.wav")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", pxy_src,
                "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", orig,
            ])
            ducked = os.path.join(wd, "ducked.wav")
            lvl = settings.duck_level
            _run([
                "ffmpeg", "-y", "-hide_banner", "-nostdin",
                "-i", orig, "-i", dialogue,
                "-filter_complex",
                # [0]=original (gets ducked), [1]=the dub (the trigger)
                f"[0:a]volume={lvl}[bg];"
                f"[1:a]asplit=2[dub][key];"
                f"[bg][key]sidechaincompress="
                f"threshold={settings.duck_threshold}:ratio={settings.duck_ratio}:"
                f"attack=25:release=380:makeup=1[bgduck];"
                f"[bgduck][dub]amix=inputs=2:duration=first:normalize=0[out]",
                "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", ducked,
            ])
            final_audio = ducked
            bed_note = (
                f"no separated bed — the ORIGINAL audio was kept and ducked under the dub "
                f"(background at {int(lvl*100)}%, pulled down while lines play). Music and "
                "atmosphere survive between lines, but the original dialogue is still faintly "
                "audible underneath. A separated bed would remove it"
            )

        pxy = pxy_src
        out = os.path.join(wd, "final.mp4")
        _run([
            "ffmpeg", "-y", "-hide_banner", "-nostdin",
            "-i", pxy, "-i", final_audio,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", out,
        ])
        key = f"{project_id}/renders/{lang}/final.mp4"
        _put_asset(project_id, f"final_{lang}", key, out, "video/mp4",
                   meta={"lang": lang, "takes": n})

        with session() as db:
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "dubbed"

        job["note"] = f"{n} lines mixed · {bed_note}"
    return project_id


def enqueue_dub(project_id: str, lang: str) -> None:
    from celery import chain

    from .models import VOICE_STAGES, Job

    with session() as db:
        db.query(Job).filter(
            Job.project_id == project_id, Job.stage.in_(VOICE_STAGES)
        ).delete(synchronize_session=False)
        db.flush()
        for stage in VOICE_STAGES:
            db.add(Job(project_id=project_id, stage=stage, state="queued",
                       capability="gpu" if stage == "synth" else "cpu"))
        proj = db.get(Project, project_id)
        if proj:
            proj.status = "dubbing"

    chain(
        synth.si(project_id, lang),
        fit.si(project_id, lang),
        mix.si(project_id, lang),
    ).apply_async()


# ── synthetic voice bank ──────────────────────────────────────────────────


@celery.task(name="voice.build_bank")
def build_bank(target_male: int = 40, target_female: int = 40, target_child: int = 40,
               replace: bool = False) -> dict:
    """Generate stock AI voices. Runs in the tts image — that's where Kokoro is."""
    from . import voicebank

    made = voicebank.build(target_male, target_female, target_child, replace)
    log.info("voice bank: %s", made)
    return made
