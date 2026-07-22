"""Phase 1 ingest pipeline: probe → proxy → thumbnails → shots.

Each stage is its own Celery task and its own Job row, so the UI can show
per-stage progress and a failure names the stage that failed. Stages are
chained; each returns project_id to feed the next.

All stages here are `capability="cpu"` and run on this box. GPU stages
(separation, TTS, lip-sync) will subscribe to a `gpu` queue instead — same
interface, different host. See docs/ARCHITECTURE.md §6.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

from celery import Celery, chain

from .config import settings
from .db import session
from .models import Asset, Job, Project, Shot

log = logging.getLogger(__name__)

celery = Celery(
    "dubbing",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks", "app.tasks_ai", "app.tasks_mt", "app.tasks_voice"],
)
celery.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # One container per model family (docs/ARCHITECTURE.md §6): whisper, pyannote
    # and ffmpeg want mutually incompatible dependency trees. Route by queue so
    # each worker image only ever runs what it can import. GPU stages will add a
    # `gpu` queue and can live on another host entirely.
    # Patterns, not exact names: an unrouted task silently lands in the default
    # `celery` queue that no worker consumes, and just sits there looking
    # "queued" forever. Wildcards mean adding a stage can't reintroduce that.
    task_routes={
        "ingest.*": {"queue": "media"},
        "analyze.separate": {"queue": "media"},  # ffmpeg only
        "analyze.*": {"queue": "ai"},
        "translate.*": {"queue": "ai"},
        # synth is GPU-shaped work; it runs on the ai queue today because this
        # box has no GPU. A GPU worker subscribes to "gpu" and takes it over.
        "voice.synth": {"queue": os.environ.get("SYNTH_QUEUE", "tts")},
        "voice.build_bank": {"queue": "tts"},
        "voice.*": {"queue": "media"},
    },
    # Backstop: if something still slips through unrouted, put it where a
    # worker will at least pick it up.
    task_default_queue="ai",
)


# ── helpers ───────────────────────────────────────────────────────────────


def _run_stderr(cmd: list[str]) -> str:
    """Some ffmpeg filters (silencedetect, volumedetect) report on stderr and
    exit non-zero-ish; we want the text, not a raise."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stderr or ""


def _run(cmd: list[str]) -> str:
    """Run a subprocess, raising with stderr attached on failure."""
    log.info("exec: %s", " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}):\n" + "\n".join(tail))
    return p.stdout


@contextmanager
def _stage(project_id: str, stage: str, skip_if=None):
    """Mark the Job row running/done/failed/skipped around a stage body.

    Yields a mutable dict; set ctx["note"] to explain a degraded or skipped
    result. Notes surface in the UI — a stage that quietly did less than the
    operator thinks it did is worse than one that failed.
    """
    ctx: dict = {"note": None}
    skipped = bool(skip_if and skip_if())

    with session() as db:
        job = (
            db.query(Job)
            .filter(Job.project_id == project_id, Job.stage == stage)
            .one_or_none()
        )
        if job is None:
            job = Job(project_id=project_id, stage=stage)
            db.add(job)
        job.state = "running"
        job.attempts = (job.attempts or 0) + 1
        job.error = None
        job.note = None
        job.started_at = datetime.now(timezone.utc)
    try:
        yield ctx
    except Exception as e:
        with session() as db:
            job = db.query(Job).filter(Job.project_id == project_id, Job.stage == stage).one()
            job.state = "failed"
            job.error = str(e)[:4000]
            job.note = ctx.get("note")
            job.finished_at = datetime.now(timezone.utc)
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "failed"
        log.exception("stage %s failed for %s", stage, project_id)
        raise
    else:
        with session() as db:
            job = db.query(Job).filter(Job.project_id == project_id, Job.stage == stage).one()
            job.state = "skipped" if skipped else "done"
            job.note = ctx.get("note")
            job.finished_at = datetime.now(timezone.utc)


def _asset(db, project_id: str, kind: str) -> Asset | None:
    return db.query(Asset).filter(Asset.project_id == project_id, Asset.kind == kind).one_or_none()


def _put_asset(project_id: str, kind: str, key: str, local: str, content_type: str, meta=None):
    from . import storage

    storage.upload_file(local, key, content_type)
    with session() as db:
        existing = _asset(db, project_id, kind)
        if existing:
            db.delete(existing)
            db.flush()
        db.add(
            Asset(
                project_id=project_id,
                kind=kind,
                key=key,
                filename=os.path.basename(local),
                size=os.path.getsize(local),
                content_type=content_type,
                meta=meta or {},
            )
        )


@contextmanager
def _workdir():
    # A worker on a fresh machine has no scratch directory — only the API used
    # to create one, so a REMOTE worker failed on its first job with a bare
    # FileNotFoundError. Every process that needs it should be able to make it.
    os.makedirs(settings.scratch_dir, exist_ok=True)
    d = tempfile.mkdtemp(dir=settings.scratch_dir)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fetch_source(project_id: str, dest_dir: str) -> str:
    from . import storage

    with session() as db:
        a = _asset(db, project_id, "source")
        if a is None:
            raise RuntimeError("no source asset for project")
        key, name = a.key, a.filename or "source"
    local = os.path.join(dest_dir, name)
    storage.download_file(key, local)
    return local


def _fetch_proxy(project_id: str, dest_dir: str) -> str:
    from . import storage

    with session() as db:
        a = _asset(db, project_id, "proxy")
        if a is None:
            raise RuntimeError("no proxy asset for project")
        key = a.key
    local = os.path.join(dest_dir, "proxy.mp4")
    storage.download_file(key, local)
    return local


def _probe_meta(project_id: str) -> dict:
    with session() as db:
        a = _asset(db, project_id, "source")
        return (a.meta or {}) if a else {}


def _duration_of(meta: dict) -> float:
    try:
        return float(meta.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _video_stream(meta: dict) -> dict:
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return {}


# ── stages ────────────────────────────────────────────────────────────────


@celery.task(name="ingest.probe")
def probe(project_id: str) -> str:
    with _stage(project_id, "probe"), _workdir() as wd:
        src = _fetch_source(project_id, wd)
        out = _run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", src,
        ])
        meta = json.loads(out)
        if not _video_stream(meta):
            raise RuntimeError("no video stream found — is this a video file?")
        with session() as db:
            a = _asset(db, project_id, "source")
            a.meta = meta
    return project_id


@celery.task(name="ingest.proxy")
def proxy(project_id: str) -> str:
    """H.264 720p CRF23 +faststart. The UI never touches the master."""
    with _stage(project_id, "proxy"), _workdir() as wd:
        src = _fetch_source(project_id, wd)
        out = os.path.join(wd, "proxy.mp4")
        h = settings.proxy_height
        _run([
            "ffmpeg", "-y", "-hide_banner", "-nostdin",
            "-threads", str(settings.ffmpeg_threads),
            "-i", src,
            # Never upscale: clamp to the source height. -2 keeps width even.
            "-vf", f"scale=-2:'min({h},ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",
            out,
        ])
        _put_asset(project_id, "proxy", f"{project_id}/proxy/proxy_{h}p.mp4", out, "video/mp4")
    return project_id


@celery.task(name="ingest.thumbnails")
def thumbnails(project_id: str) -> str:
    """Poster frame + sprite sheet + WebVTT for scrub previews."""
    with _stage(project_id, "thumbnails"), _workdir() as wd:
        pxy = _fetch_proxy(project_id, wd)
        meta = _probe_meta(project_id)
        duration = _duration_of(meta)
        if duration <= 0:
            raise RuntimeError("could not determine duration from probe")

        # Poster at 10% in — avoids black/fade-in first frames.
        poster = os.path.join(wd, "poster.jpg")
        _run([
            "ffmpeg", "-y", "-hide_banner", "-nostdin",
            "-ss", f"{max(0.0, duration * 0.1):.3f}", "-i", pxy,
            "-frames:v", "1", "-q:v", "3", poster,
        ])
        _put_asset(project_id, "poster", f"{project_id}/thumbs/poster.jpg", poster, "image/jpeg")

        # Sprite sheet: aim for <= sprite_max_tiles, one every `interval` seconds.
        interval = max(1.0, duration / settings.sprite_max_tiles)
        n_tiles = max(1, min(settings.sprite_max_tiles, int(duration / interval)))
        cols = min(settings.sprite_cols, n_tiles)
        rows = max(1, math.ceil(n_tiles / cols))

        vs = _video_stream(meta)
        sw, sh = int(vs.get("width") or 16), int(vs.get("height") or 9)
        tw = settings.sprite_tile_width
        th = max(2, int(round(tw * sh / sw)) // 2 * 2)  # even height

        sprite = os.path.join(wd, "sprite.jpg")
        _run([
            "ffmpeg", "-y", "-hide_banner", "-nostdin",
            "-i", pxy,
            "-vf", f"fps=1/{interval:.6f},scale={tw}:{th},tile={cols}x{rows}",
            "-frames:v", "1", "-q:v", "4", sprite,
        ])
        _put_asset(
            project_id, "sprite", f"{project_id}/thumbs/sprite.jpg", sprite, "image/jpeg",
            meta={"cols": cols, "rows": rows, "tile_w": tw, "tile_h": th,
                  "interval": interval, "count": n_tiles},
        )

        # WebVTT pointing at regions of the sprite.
        vtt_path = os.path.join(wd, "sprite.vtt")
        with open(vtt_path, "w") as f:
            f.write("WEBVTT\n\n")
            for i in range(n_tiles):
                t0, t1 = i * interval, min((i + 1) * interval, duration)
                x, y = (i % cols) * tw, (i // cols) * th
                f.write(f"{_ts(t0)} --> {_ts(t1)}\n")
                f.write(f"sprite.jpg#xywh={x},{y},{tw},{th}\n\n")
        _put_asset(project_id, "sprite_vtt", f"{project_id}/thumbs/sprite.vtt", vtt_path, "text/vtt")
    return project_id


def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


@celery.task(name="ingest.shots")
def shots(project_id: str) -> str:
    """Shot boundaries. Phase 5 uses these to gate lip-sync per shot; the ASR
    stage uses them as chunk boundaries for parallel transcription (§2)."""
    from scenedetect import ContentDetector, detect

    with _stage(project_id, "shots"), _workdir() as wd:
        pxy = _fetch_proxy(project_id, wd)
        scenes = detect(pxy, ContentDetector())
        with session() as db:
            db.query(Shot).filter(Shot.project_id == project_id).delete()
            db.flush()
            if not scenes:
                # Single continuous take — still record one shot.
                dur = _duration_of(_probe_meta(project_id))
                db.add(Shot(project_id=project_id, idx=0, t_start=0.0, t_end=dur))
            else:
                for i, (start, end) in enumerate(scenes):
                    db.add(Shot(
                        project_id=project_id, idx=i,
                        t_start=start.get_seconds(), t_end=end.get_seconds(),
                    ))
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "ready"
    return project_id


# ── orchestration ─────────────────────────────────────────────────────────


def enqueue_ingest(project_id: str) -> None:
    """Reset job rows and kick off the chain."""
    from .models import STAGES

    with session() as db:
        db.query(Job).filter(Job.project_id == project_id).delete()
        db.flush()
        for stage in STAGES:
            db.add(Job(project_id=project_id, stage=stage, state="queued", capability="cpu"))
        proj = db.get(Project, project_id)
        if proj:
            proj.status = "ingesting"

    chain(
        probe.si(project_id),
        proxy.si(project_id),
        thumbnails.si(project_id),
        shots.si(project_id),
    ).apply_async()
