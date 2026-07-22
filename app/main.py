import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import auth, storage
from .config import settings
from .db import Base, engine, get_db, session
from .models import (
    ANALYSIS_STAGES, FIT_TOLERANCE, FLORES, INGEST_STAGES, LANG_NAMES, STAGES,
    TRANSLATION_STAGES, VOICE_STAGES, Asset, ConsentRecord, Job, Project, ReferenceClip,
    Segment, Shot, Speaker, Take, Translation, Voice, VoiceUsage,
)
from .tasks import enqueue_ingest
from .tasks_ai import enqueue_analysis
from .tasks_mt import enqueue_translation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Dubbing Platform", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
# Assets stamped with a build token by _index_html(), so a deploy actually
# reaches the browser.
_VERSIONED_ASSETS = ("app.js", "style.css")
CHUNK = 1024 * 512


def _sync_new_columns():
    """Additive-only schema sync.

    `create_all` creates missing TABLES but never adds columns to existing
    ones — so adding a field to a live model silently 500s at query time.
    Each phase adds columns, so this closes that gap: it diffs the models
    against information_schema and issues ALTER TABLE ADD COLUMN for anything
    missing. Additive and idempotent; it never drops or retypes.

    ⚠️ This is a development convenience, not a migration tool. Before there is
    data worth keeping, replace it with Alembic — it cannot handle renames,
    type changes, backfills or rollbacks.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                if not col.nullable and col.default is None:
                    log.warning(
                        "skipping NOT NULL column %s.%s — needs a real migration",
                        table.name, col.name,
                    )
                    continue
                ddl = col.type.compile(engine.dialect)
                log.info("schema: adding %s.%s %s", table.name, col.name, ddl)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


app.middleware("http")(auth.middleware)


class LoginIn(BaseModel):
    password: str


@app.post("/api/login")
def login(body: LoginIn, response: Response):
    if not auth.enabled():
        raise HTTPException(400, "no password is configured on this instance")
    if not auth.check_password(body.password):
        raise HTTPException(401, "incorrect password")
    response.set_cookie(
        auth.COOKIE, auth.issue(),
        max_age=auth.TTL, httponly=True, samesite="lax",
        # secure=True once this is behind TLS; it would break plain-http LAN use.
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/login")
def login_page():
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/api/login-hint")
def login_hint():
    """The password, but only on a genuinely local-only instance (see auth.hint)."""
    return {"password": auth.hint(), "local_only": auth.bound_locally()}


# Every queue a worker actually consumes, plus Celery's own default — a task
# that routes nowhere lands in `celery`, which nothing reads (see the routing
# note in tasks.py) and would otherwise look queued forever.
_QUEUES = ("media", "ai", "tts", "celery")
# Covers the gap between a worker taking a message off the queue and marking
# its row running.
_ORPHAN_GRACE_SECONDS = 60


def _reconcile_orphan_jobs() -> None:
    """Fail jobs left `queued` by a restart that lost their Celery task.

    A reboot — or a broker that comes back empty — strands rows at `queued`
    forever: the message is gone, nothing will ever run it, and the UI shows a
    spinner that never resolves. Job rows carry no Celery task id, so there is
    nothing to match a row against; this reconciles at the whole-system level
    instead. If EVERY queue is empty then nothing can legitimately be waiting,
    so anything still marked `queued` is a leftover.

    Deliberately narrow:
      * `running` is never touched — a live worker can be minutes into an ASR
        pass, and its message is already off the queue.
      * a project with ANY running stage is skipped entirely. enqueue_* writes
        every downstream row as `queued` up front but publishes only the first
        task of the chain, so mid-run those rows are legitimately waiting with
        no message on any queue. Without this, restarting the API during a long
        stage would fail the rest of a perfectly healthy chain.
      * anything queued within the grace period is left alone.
      * if the broker cannot be reached, nothing is changed.

    Marked `failed` rather than deleted, so the stage explains itself in the UI
    and can be re-run like any other failure.
    """
    from .tasks import celery as celery_app

    try:
        with celery_app.connection_or_acquire() as conn:
            client = conn.default_channel.client
            if any(client.llen(q) for q in _QUEUES):
                return          # real work is waiting; nothing here is orphaned
    except Exception as e:
        log.warning("orphan job check skipped — broker unreachable (%s)", e)
        return

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_ORPHAN_GRACE_SECONDS)
    with session() as db:
        mid_run = {
            pid for (pid,) in
            db.query(Job.project_id).filter(Job.state == "running").distinct()
        }
        stale = []
        for j in db.query(Job).filter(Job.state == "queued").all():
            if j.project_id in mid_run:
                continue        # live chain — its later stages are not orphans
            ts = j.started_at or j.created_at
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                stale.append(j)
        for j in stale:
            j.state = "failed"
            j.error = ("Interrupted — the queue was lost while this stage was "
                       "waiting (worker restart or reboot). Nothing ran. Re-run it.")
        if stale:
            log.info("reconciled %d orphaned job(s): %s", len(stale),
                     ", ".join(sorted({j.stage for j in stale})))


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    _sync_new_columns()
    _reconcile_orphan_jobs()
    os.makedirs(settings.scratch_dir, exist_ok=True)
    storage.ensure_bucket()
    auth.warn_if_open()
    if auth.enabled():
        # Always safe to print here: container logs are local, and you reach them
        # with `docker compose logs api` — which already implies box access.
        log.info("─" * 62)
        log.info("  Sign in at http://localhost:%s", os.environ.get("UI_PORT", "8621"))
        log.info("  Password: %s", settings.app_password)
        log.info("  (change APP_PASSWORD in .env, then: docker compose up -d api)")
        log.info("─" * 62)
    log.info("ready (auth %s)", "on" if auth.enabled() else "OFF")


# ── schemas ───────────────────────────────────────────────────────────────


class ProjectIn(BaseModel):
    title: str
    source_lang: str = "hi"
    target_langs: list[str] = []


def _project_json(p: Project, db: Session) -> dict:
    jobs = db.query(Job).filter(Job.project_id == p.id).all()
    by_stage = {j.stage: j for j in jobs}
    assets = {a.kind: a for a in db.query(Asset).filter(Asset.project_id == p.id).all()}
    src = assets.get("source")
    meta = (src.meta or {}) if src else {}
    fmt = meta.get("format", {})
    vstream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    astream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
    shot_count = db.query(Shot).filter(Shot.project_id == p.id).count()

    return {
        "id": p.id,
        "title": p.title,
        "source_lang": p.source_lang,
        "target_langs": p.target_langs or [],
        "status": p.status,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "source": {
            "filename": src.filename,
            "size": src.size,
            "duration": float(fmt.get("duration") or 0) or None,
            "bitrate": int(fmt.get("bit_rate") or 0) or None,
            "container": fmt.get("format_long_name"),
            "video_codec": vstream.get("codec_name"),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": _fps(vstream.get("r_frame_rate")),
            "audio_codec": astream.get("codec_name"),
            "audio_channels": astream.get("channels"),
            "sample_rate": astream.get("sample_rate"),
        } if src else None,
        "jobs": [
            {
                "stage": s,
                "phase": (1 if s in INGEST_STAGES else 2 if s in ANALYSIS_STAGES
                          else 3 if s in TRANSLATION_STAGES else 4),
                "state": by_stage[s].state if s in by_stage else "pending",
                "error": by_stage[s].error if s in by_stage else None,
                "note": by_stage[s].note if s in by_stage else None,
                "capability": by_stage[s].capability if s in by_stage else "cpu",
            }
            for s in STAGES
        ],
        "media": {
            "proxy": f"/api/media/{assets['proxy'].key}" if "proxy" in assets else None,
            "poster": f"/api/media/{assets['poster'].key}" if "poster" in assets else None,
            "sprite": f"/api/media/{assets['sprite'].key}" if "sprite" in assets else None,
        },
        "sprite_meta": (assets["sprite"].meta or {}) if "sprite" in assets else None,
        "shot_count": shot_count,
        "separation": (assets["dialogue"].meta or {}).get("method") if "dialogue" in assets else None,
        "has_music_fx": "music_fx" in assets,
        "speaker_count": db.query(Speaker).filter(
            Speaker.project_id == p.id, Speaker.merged_into.is_(None)
        ).count(),
        "segment_count": db.query(Segment).filter(Segment.project_id == p.id).count(),
        "translated_langs": [
            r[0] for r in db.query(Translation.lang)
            .filter(Translation.project_id == p.id).distinct().all()
        ],
        "languages": [{"code": c, "name": LANG_NAMES.get(c, c)} for c in FLORES],
        "fit_tolerance": FIT_TOLERANCE,
        "renders": {
            k[len("final_"):]: f"/api/media/{a.key}"
            for k, a in assets.items() if k.startswith("final_")
        },
    }


def _fps(rate: str | None):
    if not rate or "/" not in rate:
        return None
    num, den = rate.split("/")
    try:
        den = float(den)
        return round(float(num) / den, 3) if den else None
    except ValueError:
        return None


# ── api ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(__import__("sqlalchemy").text("SELECT 1"))
    return {"ok": True, "stages": STAGES}


@app.get("/api/models")
def models_status():
    """Which weights are mirrored into MinIO — i.e. what this deployment can
    run without contacting HuggingFace."""
    from . import llm, mirror

    try:
        rows = mirror.status()
    except Exception as e:
        raise HTTPException(503, f"object storage unreachable: {e}")
    return {
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "hf_token": bool(settings.hf_token),
        "llm": llm.label(),
        "models": rows,
        "mirrored_bytes": sum(r["bytes"] for r in rows),
    }


@app.get("/api/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [_project_json(p, db) for p in projects]


@app.post("/api/projects", status_code=201)
def create_project(body: ProjectIn, db: Session = Depends(get_db)):
    if not body.title.strip():
        raise HTTPException(400, "title is required")
    p = Project(
        title=body.title.strip(),
        source_lang=body.source_lang,
        target_langs=body.target_langs,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _project_json(p, db)


@app.get("/api/projects/{pid}")
def get_project(pid: str, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    return _project_json(p, db)


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: str, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    storage.delete_prefix(f"{pid}/")
    db.delete(p)
    db.commit()
    return Response(status_code=204)


@app.get("/api/projects/{pid}/shots")
def get_shots(pid: str, db: Session = Depends(get_db)):
    rows = db.query(Shot).filter(Shot.project_id == pid).order_by(Shot.idx).all()
    return [
        {"idx": s.idx, "t_start": s.t_start, "t_end": s.t_end,
         "duration": round(s.t_end - s.t_start, 3)}
        for s in rows
    ]


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@app.post("/api/projects/{pid}/source", status_code=202)
async def upload_source(pid: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")

    safe = _SAFE.sub("_", os.path.basename(file.filename or "source"))[:120] or "source"
    tmp = os.path.join(settings.scratch_dir, f"upload-{uuid.uuid4()}")
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1024 * 1024 * 8):
                out.write(chunk)
                size += len(chunk)
        if size == 0:
            raise HTTPException(400, "empty upload")

        key = f"{pid}/source/{safe}"
        storage.upload_file(tmp, key, file.content_type or "application/octet-stream")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # Replace any previous source and clear derived state.
    storage.delete_prefix(f"{pid}/proxy/")
    storage.delete_prefix(f"{pid}/thumbs/")
    db.query(Asset).filter(Asset.project_id == pid).delete()
    db.query(Shot).filter(Shot.project_id == pid).delete()
    db.add(Asset(
        project_id=pid, kind="source", key=key, filename=safe,
        size=size, content_type=file.content_type, meta={},
    ))
    db.commit()

    enqueue_ingest(pid)
    return {"ok": True, "size": size, "filename": safe}


@app.post("/api/projects/{pid}/reingest", status_code=202)
def reingest(pid: str, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    src = db.query(Asset).filter(Asset.project_id == pid, Asset.kind == "source").one_or_none()
    if not src:
        raise HTTPException(400, "no source uploaded yet")
    enqueue_ingest(pid)
    return {"ok": True}


# ── phase 2 · analysis ────────────────────────────────────────────────────


class SpeakerPatch(BaseModel):
    display_name: str | None = None


class MergeIn(BaseModel):
    into: str


class SegmentPatch(BaseModel):
    text_src_edited: str | None = None
    speaker_id: str | None = None
    state: str | None = None


@app.post("/api/projects/{pid}/analyze", status_code=202)
def analyze(pid: str, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    src = db.query(Asset).filter(Asset.project_id == pid, Asset.kind == "source").one_or_none()
    if not src:
        raise HTTPException(400, "upload a video first")
    ingest_done = {
        j.stage: j.state
        for j in db.query(Job).filter(Job.project_id == pid, Job.stage.in_(INGEST_STAGES)).all()
    }
    if not all(ingest_done.get(s) == "done" for s in INGEST_STAGES):
        raise HTTPException(400, "ingest has not finished yet")
    enqueue_analysis(pid)
    return {"ok": True}


@app.get("/api/projects/{pid}/speakers")
def get_speakers(pid: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Speaker)
        .filter(Speaker.project_id == pid, Speaker.merged_into.is_(None))
        .order_by(Speaker.speech_seconds.desc())
        .all()
    )
    counts = dict(
        db.query(Segment.speaker_id, __import__("sqlalchemy").func.count(Segment.id))
        .filter(Segment.project_id == pid)
        .group_by(Segment.speaker_id)
        .all()
    )
    return [
        {
            "id": s.id,
            "label": s.label,
            "display_name": s.display_name,
            "name": s.name,
            "color": s.color,
            "speech_seconds": s.speech_seconds,
            "segment_count": counts.get(s.id, 0),
            "voice_id": s.voice_id,
        }
        for s in rows
    ]


@app.patch("/api/speakers/{sid}")
def patch_speaker(sid: str, body: SpeakerPatch, db: Session = Depends(get_db)):
    s = db.get(Speaker, sid)
    if not s:
        raise HTTPException(404, "speaker not found")
    if body.display_name is not None:
        s.display_name = body.display_name.strip() or None
    db.commit()
    return {"id": s.id, "name": s.name}


@app.post("/api/speakers/{sid}/merge")
def merge_speaker(sid: str, body: MergeIn, db: Session = Depends(get_db)):
    """Fold one diarized speaker into another. Diarization routinely splits one
    actor across several labels; merging is the operator's fix (§1 step 4)."""
    src = db.get(Speaker, sid)
    dst = db.get(Speaker, body.into)
    if not src or not dst:
        raise HTTPException(404, "speaker not found")
    if src.id == dst.id:
        raise HTTPException(400, "cannot merge a speaker into itself")
    if src.project_id != dst.project_id:
        raise HTTPException(400, "speakers belong to different projects")

    db.query(Segment).filter(Segment.speaker_id == src.id).update({"speaker_id": dst.id})
    dst.speech_seconds = (dst.speech_seconds or 0) + (src.speech_seconds or 0)
    src.merged_into = dst.id
    db.commit()
    return {"ok": True, "into": dst.id, "name": dst.name}


@app.get("/api/projects/{pid}/segments")
def get_segments(pid: str, db: Session = Depends(get_db)):
    rows = db.query(Segment).filter(Segment.project_id == pid).order_by(Segment.idx).all()
    return [
        {
            "id": s.id,
            "idx": s.idx,
            "speaker_id": s.speaker_id,
            "t_start": s.t_start,
            "t_end": s.t_end,
            "duration": round(s.duration, 3),
            "text_src": s.text_src,
            "text_src_edited": s.text_src_edited,
            "text": s.text,
            "edited": s.text_src_edited is not None,
            "asr_confidence": s.asr_confidence,
            "state": s.state,
        }
        for s in rows
    ]


@app.patch("/api/segments/{sid}")
def patch_segment(sid: str, body: SegmentPatch, db: Session = Depends(get_db)):
    s = db.get(Segment, sid)
    if not s:
        raise HTTPException(404, "segment not found")
    if body.text_src_edited is not None:
        t = body.text_src_edited.strip()
        # Never destroy text_src — it's how we measure the model and re-run safely.
        s.text_src_edited = None if t == (s.text_src or "").strip() else t
    if body.speaker_id is not None:
        if not db.get(Speaker, body.speaker_id):
            raise HTTPException(400, "unknown speaker")
        s.speaker_id = body.speaker_id
    if body.state is not None:
        if body.state not in ("draft", "approved", "needs_review"):
            raise HTTPException(400, "bad state")
        s.state = body.state
    db.commit()
    return {"id": s.id, "text": s.text, "edited": s.text_src_edited is not None,
            "state": s.state, "speaker_id": s.speaker_id}


# ── phase 3 · translation ─────────────────────────────────────────────────


class TranslateIn(BaseModel):
    lang: str


class TranslationPatch(BaseModel):
    text_edited: str | None = None
    state: str | None = None


@app.post("/api/projects/{pid}/translate", status_code=202)
def start_translate(pid: str, body: TranslateIn, db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    if body.lang not in FLORES:
        raise HTTPException(400, f"unsupported language '{body.lang}'")
    if body.lang == p.source_lang:
        raise HTTPException(400, "target language is the same as the source")
    n = db.query(Segment).filter(Segment.project_id == pid).count()
    if not n:
        raise HTTPException(400, "no transcript yet — run analysis first")
    enqueue_translation(pid, body.lang)
    return {"ok": True, "lang": body.lang, "segments": n}


@app.post("/api/projects/{pid}/refine", status_code=202)
def start_refine(pid: str, body: TranslateIn, db: Session = Depends(get_db)):
    """Re-run only the LLM pass, reusing the existing IndicTrans2 draft."""
    from . import llm
    from .tasks_mt import refine

    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    if llm.configured() is None:
        raise HTTPException(400, "no LLM key configured — set GEMINI_API_KEY in .env")
    n = db.query(Translation).filter(
        Translation.project_id == pid, Translation.lang == body.lang
    ).count()
    if not n:
        raise HTTPException(400, f"nothing translated into {body.lang} yet")

    job = db.query(Job).filter(Job.project_id == pid, Job.stage == "refine").one_or_none()
    if job is None:
        db.add(Job(project_id=pid, stage="refine", state="queued", capability="cpu"))
    else:
        job.state = "queued"
        job.error = None
    p.status = "translating"
    db.commit()
    refine.si(pid, body.lang).apply_async()
    return {"ok": True, "lang": body.lang, "engine": llm.label()}


@app.get("/api/projects/{pid}/translations")
def get_translations(pid: str, lang: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Translation, Segment)
        .join(Segment, Translation.segment_id == Segment.id)
        .filter(Translation.project_id == pid, Translation.lang == lang)
        .order_by(Segment.idx)
        .all()
    )
    return [
        {
            "id": t.id,
            "segment_id": s.id,
            "idx": s.idx,
            "speaker_id": s.speaker_id,
            "t_start": s.t_start,
            "t_end": s.t_end,
            "source_text": s.text,
            "text_mt": t.text_mt,
            "text_llm": t.text_llm,
            "text_edited": t.text_edited,
            "text": t.text,
            "edited": t.text_edited is not None,
            "refined": t.text_llm is not None and t.text_llm != t.text_mt,
            "llm_note": t.llm_note,
            "llm_engine": t.llm_engine,
            "engine": t.engine,
            "budget_ms": s.duration_budget_ms,
            "est_duration_ms": t.est_duration_ms,
            "fit_ratio": t.fit_ratio,
            "state": t.state,
        }
        for t, s in rows
    ]


@app.patch("/api/translations/{tid}")
def patch_translation(tid: str, body: TranslationPatch, db: Session = Depends(get_db)):
    from .tasks_mt import estimate_ms, fits, flores, tts_duration_model

    t = db.get(Translation, tid)
    if not t:
        raise HTTPException(404, "translation not found")
    if body.text_edited is not None:
        txt = body.text_edited.strip()
        # Neither text_mt nor text_llm is ever overwritten. Clearing the edit
        # falls back to whichever tier is beneath it.
        baseline = (t.text_llm if t.text_llm is not None else t.text_mt) or ""
        t.text_edited = None if txt == baseline.strip() else txt
        # Re-score the fit live so the operator sees the effect of their edit,
        # calibrated against the original line's actual delivery rate.
        seg = db.get(Segment, t.segment_id)
        # Same rate the refine pass budgets with — see project_tts_rate().
        model = tts_duration_model(t.project_id, t.lang)
        t.est_duration_ms = estimate_ms(t.text, flores(t.lang), model=model)
        if seg and seg.duration_budget_ms:
            t.fit_ratio = round(t.est_duration_ms / seg.duration_budget_ms, 3)
            if not fits(t.fit_ratio) and t.state != "approved":
                t.state = "needs_review"
            elif t.state == "needs_review" and fits(t.fit_ratio):
                t.state = "draft"
    if body.state is not None:
        if body.state not in ("draft", "approved", "needs_review"):
            raise HTTPException(400, "bad state")
        t.state = body.state
    db.commit()
    return {
        "id": t.id, "text": t.text, "edited": t.text_edited is not None,
        "est_duration_ms": t.est_duration_ms, "fit_ratio": t.fit_ratio, "state": t.state,
    }


# ── phase 4 · voice library ───────────────────────────────────────────────


class VoiceIn(BaseModel):
    display_name: str
    actor_name: str | None = None
    notes: str | None = None


class VoicePatch(BaseModel):
    display_name: str | None = None
    actor_name: str | None = None
    notes: str | None = None
    status: str | None = None


class ConsentIn(BaseModel):
    signatory: str
    agreement_ref: str | None = None
    scope: str | None = None
    territories: list[str] = []
    permitted_langs: list[str] = []
    expires_at: str | None = None   # ISO date


class CastIn(BaseModel):
    voice_id: str | None = None


def _voice_json(v: Voice) -> dict:
    c = v.consent
    return {
        "id": v.id,
        "display_name": v.display_name,
        "actor_name": v.actor_name,
        "kind": v.kind or "cloned",
        "category": v.category,
        "provenance": v.provenance,
        "notes": v.notes,
        "status": v.status,
        "usable": v.usable,
        "block_reason": v.block_reason,
        "clips": [
            {"id": cl.id, "filename": cl.filename, "duration": cl.duration,
             "emotion": cl.emotion, "transcript": cl.transcript,
             "is_default": bool(cl.is_default), "clipping": bool(cl.clipping),
             "peak_db": cl.peak_db, "rms_db": cl.rms_db,
             "url": f"/api/media/{cl.key}"}
            for cl in v.clips
        ],
        "consent": {
            "signatory": c.signatory,
            "agreement_ref": c.agreement_ref,
            "scope": c.scope,
            "territories": c.territories or [],
            "permitted_langs": c.permitted_langs or [],
            "signed_at": c.signed_at.isoformat() if c.signed_at else None,
            "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            "revoked_at": c.revoked_at.isoformat() if c.revoked_at else None,
        } if c else None,
    }


@app.get("/api/voices")
def list_voices(db: Session = Depends(get_db)):
    rows = db.query(Voice).order_by(Voice.created_at.desc()).all()
    return [_voice_json(v) for v in rows]


@app.post("/api/voices", status_code=201)
def create_voice(body: VoiceIn, db: Session = Depends(get_db)):
    if not body.display_name.strip():
        raise HTTPException(400, "a name is required")
    v = Voice(display_name=body.display_name.strip(),
              actor_name=(body.actor_name or "").strip() or None,
              notes=body.notes, status="draft")
    db.add(v)
    db.commit()
    db.refresh(v)
    return _voice_json(v)


@app.patch("/api/voices/{vid}")
def patch_voice(vid: str, body: VoicePatch, db: Session = Depends(get_db)):
    v = db.get(Voice, vid)
    if not v:
        raise HTTPException(404, "voice not found")
    if body.display_name is not None:
        v.display_name = body.display_name.strip() or v.display_name
    if body.actor_name is not None:
        v.actor_name = body.actor_name.strip() or None
    if body.notes is not None:
        v.notes = body.notes
    if body.status is not None:
        if body.status not in ("draft", "pending_consent", "active", "suspended", "expired"):
            raise HTTPException(400, "bad status")
        if body.status == "active" and not (v.consent and v.consent.signed_at):
            raise HTTPException(400, "cannot activate a voice with no signed consent record")
        v.status = body.status
    db.commit()
    db.refresh(v)
    return _voice_json(v)


@app.delete("/api/voices/{vid}", status_code=204)
def delete_voice(vid: str, db: Session = Depends(get_db)):
    v = db.get(Voice, vid)
    if not v:
        raise HTTPException(404, "voice not found")
    storage.delete_prefix(f"voices/{vid}/")
    db.delete(v)
    db.commit()
    return Response(status_code=204)


@app.post("/api/voices/{vid}/clips", status_code=201)
async def add_clip(
    vid: str,
    file: UploadFile = File(...),
    transcript: str = "",
    emotion: str = "neutral",
    db: Session = Depends(get_db),
):
    v = db.get(Voice, vid)
    if not v:
        raise HTTPException(404, "voice not found")

    safe = _SAFE.sub("_", os.path.basename(file.filename or "clip"))[:100] or "clip"
    raw = os.path.join(settings.scratch_dir, f"clip-{uuid.uuid4()}")
    wav = raw + ".wav"
    try:
        with open(raw, "wb") as out:
            while chunk := await file.read(1024 * 1024 * 4):
                out.write(chunk)
        # Normalise to what the cloner wants, and strip any video.
        _ff([
            "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", raw,
            "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav,
        ])
        stats = _audio_stats(wav)
        key = f"voices/{vid}/{uuid.uuid4().hex[:8]}-{safe}.wav"
        storage.upload_file(wav, key, "audio/wav")
    finally:
        for p in (raw, wav):
            if os.path.exists(p):
                os.remove(p)

    first = not v.clips
    c = ReferenceClip(
        voice_id=vid, key=key, filename=safe, transcript=transcript.strip() or None,
        emotion=emotion, duration=stats["duration"], peak_db=stats["peak_db"],
        rms_db=stats["rms_db"], clipping=1 if stats["clipping"] else 0,
        is_default=1 if first else 0,
    )
    db.add(c)
    if v.status == "draft":
        v.status = "pending_consent"
    db.commit()
    db.refresh(v)
    return _voice_json(v)


@app.post("/api/clips/{cid}/default")
def set_default_clip(cid: str, db: Session = Depends(get_db)):
    c = db.get(ReferenceClip, cid)
    if not c:
        raise HTTPException(404, "clip not found")
    db.query(ReferenceClip).filter(ReferenceClip.voice_id == c.voice_id).update({"is_default": 0})
    c.is_default = 1
    db.commit()
    return {"ok": True}


@app.delete("/api/clips/{cid}", status_code=204)
def delete_clip(cid: str, db: Session = Depends(get_db)):
    c = db.get(ReferenceClip, cid)
    if not c:
        raise HTTPException(404, "clip not found")
    was_default = c.is_default
    vid = c.voice_id
    db.delete(c)
    db.flush()
    if was_default:
        nxt = db.query(ReferenceClip).filter(ReferenceClip.voice_id == vid).first()
        if nxt:
            nxt.is_default = 1
    db.commit()
    return Response(status_code=204)


@app.post("/api/voices/{vid}/consent")
def sign_consent(vid: str, body: ConsentIn, db: Session = Depends(get_db)):
    """Record the signed grant. This is what makes the voice usable at all."""
    from datetime import datetime, timezone

    v = db.get(Voice, vid)
    if not v:
        raise HTTPException(404, "voice not found")
    if not body.signatory.strip():
        raise HTTPException(400, "signatory is required")

    exp = None
    if body.expires_at:
        try:
            exp = datetime.fromisoformat(body.expires_at).replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, "expires_at must be an ISO date, e.g. 2027-01-31")

    c = v.consent or ConsentRecord(voice_id=vid)
    c.signatory = body.signatory.strip()
    c.agreement_ref = body.agreement_ref
    c.scope = body.scope
    c.territories = body.territories
    c.permitted_langs = body.permitted_langs
    c.signed_at = datetime.now(timezone.utc)
    c.expires_at = exp
    c.revoked_at = None
    c.revoked_reason = None
    db.add(c)
    if v.clips:
        v.status = "active"
    db.commit()
    db.refresh(v)
    return _voice_json(v)


@app.post("/api/voices/{vid}/revoke")
def revoke_consent(vid: str, db: Session = Depends(get_db)):
    """The kill switch. Suspends the voice across all in-flight work."""
    from datetime import datetime, timezone

    v = db.get(Voice, vid)
    if not v or not v.consent:
        raise HTTPException(404, "no consent record to revoke")
    v.consent.revoked_at = datetime.now(timezone.utc)
    v.status = "suspended"
    db.commit()
    db.refresh(v)
    return _voice_json(v)


@app.get("/api/voices/{vid}/usage")
def voice_usage(vid: str, db: Session = Depends(get_db)):
    rows = db.query(VoiceUsage).filter(VoiceUsage.voice_id == vid).order_by(
        VoiceUsage.at.desc()).all()
    return [
        {"project_id": u.project_id, "lang": u.lang, "segments": u.segments,
         "seconds": u.seconds, "engine": u.engine,
         "at": u.at.isoformat() if u.at else None}
        for u in rows
    ]


@app.post("/api/speakers/{sid}/cast")
def cast_speaker(sid: str, body: CastIn, db: Session = Depends(get_db)):
    s = db.get(Speaker, sid)
    if not s:
        raise HTTPException(404, "speaker not found")
    if body.voice_id:
        v = db.get(Voice, body.voice_id)
        if not v:
            raise HTTPException(400, "unknown voice")
        if not v.usable:
            raise HTTPException(400, f"{v.display_name} cannot be used — {v.block_reason}")
    s.voice_id = body.voice_id
    db.commit()
    return {"ok": True, "speaker_id": s.id, "voice_id": s.voice_id}


@app.post("/api/projects/{pid}/dub", status_code=202)
def start_dub(pid: str, body: TranslateIn, db: Session = Depends(get_db)):
    from .tasks_voice import enqueue_dub

    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    n = db.query(Translation).filter(
        Translation.project_id == pid, Translation.lang == body.lang).count()
    if not n:
        raise HTTPException(400, f"nothing translated into {body.lang} yet")

    speakers = db.query(Speaker).filter(
        Speaker.project_id == pid, Speaker.merged_into.is_(None)).all()
    uncast = [s.display_name or s.label for s in speakers if not s.voice_id]
    if uncast:
        raise HTTPException(400, f"cast a voice for: {', '.join(uncast)}")
    return _dub(pid, body.lang, enqueue_dub)


def _dub(pid, lang, fn):
    fn(pid, lang)
    return {"ok": True, "lang": lang}


def _ff(cmd):
    import subprocess

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPException(400, "could not read that audio: "
                            + (p.stderr or "").strip().splitlines()[-1:][0][:200])


def _audio_stats(path: str) -> dict:
    """Peak/RMS/clipping — surfaced so a bad reference clip is caught before it
    poisons every line synthesized from it."""
    import json as _json
    import subprocess

    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True)
    duration = float(_json.loads(p.stdout)["format"]["duration"]) if p.returncode == 0 else 0.0

    v = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-af", "volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True)
    peak = rms = None
    for line in (v.stderr or "").splitlines():
        if "max_volume:" in line:
            peak = float(line.split("max_volume:")[1].strip().split()[0])
        if "mean_volume:" in line:
            rms = float(line.split("mean_volume:")[1].strip().split()[0])
    return {"duration": round(duration, 2), "peak_db": peak, "rms_db": rms,
            "clipping": peak is not None and peak >= -0.1}


@app.get("/api/media/{key:path}")
def media(key: str, request: Request):
    """Stream objects out of MinIO with HTTP Range support, so MinIO itself
    never needs a published port and video seeking still works."""
    range_header = request.headers.get("range")
    try:
        obj = storage.get_object(key, range_header)
    except Exception:
        raise HTTPException(404, "not found")

    headers = {"accept-ranges": "bytes"}
    status = 200
    if "ContentRange" in obj:
        headers["content-range"] = obj["ContentRange"]
        status = 206
    if "ContentLength" in obj:
        headers["content-length"] = str(obj["ContentLength"])

    body = obj["Body"]

    def it():
        try:
            for chunk in body.iter_chunks(CHUNK):
                yield chunk
        finally:
            body.close()

    return StreamingResponse(
        it(),
        status_code=status,
        media_type=obj.get("ContentType") or "application/octet-stream",
        headers=headers,
    )


# ── ui ────────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def _index_html() -> str:
    """index.html with a build token stamped onto its asset URLs.

    StaticFiles sends no Cache-Control, so a browser applies heuristic caching
    and goes on serving the previous app.js after a deploy — the UI silently
    does not update until someone thinks to hard-refresh. Making a new build a
    new URL is the only version of this that reliably works.

    Computed once: the files cannot change under a running container.
    """
    token = hashlib.sha256(
        b"".join(str((STATIC_DIR / n).stat().st_mtime_ns).encode()
                 for n in _VERSIONED_ASSETS)
    ).hexdigest()[:8]
    html = (STATIC_DIR / "index.html").read_text()
    for name in _VERSIONED_ASSETS:
        html = html.replace(f"/static/{name}", f"/static/{name}?v={token}")
    return html


@app.get("/")
def index():
    # The HTML carries the token, so it is the one thing that must never be
    # served stale.
    return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})


@app.exception_handler(HTTPException)
def http_exc(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
