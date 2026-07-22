import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Phase 1 — ingest. Phase 2 — analysis. Kept separate because the operator
# approves ingest before analysis spends real compute (docs/ARCHITECTURE.md §6).
INGEST_STAGES = ["probe", "proxy", "thumbnails", "shots"]
ANALYSIS_STAGES = ["separate", "diarize", "asr"]
TRANSLATION_STAGES = ["translate", "refine"]
VOICE_STAGES = ["synth", "fit", "mix"]
STAGES = INGEST_STAGES + ANALYSIS_STAGES + TRANSLATION_STAGES + VOICE_STAGES

# IndicTrans2 speaks FLORES codes, not ISO-639-1.
FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
    "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda",
    "ml": "mal_Mlym", "pa": "pan_Guru", "or": "ory_Orya", "as": "asm_Beng",
    "ur": "urd_Arab", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
    "kok": "gom_Deva", "mai": "mai_Deva", "doi": "doi_Deva", "brx": "brx_Deva",
    "mni": "mni_Beng", "sat": "sat_Olck", "ks": "kas_Arab",
}
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
}

# Fallback speaking rate in syllables/sec, used only when a line is too short to
# calibrate against. Human speech clusters around 5–6 syllables/sec across most
# languages, which is why syllables travel across a language pair far better than
# characters do.
DEFAULT_SYLL_PER_SEC = 5.5
# Clamp calibration to plausible human delivery — a mis-timestamped segment
# shouldn't produce a nonsense rate.
MIN_SYLL_PER_SEC, MAX_SYLL_PER_SEC = 2.5, 9.0

# Beyond ±15% time-stretch, audio audibly chipmunks (docs/ARCHITECTURE.md §0.4).
FIT_TOLERANCE = 0.15

# Assigned round-robin to speakers so the transcript is scannable.
SPEAKER_COLORS = [
    "#FFB000", "#4CC2FF", "#3FB950", "#F778BA",
    "#A78BFA", "#F0883E", "#56D4DD", "#D2A8FF",
]


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(String, nullable=False)
    source_lang = Column(String, default="hi")
    target_langs = Column(JSON, default=list)
    # empty | ingesting | ready | analyzing | analyzed | translating | translated | failed
    status = Column(String, default="empty", nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    assets = relationship("Asset", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    shots = relationship("Shot", back_populates="project", cascade="all, delete-orphan")
    speakers = relationship("Speaker", back_populates="project", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="project", cascade="all, delete-orphan")


class Asset(Base):
    __tablename__ = "assets"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    # source | proxy | poster | sprite | sprite_vtt | dialogue | music_fx | diarization
    kind = Column(String, nullable=False)
    key = Column(String, nullable=False)
    filename = Column(String)
    size = Column(BigInteger, default=0)
    content_type = Column(String)
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=_now)

    project = relationship("Project", back_populates="assets")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    stage = Column(String, nullable=False)
    # queued | running | done | failed | skipped
    state = Column(String, default="queued", nullable=False)
    # cpu | gpu — GPU workers subscribe by capability.
    capability = Column(String, default="cpu")
    # Why a stage was skipped/degraded — surfaced in the UI, not swallowed.
    note = Column(Text)
    error = Column(Text)
    attempts = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=_now)

    project = relationship("Project", back_populates="jobs")


class Shot(Base):
    __tablename__ = "shots"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    idx = Column(Integer, nullable=False)
    t_start = Column(Float, nullable=False)
    t_end = Column(Float, nullable=False)
    is_closeup = Column(Integer, default=0)
    lipsync_state = Column(String, default="none")

    project = relationship("Project", back_populates="shots")


class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    # Machine label from diarization, e.g. SPEAKER_00. Never overwritten.
    label = Column(String, nullable=False)
    # Human name from the review UI.
    display_name = Column(String)
    color = Column(String, default="#FFB000")
    speech_seconds = Column(Float, default=0.0)
    # Set when merged into another speaker; segments are reassigned.
    merged_into = Column(String, ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True)
    # The licensed voice cast for this character.
    voice_id = Column(String, ForeignKey("voices.id", ondelete="SET NULL"), nullable=True)

    project = relationship("Project", back_populates="speakers")
    segments = relationship("Segment", back_populates="speaker")

    @property
    def name(self) -> str:
        return self.display_name or self.label


class Segment(Base):
    __tablename__ = "segments"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    speaker_id = Column(String, ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True)
    idx = Column(Integer, nullable=False)
    t_start = Column(Float, nullable=False)
    t_end = Column(Float, nullable=False)

    # Machine output is NEVER destroyed — *_edited holds the human version.
    # Needed to measure model quality and to re-run a stage without losing edits.
    text_src = Column(Text)
    text_src_edited = Column(Text)
    asr_confidence = Column(Float)

    # The slot this line's dub must fit into. Per-segment, not per-language.
    duration_budget_ms = Column(Integer)

    # draft | approved | needs_review
    state = Column(String, default="draft")

    project = relationship("Project", back_populates="segments")
    speaker = relationship("Speaker", back_populates="segments")
    translations = relationship(
        "Translation", back_populates="segment", cascade="all, delete-orphan"
    )

    @property
    def text(self) -> str | None:
        return self.text_src_edited if self.text_src_edited is not None else self.text_src

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class Voice(Base):
    """A performer's voice, licensed for use.

    This table is the product, not paperwork. The pitch is "pay them for their
    voice", and Indian courts actively enforce voice as a personality right —
    Anil Kapoor and Amitabh Bachchan both hold Delhi HC orders protecting theirs.
    A voice with no consent record is a liability, so `usable` gates on it.

    Deliberately NOT a training pipeline: modern cloning is zero-shot from ~10s
    of reference audio. This is a curated clip library (docs/ARCHITECTURE.md §3).
    """

    __tablename__ = "voices"

    id = Column(String, primary_key=True, default=_uuid)
    display_name = Column(String, nullable=False)
    actor_name = Column(String)
    notes = Column(Text)
    languages = Column(JSON, default=list)
    # draft | pending_consent | active | suspended | expired
    status = Column(String, default="draft", nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    # cloned    — a real performer's voice. REQUIRES a signed consent record.
    # synthetic — generated by a TTS model; no human being was recorded, so
    #             there is nobody to consent and no personality right attaches.
    #             This is the whole reason the AI voice bank exists: stock voices
    #             for background and minor roles, with the consent machinery
    #             reserved for the real actors you actually pay.
    kind = Column(String, default="cloned", nullable=False)
    # male | female | child — for casting, and how the bank is organised.
    category = Column(String)
    # How a synthetic voice was made, so it is never mistaken for a person.
    provenance = Column(Text)

    clips = relationship("ReferenceClip", back_populates="voice", cascade="all, delete-orphan")
    consent = relationship("ConsentRecord", back_populates="voice",
                           uselist=False, cascade="all, delete-orphan")

    @property
    def usable(self) -> bool:
        """Cloned voices need live consent. Synthetic ones need only a clip —
        there is no person to consent, so demanding a signature would be
        theatre, and theatre devalues the signatures that are real."""
        if self.status == "suspended":
            return False
        if not self.clips:
            return False
        if self.kind == "synthetic":
            return True
        if self.status != "active":
            return False
        c = self.consent
        if not c or not c.signed_at:
            return False
        if c.revoked_at:
            return False
        if c.expires_at and c.expires_at < _now():
            return False
        return True

    @property
    def block_reason(self) -> str | None:
        if self.status == "suspended":
            return "voice suspended"
        if not self.clips:
            return "no reference clips uploaded"
        if self.kind == "synthetic":
            return None
        c = self.consent
        if not c or not c.signed_at:
            return "no signed consent record"
        if c.revoked_at:
            return "consent revoked"
        if c.expires_at and c.expires_at < _now():
            return "consent expired"
        if self.status != "active":
            return f"voice is {self.status}"
        return None


class ReferenceClip(Base):
    """Reference audio for zero-shot cloning. Curation IS the quality lever —
    clip choice moves output more than anything else, and unpredictably, which
    is why the UI makes you audition rather than guess from a waveform."""

    __tablename__ = "reference_clips"

    id = Column(String, primary_key=True, default=_uuid)
    voice_id = Column(String, ForeignKey("voices.id", ondelete="CASCADE"), index=True)
    key = Column(String, nullable=False)
    filename = Column(String)
    duration = Column(Float)
    transcript = Column(Text)
    # neutral | angry | soft | shouting | laughing
    emotion = Column(String, default="neutral")
    language = Column(String)
    is_default = Column(Integer, default=0)
    # QC — surfaced so bad clips are caught before they poison every line.
    peak_db = Column(Float)
    rms_db = Column(Float)
    clipping = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    voice = relationship("Voice", back_populates="clips")


class ConsentRecord(Base):
    """The signed grant. Scope matters: consent for one title is not consent for
    every title, and it must be revocable across in-flight work."""

    __tablename__ = "consent_records"

    id = Column(String, primary_key=True, default=_uuid)
    voice_id = Column(String, ForeignKey("voices.id", ondelete="CASCADE"), index=True, unique=True)
    signatory = Column(String)          # who signed
    agreement_ref = Column(String)      # contract id / doc url
    scope = Column(Text)                # titles, territories, media
    territories = Column(JSON, default=list)
    permitted_langs = Column(JSON, default=list)
    signed_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    revoked_at = Column(DateTime(timezone=True))
    revoked_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)

    voice = relationship("Voice", back_populates="consent")


class VoiceUsage(Base):
    """Per-use log — what the actor gets paid on, and what proves scope was kept."""

    __tablename__ = "voice_usage"

    id = Column(String, primary_key=True, default=_uuid)
    voice_id = Column(String, ForeignKey("voices.id", ondelete="CASCADE"), index=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    lang = Column(String)
    segments = Column(Integer, default=0)
    seconds = Column(Float, default=0.0)
    engine = Column(String)
    at = Column(DateTime(timezone=True), default=_now)


class Take(Base):
    """A synthesized line. Multiple takes per segment; one chosen."""

    __tablename__ = "takes"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    segment_id = Column(String, ForeignKey("segments.id", ondelete="CASCADE"), index=True)
    lang = Column(String, nullable=False, index=True)
    voice_id = Column(String, ForeignKey("voices.id", ondelete="SET NULL"), nullable=True)
    key = Column(String)
    engine = Column(String)
    raw_ms = Column(Integer)        # as synthesized
    final_ms = Column(Integer)      # after time-fit
    stretch = Column(Float)         # applied tempo ratio
    # ok | stretched | over_budget — over_budget means it could not be made to fit
    fit_state = Column(String, default="ok")
    chosen = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=_now)


class Translation(Base):
    """One row per (segment, target language).

    Keyed by language rather than a column on Segment because per-language is
    the dominant cost driver in this business — every vendor bills per target
    language, and you will dub one film into several. Modelling it as a column
    would mean losing Hindi the moment you translate to Tamil.
    """

    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("segment_id", "lang", name="uq_translation_seg_lang"),)

    id = Column(String, primary_key=True, default=_uuid)
    segment_id = Column(String, ForeignKey("segments.id", ondelete="CASCADE"), index=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    lang = Column(String, nullable=False, index=True)

    # Three tiers, none of which overwrites another. Effective text resolves
    # human > llm > mt. Keeping all three is what lets you measure whether the
    # LLM pass is actually helping, and re-run one tier without losing the rest.
    text_mt = Column(Text)       # IndicTrans2 draft
    text_llm = Column(Text)      # LLM refinement of that draft
    text_edited = Column(Text)   # human
    engine = Column(String)
    llm_engine = Column(String)
    llm_note = Column(Text)      # why the LLM changed it

    # Fit triage — see CHARS_PER_SEC above for why these are hints, not truth.
    est_duration_ms = Column(Integer)
    fit_ratio = Column(Float)

    # draft | approved | needs_review
    state = Column(String, default="draft")

    segment = relationship("Segment", back_populates="translations")

    @property
    def text(self) -> str | None:
        if self.text_edited is not None:
            return self.text_edited
        if self.text_llm is not None:
            return self.text_llm
        return self.text_mt
