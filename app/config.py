from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://dubbing:dubbing@postgres:5432/dubbing"
    redis_url: str = "redis://redis:6379/0"

    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "dubbingadmin"
    s3_secret_key: str = "dubbingadmin"
    s3_bucket: str = "media"

    scratch_dir: str = "/scratch"
    proxy_height: int = 720
    ffmpeg_threads: int = 4

    # Sprite sheet geometry
    sprite_cols: int = 10
    sprite_max_tiles: int = 100
    sprite_tile_width: int = 160

    # ── Phase 2 · analysis ────────────────────────────────────────────────
    model_dir: str = "/models"

    # faster-whisper (MIT). large-v3-turbo is the production choice; its speedup
    # does NOT transfer to CPU (encoder unchanged) — see docs/ARCHITECTURE.md §2.
    asr_model: str = "large-v3"
    asr_compute_type: str = "int8"
    asr_threads: int = 4
    # Chunk-parallel ASR. CTranslate2: "prefer increasing inter_threads over
    # intra_threads" for bulk work, and keep workers*threads <= PHYSICAL cores
    # (16 here, not the 24 that lscpu reports). Whisper needs long context, so
    # chunks stay long — this parallelises, it does not fragment.
    asr_workers: int = 4
    asr_chunk_seconds: float = 300.0

    # pyannote community-1 (CC-BY-4.0, gated). Free + commercial-OK, but the repo
    # requires accepting terms with an HF account. No token → diarization degrades
    # to a single speaker rather than guessing.
    hf_token: str = ""
    diar_model: str = "pyannote/speaker-diarization-community-1"

    # Diarization backend:
    #   diy      — Silero VAD + WeSpeaker + clustering. No gate, no account, no
    #              token. Cannot represent overlapping speech.
    #   pyannote — overlap-aware and more accurate, but gated: needs HF_TOKEN
    #              once (then it's mirrored and the token can be deleted).
    diar_backend: str = "diy"
    # Ungated (gated=False, CC-BY-4.0). pyannote's embedding model is open even
    # though its assembled pipeline is not.
    diar_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    # Cosine distance at which two windows stop being the same person. Higher =
    # fewer, broader speakers.
    #
    # 0.85 is MEASURED, not guessed. Swept on a real Hindi two-hander: 0.70 gave
    # 10 clusters (6 of them noise), 0.80 gave 6, 0.90 gave a clean 2. 0.85 sits
    # just inside the correct answer, with _absorb_specks cleaning the tail.
    # An earlier 0.70 default came from general knowledge and was badly wrong on
    # real film audio — re-measure if your material differs.
    diar_threshold: float = 0.85
    # A cluster holding less speech than this is a speck (a breath, a door), not
    # a character. Absorbed into the nearest real speaker.
    diar_min_speaker_seconds: float = 2.0
    # 0 = infer the count. Set it when you know the cast — it beats a threshold.
    diar_num_speakers: int = 0
    diar_window: float = 2.0
    diar_hop: float = 1.0
    diar_min_window: float = 0.6

    # Bandit v2 — the only commercially-licensed cinematic D/M/E separator.
    # Weights are CC BY-SA 4.0 and pending legal review (§7), so this is opt-in.
    bandit_checkpoint: str = ""

    # ── Phase 3 · translation ─────────────────────────────────────────────
    # IndicTrans2 (MIT). Distilled 200M/320M by default — the 1B variants are
    # better but ~5× the CPU cost.
    mt_batch_size: int = 8
    mt_threads: int = 8
    mt_max_length: int = 128

    # LLM refinement pass. IndicTrans2 does the En→Indic translation (the LLM's
    # weak direction); the LLM only *refines* its draft for document context,
    # character register and length. It never translates from scratch.
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    xai_api_key: str = ""
    llm_batch_size: int = 20
    # Extra compression rounds over lines that are still too long. The prompt
    # asks for a syllable budget; this verifies it and re-asks with the actual
    # overrun, which converges much better than one request ever does.
    refine_passes: int = 2

    # ── Phase 4 · voice ───────────────────────────────────────────────────
    # Chatterbox (Resemble AI) — MIT code AND weights, 23 langs incl. Hindi,
    # zero-shot from ~10s reference. Measured ~7.2x slower than realtime on
    # this CPU — a 2-hour film is an overnight job, not an impossible one.
    tts_threads: int = 8
    # Chatterbox's MEASURED speaking rate, with reference clips generated at
    # VOICEBANK_SPEED: ~4.6 syllables/sec. Hindi actors in real film run 7.26,
    # so a dub is inherently ~1.6x longer than its slot unless the translation
    # is compressed. That is a property of the engine, not a bug to tune away.
    # The fit estimate must use THIS, not the source actor's tempo.
    # Affine duration model, MEASURED over 58 real takes:
    #     duration_ms ≈ 510 + 421 × syllables
    # Speech is not syllables÷rate — every utterance has a fixed cost (onset,
    # offset, breath) and only the rest scales. A rate model is 1.4x less
    # accurate overall and badly wrong at the short end, where film dialogue
    # lives. These are the fallback; each project refits on its own takes.
    tts_overhead_ms: float = 510.0
    tts_ms_per_syllable: float = 421.0
    tts_syll_per_sec: float = 4.6   # legacy fallback only
    # Chatterbox pacing — lever 2 of the time-fit budget (§0.4).
    # ⚠️ Measured, and the opposite of what the docs implied: HIGHER is FASTER.
    #     cfg_weight=0.5 → 5.00 syl/s
    #     cfg_weight=0.3 → 3.95 syl/s
    #     cfg_weight=0.2 → 3.09 syl/s
    # 0.5 is the default and the fastest useful value; do not lower it hoping
    # for speed.
    tts_cfg_weight: float = 0.5
    tts_exaggeration: float = 0.5
    # Kokoro speed for synthetic reference clips. 1.3 measured best; 1.6 makes
    # the clone WORSE (2.79 syl/s) — do not raise it without re-measuring.
    voicebank_speed: float = 1.3
    # Runaway rejection. Chatterbox is autoregressive and sometimes fails to
    # stop — measured: 7/58 takes overran, 24% of all audio, worst case a
    # 1-syllable line generating 5.4s. Retry when a take exceeds the affine
    # model by this factor, and keep the shortest attempt.
    tts_runaway_factor: float = 1.8
    tts_retries: int = 3

    # Ducking, used when no separated music/FX bed exists. The original audio is
    # kept at duck_level and pulled further down while the dub speaks, so the
    # score and atmosphere survive between lines instead of being discarded.
    # NOT separation — the original dialogue remains faintly audible.
    duck_level: float = 0.55
    duck_threshold: float = 0.03
    duck_ratio: float = 9.0

    # ── auth ──────────────────────────────────────────────────────────────
    # Empty = no authentication. The app binds 0.0.0.0 and holds licensed
    # voices, consent records and pre-release footage — set this.
    app_password: str = ""
    # Signing key for session cookies. Rotating it logs everyone out.
    secret_key: str = ""
    # Show the password on the login page for copy-paste. ONLY honoured when
    # UI_BIND is loopback — otherwise it would hand the password to the LAN.
    dev_show_password: bool = False


settings = Settings()
