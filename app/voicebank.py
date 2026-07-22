"""Synthetic AI voice bank — stock voices that belong to nobody.

Why this exists
---------------
You pay real actors for the roles that matter. You do not want to pay, or get
consent from, a human being for "Guard #3". This bank fills that gap with voices
that were never a person: no consent record, no personality right, no expiry,
no revocation risk — because there is no one to revoke.

How a voice is made
-------------------
Kokoro (Apache-2.0 code AND weights, ungated) generates a reference clip. That
clip becomes a Chatterbox reference — and Chatterbox clones *timbre*, so a voice
whose reference clip is in English can perform in Hindi. The reference language
is an identity, not a limit. That is what lets 54 Kokoro voices serve an Indic
dub.

What is honestly derived, not generated
---------------------------------------
Kokoro ships **54 voices and no children**. To reach a larger bank we pitch- and
formant-shift the base voices with rubberband. That is a real technique — shifting
formants changes the apparent size of the vocal tract, which is what actually
distinguishes a child from a chipmunk — but derived voices SHARE DNA with their
base and will sound related. Each one records what it came from in `provenance`.

Child voices are ALL derived. A convincing child voice needs a model trained on
child speech; none of the commercially-licensed options here has one. Treat them
as usable for background roles and audition them before casting a speaking part.

Deliberately NOT used as a source
---------------------------------
Public voice corpora (Common Voice, LibriVox). They are CC0/PD for *copyright*,
which is not consent to have your voice cloned and sold. Those contributors gave
their voice to speech research, not to a dubbing product. Building the bank from
them would contradict the entire consent model this system exists to enforce.
"""

import logging
import os
import subprocess

from .config import settings

log = logging.getLogger(__name__)

# Kokoro's lang_code is inferred from the voice prefix. A reference clip should
# be in the voice's native language so delivery is natural — its timbre then
# transfers to any language Chatterbox supports.
PASSAGES = {
    "a": "Hello. This is a reference recording of my voice, spoken clearly and "
         "calmly, so the system can learn exactly how I sound in conversation.",
    "b": "Hello there. This is a reference recording of my voice, spoken clearly "
         "and calmly, so the system can learn exactly how I sound when I speak.",
    "h": "नमस्ते। यह मेरी आवाज़ की एक रिकॉर्डिंग है। मैं धीरे और साफ़ बोल रहा हूँ "
         "ताकि सिस्टम मेरी आवाज़ को अच्छी तरह से पहचान सके।",
    "e": "Hola. Esta es una grabación de referencia de mi voz, hablando con "
         "claridad y calma para que el sistema aprenda cómo sueno.",
    "f": "Bonjour. Ceci est un enregistrement de référence de ma voix, parlé "
         "clairement et calmement afin que le système apprenne mon timbre.",
    "i": "Ciao. Questa è una registrazione di riferimento della mia voce, "
         "parlata con calma e chiarezza per far imparare al sistema il mio timbro.",
    "p": "Olá. Esta é uma gravação de referência da minha voz, falada com clareza "
         "e calma para que o sistema aprenda como eu soo.",
    "j": "こんにちは。これは私の声の参照録音です。システムが私の声を学習できるように、"
         "ゆっくりとはっきりと話しています。",
    "z": "你好。这是我声音的参考录音。我正在缓慢而清晰地说话，以便系统能够学习我的声音。",
}

PREFIX_LANG = {"a": "a", "b": "b", "h": "h", "e": "e", "f": "f",
               "i": "i", "p": "p", "j": "j", "z": "z"}

# (semitone-ish pitch ratio, label). formant=shifted moves the formants with the
# pitch, which changes apparent vocal-tract SIZE — i.e. a different person,
# rather than the same person on helium.
ADULT_VARIANTS = [
    (0.92, "deeper"),
    (1.08, "lighter"),
    (0.86, "deep"),
    (1.14, "bright"),
]
# Children: pitch AND formants up hard. Honestly an approximation.
CHILD_VARIANTS = [
    (1.32, "child"),
    (1.42, "child-young"),
    (1.26, "child-older"),
]


def kokoro_voices() -> list[str]:
    """Voice ids shipped with Kokoro, read from the mirrored repo."""
    from . import mirror
    from .tasks_ai import _snapshot

    path = _snapshot(mirror.ensure("hexgrad/Kokoro-82M"))
    vdir = os.path.join(path, "voices")
    if not os.path.isdir(vdir):
        raise RuntimeError(f"no voices/ directory in {path}")
    return sorted(f[:-3] for f in os.listdir(vdir) if f.endswith(".pt"))


def category_of(voice_id: str) -> str:
    return "female" if voice_id[1] == "f" else "male"


def _generate_base(voice_id: str, out_path: str) -> float:
    """Kokoro → a reference clip for one voice. Returns duration."""
    import soundfile as sf
    from kokoro import KPipeline

    lang = PREFIX_LANG.get(voice_id[0], "a")
    text = PASSAGES.get(lang, PASSAGES["a"])
    pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")

    # speed matters more than it looks: the reference clip's OWN pacing
    # transfers to the clone. Measured, same text and cfg_weight, only the
    # reference changed:
    #     kokoro speed=1.0 → dub 3.87 syl/s
    #     kokoro speed=1.3 → dub 4.62 syl/s   ← best
    #     kokoro speed=1.6 → dub 2.79 syl/s   ← degrades; the clone fights an
    #                                            unnaturally fast reference
    chunks = [audio for _, _, audio in pipe(text, voice=voice_id,
                                            speed=settings.voicebank_speed)]
    if not chunks:
        raise RuntimeError(f"kokoro produced no audio for {voice_id}")
    import numpy as np

    wav = np.concatenate([c.detach().cpu().numpy() if hasattr(c, "detach") else c
                          for c in chunks])
    sf.write(out_path, wav, 24000)
    return len(wav) / 24000


def _derive(src: str, dst: str, pitch: float) -> None:
    """Pitch + formant shift. formant=shifted is deliberate: it moves the
    formants with the pitch, changing apparent vocal-tract size — which is what
    makes it a different person instead of the same person sped up."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-filter:a", f"rubberband=pitch={pitch}:formant=shifted",
         "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", dst],
        check=True,
    )


def plan(target_male: int, target_female: int, target_child: int) -> dict:
    """Work out what can be generated vs what must be derived — before spending
    an hour on it, and honestly about which is which."""
    voices = kokoro_voices()
    base_m = [v for v in voices if category_of(v) == "male"]
    base_f = [v for v in voices if category_of(v) == "female"]
    return {
        "base_voices": len(voices),
        "male": {"base": len(base_m), "derived": max(0, target_male - len(base_m)),
                 "target": target_male},
        "female": {"base": len(base_f), "derived": max(0, target_female - len(base_f)),
                   "target": target_female},
        # Kokoro has no child voices at all — every one is derived.
        "child": {"base": 0, "derived": target_child, "target": target_child},
    }


def build(target_male: int = 40, target_female: int = 40, target_child: int = 40,
          replace: bool = False) -> dict:
    """Generate the bank. Idempotent unless replace=True."""
    import tempfile

    from . import storage
    from .db import session
    from .models import ReferenceClip, Voice

    voices = kokoro_voices()
    base_m = [v for v in voices if category_of(v) == "male"]
    base_f = [v for v in voices if category_of(v) == "female"]

    with session() as db:
        if replace:
            for v in db.query(Voice).filter(Voice.kind == "synthetic").all():
                storage.delete_prefix(f"voices/{v.id}/")
                db.delete(v)
            db.flush()
        existing = {v.provenance for v in db.query(Voice).filter(Voice.kind == "synthetic").all()}

    wd = tempfile.mkdtemp(dir=settings.scratch_dir)
    made = {"male": 0, "female": 0, "child": 0, "skipped": 0}
    base_clips: dict[str, str] = {}

    def add(name: str, category: str, prov: str, wav: str, dur: float) -> bool:
        if prov in existing:
            made["skipped"] += 1
            return False
        with session() as db:
            v = Voice(display_name=name, kind="synthetic", category=category,
                      status="active", provenance=prov,
                      notes="Synthetic voice — no human was recorded, so no consent "
                            "record applies. See app/voicebank.py.")
            db.add(v)
            db.flush()
            key = f"voices/{v.id}/reference.wav"
            storage.upload_file(wav, key, "audio/wav")
            db.add(ReferenceClip(
                voice_id=v.id, key=key, filename="reference.wav", duration=round(dur, 2),
                transcript=None, emotion="neutral", is_default=1,
            ))
        made[category] += 1
        existing.add(prov)
        return True

    # ── base voices: genuinely distinct, generated by Kokoro ──────────────
    for cat, pool, target in (("male", base_m, target_male), ("female", base_f, target_f := target_female)):
        for i, vid in enumerate(pool[:target]):
            p = os.path.join(wd, f"{vid}.wav")
            try:
                dur = _generate_base(vid, p)
            except Exception as e:
                log.warning("kokoro failed for %s: %s", vid, e)
                continue
            base_clips[vid] = p
            add(f"AI {cat.title()} {i+1:02d}", cat, f"kokoro:{vid}", p, dur)
            log.info("base %s %s (%.1fs)", cat, vid, dur)

    # ── derived: pitch/formant shifted. Shares DNA with its base — say so. ─
    def derive_pool(cat: str, pool: list[str], target: int, variants):
        """Top up to `target` counting what was ACTUALLY added — base voices can
        fail (a missing phonemiser for one language), and assuming pool size
        equals voices-added silently undershoots the target."""
        vi = 0
        guard = 0
        pool = [v for v in pool if v in base_clips] or pool
        while made[cat] < target and pool and guard < target * 12:
            for vid in pool:
                guard += 1
                if made[cat] >= target:
                    break
                pitch, label = variants[vi % len(variants)]
                vi += 1
                src = base_clips.get(vid)
                if not src:
                    src = os.path.join(wd, f"{vid}.wav")
                    if not os.path.exists(src):
                        try:
                            _generate_base(vid, src)
                            base_clips[vid] = src
                        except Exception:
                            continue
                dst = os.path.join(wd, f"{vid}-{label}-{made[cat]}.wav")
                try:
                    _derive(src, dst, pitch)
                except Exception as e:
                    log.warning("derive failed %s: %s", vid, e)
                    continue
                import soundfile as sf
                add(f"AI {cat.title()} {made[cat] + 1:02d}", cat,
                    f"kokoro:{vid} + {label} (pitch {pitch})", dst, sf.info(dst).duration)

    derive_pool("male", base_m, target_male, ADULT_VARIANTS)
    derive_pool("female", base_f, target_female, ADULT_VARIANTS)

    # ── children: ALL derived. Kokoro has none. ───────────────────────────
    # Lighter voices shift into a child register more convincingly than deep ones.
    child_sources = [v for v in base_f if v in base_clips] + \
                    [v for v in base_m if v[0] in "abh" and v in base_clips]
    child_sources = child_sources or base_f
    n = 0
    vi = 0
    guard = 0
    while made["child"] < target_child and child_sources and guard < target_child * 12:
        guard += 1
        vid = child_sources[n % len(child_sources)]
        pitch, label = CHILD_VARIANTS[vi % len(CHILD_VARIANTS)]
        vi += 1
        src = base_clips.get(vid)
        if not src or not os.path.exists(src):
            src = os.path.join(wd, f"{vid}.wav")
            if not os.path.exists(src):
                try:
                    _generate_base(vid, src)
                    base_clips[vid] = src
                except Exception:
                    n += 1
                    continue
        dst = os.path.join(wd, f"child-{made['child']}-{n}.wav")
        try:
            _derive(src, dst, pitch)
        except Exception:
            n += 1
            continue
        import soundfile as sf
        add(f"AI Child {made['child'] + 1:02d}", "child",
            f"kokoro:{vid} + {label} (pitch {pitch})", dst, sf.info(dst).duration)
        n += 1

    import shutil
    shutil.rmtree(wd, ignore_errors=True)
    return made
