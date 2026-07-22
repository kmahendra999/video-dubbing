"""Phase 3 · translation — IndicTrans2 (MIT).

Why IndicTrans2 and not the obvious alternatives (docs/ARCHITECTURE.md §5):

  · NLLB-200 weights are CC-BY-NC. Unusable in a commercial product.
  · Google Translate has no document context, no length control, no register
    control. Dubbing needs all three.
  · Raw LLMs are markedly WEAKER at English→Indic than Indic→English
    (IndicGenBench: GPT-4 scores 32.1 vs 54.5 char-F1). English→Hindi is
    exactly the LLM's weak direction and IndicTrans2's strong one.

IndicTrans2 has no length control of its own, so this stage translates, then
*measures the fit*, then flags lines the operator must shorten. That split is
deliberate: the machine proposes, the human decides.

The documented upgrade is an LLM pass on top for document-level context,
character register and length rewriting — deferred until an API key exists.
"""

import json
import logging
import re
import unicodedata

from .config import settings
from .db import session
from .models import (
    DEFAULT_SYLL_PER_SEC, FIT_TOLERANCE, FLORES, LANG_NAMES, MAX_SYLL_PER_SEC,
    MIN_SYLL_PER_SEC, Project, Segment, Speaker, Translation,
)
from .tasks import _stage, celery

log = logging.getLogger(__name__)

# Everything here is MIT — but the OFFICIAL AI4Bharat repos are `gated=auto` on
# HuggingFace: instantly auto-approved, yet still requiring a logged-in account.
# MIT licence, gated distribution. Those two are independent, and conflating them
# is how you discover the wall at runtime.
#
# So there are two tiers:
#   · with HF_TOKEN  → official AI4Bharat checkpoints. The benchmarked ones.
#   · without        → prajdabre's ungated MIT mirrors. Raj Dabre is an
#     IndicTrans2 co-author, and these swap sinusoidal for rotary positional
#     embeddings (long-context — genuinely useful for dialogue). But the card
#     calls them "my independent reproduction", so they are NOT the checkpoints
#     the published Vistaar/IN22 numbers describe. Treat those benchmarks as
#     not applying here.
GATED = {
    ("eng_Latn", "indic"): "ai4bharat/indictrans2-en-indic-dist-200M",
    ("indic", "eng_Latn"): "ai4bharat/indictrans2-indic-en-dist-200M",
    ("indic", "indic"): "ai4bharat/indictrans2-indic-indic-dist-320M",
}
OPEN = {
    ("eng_Latn", "indic"): "prajdabre/rotary-indictrans2-en-indic-dist-200M",
    ("indic", "eng_Latn"): "prajdabre/rotary-indictrans2-indic-en-dist-200M",
}


def flores(code: str) -> str:
    return FLORES.get((code or "").strip().lower(), code)


def pick_model(src: str, tgt: str, token: str | None) -> tuple[str, bool]:
    """Returns (model_name, is_official)."""
    if src == "eng_Latn":
        key = ("eng_Latn", "indic")
    elif tgt == "eng_Latn":
        key = ("indic", "eng_Latn")
    else:
        key = ("indic", "indic")

    if token:
        return GATED[key], True
    if key in OPEN:
        return OPEN[key], False
    raise RuntimeError(
        "Indic→Indic translation has no ungated mirror. Set HF_TOKEN in .env "
        "(accept the terms at https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M), "
        "or translate via English in two passes."
    )


_LATIN_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.I)
_LATIN_WORD = re.compile(r"[A-Za-z']+")

# Virama / halant — one per Indic script. It binds two consonants into a single
# conjunct akshara (क् + त → क्त), i.e. one spoken unit, not two.
VIRAMAS = frozenset(
    "्"  # Devanagari  (hi, mr, ne, sa, kok, mai, doi, brx)
    "্"  # Bengali     (bn, as, mni)
    "੍"  # Gurmukhi    (pa)
    "્"  # Gujarati    (gu)
    "୍"  # Oriya       (or)
    "்"  # Tamil       (ta)
    "్"  # Telugu      (te)
    "್"  # Kannada     (kn)
    "്"  # Malayalam   (ml)
    "්"  # Sinhala
)


def count_syllables(text: str, flores_code: str) -> int:
    """Approximate syllable count — the unit that actually travels across a
    language pair.

    Character count is the naive choice and it is wrong, worst of all across
    Latin↔Devanagari (VideoDubber, AAAI 2023). Devanagari encodes vowels as
    combining marks (matras) and the virama as a combining char, so `len()`
    counts क + ु as two units where a speaker utters one. Stripping combining
    marks approximates aksharas, which track speaking time far better.

    Conjuncts are handled by subtracting viramas: क् + त renders as क्त and is
    spoken as ONE akshara, but drops two base consonants into the count. Each
    virama collapses exactly one such pair, so subtracting them is exact rather
    than a fudge factor — verified against hand counts (22/22, 24/24, 13/13).

    Still an approximation for schwa deletion ("मौसम" is mau-sam, not mau-sa-ma),
    which trims real Hindi further. The real answer remains a duration predictor
    over forced-aligned phonemes (§5).
    """
    if not text:
        return 0
    if flores_code.endswith("_Latn"):
        n = 0
        for w in _LATIN_WORD.findall(text):
            groups = len(_LATIN_VOWEL_GROUPS.findall(w))
            if len(w) > 2 and w.lower().endswith("e") and groups > 1:
                groups -= 1  # silent terminal 'e'
            n += max(1, groups)
        return n
    # Indic scripts: base letters (vowels are combining matras), minus the
    # conjuncts that two base consonants collapse into.
    base = sum(1 for ch in text if ch.isalpha() and not unicodedata.combining(ch))
    conjuncts = sum(1 for ch in text if ch in VIRAMAS)
    return max(1, base - conjuncts)


def syllable_budget(budget_ms: int, model: tuple[float, float]) -> int:
    """Syllables that fit in `budget_ms`. Returns 0 when the slot is shorter
    than the fixed overhead — no wording can rescue that line, and telling the
    operator to "shorten it more" would be a lie."""
    overhead, per_syl = model
    usable = budget_ms * (1.0 + FIT_TOLERANCE) - overhead
    return max(0, int(usable // per_syl))


def fits(ratio: float | None) -> bool:
    """Does this line fit its slot?

    Only OVERRUN is a problem. A short line leaves silence, and the original
    already has silence around it — lever 4 of the time-fit budget is literally
    "absorb the slack into the pauses" (docs/ARCHITECTURE.md §0.4). Flagging
    short lines sends the operator to rewrite dialogue that is already fine, and
    buries the lines that genuinely overrun.
    """
    if ratio is None:
        return True
    return ratio <= 1.0 + FIT_TOLERANCE


def estimate_ms(
    text: str,
    tgt_flores: str,
    ref_text: str | None = None,   # kept for call compatibility; see below
    ref_flores: str | None = None,
    ref_ms: int | None = None,
    tts_rate: float | None = None,   # deprecated; use model=
    model: tuple[float, float] | None = None,
) -> int:
    """Estimate how long the TTS ENGINE will take to speak `text`.

    ⚠️ This used to calibrate against the ORIGINAL line's delivery rate, which
    sounds clever and is wrong. The dub is not spoken by the original actor — it
    is spoken by Chatterbox, and the source's tempo says nothing about the
    engine's. Measured on a real Hindi clip:

        Hindi actors : 7.26 syllables/sec
        Chatterbox   : 3.21 syllables/sec   ← 2.26x slower

    The old method therefore under-predicted TTS duration by ~2.5x, told the
    operator lines would come in 17-36% SHORT, and then produced audio 2.22x
    over budget. Predict the engine that will actually speak.

    ref_* are ignored. They remain so existing callers keep working.
    """
    syl = count_syllables(text, tgt_flores)
    if not syl:
        return 0
    if model:
        overhead, per_syl = model
    else:
        overhead, per_syl = settings.tts_overhead_ms, settings.tts_ms_per_syllable
    return int(overhead + per_syl * syl)


def tts_duration_model(project_id: str, lang: str) -> tuple[float, float]:
    """(overhead_ms, ms_per_syllable) for this project, fitted on its own takes.

    Speech duration is NOT syllables ÷ rate. Every utterance carries a fixed
    cost — onset, offset, the breath around it — and only the remainder scales
    with length. Least-squares over 58 real takes:

        duration_ms ≈ 510 + 421 × syllables

    A pure-rate model is 1.4x less accurate overall and catastrophically wrong
    at the short end, where most film dialogue lives: 38 of those 58 lines were
    ≤3 syllables, and a rate model predicts ~220ms for a line that really takes
    820ms. That mismatch is why the estimate said "56 of 58 fit" while the audio
    came out 1.47x over.
    """
    from .models import Take

    with session() as db:
        rows = (
            db.query(Take, Translation)
            .join(Translation, (Translation.segment_id == Take.segment_id)
                  & (Translation.lang == lang))
            .filter(Take.project_id == project_id, Take.lang == lang)
            .all()
        )
        pts = [
            (count_syllables(tr.text or "", flores(lang)), t.raw_ms)
            for t, tr in rows if t.raw_ms and (tr.text or "").strip()
        ]

    pts = [(x, y) for x, y in pts if x >= 1 and y > 100]
    if len(pts) >= 8 and len({x for x, _ in pts}) >= 3:
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        denom = sum((x - mx) ** 2 for x, _ in pts)
        if denom > 0:
            slope = sum((x - mx) * (y - my) for x, y in pts) / denom
            icept = my - slope * mx
            # Guard against a degenerate fit on odd data.
            if 50 <= icept <= 2000 and 80 <= slope <= 1200:
                return icept, slope
    return settings.tts_overhead_ms, settings.tts_ms_per_syllable


def project_tts_rate(project_id: str, lang: str) -> float:
    """The ONE rate this project budgets and measures with.

    Budgeting at one rate and judging at another is how correctly-compressed
    lines end up flagged "too short" — measured: a 4-syllable line hitting a
    4-syllable budget scored 0.54 and went to needs_review. Every caller must
    take its rate from here.
    """
    return observed_tts_rate(project_id, lang) or settings.tts_syll_per_sec


def observed_tts_rate(project_id: str, lang: str) -> float | None:
    """The engine's real syllables/sec on THIS project, from finished takes.

    Ground truth beats a constant: pacing shifts with the reference voice, so
    once a project has takes, believe them.
    """
    from .models import Take

    with session() as db:
        rows = (
            db.query(Take, Translation)
            .join(Translation, (Translation.segment_id == Take.segment_id)
                  & (Translation.lang == lang))
            .filter(Take.project_id == project_id, Take.lang == lang)
            .all()
        )
        # Read every attribute INSIDE the session. ORM objects detach when it
        # closes and touching them then raises DetachedInstanceError.
        pairs = [
            (count_syllables(tr.text or "", flores(lang)), t.raw_ms)
            for t, tr in rows if t.raw_ms and (tr.text or "").strip()
        ]
    pairs = [(s, ms) for s, ms in pairs if s >= 2 and ms > 200]
    if len(pairs) < 5:
        return None
    rates = sorted(s / (ms / 1000) for s, ms in pairs)
    return rates[len(rates) // 2]  # median — robust to a few odd takes


@celery.task(name="translate.run")
def translate(project_id: str, target_lang: str) -> str:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Celery's prefork children do NOT inherit a usable torch thread count, so
    # torch silently runs SINGLE-threaded here. Measured: 58 lines took 27 min
    # (~28s/line) without this; the same code with threads set does ~0.5s/line.
    # The container's OMP_NUM_THREADS is not enough — set it explicitly.
    torch.set_num_threads(settings.mt_threads)

    with _stage(project_id, "translate") as job:
        with session() as db:
            proj = db.get(Project, project_id)
            if not proj:
                raise RuntimeError("project gone")
            src_code = proj.source_lang
            rows = (
                db.query(Segment)
                .filter(Segment.project_id == project_id)
                .order_by(Segment.idx)
                .all()
            )
            items = [
                {"id": s.id, "text": s.text or "", "budget": s.duration_budget_ms or 0}
                for s in rows
                if (s.text or "").strip()
            ]

        if not items:
            job["note"] = "No transcript to translate — run analysis first."
            return project_id

        s_f, t_f = flores(src_code), flores(target_lang)
        if s_f == t_f:
            raise RuntimeError(f"source and target are the same language ({s_f})")

        token = settings.hf_token or None
        model_name, official = pick_model(s_f, t_f, token)
        log.info("translating %d segments %s → %s via %s", len(items), s_f, t_f, model_name)

        # Mirror-first: MinIO, else HuggingFace once, then never again.
        from . import mirror

        try:
            mirror.ensure(model_name, token)
        except Exception as e:
            log.warning("mirror unavailable for %s (%s) — using HF directly", model_name, e)

        # IndicTrans2 ships custom modelling code, so trust_remote_code is
        # unavoidable — note that this executes Python fetched from HF.
        kw = {"trust_remote_code": True}
        if token:
            kw["token"] = token
        tok = AutoTokenizer.from_pretrained(model_name, **kw)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kw)
        model.eval()

        # IndicProcessor does script normalization + tagging. IndicTrans2's
        # quality depends on it; the tokenizer alone is not enough.
        proc = _processor()

        out_texts: list[str] = []
        batch_size = settings.mt_batch_size
        for i in range(0, len(items), batch_size):
            chunk = [it["text"] for it in items[i : i + batch_size]]
            prepped = proc.preprocess_batch(chunk, src_lang=s_f, tgt_lang=t_f)
            enc = tok(prepped, truncation=True, padding="longest",
                      return_tensors="pt", max_length=256)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    num_beams=5,
                    # Dialogue lines are short. 256 let the model ramble to the
                    # cap on garbage ASR input, which is where the real time
                    # went — a bad transcript line cost minutes, not seconds.
                    max_length=settings.mt_max_length,
                    min_length=0,
                    num_return_sequences=1,
                    # IndicTrans2's custom modelling code predates the Cache API
                    # and does `past_key_values[0][0].shape[2] if past_key_values
                    # is not None else 0`. Modern transformers passes an EMPTY
                    # Cache object (non-None) on step 0, so [0][0] is None and
                    # that line raises. Disabling the cache keeps it None and
                    # takes the else branch.
                    #
                    # Cost: no KV reuse, so generation is slower. Fine for 200M
                    # over short dialogue lines; revisit if it bites on a feature
                    # film, where the fix is patching the remote code or pinning
                    # transformers to the 4.4x cache format.
                    use_cache=False,
                )
            decoded = tok.batch_decode(gen, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
            out_texts.extend(proc.postprocess_batch(decoded, lang=t_f))
            log.info("  %d/%d", min(i + batch_size, len(items)), len(items))

        over = 0
        with session() as db:
            db.query(Translation).filter(
                Translation.project_id == project_id, Translation.lang == target_lang
            ).delete(synchronize_session=False)
            db.flush()
            for it, txt in zip(items, out_texts):
                # Calibrate on this line's own delivery rate, not a constant.
                est = estimate_ms(txt, t_f, ref_text=it["text"], ref_flores=s_f,
                                  ref_ms=it["budget"])
                budget = it["budget"] or 0
                ratio = (est / budget) if budget else None
                needs = ratio is not None and abs(ratio - 1.0) > FIT_TOLERANCE
                if needs:
                    over += 1
                db.add(Translation(
                    project_id=project_id,
                    segment_id=it["id"],
                    lang=target_lang,
                    text_mt=txt,
                    engine=model_name,
                    est_duration_ms=est,
                    fit_ratio=round(ratio, 3) if ratio else None,
                    state="needs_review" if needs else "draft",
                ))
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "translating"
                langs = list(proj.target_langs or [])
                if target_lang not in langs:
                    proj.target_langs = langs + [target_lang]

        provenance = (
            "official AI4Bharat checkpoint"
            if official else
            "ungated community mirror (prajdabre, MIT) — an independent RoPE reproduction, "
            "NOT the checkpoint AI4Bharat's published benchmarks describe. Set HF_TOKEN to "
            "use the official one."
        )
        job["note"] = (
            f"{len(items)} lines {s_f} → {t_f} · {model_name} · {provenance} · "
            f"{over} of {len(items)} estimated outside ±{int(FIT_TOLERANCE * 100)}% of the "
            "original's duration. Fit is estimated from conjunct-aware syllable count, "
            "calibrated against each line's own delivery rate. Still an approximation — it "
            "ignores schwa deletion, so real Hindi runs a little shorter than estimated. "
            "Trust your ear over the badge (docs §5)."
        )
    return project_id


def _processor():
    """IndicTransToolkit moved its import path between releases."""
    try:
        from IndicTransToolkit.processor import IndicProcessor  # newer
    except ImportError:
        from IndicTransToolkit import IndicProcessor  # older
    return IndicProcessor(inference=True)


# ── refine (LLM pass) ─────────────────────────────────────────────────────

REFINE_PROMPT = """You are a dubbing script editor cutting {tgt_name} lines to length.

These lines will be SPOKEN ALOUD over picture, in a slot of fixed length. A line that runs long
cannot be fixed later: stretching audio past ~15% makes the voice sound artificial, so an
over-length line either talks over the next shot or gets squashed until it sounds wrong.

**Your first duty is the syllable budget.** Each line has "max_syllables". Going over is a
failure, even if the result reads beautifully. Going UNDER is fine and often better — silence
is free, overrun is not.

How to cut, in order:
1. Drop words that carry no meaning: "actually", "you know", "I mean", "just", "well", "so".
2. Contract everything: "I am" → "I'm", "it is" → "it's", "do not" → "don't".
3. Replace long words with short ones: "immediately" → "now", "approximately" → "about",
   "physician" → "doctor", "residence" → "home".
4. Drop redundancy the picture already tells us. If a character points at a chart, "the chart
   over there" becomes "that".
5. Cut whole clauses. Keep the ONE idea the line exists to deliver.
6. Vocatives and politeness go last: "ma'am", "sir", "please" are the first words to lose when
   nothing else can go.

Never do these:
- Never drop a proper noun, a number, a negation ("not", "never"), or a question mark.
- Never merge two speakers' lines.
- Never invent information that is not in the source.
- Never leave a line empty.

Also make it natural {tgt_name} that an actor can say — contractions, spoken word order, the
register the speaker would actually use with the person they are addressing. Keep each
character's voice consistent across the scene.

Return JSON only:
{{"lines": [{{"idx": <int>, "text": "<line>", "syllables": <your count>, "note": "<max 6 words>"}}]}}

Include every idx exactly once. Count syllables as spoken, and check the count before you answer.

LINES:
{lines}
"""


@celery.task(name="translate.refine")
def refine(project_id: str, target_lang: str) -> str:
    from . import llm

    with _stage(project_id, "refine", skip_if=lambda: llm.configured() is None) as job:
        with session() as db:
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "translated"
        if llm.configured() is None:
            job["note"] = (
                "Skipped — no LLM key. IndicTrans2's draft is used as-is, so there is no "
                "document-level context, character register, or length rewriting. "
                "Set GEMINI_API_KEY (or XAI_API_KEY) in .env and re-run."
            )
            return project_id

        with session() as db:
            proj = db.get(Project, project_id)
            src_code = proj.source_lang
            rows = (
                db.query(Translation, Segment, Speaker)
                .join(Segment, Translation.segment_id == Segment.id)
                .outerjoin(Speaker, Segment.speaker_id == Speaker.id)
                .filter(Translation.project_id == project_id, Translation.lang == target_lang)
                .order_by(Segment.idx)
                .all()
            )
            items = [
                {
                    "id": t.id,
                    "idx": s.idx,
                    "speaker": (sp.display_name or sp.label) if sp else "Unknown",
                    "source": s.text or "",
                    # Refine the MT draft, not a previous refinement — so re-running
                    # is idempotent rather than drifting further each pass.
                    "draft": t.text_mt or "",
                    "budget_ms": s.duration_budget_ms or 0,
                    "src_text": s.text or "",
                }
                for t, s, sp in rows
            ]

        if not items:
            job["note"] = f"Nothing translated into {target_lang} yet."
            return project_id

        s_f, t_f = flores(src_code), flores(target_lang)
        src_name = LANG_NAMES.get(src_code, s_f)
        tgt_name = LANG_NAMES.get(target_lang, t_f)

        # Syllable budget from the TTS ENGINE's measured rate — not the original
        # actor's. Chatterbox speaks ~4.6 syl/s where Hindi actors run 7.26, so
        # budgeting at the actor's tempo asks for lines ~1.6x too long and every
        # one of them overruns. Prefer a rate observed on this project's own
        # takes when we have them.
        model = tts_duration_model(project_id, target_lang)
        log.info("refine: duration model = %.0fms + %.0fms/syllable", *model)
        impossible = 0
        for it in items:
            b = syllable_budget(it["budget_ms"], model)
            if b < 1:
                impossible += 1
            it["target_syllables"] = max(1, b)

        out: dict[int, dict] = {}
        bs = settings.llm_batch_size
        for i in range(0, len(items), bs):
            # Overlap one line either side so the model sees conversational context
            # across batch seams instead of restarting cold each time.
            lo, hi = max(0, i - 1), min(len(items), i + bs + 1)
            window = items[lo:hi]
            payload = []
            for w in window:
                cur = count_syllables(w["draft"], t_f)
                payload.append({
                    "idx": w["idx"],
                    "speaker": w["speaker"],
                    "source": w["source"],
                    "draft": w["draft"],
                    "draft_syllables": cur,
                    "max_syllables": w["target_syllables"],
                    "over_by": max(0, cur - w["target_syllables"]),
                })
            prompt = REFINE_PROMPT.format(
                src_name=src_name, tgt_name=tgt_name,
                lines=json.dumps(payload, ensure_ascii=False, indent=1),
            )
            data = llm.complete_json(prompt)
            lines = data.get("lines", []) if isinstance(data, dict) else []
            for ln in lines:
                try:
                    out[int(ln["idx"])] = {
                        "text": str(ln["text"]).strip(),
                        "note": str(ln.get("note") or "").strip()[:120],
                    }
                except (KeyError, TypeError, ValueError):
                    continue
            log.info("refined %d/%d", min(i + bs, len(items)), len(items))

        # The prompt asks for a budget; asking is not the same as getting it.
        # Re-submit only the lines that are STILL over, up to a few rounds. Each
        # round sees its own failure, which converges far better than one plea.
        for attempt in range(settings.refine_passes):
            stubborn = []
            for it in items:
                r = out.get(it["idx"])
                if not r or not r["text"]:
                    continue
                if count_syllables(r["text"], t_f) > it["target_syllables"]:
                    stubborn.append({**it, "draft": r["text"]})
            if not stubborn:
                break
            log.info("refine pass %d: %d line(s) still over budget", attempt + 2, len(stubborn))
            for i in range(0, len(stubborn), bs):
                chunk = stubborn[i:i + bs]
                payload = [{
                    "idx": w["idx"], "speaker": w["speaker"], "source": w["source"],
                    "draft": w["draft"],
                    "draft_syllables": count_syllables(w["draft"], t_f),
                    "max_syllables": w["target_syllables"],
                    "over_by": count_syllables(w["draft"], t_f) - w["target_syllables"],
                } for w in chunk]
                prompt = REFINE_PROMPT.format(
                    src_name=src_name, tgt_name=tgt_name,
                    lines=json.dumps(payload, ensure_ascii=False, indent=1),
                ) + (
                    "\n\nThese lines are STILL over budget after an earlier edit. Cut harder. "
                    "Losing a politeness word or a whole subordinate clause is correct here."
                )
                try:
                    data = llm.complete_json(prompt, temperature=0.5)
                except Exception as e:
                    log.warning("refine retry failed: %s", e)
                    continue
                for ln in (data.get("lines", []) if isinstance(data, dict) else []):
                    try:
                        idx = int(ln["idx"])
                        txt = str(ln["text"]).strip()
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not txt:
                        continue
                    # Only accept the retry if it is genuinely shorter.
                    if count_syllables(txt, t_f) < count_syllables(out[idx]["text"], t_f):
                        out[idx] = {"text": txt, "note": str(ln.get("note") or "").strip()[:120]}

        engine = llm.label()
        changed = fixed = 0
        with session() as db:
            for it in items:
                r = out.get(it["idx"])
                if not r or not r["text"]:
                    continue
                t = db.get(Translation, it["id"])
                if not t:
                    continue
                t.text_llm = r["text"]
                t.llm_engine = engine
                t.llm_note = r["note"] or None
                if r["text"] != it["draft"]:
                    changed += 1

                seg = db.get(Segment, t.segment_id)
                t.est_duration_ms = estimate_ms(t.text, t_f, model=model)
                if seg and seg.duration_budget_ms:
                    t.fit_ratio = round(t.est_duration_ms / seg.duration_budget_ms, 3)
                    if fits(t.fit_ratio):
                        fixed += 1
                    if t.state != "approved":
                        t.state = "draft" if fits(t.fit_ratio) else "needs_review"

        with session() as db:
            proj = db.get(Project, project_id)
            if proj:
                proj.status = "translated"

        job["note"] = (
            f"{engine} refined {changed} of {len(items)} lines · {fixed} now fit within "
            f"±{int(FIT_TOLERANCE * 100)}%"
            + (f" · {impossible} slot(s) are shorter than the {model[0]:.0f}ms floor a spoken "
               "utterance takes at all — those cannot be fixed by shortening and will run into "
               "the following pause" if impossible else "")
            + ". The LLM only edits IndicTrans2's draft — it never "
            "translates from scratch, because English→Indic is its weak direction."
        )
    return project_id


def enqueue_translation(project_id: str, target_lang: str, do_refine: bool = True) -> None:
    from celery import chain

    from .models import TRANSLATION_STAGES, Job

    with session() as db:
        db.query(Job).filter(
            Job.project_id == project_id, Job.stage.in_(TRANSLATION_STAGES)
        ).delete(synchronize_session=False)
        db.flush()
        for stage in TRANSLATION_STAGES:
            db.add(Job(project_id=project_id, stage=stage, state="queued", capability="cpu"))
        proj = db.get(Project, project_id)
        if proj:
            proj.status = "translating"

    if do_refine:
        chain(
            translate.si(project_id, target_lang),
            refine.si(project_id, target_lang),
        ).apply_async()
    else:
        translate.si(project_id, target_lang).apply_async()
