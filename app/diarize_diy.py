"""Ungated speaker diarization, assembled from parts that need no account.

Every component downloads anonymously and permits commercial use:

  · Silero VAD    — MIT. The ONNX model ships inside the pip package, so there
                    is no download at all. Nothing to gate.
  · WeSpeaker     — pyannote/wespeaker-voxceleb-resnet34-LM, CC-BY-4.0,
                    gated=False (verified: anonymous HTTP 200). The irony worth
                    knowing is that pyannote's *embedding* model is open; only
                    the assembled pipeline is gated.
  · Clustering    — scikit-learn agglomerative. No model, no weights.

WHAT THIS COSTS, said plainly
-----------------------------
This is clustering-based diarization: VAD → window → embed → cluster. It
assigns every instant to exactly ONE speaker, so it **cannot represent
overlapping speech** — two people talking at once become one of them. Films are
full of that. pyannote's gated `segmentation-3.0` exists precisely to model up
to 3 simultaneous speakers (its "powerset" formulation), and that capability is
what the gate buys.

Published DER for reference: pyannote community-1 ≈ 17.0 on AMI-IHM. Expect
this to land worse — clustering pipelines typically sit ~20-25 — and worse
again on overlap-heavy material.

Use it because you want zero dependencies. Switch DIAR_BACKEND=pyannote (with a
token, mirrored once) when accuracy matters more than independence.
"""

import logging

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

SR = 16000


def _load_audio(path: str):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        # Resample rather than assume: a wrong rate silently wrecks both the VAD
        # and the embeddings.
        import subprocess
        import tempfile

        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ar", str(SR), "-ac", "1", tmp],
            check=True,
        )
        data, sr = sf.read(tmp, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    return mono, sr


def speech_regions(path: str) -> list[tuple[float, float]]:
    """Silero VAD → (start, end) speech regions in seconds."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    audio, sr = _load_audio(path)
    ts = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=sr,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200,
        speech_pad_ms=30,
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def _windows(regions, win: float, hop: float, min_win: float):
    """Slice speech into embedding windows.

    Windows must be long enough for a speaker embedding to be meaningful
    (~1s+) but short enough not to straddle a speaker change. Anything shorter
    than min_win is kept whole rather than dropped — a short interjection is
    still a line someone has to dub.
    """
    out = []
    for start, end in regions:
        dur = end - start
        if dur < min_win:
            out.append((start, end))
            continue
        t = start
        while t + win <= end + 1e-6:
            out.append((t, t + win))
            t += hop
        # Keep the tail if it's substantial.
        if end - t > min_win:
            out.append((t, end))
    return out


def _embed(path: str, windows: list[tuple[float, float]]) -> np.ndarray:
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    from . import mirror

    repo = settings.diar_embedding_model
    try:
        mirror.ensure(repo)  # ungated — no token passed, none needed
    except Exception as e:
        log.warning("mirror unavailable for %s (%s)", repo, e)

    model = Model.from_pretrained(repo)
    inference = Inference(model, window="whole")

    audio, sr = _load_audio(path)
    import torch

    file = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}

    vecs = []
    for a, b in windows:
        try:
            v = inference.crop(file, Segment(a, b))
        except Exception:
            v = np.zeros(256, dtype="float32")
        vecs.append(np.asarray(v).reshape(-1))
    dim = max(len(v) for v in vecs)
    return np.stack([
        v if len(v) == dim else np.pad(v, (0, dim - len(v))) for v in vecs
    ])


def _cluster(emb: np.ndarray, num_speakers: int | None) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    if len(emb) == 1:
        return np.zeros(1, dtype=int)

    # Length-normalise so euclidean distance ≈ cosine distance, which is what
    # speaker embeddings are trained for.
    x = normalize(emb)

    if num_speakers and num_speakers > 1:
        model = AgglomerativeClustering(
            n_clusters=min(num_speakers, len(x)), metric="cosine", linkage="average"
        )
    else:
        # Unknown count — threshold instead of guessing k. Films have as many
        # speakers as they have; forcing a number is worse than a loose cut.
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=settings.diar_threshold,
            metric="cosine",
            linkage="average",
        )
    return model.fit_predict(x)


def _absorb_specks(emb: np.ndarray, labels: np.ndarray, windows, min_seconds: float) -> np.ndarray:
    """Fold tiny clusters into their nearest real one.

    Clustering on real film audio produces a long tail of specks — a cluster
    holding 0.4s of a door slam or a breath. Measured on a real two-hander:
    threshold 0.70 gave 10 clusters of which 6 held ~8s combined; the two real
    speakers held 54s and 51s.

    Chasing this with the threshold alone is fragile — it trades specks for
    merging your actual cast. Filtering by how much a cluster actually SPEAKS is
    the robust axis, and it holds at any threshold.
    """
    from sklearn.preprocessing import normalize

    secs: dict[int, float] = {}
    for i, (a, b) in enumerate(windows):
        secs[labels[i]] = secs.get(labels[i], 0.0) + (b - a)

    real = [c for c, s in secs.items() if s >= min_seconds]
    specks = [c for c, s in secs.items() if s < min_seconds]
    if not real or not specks:
        return labels

    x = normalize(emb)
    centroids = {c: normalize(x[labels == c].mean(axis=0).reshape(1, -1))[0] for c in real}
    out = labels.copy()
    for c in specks:
        idx = np.where(labels == c)[0]
        v = normalize(x[idx].mean(axis=0).reshape(1, -1))[0]
        nearest = max(real, key=lambda r: float(np.dot(v, centroids[r])))
        out[idx] = nearest
    log.info("absorbed %d speck cluster(s) below %.1fs into %d real speakers",
             len(specks), min_seconds, len(real))
    return out


def _to_turns(windows, labels, merge_gap: float = 0.35) -> list[dict]:
    """Collapse per-window labels into contiguous speaker turns."""
    order = np.argsort([w[0] for w in windows])
    turns: list[dict] = []
    for i in order:
        a, b = windows[i]
        spk = f"SPEAKER_{int(labels[i]):02d}"
        if turns and turns[-1]["speaker"] == spk and a - turns[-1]["end"] <= merge_gap:
            turns[-1]["end"] = max(turns[-1]["end"], b)
        else:
            turns.append({"speaker": spk, "start": float(a), "end": float(b)})
    return turns


def diarize(path: str, num_speakers: int | None = None) -> tuple[list[dict], dict]:
    """Returns (turns, info). Turns are non-overlapping by construction."""
    regions = speech_regions(path)
    if not regions:
        return [], {"reason": "no speech detected"}

    wins = _windows(
        regions,
        win=settings.diar_window,
        hop=settings.diar_hop,
        min_win=settings.diar_min_window,
    )
    log.info("diy diarize: %d speech regions → %d windows", len(regions), len(wins))

    emb = _embed(path, wins)
    labels = _cluster(emb, num_speakers)
    if not num_speakers:
        labels = _absorb_specks(emb, labels, wins, settings.diar_min_speaker_seconds)
    turns = _to_turns(wins, labels)
    n = len(set(labels))
    speech = sum(b - a for a, b in regions)
    return turns, {
        "speakers": n,
        "windows": len(wins),
        "regions": len(regions),
        "speech_seconds": round(speech, 2),
    }
