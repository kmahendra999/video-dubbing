"""Model mirror — HuggingFace → your MinIO → every worker.

Why this exists
---------------
The models are already self-hosted: they run in your containers, on your box.
HuggingFace is only a *file server*. But depending on it at deploy time means
depending on someone else's uptime, rate limits, gates and continued goodwill —
and gated repos need a token on every fresh host.

The licences permit fixing that properly. pyannote 3.1 is MIT, community-1 is
CC-BY-4.0, IndicTrans2 is MIT, Whisper is MIT — all of them allow redistribution
(MIT: keep the notice; CC-BY: attribute). The gate is lead capture, not a licence
term: pyannote's own gate text says the pipeline "will always remain open-source".

So: accept a gate once, mirror the weights into your object store, and every
worker afterwards pulls from MinIO. No token, no HuggingFace, no egress.

How it works
------------
We mirror the HuggingFace *cache directory* rather than a flat snapshot, because
that's the one format every library here already understands — transformers,
faster-whisper and pyannote all read it without special-casing. Once mirrored,
set HF_HUB_OFFLINE=1 and the whole stack runs with no network at all.
"""

import logging
import os
import shutil

from . import storage
from .config import settings

log = logging.getLogger(__name__)

PREFIX = "models"


def hub_cache() -> str:
    return os.path.join(os.environ.get("HF_HOME", settings.model_dir + "/hf"), "hub")


def cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def cache_path(repo_id: str) -> str:
    return os.path.join(hub_cache(), cache_name(repo_id))


def _prefix(repo_id: str) -> str:
    return f"{PREFIX}/{cache_name(repo_id)}/"


def is_cached(repo_id: str) -> bool:
    d = cache_path(repo_id)
    return os.path.isdir(d) and any(os.scandir(d))


def is_mirrored(repo_id: str) -> bool:
    r = storage.client().list_objects_v2(
        Bucket=settings.s3_bucket, Prefix=_prefix(repo_id), MaxKeys=1
    )
    return bool(r.get("KeyCount"))


def mirror_size(repo_id: str) -> tuple[int, int]:
    """(files, bytes) held in MinIO for this repo."""
    c = storage.client()
    files = total = 0
    for page in c.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=_prefix(repo_id)
    ):
        for o in page.get("Contents") or []:
            files += 1
            total += o["Size"]
    return files, total


def push(repo_id: str) -> tuple[int, int]:
    """Upload an already-cached repo into MinIO."""
    src = cache_path(repo_id)
    if not is_cached(repo_id):
        raise RuntimeError(f"{repo_id} is not in the local cache — nothing to push")
    n = total = 0
    for root, _, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src)

            # The HF cache holds every file TWICE: the real bytes in blobs/<sha>,
            # and a symlink at snapshots/<rev>/<name> pointing to it. S3 has no
            # symlinks, so uploading both stores everything twice. Skip blobs/
            # and materialise the real bytes under snapshots/ instead — that's
            # the path every library actually resolves.
            if rel.split(os.sep)[0] == "blobs":
                continue

            real = os.path.realpath(full)
            if not os.path.isfile(real):
                continue
            storage.upload_file(real, _prefix(repo_id) + rel)
            n += 1
            total += os.path.getsize(real)
    log.info("mirrored %s → minio (%d files, %.1f MB)", repo_id, n, total / 1e6)
    return n, total


def pull(repo_id: str) -> str:
    """Download a mirrored repo from MinIO into the local HF cache."""
    dest = cache_path(repo_id)
    c = storage.client()
    n = 0
    for page in c.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=_prefix(repo_id)
    ):
        for o in page.get("Contents") or []:
            rel = o["Key"][len(_prefix(repo_id)):]
            if not rel:
                continue
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out):
                continue
            storage.download_file(o["Key"], out)
            n += 1
    if not n and not is_cached(repo_id):
        raise RuntimeError(f"{repo_id} is not mirrored in MinIO")
    log.info("pulled %s from minio (%d files)", repo_id, n)
    return dest


def ensure(repo_id: str, token: str | None = None, allow_patterns=None) -> str:
    """Make repo_id usable locally, preferring the mirror. Returns its cache dir.

    Order: local cache → MinIO mirror → HuggingFace (then mirror it).
    Only the last step needs the network or a token, and only ever once.
    """
    if is_cached(repo_id):
        # Cached but not mirrored is a trap: it works on THIS host and fails on
        # the next one. Backfill the mirror so the cache is never the only copy.
        if not is_mirrored(repo_id):
            try:
                push(repo_id)
            except Exception as e:
                log.warning("could not backfill mirror for %s: %s", repo_id, e)
        return cache_path(repo_id)

    if is_mirrored(repo_id):
        log.info("%s: pulling from MinIO mirror (no HF contact)", repo_id)
        return pull(repo_id)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            f"{repo_id} is neither cached nor mirrored, and HF_HUB_OFFLINE=1. "
            f"Mirror it first: docker compose run --rm worker-ai python -m app.mirror add {repo_id}"
        )

    log.info("%s: not mirrored — fetching from HuggingFace once", repo_id)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, token=token or None, allow_patterns=allow_patterns)
    try:
        push(repo_id)
    except Exception as e:  # a failed mirror must not fail the job
        log.warning("could not mirror %s: %s", repo_id, e)
    return cache_path(repo_id)


def drop_local(repo_id: str) -> None:
    shutil.rmtree(cache_path(repo_id), ignore_errors=True)


# Everything this deployment can need. Kept explicit so `mirror all` is a
# complete, auditable list rather than whatever happened to get downloaded.
CATALOG = {
    # ASR — MIT
    "Systran/faster-whisper-small": {"gated": False, "why": "ASR (default)"},
    "Systran/faster-whisper-large-v3": {"gated": False, "why": "ASR (production)"},
    # Translation — MIT weights, ungated community mirror
    "prajdabre/rotary-indictrans2-en-indic-dist-200M": {"gated": False, "why": "MT en→indic (ungated)"},
    "prajdabre/rotary-indictrans2-indic-en-dist-200M": {"gated": False, "why": "MT indic→en (ungated)"},
    # Translation — MIT weights, official but gated
    "ai4bharat/indictrans2-en-indic-dist-200M": {"gated": True, "why": "MT en→indic (official)"},
    "ai4bharat/indictrans2-indic-en-dist-200M": {"gated": True, "why": "MT indic→en (official)"},
    # Diarization — the DIY backend needs only the ungated embedding model.
    # Silero VAD ships inside its pip package, so there is nothing to mirror.
    "pyannote/wespeaker-voxceleb-resnet34-LM": {"gated": False, "why": "speaker embeddings (DIY backend)"},
    # Optional overlap-aware upgrade — gated, needs HF_TOKEN once.
    "pyannote/speaker-diarization-community-1": {"gated": True, "why": "diarization pipeline (optional upgrade)"},
    "pyannote/segmentation-3.0": {"gated": True, "why": "overlap-aware VAD (optional upgrade)"},
    # TTS — MIT code AND weights.
    "ResembleAI/chatterbox": {"gated": False, "why": "voice cloning (Chatterbox)"},
    # Synthetic voice bank source — Apache-2.0, ungated, 54 fixed voices.
    "hexgrad/Kokoro-82M": {"gated": False, "why": "synthetic AI voice bank (Kokoro)"},
}


def status() -> list[dict]:
    out = []
    for repo, meta in CATALOG.items():
        mirrored = is_mirrored(repo)
        files, size = mirror_size(repo) if mirrored else (0, 0)
        out.append({
            "repo": repo,
            "why": meta["why"],
            "gated": meta["gated"],
            "cached": is_cached(repo),
            "mirrored": mirrored,
            "files": files,
            "bytes": size,
        })
    return out


def _cli():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    storage.ensure_bucket()
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    token = settings.hf_token or None

    if cmd == "status":
        print(f"{'repo':<52} {'gate':<6} {'cached':<7} {'mirrored':<9} size")
        for r in status():
            size = f"{r['bytes']/1e6:.0f} MB" if r["bytes"] else "—"
            print(f"{r['repo']:<52} {'yes' if r['gated'] else 'no':<6} "
                  f"{'yes' if r['cached'] else 'no':<7} {'yes' if r['mirrored'] else 'no':<9} {size}")
        return

    if cmd == "add":
        for repo in args[1:]:
            try:
                ensure(repo, token)
                print(f"✅ {repo}")
            except Exception as e:
                print(f"❌ {repo}: {e}")
        return

    if cmd == "all":
        for repo, meta in CATALOG.items():
            if meta["gated"] and not token:
                print(f"⏭️  {repo} — gated, needs HF_TOKEN")
                continue
            try:
                ensure(repo, token)
                print(f"✅ {repo}")
            except Exception as e:
                print(f"❌ {repo}: {str(e)[:120]}")
        return

    print("usage: python -m app.mirror [status|add <repo>...|all]")


if __name__ == "__main__":
    _cli()
