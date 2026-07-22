import os

# docker-compose passes HF_TOKEN through unconditionally, so an unset token
# arrives as the EMPTY STRING rather than being absent. huggingface_hub treats
# "" as a real token and builds the header `Bearer ` — which httpx rejects as an
# illegal header value, breaking every download with an error that names neither
# the token nor the cause.
#
# Empty-but-set is worse than unset. Normalise it here, before any library reads
# the environment: importing `app` is the first thing every worker does.
for _var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
    if os.environ.get(_var, "").strip() == "":
        os.environ.pop(_var, None)

# Same story: HF_HUB_OFFLINE="" is falsy to us but truthy to some readers.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if os.environ.get(_var, "").strip() == "":
        os.environ.pop(_var, None)

del _var
