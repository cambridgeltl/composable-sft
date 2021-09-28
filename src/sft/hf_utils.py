from huggingface_hub import snapshot_download


def pull_from_hf_model_hub(specifier, version=None, cache_dir=None):
    return snapshot_download(
        specifier,
        revision=version,
        cache_dir=cache_dir,
    )

