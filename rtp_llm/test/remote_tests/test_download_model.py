"""Temporary test to download models to NAS for smoke tests.

Run with:
    pytest --remote --rtp-ci-profile=smoke_light_sm9x \
        rtp_llm/test/remote_tests/test_download_model.py -k "gemma4" --timeout=7200

Delete this file after models are downloaded.
"""
import os
import pytest


@pytest.mark.manual
@pytest.mark.smoke
@pytest.mark.timeout(7200)
@pytest.mark.gpu(type="H20", count=1)
def test_download_gemma4_31b(gpu_lock):
    """Download gemma-4-31B-it to /mnt/nas1/hf/ if not already present."""
    model_id = "google/gemma-4-31B-it"
    target_dir = "/mnt/nas1/hf/gemma-4-31B-it"

    if os.path.exists(os.path.join(target_dir, "config.json")):
        print(f"Model already exists at {target_dir}, skipping download")
        return

    # Use the internal HF mirror for faster download
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://whale-hf-mirror.alibaba-inc.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint

    print(f"Downloading {model_id} to {target_dir} via {hf_endpoint}")

    from huggingface_hub import snapshot_download
    snapshot_download(
        model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    # Verify download
    assert os.path.exists(os.path.join(target_dir, "config.json")), \
        f"config.json not found in {target_dir}"
    print(f"Successfully downloaded {model_id} to {target_dir}")
