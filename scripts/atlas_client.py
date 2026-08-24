#!/usr/bin/env python3
"""Minimal Atlas Cloud client for asynchronous media generation."""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request


class AtlasCloudError(RuntimeError):
    pass


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _key():
    _load_env()
    key = os.environ.get("ATLASCLOUD_API_KEY")
    if not key:
        raise AtlasCloudError(
            "ATLASCLOUD_API_KEY environment variable is not set. "
            "Create a key at https://www.atlascloud.ai/console/api-key."
        )
    return key


def _base_url():
    return os.environ.get(
        "ATLASCLOUD_MEDIA_API_BASE", "https://api.atlascloud.ai/api/v1/model"
    ).rstrip("/")


def _request(req, timeout=120):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:400]
        raise AtlasCloudError(
            f"{req.method} {req.full_url} -> {error.code}: {detail}"
        ) from error
    except Exception as error:
        raise AtlasCloudError(f"{req.method} {req.full_url} failed: {error}") from error


def post(endpoint, payload):
    """Submit one generation request. This function deliberately never retries."""
    req = urllib.request.Request(
        f"{_base_url()}/{endpoint.lstrip('/')}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _request(req)


def submit(endpoint, payload):
    response = post(endpoint, payload)
    data = response.get("data") if isinstance(response.get("data"), dict) else response
    request_id = data.get("id") or data.get("request_id")
    if not request_id:
        raise AtlasCloudError(f"Atlas Cloud submit returned no prediction ID: {response}")
    return request_id


def get_prediction(request_id, retries=3):
    """Fetch prediction state with bounded backoff for transient GET failures."""
    req = urllib.request.Request(
        f"{_base_url()}/prediction/{request_id}",
        headers={"Authorization": f"Bearer {_key()}"},
        method="GET",
    )
    last_error = None
    for attempt in range(retries):
        try:
            return _request(req)
        except AtlasCloudError as error:
            if isinstance(error.__cause__, urllib.error.HTTPError):
                raise
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def extract_output(result):
    if not isinstance(result, dict):
        return None
    data = result.get("data") if isinstance(result.get("data"), dict) else result
    structured = data.get("stt_result")
    if isinstance(structured, dict):
        return structured
    outputs = data.get("outputs") or data.get("output")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("url", "image_url", "video_url", "audio_url"):
                if first.get(key):
                    return first[key]
    if isinstance(outputs, str):
        return outputs
    for key in ("image_url", "video_url", "audio_url", "url"):
        if isinstance(data.get(key), str):
            return data[key]
    return None


def prediction_status(request_id):
    result = get_prediction(request_id)
    data = result.get("data") if isinstance(result.get("data"), dict) else result
    status = str(data.get("status") or "").lower()
    if status in ("completed", "success", "succeeded"):
        return {"status": "completed", "output": extract_output(result), "error": None}
    if status in ("failed", "cancelled", "canceled"):
        error = data.get("error") or data.get("error_message") or "Prediction failed"
        return {"status": "failed", "output": None, "error": str(error)}
    return {"status": "pending", "output": None, "error": None}


def upload(file_path):
    """Stream one local media file without exposing the API key in process arguments."""
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--header",
        "@-",
        "--form",
        f"file=@{file_path}",
        f"{_base_url()}/uploadMedia",
    ]
    try:
        completed = subprocess.run(
            command,
            input=f"Authorization: Bearer {_key()}\n",
            capture_output=True,
            text=True,
            check=True,
        )
        response = json.loads(completed.stdout)
    except Exception as error:
        raise AtlasCloudError(f"Upload failed for {file_path}: {error}") from error
    data = response.get("data") if isinstance(response.get("data"), dict) else response
    url = data.get("download_url")
    if not url:
        raise AtlasCloudError(f"Atlas Cloud upload returned no download URL: {response}")
    return url


def download(url, dest):
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as target:
            shutil.copyfileobj(response, target)
    except Exception as error:
        raise AtlasCloudError(f"Download failed for {url}: {error}") from error
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise AtlasCloudError(f"Download produced an empty file: {url}")
    return dest
