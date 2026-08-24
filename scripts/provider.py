#!/usr/bin/env python3
"""Pluggable media backends for Vox Director pipeline stages."""

import time
from abc import ABC, abstractmethod

import atlas_client
import muapi_client


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    """Abstract provider surface for Vox Director pipeline stages."""
    name = "base"
    allows_resubmission = True

    @abstractmethod
    def submit_image(self, model, prompt, **params): ...
    @abstractmethod
    def submit_video(self, model, prompt, **params): ...
    @abstractmethod
    def submit_audio(self, model, **params): ...
    @abstractmethod
    def remove_bg(self, model, image_url, **params): ...
    @abstractmethod
    def get_status(self, job_id): ...
    @abstractmethod
    def upload(self, path): ...
    @abstractmethod
    def download(self, url, dest): ...


class MuapiProvider(Provider):
    """Wraps muapi_client — interfaces with api.muapi.ai"""
    name = "muapi"

    def submit_image(self, model, prompt, **params):
        return muapi_client.submit(model, {"prompt": prompt, **params})

    def submit_video(self, model, prompt, **params):
        return muapi_client.submit(model, {"prompt": prompt, **params})

    def submit_audio(self, model, **params):
        return muapi_client.submit(model, params)

    def remove_bg(self, model, image_url, **params):
        return muapi_client.submit(model or "remove-background", {"image_url": image_url, **params})

    def get_status(self, job_id):
        res = None
        try:
            res = muapi_client._get(f"/predictions/{job_id}/result")
        except Exception as e:
            try:
                res = muapi_client._get(f"/predictions/{job_id}")
            except Exception as e2:
                return {"status": "pending", "output": None, "error": str(e2)}

        if not res:
            return {"status": "pending", "output": None, "error": None}

        status = (res.get("status") or (res.get("data") or {}).get("status") or "").lower()
        if status in ("completed", "success", "succeeded"):
            url = muapi_client.extract_url(res)
            return {"status": "completed", "output": url, "error": None}
        if status in ("failed", "cancelled"):
            err = res.get("error") or (res.get("data") or {}).get("error") or "Job failed"
            return {"status": "failed", "output": None, "error": str(err)}
        return {"status": "pending", "output": None, "error": None}

    def upload(self, path):
        return muapi_client.upload(path)

    def download(self, url, dest):
        return muapi_client.download(url, dest)


class AtlasCloudProvider(Provider):
    """Atlas Cloud backend with one-shot generation submissions."""

    name = "atlas_cloud"
    allows_resubmission = False

    MODEL_ALIASES = {
        "nano-banana-2": "google/nano-banana-2-lite/text-to-image",
        "flux-dev": "black-forest-labs/flux-dev",
        "veo3.1-image-to-video": "google/veo3.1/image-to-video",
        "veo31-image-to-video": "google/veo3.1/image-to-video",
        "minimax-speech-2.6-turbo": "minimax/speech-2.6-turbo",
        "suno-create-music": "suno/chirp-v5",
        "remove-background": "youchuan/v8.2/remove-background",
        "whisper": "bytedance/seed-asr-2.0",
    }
    IMAGE_FIELDS = {
        "google/nano-banana-2-lite/text-to-image": {
            "aspect_ratio", "enable_base64_output", "enable_sync_mode", "prompt",
            "resolution", "thinking_level",
        },
        "black-forest-labs/flux-dev": {
            "seed", "size", "image", "prompt", "strength", "mask_image", "num_images",
            "guidance_scale", "num_inference_steps", "enable_base64_output",
            "enable_safety_checker",
        },
    }
    VIDEO_FIELDS = {
        "google/veo3.1/image-to-video": {
            "aspect_ratio", "duration", "generate_audio", "image", "last_image",
            "negative_prompt", "prompt", "resolution", "seed",
        },
    }
    AUDIO_FIELDS = {
        "minimax/speech-2.6-turbo": {
            "text", "voice", "speed", "vol", "pitch", "emotion", "language_boost",
            "format", "sample_rate", "bitrate",
        },
        "suno/chirp-v5": {
            "prompt", "custom", "instrumental", "vocal_gender", "title", "style",
            "negative_tags", "auto_lyrics", "style_weight", "weirdness_constraint",
            "audio_weight", "persona_id",
        },
        "bytedance/seed-asr-2.0": {
            "audio_url", "format", "language", "enable_itn", "enable_punc", "enable_ddc",
            "enable_speaker_info", "show_utterances", "boosting_table_name",
            "correct_table_name", "context",
        },
    }

    @classmethod
    def _model(cls, model):
        resolved = cls.MODEL_ALIASES.get(model, model)
        supported = set(cls.IMAGE_FIELDS) | set(cls.VIDEO_FIELDS) | set(cls.AUDIO_FIELDS) | {
            "youchuan/v8.2/remove-background"
        }
        if resolved not in supported:
            raise ProviderError(
                f"Atlas Cloud model '{model}' is not mapped by this pipeline; "
                "choose a documented Atlas model in beats.json"
            )
        return resolved

    @staticmethod
    def _select(params, fields):
        return {key: value for key, value in params.items() if key in fields and value is not None}

    def submit_image(self, model, prompt, **params):
        model = self._model(model)
        if model not in self.IMAGE_FIELDS:
            raise ProviderError(f"Atlas Cloud model '{model}' is not an image-generation model")
        params = dict(params)
        if "image_url" in params:
            params["image"] = params.pop("image_url")
        payload = self._select({"prompt": prompt, **params}, self.IMAGE_FIELDS[model])
        return atlas_client.submit("generateImage", {"model": model, **payload})

    def submit_video(self, model, prompt, **params):
        model = self._model(model)
        if model not in self.VIDEO_FIELDS:
            raise ProviderError(f"Atlas Cloud model '{model}' is not a video-generation model")
        params = dict(params)
        if "image_url" in params:
            params["image"] = params.pop("image_url")
        payload = self._select({"prompt": prompt, **params}, self.VIDEO_FIELDS[model])
        return atlas_client.submit("generateVideo", {"model": model, **payload})

    def submit_audio(self, model, **params):
        model = self._model(model)
        if model not in self.AUDIO_FIELDS:
            raise ProviderError(f"Atlas Cloud model '{model}' is not an audio model")
        params = dict(params)
        if params.get("references"):
            raise ProviderError(f"Atlas Cloud model '{model}' does not support voice references")
        if "is_instrumental" in params:
            params["instrumental"] = params.pop("is_instrumental")
            params.setdefault("custom", True)
        language_map = {"en": "en-US", "zh": "zh-CN", "ja": "ja-JP", "es": "es-MX"}
        if model == "bytedance/seed-asr-2.0" and params.get("language") in language_map:
            params["language"] = language_map[params["language"]]
        payload = self._select(params, self.AUDIO_FIELDS[model])
        return atlas_client.submit("generateAudio", {"model": model, **payload})

    def remove_bg(self, model, image_url, **params):
        model = self._model(model or "remove-background")
        if model != "youchuan/v8.2/remove-background":
            raise ProviderError(f"Atlas Cloud model '{model}' is not configured for background removal")
        payload = {"model": model, "image": image_url}
        if "enable_base64_output" in params:
            payload["enable_base64_output"] = params["enable_base64_output"]
        return atlas_client.submit("generateImage", payload)

    def get_status(self, job_id):
        return atlas_client.prediction_status(job_id)

    def upload(self, path):
        return atlas_client.upload(path)

    def download(self, url, dest):
        return atlas_client.download(url, dest)


_REGISTRY = {
    "muapi": MuapiProvider,
    "atlas_cloud": AtlasCloudProvider,
}


def get_provider(name=None):
    """Return a Provider instance by name (default 'muapi')."""
    name = (name or "muapi").lower()
    if name not in _REGISTRY:
        raise ProviderError(f"Unknown provider '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def run_jobs(prov, specs, *, poll_s=3, stall_s=90, max_retries=2, deadline_s=900):
    """Submit and poll jobs, honoring the provider's resubmission policy."""
    state = {}
    for key, submit_fn in specs.items():
        try:
            jid = submit_fn()
            state[key] = {
                "submit_fn": submit_fn,
                "id": jid,
                "tries": 1,
                "last_change": time.monotonic(),
                "out": None,
                "error": None,
                "done": False,
            }
            print(f"  [{key}] submitted -> {jid}")
        except Exception as e:
            print(f"  [{key}] submit failed: {e}")
            state[key] = {
                "submit_fn": submit_fn,
                "id": None,
                "tries": 1,
                "last_change": time.monotonic(),
                "out": None,
                "error": str(e),
                "done": True,
            }

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        all_done = True
        for key, s in state.items():
            if s["done"]:
                continue
            all_done = False
            res = prov.get_status(s["id"])
            st = res["status"]
            if st == "completed":
                s["out"] = res["output"]
                s["done"] = True
                print(f"  [{key}] completed -> {s['out']}")
            elif st == "failed" or (time.monotonic() - s["last_change"] > stall_s):
                reason = res["error"] if st == "failed" else f"stalled >{stall_s}s"
                if prov.allows_resubmission and s["tries"] <= max_retries:
                    print(f"  [{key}] {reason}; retry {s['tries']}/{max_retries}...")
                    try:
                        new_id = s["submit_fn"]()
                        s["id"] = new_id
                        s["tries"] += 1
                        s["last_change"] = time.monotonic()
                        print(f"  [{key}] re-submitted -> {new_id}")
                    except Exception as e:
                        s["error"] = str(e)
                        s["done"] = True
                        print(f"  [{key}] re-submit failed: {e}")
                else:
                    s["error"] = reason
                    s["done"] = True
                    print(f"  [{key}] failed permanently: {reason}")

        if all_done:
            break
        time.sleep(poll_s)

    results = {}
    for key, s in state.items():
        results[key] = s["out"]
        if not s["out"]:
            print(f"  WARNING: [{key}] produced no output (error: {s['error']})")
    return results
