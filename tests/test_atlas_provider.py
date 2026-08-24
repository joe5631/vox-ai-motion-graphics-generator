import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import atlas_client
from provider import AtlasCloudProvider, run_jobs


class AtlasCloudProviderTests(unittest.TestCase):
    @mock.patch.object(atlas_client.subprocess, "run")
    def test_upload_keeps_api_key_out_of_process_arguments(self, run):
        run.return_value.stdout = (
            '{"code":200,"data":{"download_url":"https://uploads.example/frame.jpg"}}'
        )

        with mock.patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "secret-test-key"}):
            result = atlas_client.upload("frame.jpg")

        self.assertEqual(result, "https://uploads.example/frame.jpg")
        command = run.call_args.args[0]
        self.assertNotIn("secret-test-key", " ".join(command))
        self.assertEqual(run.call_args.kwargs["input"], "Authorization: Bearer secret-test-key\n")

    @mock.patch.object(atlas_client.time, "sleep")
    @mock.patch.object(
        atlas_client,
        "_request",
        side_effect=[atlas_client.AtlasCloudError("temporary"), {"status": "processing"}],
    )
    def test_prediction_get_retries_transient_failure(self, request, sleep):
        with mock.patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}):
            result = atlas_client.get_prediction("pred-123")

        self.assertEqual(result, {"status": "processing"})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1)

    @mock.patch.object(atlas_client, "submit", return_value="img-123")
    def test_image_alias_builds_atlas_payload(self, submit):
        provider = AtlasCloudProvider()

        result = provider.submit_image(
            "nano-banana-2",
            "paper collage",
            aspect_ratio="16:9",
            resolution="1k",
            unsupported="ignored",
        )

        self.assertEqual(result, "img-123")
        submit.assert_called_once_with(
            "generateImage",
            {
                "model": "google/nano-banana-2-lite/text-to-image",
                "prompt": "paper collage",
                "aspect_ratio": "16:9",
                "resolution": "1k",
            },
        )

    @mock.patch.object(atlas_client, "submit", return_value="vid-123")
    def test_video_maps_image_url_to_schema_field(self, submit):
        provider = AtlasCloudProvider()

        provider.submit_video(
            "veo3.1-image-to-video",
            "slow push in",
            image_url="https://example.test/frame.jpg",
            duration=8,
            aspect_ratio="16:9",
        )

        submit.assert_called_once_with(
            "generateVideo",
            {
                "model": "google/veo3.1/image-to-video",
                "prompt": "slow push in",
                "image": "https://example.test/frame.jpg",
                "duration": 8,
                "aspect_ratio": "16:9",
            },
        )

    @mock.patch.object(atlas_client, "submit", return_value="audio-123")
    def test_music_maps_instrumental_flag(self, submit):
        provider = AtlasCloudProvider()

        provider.submit_audio(
            "suno-create-music",
            prompt="cinematic collage score",
            style="cinematic collage score",
            is_instrumental=True,
            format="mp3",
        )

        submit.assert_called_once_with(
            "generateAudio",
            {
                "model": "suno/chirp-v5",
                "prompt": "cinematic collage score",
                "style": "cinematic collage score",
                "instrumental": True,
                "custom": True,
            },
        )

    def test_failed_atlas_job_is_not_resubmitted(self):
        provider = AtlasCloudProvider()
        provider.get_status = mock.Mock(
            return_value={"status": "failed", "output": None, "error": "rejected"}
        )
        submit = mock.Mock(return_value="job-1")

        result = run_jobs(provider, {"shot": submit}, poll_s=0, max_retries=2, deadline_s=1)

        self.assertEqual(result, {"shot": None})
        submit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
