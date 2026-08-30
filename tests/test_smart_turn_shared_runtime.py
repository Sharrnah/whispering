import unittest
from unittest import mock

import numpy as np

from Models.STS import SmartTurn as smart_turn_module


class SmartTurnSharedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = smart_turn_module.SmartTurn
        self.previous = (
            self.runtime.feature_extractor,
            self.runtime.session,
            self.runtime.providers,
        )
        self.runtime.feature_extractor = None
        self.runtime.session = None
        self.runtime.providers = None

    def tearDown(self):
        (
            self.runtime.feature_extractor,
            self.runtime.session,
            self.runtime.providers,
        ) = self.previous

    def test_instances_share_model_runtime_but_keep_separate_audio(self):
        fake_extractor = object()
        fake_session = object()
        with mock.patch.object(
            smart_turn_module.downloader, "download_model"
        ) as download_model, mock.patch.object(
            smart_turn_module, "WhisperFeatureExtractor", return_value=fake_extractor
        ) as extractor_type, mock.patch.object(
            smart_turn_module.ort,
            "get_available_providers",
            return_value=["CPUExecutionProvider"],
        ), mock.patch.object(
            self.runtime, "build_session", return_value=fake_session
        ) as build_session:
            first = self.runtime()
            second = self.runtime()

        download_model.assert_called_once()
        extractor_type.assert_called_once_with(chunk_length=8)
        build_session.assert_called_once()
        self.assertIs(self.runtime.feature_extractor, fake_extractor)
        self.assertIs(self.runtime.session, fake_session)

        first.add_audio(np.ones(160, dtype=np.float32))
        self.assertEqual(len(first.audio_array), 160)
        self.assertEqual(len(second.audio_array), 0)


if __name__ == "__main__":
    unittest.main()
