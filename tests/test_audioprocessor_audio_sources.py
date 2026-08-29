import unittest
from unittest import mock

import audioprocessor


class _Settings:
    values = {
        "verbose": False,
        "osc_ip": "0",
        "txt_translate": False,
        "transcription_auto_save_file": "",
        "transcription_auto_save_continuous_text": False,
        "max_sentence_repetition": -1,
        "websocket_ip": "0",
        "realtime": False,
    }

    def GetOption(self, name):
        return self.values.get(name)


class AudioSourceResultRoutingTests(unittest.TestCase):
    def setUp(self):
        with audioprocessor.last_audio_timestamps_lock:
            audioprocessor.last_audio_timestamps.clear()

    def test_intermediate_timestamp_ordering_is_source_local(self):
        settings = _Settings()
        with mock.patch.object(audioprocessor, "send_message") as send:
            audioprocessor.whisper_result_handling(
                {
                    "text": "microphone",
                    "language": "en",
                    "audio_source_id": "main",
                    "audio_source_name": "Microphone",
                },
                200,
                False,
                settings,
                [],
            )
            audioprocessor.whisper_result_handling(
                {
                    "text": "game",
                    "language": "en",
                    "audio_source_id": "game",
                    "audio_source_name": "Game audio",
                },
                100,
                False,
                settings,
                [],
            )

        self.assertEqual(send.call_count, 2)

    def test_older_intermediate_from_the_same_source_is_ignored(self):
        settings = _Settings()
        with mock.patch.object(audioprocessor, "send_message") as send:
            for timestamp, text in ((200, "new"), (100, "old")):
                audioprocessor.whisper_result_handling(
                    {
                        "text": text,
                        "language": "en",
                        "audio_source_id": "game",
                        "audio_source_name": "Game audio",
                    },
                    timestamp,
                    False,
                    settings,
                    [],
                )

        send.assert_called_once()

    def test_result_thread_attaches_source_identity(self):
        result = {"text": "hello", "language": "en"}
        with mock.patch.object(audioprocessor, "whisper_result_handling") as handling:
            audioprocessor.whisper_result_thread(
                result, 1, True, _Settings(), [], "game", "Game audio"
            )

        self.assertEqual(result["audio_source_id"], "game")
        self.assertEqual(result["audio_source_name"], "Game audio")
        handling.assert_called_once()


class WhisperLanguageNormalizationTests(unittest.TestCase):
    def test_auto_language_is_passed_as_none(self):
        for value in (None, "", "auto", "Auto", "null"):
            with self.subTest(value=value):
                self.assertIsNone(audioprocessor.normalize_whisper_language(value))

    def test_cross_backend_iso3_languages_are_converted_for_whisper(self):
        expected = {
            "deu": "de",
            "deu_Latn": "de",
            "German": "de",
            "de-DE": "de",
            "eng": "en",
            "jpn": "ja",
        }
        for value, language_code in expected.items():
            with self.subTest(value=value):
                self.assertEqual(
                    audioprocessor.normalize_whisper_language(value),
                    language_code,
                )

    def test_transformer_whisper_receives_the_normalized_language(self):
        values = {
            "stt_type": "transformer_whisper",
            "whisper_task": "transcribe",
            "current_language": "deu",
            "target_language": "eng",
            "condition_on_previous_text": False,
            "logprob_threshold": None,
            "no_speech_threshold": None,
            "beam_size": 1,
            "realtime_whisper_beam_size": 1,
            "word_timestamps": False,
            "faster_without_timestamps": False,
            "length_penalty": 1.0,
            "beam_search_patience": 1.0,
            "temperature_fallback": False,
            "realtime_temperature_fallback": False,
            "initial_prompt": "",
            "realtime": False,
            "prompt_reset_on_temperature": False,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "whisper_precision": "float16",
            "ai_device": "cuda",
            "model": "large-v3",
        }

        class _InferenceSettings:
            def GetOption(self, name):
                return values.get(name)

        model = mock.Mock()
        model.transcribe.return_value = {"text": "hallo", "language": "de"}
        with mock.patch.object(audioprocessor, "convert_audio", return_value=[]), \
                mock.patch.object(audioprocessor, "whisper_result_thread"):
            audioprocessor.whisper_ai_thread(
                b"audio", 1, model, None, "", True,
                _InferenceSettings(), [], "game", "Game audio",
            )

        self.assertEqual(model.transcribe.call_args.kwargs["language"], "de")


if __name__ == "__main__":
    unittest.main()
