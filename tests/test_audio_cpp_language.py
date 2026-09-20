import unittest
from unittest import mock

from Models.TTS.speech_language import language_code, spoken_language
from Models.TTS.audio_cpp import AudioCppTTS
from Models.TTS import audio_cpp


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(AudioCppTTS)

    def test_language_names_iso3_nllb_and_locales(self):
        for value, expected in (("German", "de"), ("deu_Latn", "de"),
                                ("eng", "en"), ("de-DE", "de-de"),
                                (None, ""), ("Auto", ""), ("und", "")):
            with self.subTest(value=value):
                self.assertEqual(language_code(value), expected)

    def test_known_stt_language_bypasses_unreliable_text_detection(self):
        with mock.patch.object(self.adapter, "_detect_language") as detection:
            self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {}, "de-DE"), "de")
            self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {}, "zh"), "en")
            detection.assert_not_called()

    def test_explicit_tts_language_wins_and_is_validated(self):
        self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {"language": "fr"}, "de"), "fr")
        self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {"language": "zh"}, "de"), "en")
        self.assertEqual(self.adapter._language("Hi", "IndexTTS2-GGUF", {}, "ja"), "en")
        self.assertEqual(self.adapter._language("Hi", "IndexTTS2.5-GGUF", {}, "ja"), "ja")

    def test_detection_failure_or_unsupported_result_uses_english(self):
        from Models import languageClassification
        for classification in ("zh", "und", ("zho_Hans", 0.9), ""):
            with mock.patch.object(languageClassification, "classify", return_value=classification):
                self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {}), "en")
        with mock.patch.object(languageClassification, "classify", side_effect=ValueError("failed")):
            self.assertEqual(self.adapter._language("Hi", "Supertonic-3-GGUF", {}), "en")

    def test_locale_aliases_and_fixed_language_voices(self):
        for model, hint, expected in (("MagpieTTS-Multilingual-357M-GGUF", "pt", "pt-br"),
                                      ("MagpieTTS-Multilingual-357M-GGUF", "ar", "ar-msa"),
                                      ("Kokoro-82M-GGUF", "de", "en-us"),
                                      ("sanoTTS-de-GGUF", "en", "de")):
            self.assertEqual(self.adapter._language("Hi", model, {}, hint), expected)

    def test_kokoro_selects_a_voice_matching_the_utterance_language(self):
        self.adapter.server = mock.Mock(model_id="test")
        values = {"tts_model": ["Preset voices", "Kokoro-82M-GGUF"], "tts_voice": "af_heart"}
        with mock.patch.object(audio_cpp.settings, "GetOption", side_effect=values.get):
            payload, *references = self.adapter._request_payload("Bonjour", language="fr")
        self.assertEqual(payload["language"], "fr-fr")
        self.assertEqual(payload["voice"], "ff_siwis")
        self.assertEqual(references, [None, None, None])

    def test_outgoing_translation_language_and_fixed_stt_setting(self):
        values = {"current_language": "de", "whisper_task": "transcribe"}
        settings = mock.Mock(GetOption=values.get)
        self.assertEqual(spoken_language({"language": "auto"}, settings), "de")
        self.assertEqual(spoken_language({"language": "fr"}, settings), "fr")
        self.assertEqual(spoken_language({"language": "de", "txt_translation": "Hello",
                                         "txt_translation_target": "eng_Latn|ja"}, settings), "en")
        values["whisper_task"] = "translate"
        self.assertEqual(spoken_language({"language": "de"}, settings), "en")

    def test_streaming_fallback_keeps_language(self):
        import threading
        import torch
        self.adapter.stop_event = threading.Event()
        with mock.patch.object(self.adapter, "_selected_model", return_value="IndexTTS2-GGUF"), \
                mock.patch.object(self.adapter, "tts", return_value=(torch.zeros((1, 0)), 22050)) as synthesize:
            self.adapter.tts_streaming("Hello", language="en")
        synthesize.assert_called_once_with("Hello", None, language="en")


if __name__ == "__main__":
    unittest.main()
