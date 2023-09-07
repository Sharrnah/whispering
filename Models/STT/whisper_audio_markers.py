import os
import re

import numpy as np

import audio_tools


class WhisperVoiceMarker:
    audio_sample = None
    audio_model = None
    try_count = 0
    last_result = ""
    verbose = False

    def __init__(self, audio_sample, audio_model):
        self.audio_sample = audio_sample
        self.audio_model = audio_model
        self.try_count = 0
        self.last_result = ""

    def get_voice_marker_prompt(self, lng: str, task: str):
        voice_marker_prompt = ""
        if self.try_count == -1:
            return voice_marker_prompt

        if lng == "en" or task == "translate":
            # use english prompt for translation task, so we don't lose translation capabilities.
            voice_marker_prompt = "Whisper, Ok. " \
                                  + "A pertinent sentence for your purpose in your language. " \
                                  + "Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. " \
                                  + "Please find here, an unlikely ordinary sentence. " \
                                  + "This is to avoid a repetition to be deleted. " \
                                  + "Ok, Whisper. "
            return voice_marker_prompt

        if lng == "zh":
            voice_marker_prompt = "Whisper, Ok. " \
                                  + "用你的语言为你的目的写一个相关的句子。 " \
                                  + "Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. " \
                                  + "请看这里，一个不太可能的普通句子。 " \
                                  + "这是为了避免重复的内容被删除。 " \
                                  + "Ok, Whisper. "

        if lng == "fr":
            voice_marker_prompt = "Whisper, Ok. " \
                                  + "Une phrase pertinente pour votre propos dans votre langue. " \
                                  + "Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. " \
                                  + "Merci de trouver ci-joint, une phrase ordinaire improbable. " \
                                  + "Pour éviter une répétition à être supprimée. " \
                                  + "Ok, Whisper. "

        if lng == "uk":
            voice_marker_prompt = "Whisper, Ok. " \
                                  + "Доречне речення вашою мовою для вашої мети. " \
                                  + "Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. " \
                                  + "Будь ласка, знайдіть тут навряд чи звичайне речення. " \
                                  + "Це зроблено для того, щоб уникнути повторення, яке потрібно видалити. " \
                                  + "Ok, Whisper. "

        if lng == "hi":
            voice_marker_prompt = "विस्पर, ओके. " \
                                  + "आपकी भाषा में आपके उद्देश्य के लिए एक प्रासंगिक वाक्य। " \
                                  + "ओके, विस्पर. विस्पर, ओके. ओके, विस्पर. विस्पर, ओके. " \
                                  + "कृपया यहां खोजें, एक असंभावित सामान्य वाक्य। " \
                                  + "यह हटाए जाने की पुनरावृत्ति से बचने के लिए है। " \
                                  + "ओके, विस्पर. "

        if lng == "de":
            voice_marker_prompt = "Whisper, Okay. " \
                                  + "Ein passender Satz für Ihren Zweck in Ihrer Sprache. " \
                                  + "Okay, Whisper. Whisper, Okay. Okay, Whisper. Whisper, Okay. " \
                                  + "Finden Sie hier einen ungewöhnlich gewöhnlichen Satz. " \
                                  + "Dadurch soll verhindert werden, dass eine Wiederholung gelöscht wird. " \
                                  + "Okay, Whisper. "

        if lng == "ja":
            voice_marker_prompt = "Whisper, Ok. " \
                                  + "あなたの言語であなたの目的に適した文章。 " \
                                  + "Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. " \
                                  + "ここで、ありそうでない普通の文を見つけてください。 " \
                                  + "これは、重複して削除されないようにするためです。 " \
                                  + "Ok, Whisper. "

        return voice_marker_prompt

    def apply_voice_markers(self, lng: str):
        if self.try_count == -1:
            return self.audio_sample

        lngInput = lng

        if lng == "":
            lngInput = lng

        if lngInput is not None and os.path.exists("markers/WOK-MRK-" + lngInput + ".wav"):
            mark1 = "markers/WOK-MRK-" + lngInput + ".wav"
        else:
            mark1 = "markers/WOK-MRK.wav"
        if lngInput is not None and os.path.exists("markers/OKW-MRK-" + lngInput + ".wav"):
            mark2 = "markers/OKW-MRK-" + lngInput + ".wav"
        else:
            mark2 = "markers/OKW-MRK.wav"

        # switch markers if try_count is 1
        if self.try_count == 1:
            mark = mark1
            mark1 = mark2
            mark2 = mark

        if self.verbose:
            print("Using markers: start: " + mark1 + " and end: " + mark2)

        marker1_audio = audio_tools.load_wav_to_bytes(mark1)

        # convert audio to 16 bit numpy array
        marker1_audio = np.frombuffer(marker1_audio, np.int16).flatten().astype(np.float32) / 32768.0

        marker2_audio = audio_tools.load_wav_to_bytes(mark2)

        # convert audio to 16 bit numpy array
        marker2_audio = np.frombuffer(marker2_audio, np.int16).flatten().astype(np.float32) / 32768.0

        # prepend and append markers
        audio_sample = np.concatenate([marker1_audio, self.audio_sample, marker2_audio])

        mark = None
        del mark
        marker1_audio = None
        del marker1_audio
        marker2_audio = None
        del marker2_audio

        return audio_sample

    def transcribe(self, **kwargs):
        result = None

        whisper_task = kwargs["task"]
        whisper_language = kwargs["language"]
        whisper_condition_on_previous_text = kwargs["condition_on_previous_text"]
        # whisper_initial_prompt = kwargs["initial_prompt"]
        whisper_logprob_threshold = kwargs["logprob_threshold"]
        whisper_no_speech_threshold = kwargs["no_speech_threshold"]
        whisper_temperature_fallback_option = kwargs["temperature"]
        whisper_beam_size = kwargs["beam_size"]
        whisper_word_timestamps = kwargs["word_timestamps"]
        whisper_faster_without_timestamps = kwargs["without_timestamps"]
        whisper_faster_beam_search_patience = kwargs["patience"]
        whisper_faster_length_penalty = kwargs["length_penalty"]
        prompt_reset_on_temperature = kwargs["prompt_reset_on_temperature"]
        repetition_penalty = kwargs["repetition_penalty"]
        no_repeat_ngram_size = kwargs["no_repeat_ngram_size"]

        whisper_initial_prompt = self.get_voice_marker_prompt(whisper_language, whisper_task)
        if kwargs["initial_prompt"] is not None and kwargs["initial_prompt"] != "":
            whisper_initial_prompt += " " + kwargs["initial_prompt"]

        audio_sample = self.apply_voice_markers(whisper_language)

        result = self.audio_model.transcribe(audio_sample, task=whisper_task,
                                             language=whisper_language,
                                             condition_on_previous_text=whisper_condition_on_previous_text,
                                             prompt_reset_on_temperature=prompt_reset_on_temperature,
                                             initial_prompt=whisper_initial_prompt,
                                             logprob_threshold=whisper_logprob_threshold,
                                             no_speech_threshold=whisper_no_speech_threshold,
                                             temperature=whisper_temperature_fallback_option,
                                             beam_size=whisper_beam_size,
                                             word_timestamps=whisper_word_timestamps,
                                             without_timestamps=whisper_faster_without_timestamps,
                                             patience=whisper_faster_beam_search_patience,
                                             length_penalty=whisper_faster_length_penalty,
                                             repetition_penalty=repetition_penalty,
                                             no_repeat_ngram_size=no_repeat_ngram_size)
        if self.verbose:
            print("Result: " + str(result))

        return result

    def voice_marker_transcribe(self, **kwargs):
        result = {}
        result["text"] = ""

        result = self.transcribe(**kwargs)
        aWhisper = "(Whisper|Wisper|Wyspę|Wysper|Wispa|Уіспер|Ου ίσπερ|위스퍼드|ウィスパー|विस्पर|विसपर)"
        aOk = "(okay|oké|okej|Окей|οκέι|오케이|オーケー|ओके|o[.]?k[.]?)"
        aSep = "[.,!?。， ]*"

        if self.try_count == -1:
            if self.verbose:
                print("try_count == -1")
            return result

        if self.try_count == 0:
            aCleaned = re.sub(r"(^ *" + aWhisper + aSep + aOk + aSep + "|" + aOk + aSep + aWhisper + aSep + " *$)", "",
                              result["text"], 2, re.IGNORECASE)
            if re.match(
                    r"^ *(" + aOk + "|" + aSep + "|" + aWhisper + ")*" + aWhisper + "(" + aOk + "|" + aSep + "|" + aWhisper + ")* *$",
                    result["text"], re.IGNORECASE):
                # Empty sound ?
                self.try_count = 1
                self.last_result = ""
                if self.verbose:
                    print("Empty sound ? 1")
                return self.voice_marker_transcribe(**kwargs)

            if re.match(r"^ *" + aWhisper + aSep + aOk + aSep + ".*" + aOk + aSep + aWhisper + aSep + " *$",
                        result["text"], re.IGNORECASE):
                # GOOD!
                result["text"] = aCleaned
                if self.verbose:
                    print("GOOD! 1")
                return result

            self.try_count = 1
            self.last_result = aCleaned
            return self.voice_marker_transcribe(**kwargs)

        if self.try_count == 1:
            aCleaned = re.sub(r"(^ *" + aOk + aSep + aWhisper + aSep + "|" + aWhisper + aSep + aOk + aSep + " *$)", "",
                              result["text"], 2, re.IGNORECASE)
            if aCleaned == self.last_result:
                # CONFIRMED!
                result["text"] = aCleaned
                if self.verbose:
                    print("CONFIRMED!")
                return result

            if re.match(
                    r"^ *(" + aOk + "|" + aSep + "|" + aWhisper + ")*" + aWhisper + "(" + aOk + "|" + aSep + "|" + aWhisper + ")* *$",
                    result["text"], re.IGNORECASE):
                # Empty sound ?
                result["text"] = ""
                if self.verbose:
                    print("Empty sound ? 2")
                return result

            if re.match(r"^ *" + aOk + aSep + aWhisper + aSep + ".*" + aWhisper + aSep + aOk + aSep + " *$",
                        result["text"], re.IGNORECASE):
                # GOOD!
                result["text"] = aCleaned
                if self.verbose:
                    print("GOOD! 2")
                return result

            # retry
            self.try_count = -1
            self.last_result = aCleaned
            if self.verbose:
                print("retry count -1")
            return self.voice_marker_transcribe(**kwargs)
