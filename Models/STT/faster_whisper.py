from faster_whisper import WhisperModel


class FasterWhisper:
    model = None

    def __init__(self, model_path: str, device: str = "cpu", compute_type: str = "float32"):
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def transcribe(self, audio_sample, task, language, condition_on_previous_text,
                   initial_prompt, logprob_threshold, no_speech_threshold,
                   temperature) -> dict:

        result_segments, audio_info = self.model.transcribe(audio_sample, task=task,
                                                            language=language,
                                                            condition_on_previous_text=condition_on_previous_text,
                                                            initial_prompt=initial_prompt,
                                                            log_prob_threshold=logprob_threshold,
                                                            no_speech_threshold=no_speech_threshold,
                                                            temperature=temperature,
                                                            without_timestamps=True
                                                            )

        result = {
            'text': " ".join([segment.text for segment in result_segments]),
            'type': task,
            'language': audio_info.language
        }

        return result
