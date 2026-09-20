"""Additional speech models verified against audio.cpp v0.8.1.

Source: 0xShug0/audio.cpp@f2b4937306daa25f5c78520f3c626ed31495a37a.
Download URLs pin immutable model revisions; SHA-256 values are the published
LFS hashes (small sidecars were independently hashed). No runtime catalogue fetch.
"""

STT_MODELS = {'Canary-180M-Flash-GGUF': {'family': 'canary_asr',
                            'description': 'NVIDIA Canary multilingual speech recognition and speech '
                                           'translation.',
                            'settings_key': 'canary_asr',
                            'streaming': False,
                            'default_precision': 'f32',
                            'variants': {'f32': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Canary-180M-Flash-GGUF/canary-180m-flash-f32.gguf'],
                                                 'checksum': '03ca359d7db1de31e71a344307aa6c649a8c071717503cae818e4f3e3ce21dde',
                                                 'filename': 'canary-180m-flash-f32.gguf',
                                                 'size': 757871168,
                                                 'file_checksums': {'canary-180m-flash-f32.gguf': '03ca359d7db1de31e71a344307aa6c649a8c071717503cae818e4f3e3ce21dde'}},
                                         'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Canary-180M-Flash-GGUF/canary-180m-flash-q8_0.gguf'],
                                                  'checksum': '8a6d06be2b6c8e44b7810e5ad1277db571d3152a7eea8b699974516907a6ef57',
                                                  'filename': 'canary-180m-flash-q8_0.gguf',
                                                  'size': 249496128,
                                                  'file_checksums': {'canary-180m-flash-q8_0.gguf': '8a6d06be2b6c8e44b7810e5ad1277db571d3152a7eea8b699974516907a6ef57'}}},
                            'timestamps': False,
                            'languages': ['en', 'de', 'es', 'fr']},
 'Cohere-Transcribe-GGUF': {'family': 'cohere_asr',
                            'description': 'Cohere multilingual speech transcription with punctuation '
                                           'control.',
                            'settings_key': 'cohere_asr',
                            'streaming': False,
                            'default_precision': 'bf16',
                            'variants': {'bf16': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-bf16.gguf'],
                                                  'checksum': '088f97ab2aa1dbd40230c2eaf9dc7956e574a1266206938fbbaf4ecab55aff04',
                                                  'filename': 'cohere-transcribe-03-2026-bf16.gguf',
                                                  'size': 4134852224,
                                                  'file_checksums': {'cohere-transcribe-03-2026-bf16.gguf': '088f97ab2aa1dbd40230c2eaf9dc7956e574a1266206938fbbaf4ecab55aff04'}},
                                         'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-q8_0.gguf'],
                                                  'checksum': 'f3da2899fe5d918b3aefb0d3c0d4c963ca7171c42cb98439000d1228b224d51c',
                                                  'filename': 'cohere-transcribe-03-2026-q8_0.gguf',
                                                  'size': 2438863008,
                                                  'file_checksums': {'cohere-transcribe-03-2026-q8_0.gguf': 'f3da2899fe5d918b3aefb0d3c0d4c963ca7171c42cb98439000d1228b224d51c'}},
                                         'q4_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-q4_0.gguf'],
                                                  'checksum': 'a39d6237bf20030c1422d2046dee0d2c7ebf6de729aab9ddd69672bf9afce29e',
                                                  'filename': 'cohere-transcribe-03-2026-q4_0.gguf',
                                                  'size': 1534335392,
                                                  'file_checksums': {'cohere-transcribe-03-2026-q4_0.gguf': 'a39d6237bf20030c1422d2046dee0d2c7ebf6de729aab9ddd69672bf9afce29e'}}},
                            'timestamps': False,
                            'languages': ['en',
                                          'fr',
                                          'de',
                                          'es',
                                          'it',
                                          'pt',
                                          'nl',
                                          'pl',
                                          'el',
                                          'ar',
                                          'ja',
                                          'zh',
                                          'vi',
                                          'ko']},
 'Moonshine-Streaming-Tiny-GGUF': {'family': 'moonshine_asr',
                                   'description': 'Moonshine tiny/small/medium streaming English speech '
                                                  'recognition models.',
                                   'settings_key': 'moonshine_asr',
                                   'streaming': True,
                                   'default_precision': 'q8_0',
                                   'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf'],
                                                         'checksum': 'e9a342a07327f4e1745874f137f45350697e91699a91e4eb2ac60c223718f8c3',
                                                         'filename': 'moonshine-streaming-tiny-q8_0.gguf',
                                                         'size': 60407904,
                                                         'file_checksums': {'moonshine-streaming-tiny-q8_0.gguf': 'e9a342a07327f4e1745874f137f45350697e91699a91e4eb2ac60c223718f8c3'}}},
                                   'timestamps': False,
                                   'languages': ['en']},
 'Moonshine-Streaming-Small-GGUF': {'family': 'moonshine_asr',
                                    'description': 'Moonshine tiny/small/medium streaming English speech '
                                                   'recognition models.',
                                    'settings_key': 'moonshine_asr',
                                    'streaming': True,
                                    'default_precision': 'q8_0',
                                    'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0.gguf'],
                                                          'checksum': '61036888fbc8fc685ef49ff075f1eaa7d1a68bf03dbc3ef80bc12189022c44e3',
                                                          'filename': 'moonshine-streaming-small-q8_0.gguf',
                                                          'size': 300621248,
                                                          'file_checksums': {'moonshine-streaming-small-q8_0.gguf': '61036888fbc8fc685ef49ff075f1eaa7d1a68bf03dbc3ef80bc12189022c44e3'}}},
                                    'timestamps': False,
                                    'languages': ['en']},
 'Moonshine-Streaming-Medium-GGUF': {'family': 'moonshine_asr',
                                     'description': 'Moonshine tiny/small/medium streaming English speech '
                                                    'recognition models.',
                                     'settings_key': 'moonshine_asr',
                                     'streaming': True,
                                     'default_precision': 'q8_0',
                                     'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Moonshine-Streaming-GGUF/moonshine-streaming-medium-q8_0.gguf'],
                                                           'checksum': 'cc242a59dd7aa3cf9f688a68b52f7e22e93ab19fa1a71a38927d316ad1b349dd',
                                                           'filename': 'moonshine-streaming-medium-q8_0.gguf',
                                                           'size': 315583648,
                                                           'file_checksums': {'moonshine-streaming-medium-q8_0.gguf': 'cc242a59dd7aa3cf9f688a68b52f7e22e93ab19fa1a71a38927d316ad1b349dd'}}},
                                     'timestamps': False,
                                     'languages': ['en']},
 'Niagara-19M-Batch-English-GGUF': {'family': 'niagara_asr',
                                    'description': 'ABR Niagara English batch ASR state-space models with '
                                                   'attention, greedy CTC decoding, and SentencePiece '
                                                   'tokenization.',
                                    'settings_key': 'niagara_asr',
                                    'streaming': False,
                                    'default_precision': 'f32',
                                    'variants': {'f32': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Niagara-ASR-GGUF/niagara-19m-batch.en-f32.gguf'],
                                                         'checksum': '5c4b0b66e1f9bbe8270d378671d4cc71b1c4ce4b7d045cc787607b5b10f2e1a7',
                                                         'filename': 'niagara-19m-batch.en-f32.gguf',
                                                         'size': 255372352,
                                                         'file_checksums': {'niagara-19m-batch.en-f32.gguf': '5c4b0b66e1f9bbe8270d378671d4cc71b1c4ce4b7d045cc787607b5b10f2e1a7'}}},
                                    'timestamps': False,
                                    'languages': ['en']},
 'Niagara-38M-Batch-English-GGUF': {'family': 'niagara_asr',
                                    'description': 'ABR Niagara English batch ASR state-space models with '
                                                   'attention, greedy CTC decoding, and SentencePiece '
                                                   'tokenization.',
                                    'settings_key': 'niagara_asr',
                                    'streaming': False,
                                    'default_precision': 'f32',
                                    'variants': {'f32': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Niagara-ASR-GGUF/niagara-38m-batch.en-f32.gguf'],
                                                         'checksum': '0ffe882781d26574d735350270f6ed7f42ecc7de051aaa4bcb15c53b20a5a1c0',
                                                         'filename': 'niagara-38m-batch.en-f32.gguf',
                                                         'size': 427067840,
                                                         'file_checksums': {'niagara-38m-batch.en-f32.gguf': '0ffe882781d26574d735350270f6ed7f42ecc7de051aaa4bcb15c53b20a5a1c0'}}},
                                    'timestamps': False,
                                    'languages': ['en']},
 'MOSS-Transcribe-Diarize-GGUF': {'family': 'moss_transcribe_diarize',
                                  'description': 'Joint transcription, speaker diarization, and timestamps '
                                                 'across 50+ languages with a Whisper encoder and Qwen3 '
                                                 'decoder.',
                                  'settings_key': 'moss_transcribe_diarize',
                                  'streaming': True,
                                  'default_precision': 'bf16',
                                  'variants': {'bf16': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-bf16.gguf'],
                                                        'checksum': '5bc627289e2586fc2d9269afda15a305e545a4ff3512c902889be71d58e56ad5',
                                                        'filename': 'moss-transcribe-diarize-bf16.gguf',
                                                        'size': 1833018080,
                                                        'file_checksums': {'moss-transcribe-diarize-bf16.gguf': '5bc627289e2586fc2d9269afda15a305e545a4ff3512c902889be71d58e56ad5'}},
                                               'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-q8_0.gguf'],
                                                        'checksum': '93eea5865615e270b827752945f2dfd0f522ed673f4f551675e9013170173679',
                                                        'filename': 'moss-transcribe-diarize-q8_0.gguf',
                                                        'size': 1132110560,
                                                        'file_checksums': {'moss-transcribe-diarize-q8_0.gguf': '93eea5865615e270b827752945f2dfd0f522ed673f4f551675e9013170173679'}},
                                               'q4_k': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-q4_k.gguf'],
                                                        'checksum': '4e3020379ff592bbb22a4e3d47a7cd6709951cc5d451c561b69df955b5f0a220',
                                                        'filename': 'moss-transcribe-diarize-q4_k.gguf',
                                                        'size': 758293216,
                                                        'file_checksums': {'moss-transcribe-diarize-q4_k.gguf': '4e3020379ff592bbb22a4e3d47a7cd6709951cc5d451c561b69df955b5f0a220'}}},
                                  'timestamps': True,
                                  'languages': ['auto', '50+ languages']},
 'VibeVoice-ASR-Streaming-7B-GGUF': {'family': 'vibevoice_asr_streaming',
                                     'description': 'Microsoft VibeVoice ASR Streaming 7B model for chunked '
                                                    'streaming speech-to-text with persistent decoder state.',
                                     'settings_key': 'vibevoice_asr_streaming',
                                     'streaming': True,
                                     'default_precision': 'q8_0',
                                     'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF/resolve/f1c45856434b200afc049a57d1c986e8d30052db/vibevoice-asr-streaming-7b-q8_0.gguf'],
                                                           'checksum': '7e4c86034c1386d51d2eecb5af12c2f9c676d2156e8e8079c2297326461bf17d',
                                                           'filename': 'vibevoice-asr-streaming-7b-q8_0.gguf',
                                                           'size': 9858232896,
                                                           'file_checksums': {'vibevoice-asr-streaming-7b-q8_0.gguf': '7e4c86034c1386d51d2eecb5af12c2f9c676d2156e8e8079c2297326461bf17d'}},
                                                  'bf16': {'urls': ['https://huggingface.co/audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF/resolve/f1c45856434b200afc049a57d1c986e8d30052db/vibevoice-asr-streaming-7b-bf16.gguf'],
                                                           'checksum': '41c998202026ee60d2a47430e1720555ebd8a0f451d18bbf12e35fcf1172a28c',
                                                           'filename': 'vibevoice-asr-streaming-7b-bf16.gguf',
                                                           'size': 17360678976,
                                                           'file_checksums': {'vibevoice-asr-streaming-7b-bf16.gguf': '41c998202026ee60d2a47430e1720555ebd8a0f451d18bbf12e35fcf1172a28c'}},
                                                  'q4_k': {'urls': ['https://huggingface.co/audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF/resolve/f1c45856434b200afc049a57d1c986e8d30052db/vibevoice-asr-streaming-7b-q4_k.gguf'],
                                                           'checksum': 'ab0be81dd7522903ab0b9dd2bfdb4c08ab59debe84f0c1231642fd5188de47f6',
                                                           'filename': 'vibevoice-asr-streaming-7b-q4_k.gguf',
                                                           'size': 5859083328,
                                                           'file_checksums': {'vibevoice-asr-streaming-7b-q4_k.gguf': 'ab0be81dd7522903ab0b9dd2bfdb4c08ab59debe84f0c1231642fd5188de47f6'}}},
                                     'timestamps': True,
                                     'languages': ['en',
                                                   'zh',
                                                   'es',
                                                   'pt',
                                                   'de',
                                                   'ja',
                                                   'ko',
                                                   'fr',
                                                   'ru',
                                                   'it']}}

TTS_MODELS = {'Breeze-TTS-2-GGUF': {'family': 'breeze_tts',
                       'description': 'BreezeTTS 2 native GGUF package for instruction-conditioned '
                                      'text-to-speech and prompt-audio voice cloning with T5Gemma text '
                                      'conditioning, Qwen-style acoustic code generation, depth codebook '
                                      'decoding, and Mimi waveform decoding.',
                       'settings_key': 'breeze_tts',
                       'streaming': True,
                       'default_precision': 'q8_0',
                       'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf'],
                                             'checksum': '0de52d61560f9f6b2dfeca79f9100f8fce0c2b17c52ec30622e23e150df1ad88',
                                             'filename': 'breeze-tts-2-q8_0.gguf',
                                             'size': 5079668352,
                                             'file_checksums': {'breeze-tts-2-q8_0.gguf': '0de52d61560f9f6b2dfeca79f9100f8fce0c2b17c52ec30622e23e150df1ad88'}},
                                    'bf16': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Breeze-TTS-2-GGUF/breeze-tts-2-bf16.gguf'],
                                             'checksum': 'a00c9f678b4c5ae03d1dcd228f636b329352cda200823faef4e01d3bd97c0a89',
                                             'filename': 'breeze-tts-2-bf16.gguf',
                                             'size': 7342916800,
                                             'file_checksums': {'breeze-tts-2-bf16.gguf': 'a00c9f678b4c5ae03d1dcd228f636b329352cda200823faef4e01d3bd97c0a89'}}},
                       'group': 'Voice cloning',
                       'task': 'tts',
                       'sample_rate': 24000,
                       'languages': ['zh', 'en']},
 'Chatterbox-Turbo-GGUF': {'family': 'chatterbox_turbo',
                           'description': "Resemble AI's distilled 350M Chatterbox Turbo: GPT2 T3 backbone, "
                                          'GPT2 BPE tokenizer, and a 2-step meanflow S3Gen decoder for fast '
                                          'English TTS with a built-in voice.',
                           'settings_key': 'chatterbox_turbo',
                           'streaming': False,
                           'default_precision': 'q8_0',
                           'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf'],
                                                 'checksum': '6eed51ff0b2993fec67db6211dca92de5819d83ac50513ba7c065c467eb75910',
                                                 'filename': 'chatterbox-turbo-q8_0.gguf',
                                                 'size': 699101408,
                                                 'file_checksums': {'chatterbox-turbo-q8_0.gguf': '6eed51ff0b2993fec67db6211dca92de5819d83ac50513ba7c065c467eb75910'}}},
                           'group': 'Preset voices',
                           'task': 'tts',
                           'sample_rate': 24000,
                           'languages': ['en'],
                           'voices': ['default'],
                           'default_voice': 'default'},
 'CosyVoice3-GGUF': {'family': 'cosyvoice3',
                     'description': 'Fun-CosyVoice3 native GGUF package for zero-shot, cross-lingual, and '
                                    'instruction-conditioned text-to-speech using CosyVoice3 speech-token AR '
                                    'generation, causal masked flow mel decoding, CAM++ speaker '
                                    'conditioning, and causal HiFT waveform decoding.',
                     'settings_key': 'cosyvoice3',
                     'streaming': False,
                     'default_precision': 'q8_0',
                     'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf'],
                                           'checksum': 'ff31bb29ba5723809ec817c9fcc09a5f88d5d8ef6cfbc623edb5a7d7be8a6fca',
                                           'filename': 'cosyvoice3-q8_0.gguf',
                                           'size': 2257658080,
                                           'file_checksums': {'cosyvoice3-q8_0.gguf': 'ff31bb29ba5723809ec817c9fcc09a5f88d5d8ef6cfbc623edb5a7d7be8a6fca'}},
                                  'f32': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/CosyVoice3-GGUF/cosyvoice3-f32.gguf'],
                                          'checksum': '58f7f6c782268315c1d1f0783b6e652162d894dd4b94bde2153173218ab9bfe0',
                                          'filename': 'cosyvoice3-f32.gguf',
                                          'size': 6995036608,
                                          'file_checksums': {'cosyvoice3-f32.gguf': '58f7f6c782268315c1d1f0783b6e652162d894dd4b94bde2153173218ab9bfe0'}}},
                     'group': 'Voice cloning',
                     'task': 'clon',
                     'sample_rate': 24000,
                     'languages': ['zh', 'en', 'ja', 'ko', 'de', 'es', 'fr', 'it', 'ru', 'yue'],
                     'requires_reference': True},
 'Kokoro-82M-GGUF': {'family': 'kokoro_tts',
                     'description': 'Kokoro 82M multilingual text-to-speech with 54 preset voices, optimized '
                                    'native CPU inference, and shared eSpeak-ng phonemization.',
                     'settings_key': 'kokoro_tts',
                     'streaming': False,
                     'default_precision': 'q8_0',
                     'variants': {'q8_0': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf'],
                                           'checksum': 'ed1bee24744a27c331e53e78eca4a052c9ad319dce8b536c6319bc58b500a03f',
                                           'filename': 'kokoro-82m-q8_0.gguf',
                                           'size': 189549056,
                                           'file_checksums': {'kokoro-82m-q8_0.gguf': 'ed1bee24744a27c331e53e78eca4a052c9ad319dce8b536c6319bc58b500a03f'}},
                                  'bf16': {'urls': ['https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/bd2f3e26c1a74fa359d712b4e46919fc244722c5/Kokoro-82M-GGUF/kokoro-82m-bf16.gguf'],
                                           'checksum': '24c50bd6d09427a857744926f676e43d2484655c30d13eebce3b07efda0185b3',
                                           'filename': 'kokoro-82m-bf16.gguf',
                                           'size': 211954464,
                                           'file_checksums': {'kokoro-82m-bf16.gguf': '24c50bd6d09427a857744926f676e43d2484655c30d13eebce3b07efda0185b3'}}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 24000,
                     'languages': ['en-us', 'en-gb', 'es', 'fr-fr', 'hi', 'it', 'ja', 'pt-br', 'zh'],
                     'voices': ['af_alloy',
                                'af_aoede',
                                'af_bella',
                                'af_heart',
                                'af_jessica',
                                'af_kore',
                                'af_nicole',
                                'af_nova',
                                'af_river',
                                'af_sarah',
                                'af_sky',
                                'am_adam',
                                'am_echo',
                                'am_eric',
                                'am_fenrir',
                                'am_liam',
                                'am_michael',
                                'am_onyx',
                                'am_puck',
                                'am_santa',
                                'bf_alice',
                                'bf_emma',
                                'bf_isabella',
                                'bf_lily',
                                'bm_daniel',
                                'bm_fable',
                                'bm_george',
                                'bm_lewis',
                                'ef_dora',
                                'em_alex',
                                'em_santa',
                                'ff_siwis',
                                'hf_alpha',
                                'hf_beta',
                                'hm_omega',
                                'hm_psi',
                                'if_sara',
                                'im_nicola',
                                'jf_alpha',
                                'jf_gongitsune',
                                'jf_nezumi',
                                'jf_tebukuro',
                                'jm_kumo',
                                'pf_dora',
                                'pm_alex',
                                'pm_santa',
                                'zf_xiaobei',
                                'zf_xiaoni',
                                'zf_xiaoxiao',
                                'zf_xiaoyi',
                                'zm_yunjian',
                                'zm_yunxi',
                                'zm_yunxia',
                                'zm_yunyang'],
                     'default_voice': 'af_heart'},
 'Audio8-TTS-Preview-0.6B-GGUF': {'family': 'audio8_tts',
                                  'description': 'Audio8 TTS Preview 0.6B DualAR text-to-speech model with '
                                                 'expressive multilingual speech across yue, zh, nl, en, fr, '
                                                 'de, it, ja, ko, pl, and es, automatic language handling, '
                                                 'and rapid voice cloning from short reference samples.',
                                  'settings_key': 'audio8_tts',
                                  'streaming': True,
                                  'default_precision': 'q8_0',
                                  'variants': {'q8_0': {'urls': ['https://huggingface.co/js-byte/Audio8-TTS-Preview-0.6b-GGUF/resolve/788f6fdb0bbdbbc407c63f3265cea9875b4a7c14/audio8-tts-preview-0.6b-q8_0.gguf'],
                                                        'checksum': '7fa4d2ce5ef37a0a2526cad814bb271b1255d50681297ba56407c6307148652f',
                                                        'filename': 'audio8-tts-preview-0.6b-q8_0.gguf',
                                                        'size': 1429545312,
                                                        'file_checksums': {'audio8-tts-preview-0.6b-q8_0.gguf': '7fa4d2ce5ef37a0a2526cad814bb271b1255d50681297ba56407c6307148652f'}}},
                                  'group': 'Voice cloning',
                                  'task': 'tts',
                                  'sample_rate': 44100,
                                  'languages': ['yue',
                                                'zh',
                                                'nl',
                                                'en',
                                                'fr',
                                                'de',
                                                'it',
                                                'ja',
                                                'ko',
                                                'pl',
                                                'es']},
 'sanoTTS-heart-nano-GGUF': {'family': 'sanotts',
                             'description': 'Very small multilingual text-to-speech across fourteen '
                                            'languages. Two graphs: the nano lineage (duration student, '
                                            'contextual acoustic student to mel-100, noise-fed ConvNeXt-1D '
                                            'decoder with an iSTFT head; voices heart 2.27M and heart-nano '
                                            '294k, English, 24 kHz) and the deterministic piperlite lineage '
                                            '(duration student, acoustic student to a 192-channel latent, '
                                            'optionally through a calibration adapter, then a 3-stage '
                                            'ConvTranspose1d decoder with dilated residual banks; voices '
                                            'amy, hfc and kristin for English, vi, id, cs, de, es, fr, it, '
                                            'pt, ro, ru, tr, ne and hi at 1.1-1.8M parameters, 22.05 kHz). '
                                            'Uses an external eSpeak-ng phonemizer.',
                             'settings_key': 'sanotts',
                             'streaming': False,
                             'default_precision': 'orig',
                             'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/heart-nano-f32.gguf'],
                                                   'checksum': '9e9fddfc10c26e89f080c548be4778f191a3fb0ba877f35e05baf67da42f7476',
                                                   'filename': 'heart-nano-f32.gguf',
                                                   'size': 1197376,
                                                   'file_checksums': {'heart-nano-f32.gguf': '9e9fddfc10c26e89f080c548be4778f191a3fb0ba877f35e05baf67da42f7476'},
                                                   'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/config.json'],
                                                                         'checksum': '58eabea7a1156d436bf30731a47c86c3f1ec02623ee2a545a7cf042a8517b4dd',
                                                                         'filename': 'config.json',
                                                                         'size': 489,
                                                                         'file_checksums': {'config.json': '58eabea7a1156d436bf30731a47c86c3f1ec02623ee2a545a7cf042a8517b4dd'}}]}},
                             'group': 'Preset voices',
                             'task': 'tts',
                             'sample_rate': 24000,
                             'languages': ['en'],
                             'voices': ['heart-nano'],
                             'default_voice': 'heart-nano'},
 'sanoTTS-heart-GGUF': {'family': 'sanotts',
                        'description': 'Very small multilingual text-to-speech across fourteen languages. '
                                       'Two graphs: the nano lineage (duration student, contextual acoustic '
                                       'student to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT '
                                       'head; voices heart 2.27M and heart-nano 294k, English, 24 kHz) and '
                                       'the deterministic piperlite lineage (duration student, acoustic '
                                       'student to a 192-channel latent, optionally through a calibration '
                                       'adapter, then a 3-stage ConvTranspose1d decoder with dilated '
                                       'residual banks; voices amy, hfc and kristin for English, vi, id, cs, '
                                       'de, es, fr, it, pt, ro, ru, tr, ne and hi at 1.1-1.8M parameters, '
                                       '22.05 kHz). Uses an external eSpeak-ng phonemizer.',
                        'settings_key': 'sanotts',
                        'streaming': False,
                        'default_precision': 'orig',
                        'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/heart/heart-f32.gguf'],
                                              'checksum': 'b2096b1931f7a3ee238e1fc7794502ec6e10a33f0848fd05bb2c1ee90cb9398c',
                                              'filename': 'heart-f32.gguf',
                                              'size': 9109600,
                                              'file_checksums': {'heart-f32.gguf': 'b2096b1931f7a3ee238e1fc7794502ec6e10a33f0848fd05bb2c1ee90cb9398c'},
                                              'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/heart/config.json'],
                                                                    'checksum': '0b3ffed9db2975352d662b4648006e1e6dfde9892bd30dcac6fea3d013607dd6',
                                                                    'filename': 'config.json',
                                                                    'size': 489,
                                                                    'file_checksums': {'config.json': '0b3ffed9db2975352d662b4648006e1e6dfde9892bd30dcac6fea3d013607dd6'}}]}},
                        'group': 'Preset voices',
                        'task': 'tts',
                        'sample_rate': 24000,
                        'languages': ['en'],
                        'voices': ['heart'],
                        'default_voice': 'heart'},
 'sanoTTS-amy-GGUF': {'family': 'sanotts',
                      'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                     'graphs: the nano lineage (duration student, contextual acoustic '
                                     'student to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; '
                                     'voices heart 2.27M and heart-nano 294k, English, 24 kHz) and the '
                                     'deterministic piperlite lineage (duration student, acoustic student to '
                                     'a 192-channel latent, optionally through a calibration adapter, then a '
                                     '3-stage ConvTranspose1d decoder with dilated residual banks; voices '
                                     'amy, hfc and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, '
                                     'ru, tr, ne and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external '
                                     'eSpeak-ng phonemizer.',
                      'settings_key': 'sanotts',
                      'streaming': False,
                      'default_precision': 'orig',
                      'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/amy/amy-f32.gguf'],
                                            'checksum': 'd0e0a8dc2d2a1eab15e1093d3fcfe9cb3cad03e36a7e890487b75480312dcf2a',
                                            'filename': 'amy-f32.gguf',
                                            'size': 5837824,
                                            'file_checksums': {'amy-f32.gguf': 'd0e0a8dc2d2a1eab15e1093d3fcfe9cb3cad03e36a7e890487b75480312dcf2a'},
                                            'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/amy/config.json'],
                                                                  'checksum': 'f974432ca8bab93ea07abcee768c649a4e4ee0cfe1b1e88efa886e68702073a4',
                                                                  'filename': 'config.json',
                                                                  'size': 2660,
                                                                  'file_checksums': {'config.json': 'f974432ca8bab93ea07abcee768c649a4e4ee0cfe1b1e88efa886e68702073a4'}}]}},
                      'group': 'Preset voices',
                      'task': 'tts',
                      'sample_rate': 22050,
                      'languages': ['en'],
                      'voices': ['amy'],
                      'default_voice': 'amy'},
 'sanoTTS-hfc-GGUF': {'family': 'sanotts',
                      'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                     'graphs: the nano lineage (duration student, contextual acoustic '
                                     'student to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; '
                                     'voices heart 2.27M and heart-nano 294k, English, 24 kHz) and the '
                                     'deterministic piperlite lineage (duration student, acoustic student to '
                                     'a 192-channel latent, optionally through a calibration adapter, then a '
                                     '3-stage ConvTranspose1d decoder with dilated residual banks; voices '
                                     'amy, hfc and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, '
                                     'ru, tr, ne and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external '
                                     'eSpeak-ng phonemizer.',
                      'settings_key': 'sanotts',
                      'streaming': False,
                      'default_precision': 'orig',
                      'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/hfc/hfc-f32.gguf'],
                                            'checksum': '0cfdf86140c83822bf5d5c1aecffa1081bb98c6b0b6c15ab3493426d50315629',
                                            'filename': 'hfc-f32.gguf',
                                            'size': 7358208,
                                            'file_checksums': {'hfc-f32.gguf': '0cfdf86140c83822bf5d5c1aecffa1081bb98c6b0b6c15ab3493426d50315629'},
                                            'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/hfc/config.json'],
                                                                  'checksum': 'efe89a538552742252b1a94e404d30bcebe907829669298b5783ca3b874a745f',
                                                                  'filename': 'config.json',
                                                                  'size': 2722,
                                                                  'file_checksums': {'config.json': 'efe89a538552742252b1a94e404d30bcebe907829669298b5783ca3b874a745f'}}]}},
                      'group': 'Preset voices',
                      'task': 'tts',
                      'sample_rate': 22050,
                      'languages': ['en'],
                      'voices': ['hfc'],
                      'default_voice': 'hfc'},
 'sanoTTS-kristin-GGUF': {'family': 'sanotts',
                          'description': 'Very small multilingual text-to-speech across fourteen languages. '
                                         'Two graphs: the nano lineage (duration student, contextual '
                                         'acoustic student to mel-100, noise-fed ConvNeXt-1D decoder with an '
                                         'iSTFT head; voices heart 2.27M and heart-nano 294k, English, 24 '
                                         'kHz) and the deterministic piperlite lineage (duration student, '
                                         'acoustic student to a 192-channel latent, optionally through a '
                                         'calibration adapter, then a 3-stage ConvTranspose1d decoder with '
                                         'dilated residual banks; voices amy, hfc and kristin for English, '
                                         'vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne and hi at 1.1-1.8M '
                                         'parameters, 22.05 kHz). Uses an external eSpeak-ng phonemizer.',
                          'settings_key': 'sanotts',
                          'streaming': False,
                          'default_precision': 'orig',
                          'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/kristin/kristin-f32.gguf'],
                                                'checksum': '7ae11f19a04e38ae4844d5d198a0f91f245676a1c44edc55e3430ecafca5365c',
                                                'filename': 'kristin-f32.gguf',
                                                'size': 5607296,
                                                'file_checksums': {'kristin-f32.gguf': '7ae11f19a04e38ae4844d5d198a0f91f245676a1c44edc55e3430ecafca5365c'},
                                                'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/kristin/config.json'],
                                                                      'checksum': '299f51b020fc0982da562e30befd7d88fdb9a384d2783721e314bc473bc1b494',
                                                                      'filename': 'config.json',
                                                                      'size': 2700,
                                                                      'file_checksums': {'config.json': '299f51b020fc0982da562e30befd7d88fdb9a384d2783721e314bc473bc1b494'}}]}},
                          'group': 'Preset voices',
                          'task': 'tts',
                          'sample_rate': 22050,
                          'languages': ['en'],
                          'voices': ['kristin'],
                          'default_voice': 'kristin'},
 'sanoTTS-vi-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/vi/vi-f32.gguf'],
                                           'checksum': 'a0bb03936b3f1be81e412e24d620cf273f9999264c89ecdb8fffc6b9f6200a58',
                                           'filename': 'vi-f32.gguf',
                                           'size': 6282752,
                                           'file_checksums': {'vi-f32.gguf': 'a0bb03936b3f1be81e412e24d620cf273f9999264c89ecdb8fffc6b9f6200a58'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/vi/config.json'],
                                                                 'checksum': 'd72b29f8c5dad7e442ba7fdf40232d6fd7cab0a9afd306e6d8ab2894ca0722f0',
                                                                 'filename': 'config.json',
                                                                 'size': 2661,
                                                                 'file_checksums': {'config.json': 'd72b29f8c5dad7e442ba7fdf40232d6fd7cab0a9afd306e6d8ab2894ca0722f0'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['vi'],
                     'voices': ['vi'],
                     'default_voice': 'vi'},
 'sanoTTS-id-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/id/id-f32.gguf'],
                                           'checksum': '6eb7c447e31e337987a05c71c00e5d9dfd06368f8cc012fbe9b0cacbb1b77488',
                                           'filename': 'id-f32.gguf',
                                           'size': 6269312,
                                           'file_checksums': {'id-f32.gguf': '6eb7c447e31e337987a05c71c00e5d9dfd06368f8cc012fbe9b0cacbb1b77488'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/id/config.json'],
                                                                 'checksum': 'e93d194707cd96a810c455d18ac78398457cc6452c897dcdeb5f7592cb4e248b',
                                                                 'filename': 'config.json',
                                                                 'size': 2749,
                                                                 'file_checksums': {'config.json': 'e93d194707cd96a810c455d18ac78398457cc6452c897dcdeb5f7592cb4e248b'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['id'],
                     'voices': ['id'],
                     'default_voice': 'id'},
 'sanoTTS-cs-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/cs/cs-f32.gguf'],
                                           'checksum': 'de03447471ff055c9e61e1bb39b2bc3bda8edd89e84c270f5caf4d326371c5c4',
                                           'filename': 'cs-f32.gguf',
                                           'size': 6296352,
                                           'file_checksums': {'cs-f32.gguf': 'de03447471ff055c9e61e1bb39b2bc3bda8edd89e84c270f5caf4d326371c5c4'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/cs/config.json'],
                                                                 'checksum': '02e1e77796be139bb4dea3d1e0784a717a8d3e86463aa8e69ef3595de94822a3',
                                                                 'filename': 'config.json',
                                                                 'size': 2727,
                                                                 'file_checksums': {'config.json': '02e1e77796be139bb4dea3d1e0784a717a8d3e86463aa8e69ef3595de94822a3'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['cs'],
                     'voices': ['cs'],
                     'default_voice': 'cs'},
 'sanoTTS-de-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/de/de-f32.gguf'],
                                           'checksum': '267c56420162d949994912eef5e1c3b4543fcf915b94258ca4bc1066a2f8d39b',
                                           'filename': 'de-f32.gguf',
                                           'size': 6285472,
                                           'file_checksums': {'de-f32.gguf': '267c56420162d949994912eef5e1c3b4543fcf915b94258ca4bc1066a2f8d39b'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/de/config.json'],
                                                                 'checksum': '6847f38cb3028d1d5854123a879660efba959a2e24b980cb37b2ae1fe0eabb17',
                                                                 'filename': 'config.json',
                                                                 'size': 2638,
                                                                 'file_checksums': {'config.json': '6847f38cb3028d1d5854123a879660efba959a2e24b980cb37b2ae1fe0eabb17'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['de'],
                     'voices': ['de'],
                     'default_voice': 'de'},
 'sanoTTS-es-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/es/es-f32.gguf'],
                                           'checksum': '2d7815e340ad0d95c8f27acbf4ca2bbe8e872e5fe1f7db2b65b6dfe5b0a25ce6',
                                           'filename': 'es-f32.gguf',
                                           'size': 6275872,
                                           'file_checksums': {'es-f32.gguf': '2d7815e340ad0d95c8f27acbf4ca2bbe8e872e5fe1f7db2b65b6dfe5b0a25ce6'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/es/config.json'],
                                                                 'checksum': '09b7a90a284e36ffaf0e066bbd87f80cfe284bbb7f53cb5b47bdaae121223206',
                                                                 'filename': 'config.json',
                                                                 'size': 2638,
                                                                 'file_checksums': {'config.json': '09b7a90a284e36ffaf0e066bbd87f80cfe284bbb7f53cb5b47bdaae121223206'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['es'],
                     'voices': ['es'],
                     'default_voice': 'es'},
 'sanoTTS-fr-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/fr/fr-f32.gguf'],
                                           'checksum': '00270dfec1b04c38119e18cd601a18f54344d192b4cdbda750f42ae70e39a161',
                                           'filename': 'fr-f32.gguf',
                                           'size': 6285472,
                                           'file_checksums': {'fr-f32.gguf': '00270dfec1b04c38119e18cd601a18f54344d192b4cdbda750f42ae70e39a161'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/fr/config.json'],
                                                                 'checksum': '925a7bb3337f0d0d86076d53601e8de5273dad11d6736446e423453066dcb157',
                                                                 'filename': 'config.json',
                                                                 'size': 2664,
                                                                 'file_checksums': {'config.json': '925a7bb3337f0d0d86076d53601e8de5273dad11d6736446e423453066dcb157'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['fr'],
                     'voices': ['fr'],
                     'default_voice': 'fr'},
 'sanoTTS-it-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/it/it-f32.gguf'],
                                           'checksum': 'e462520e040405b6b0b69bb5f5d31a1edb422c608f8883182685a6573570b331',
                                           'filename': 'it-f32.gguf',
                                           'size': 6286112,
                                           'file_checksums': {'it-f32.gguf': 'e462520e040405b6b0b69bb5f5d31a1edb422c608f8883182685a6573570b331'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/it/config.json'],
                                                                 'checksum': '020842d2b0e35184413d5c9eddc081b8f6220d938adab5483246d5de2c220133',
                                                                 'filename': 'config.json',
                                                                 'size': 2702,
                                                                 'file_checksums': {'config.json': '020842d2b0e35184413d5c9eddc081b8f6220d938adab5483246d5de2c220133'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['it'],
                     'voices': ['it'],
                     'default_voice': 'it'},
 'sanoTTS-pt-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/pt/pt-f32.gguf'],
                                           'checksum': 'aa80b204bc6ebaad799e03e3db013b141974aad2d1088cd614bb44203aeac55e',
                                           'filename': 'pt-f32.gguf',
                                           'size': 6285472,
                                           'file_checksums': {'pt-f32.gguf': 'aa80b204bc6ebaad799e03e3db013b141974aad2d1088cd614bb44203aeac55e'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/pt/config.json'],
                                                                 'checksum': 'afeae6f79747d31f0110c40f629e06457301bb7f3e43ce17b60a2e5d1c85e376',
                                                                 'filename': 'config.json',
                                                                 'size': 2757,
                                                                 'file_checksums': {'config.json': 'afeae6f79747d31f0110c40f629e06457301bb7f3e43ce17b60a2e5d1c85e376'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['pt'],
                     'voices': ['pt'],
                     'default_voice': 'pt'},
 'sanoTTS-ro-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ro/ro-f32.gguf'],
                                           'checksum': 'd15a3b02b57a0f5d2ed4bd0cd530e5284f99b3f2a9808847a593c46764fa28ee',
                                           'filename': 'ro-f32.gguf',
                                           'size': 6284832,
                                           'file_checksums': {'ro-f32.gguf': 'd15a3b02b57a0f5d2ed4bd0cd530e5284f99b3f2a9808847a593c46764fa28ee'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ro/config.json'],
                                                                 'checksum': '43513008d42f78f3e9a2666ff3d89d9f7ba43b1c0ec79c31b6932aa6cefe8a08',
                                                                 'filename': 'config.json',
                                                                 'size': 2664,
                                                                 'file_checksums': {'config.json': '43513008d42f78f3e9a2666ff3d89d9f7ba43b1c0ec79c31b6932aa6cefe8a08'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['ro'],
                     'voices': ['ro'],
                     'default_voice': 'ro'},
 'sanoTTS-ru-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ru/ru-f32.gguf'],
                                           'checksum': '9e84e999f1a14ae25cee0070bba70cb92a11fc83a2185422756573bf6cd0b6ed',
                                           'filename': 'ru-f32.gguf',
                                           'size': 6291232,
                                           'file_checksums': {'ru-f32.gguf': '9e84e999f1a14ae25cee0070bba70cb92a11fc83a2185422756573bf6cd0b6ed'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ru/config.json'],
                                                                 'checksum': '3cdc42083c7a14da864c650df6dfc7dff47919f5a54ad5722ffa65ecda20e11b',
                                                                 'filename': 'config.json',
                                                                 'size': 2625,
                                                                 'file_checksums': {'config.json': '3cdc42083c7a14da864c650df6dfc7dff47919f5a54ad5722ffa65ecda20e11b'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['ru'],
                     'voices': ['ru'],
                     'default_voice': 'ru'},
 'sanoTTS-tr-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/tr/tr-f32.gguf'],
                                           'checksum': '0edf477899dbdaf2009cf87c2cf4fd601ebdb2c1063414bcea11d79e4dd46534',
                                           'filename': 'tr-f32.gguf',
                                           'size': 6273312,
                                           'file_checksums': {'tr-f32.gguf': '0edf477899dbdaf2009cf87c2cf4fd601ebdb2c1063414bcea11d79e4dd46534'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/tr/config.json'],
                                                                 'checksum': '7895212d23ba27df600192e53b50a83de53231eab661c71a628be2a5a034a35f',
                                                                 'filename': 'config.json',
                                                                 'size': 2664,
                                                                 'file_checksums': {'config.json': '7895212d23ba27df600192e53b50a83de53231eab661c71a628be2a5a034a35f'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['tr'],
                     'voices': ['tr'],
                     'default_voice': 'tr'},
 'sanoTTS-ne-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ne/ne-f32.gguf'],
                                           'checksum': '4933083abfe7031a2db8af3d88162fa00e30ae69da3a8a3a9efb9c4196ac46fb',
                                           'filename': 'ne-f32.gguf',
                                           'size': 5924640,
                                           'file_checksums': {'ne-f32.gguf': '4933083abfe7031a2db8af3d88162fa00e30ae69da3a8a3a9efb9c4196ac46fb'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/ne/config.json'],
                                                                 'checksum': '334d33da82e46f4a39505fa4adde1696ff5c9772b3c023bf996c1e5338411309',
                                                                 'filename': 'config.json',
                                                                 'size': 2754,
                                                                 'file_checksums': {'config.json': '334d33da82e46f4a39505fa4adde1696ff5c9772b3c023bf996c1e5338411309'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['ne'],
                     'voices': ['ne'],
                     'default_voice': 'ne'},
 'sanoTTS-hi-GGUF': {'family': 'sanotts',
                     'description': 'Very small multilingual text-to-speech across fourteen languages. Two '
                                    'graphs: the nano lineage (duration student, contextual acoustic student '
                                    'to mel-100, noise-fed ConvNeXt-1D decoder with an iSTFT head; voices '
                                    'heart 2.27M and heart-nano 294k, English, 24 kHz) and the deterministic '
                                    'piperlite lineage (duration student, acoustic student to a 192-channel '
                                    'latent, optionally through a calibration adapter, then a 3-stage '
                                    'ConvTranspose1d decoder with dilated residual banks; voices amy, hfc '
                                    'and kristin for English, vi, id, cs, de, es, fr, it, pt, ro, ru, tr, ne '
                                    'and hi at 1.1-1.8M parameters, 22.05 kHz). Uses an external eSpeak-ng '
                                    'phonemizer.',
                     'settings_key': 'sanotts',
                     'streaming': False,
                     'default_precision': 'orig',
                     'variants': {'orig': {'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/hi/hi-f32.gguf'],
                                           'checksum': '0c1ec8e0ac281e405874c1068962bad41a8609f2e6c5c5fbfca9ecc2dcb4a504',
                                           'filename': 'hi-f32.gguf',
                                           'size': 6024096,
                                           'file_checksums': {'hi-f32.gguf': '0c1ec8e0ac281e405874c1068962bad41a8609f2e6c5c5fbfca9ecc2dcb4a504'},
                                           'additional_files': [{'urls': ['https://huggingface.co/ampixa/sanoTTS/resolve/c532a5d21c078a16cb633718e9182bfd71a5b760/gguf/hi/config.json'],
                                                                 'checksum': '0f4a254235b91de1370519497493a3d118df43acdd3a1922a2069b9c518114d9',
                                                                 'filename': 'config.json',
                                                                 'size': 2784,
                                                                 'file_checksums': {'config.json': '0f4a254235b91de1370519497493a3d118df43acdd3a1922a2069b9c518114d9'}}]}},
                     'group': 'Preset voices',
                     'task': 'tts',
                     'sample_rate': 22050,
                     'languages': ['hi'],
                     'voices': ['hi'],
                     'default_voice': 'hi'}}

SETTINGS_DEFAULTS = {'canary_asr': {'pnc': True,
                'max_tokens': 0,
                'audio_chunk_mode': 'auto',
                'audio_chunk_duration_sec': 40,
                'mode': 'offline'},
 'cohere_asr': {'pnc': True,
                'max_tokens': 256,
                'audio_chunk_mode': 'auto',
                'audio_chunk_duration_sec': 35,
                'mode': 'offline'},
 'moonshine_asr': {'max_tokens': 0, 'mode': 'auto'},
 'niagara_asr': {'audio_chunk_mode': 'none', 'mode': 'offline'},
 'moss_transcribe_diarize': {'max_tokens': 5120, 'mode': 'auto'},
 'vibevoice_asr_streaming': {'context': '',
                             'max_tokens': 256,
                             'temperature': 0,
                             'top_p': 1,
                             'top_k': 0,
                             'num_beams': 1,
                             'repetition_penalty': 1,
                             'seed': 42,
                             'audio_chunk_mode': 'auto',
                             'audio_chunk_duration_sec': 1200,
                             'mode': 'auto'},
 'breeze_tts': {'instruction': 'Speak clearly and naturally.',
                'text_chunk_size': 600,
                'text_chunk_mode': 'default',
                'max_tokens': 1500,
                'guidance_scale': 1.0,
                'temperature': 0.9,
                'depth_temperature': 0.9,
                'top_k': 50,
                'top_p': 1.0,
                'seed': 0,
                'stream_frames_per_event': 16,
                'stream_lookahead_margin': 12,
                'language': 'auto'},
 'chatterbox_turbo': {'language': 'auto'},
 'cosyvoice3': {'template_name': 'cross_lingual',
                'text_chunk_size': 600,
                'text_chunk_mode': 'default',
                'max_tokens': 1600,
                'min_tokens': 0,
                'top_k': 25,
                'num_inference_steps': 10,
                'seed': 1986,
                'language': 'auto'},
 'kokoro_tts': {'text_chunk_size': 240, 'language': 'auto'},
 'audio8_tts': {'max_tokens': 1024,
                'text_chunk_size': 200,
                'text_chunk_mode': 'word_budget',
                'top_p': 0.9,
                'top_k': 50,
                'temperature': 0.7,
                'language': 'auto'},
 'sanotts': {'speaking_rate': 1.0,
             'seed': 0,
             'text_chunk_mode': 'word_budget',
             'text_chunk_size': 280,
             'language': 'auto'}}

REQUEST_KEYS = {'canary_asr': ('pnc', 'max_tokens', 'audio_chunk_mode', 'audio_chunk_duration_sec'),
 'cohere_asr': ('pnc', 'max_tokens', 'audio_chunk_mode', 'audio_chunk_duration_sec'),
 'moonshine_asr': ('max_tokens',),
 'niagara_asr': ('audio_chunk_mode',),
 'moss_transcribe_diarize': ('max_tokens', 'instruct'),
 'vibevoice_asr_streaming': ('context',
                             'max_tokens',
                             'temperature',
                             'top_p',
                             'top_k',
                             'num_beams',
                             'repetition_penalty',
                             'seed',
                             'audio_chunk_mode',
                             'audio_chunk_duration_sec'),
 'breeze_tts': ('instruction',
                'text_chunk_size',
                'text_chunk_mode',
                'max_tokens',
                'guidance_scale',
                'temperature',
                'depth_temperature',
                'top_k',
                'top_p',
                'seed',
                'stream_frames_per_event',
                'stream_lookahead_margin'),
 'chatterbox_turbo': (),
 'cosyvoice3': ('template_name',
                'instruction',
                'text_chunk_size',
                'text_chunk_mode',
                'max_tokens',
                'min_tokens',
                'top_k',
                'num_inference_steps',
                'seed'),
 'kokoro_tts': ('seed', 'text_chunk_size'),
 'audio8_tts': ('max_tokens', 'text_chunk_size', 'text_chunk_mode', 'top_p', 'top_k', 'temperature', 'seed'),
 'sanotts': ('speaking_rate', 'seed', 'text_chunk_mode', 'text_chunk_size')}

SESSION_KEYS = {'canary_asr': (),
 'cohere_asr': (),
 'moonshine_asr': ('encoder_gelu', 'cpu_blas_scheduler', 'weight_context_mb', 'graph_arena_mb'),
 'niagara_asr': ('graph_arena_mb', 'weight_context_mb'),
 'moss_transcribe_diarize': (),
 'vibevoice_asr_streaming': (),
 'breeze_tts': ('graph_arena_mb', 'weight_context_mb', 'reference_cache_slots', 'attention'),
 'chatterbox_turbo': (),
 'cosyvoice3': ('graph_arena_mb', 'weight_context_mb', 'reference_cache_slots', 'mem_saver'),
 'kokoro_tts': ('weight_context_mb',
                'predictor_duration_graph_mb',
                'predictor_text_graph_mb',
                'predictor_tail_graph_mb',
                'graph_capacity_mode',
                'max_input_tokens',
                'pre_tail_tokens'),
 'audio8_tts': ('mem_saver',
                'reference_cache_slots',
                'ar_graph_arena_mb',
                'ar_weight_context_mb',
                'codec_graph_arena_mb',
                'codec_weight_context_mb'),
 'sanotts': ('espeak_library_path', 'espeak_data_path')}

