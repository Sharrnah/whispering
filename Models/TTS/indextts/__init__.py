"""Pinned, inference-only IndexTTS 2.5 runtime.

Source: https://github.com/index-tts/index-tts
Revision: 4f8792ff120cd3ea470dd511e997a17c86cddd10

The upstream license and disclaimer are included beside this file. Training,
web UI, remote download, and optional acceleration modules are intentionally
not vendored into Whispering Tiger.

Local compatibility changes are documented in the repository AGENTS.md and
knowledge graph. In brief: the GPT inference bridge targets Transformers 5
GenerationMixin/DynamicCache, BigVGAN accepts the current Hub mixin signature
but is loaded local-only, and model_download.py is an offline failure stub.
"""
