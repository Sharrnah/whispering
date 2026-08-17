"""Transformers 5 compatibility surface for the pinned IndexTTS GPT code.

IndexTTS 2.5 vendors a modified copy of Transformers 4.46 GPT-2 and generation
internals. Whispering Tiger already requires Transformers 5, so importing that
copy would depend on APIs removed in v5. The IndexTTS customization lives in
``model_v2.GPT2InferenceModel``; its transformer backbone can use the public
v5 GPT-2 classes directly once generation is mixed back into the base class.
"""

from transformers import GPT2Model, GPT2PreTrainedModel
