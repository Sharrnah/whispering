from __future__ import annotations
import types
from contextlib import contextmanager
from typing import Any, Dict, Tuple, Optional

import torch
from transformers import GenerationConfig
from parler_tts.configuration_parler_tts import ParlerTTSConfig
from parler_tts import ParlerTTSForConditionalGeneration


# ---------- tiny helpers ----------

@contextmanager
def parler_safe_repr():
    """Avoid ParlerTTSConfig.__repr__ -> to_json_string() crash during from_pretrained()."""
    _orig = ParlerTTSConfig.__repr__
    ParlerTTSConfig.__repr__ = lambda self: f"{self.__class__.__name__}(…)"
    try:
        yield
    finally:
        ParlerTTSConfig.__repr__ = _orig


def _as_1d_tensor(val, device, dtype=torch.long) -> Optional[torch.Tensor]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return torch.tensor(list(val), device=device, dtype=dtype)
    return torch.tensor([val], device=device, dtype=dtype)


def _try_import_logits_bits():
    LPList = TempWarp = TopPWarp = TopKWarp = RepPen = NoRep = EncNoRep = MinLen = FBos = FEos = Suppress = SuppressAtBeg = PrefixProc = None
    try:
        from transformers.generation.logits_process import (
            LogitsProcessorList as LPList,
            TemperatureLogitsWarper as TempWarp,
            TopPLogitsWarper as TopPWarp,
            TopKLogitsWarper as TopKWarp,
            RepetitionPenaltyLogitsProcessor as RepPen,
            NoRepeatNGramLogitsProcessor as NoRep,
            EncoderNoRepeatNGramLogitsProcessor as EncNoRep,
            MinLengthLogitsProcessor as MinLen,
            ForcedBOSTokenLogitsProcessor as FBos,
            ForcedEOSTokenLogitsProcessor as FEos,
            SuppressTokensLogitsProcessor as Suppress,
            SuppressTokensAtBeginLogitsProcessor as SuppressAtBeg,
            PrefixConstrainedLogitsProcessor as PrefixProc,
        )
    except Exception:
        try:
            # noinspection PyUnresolvedReferences
            from transformers.generation_utils import LogitsProcessorList as LPList  # very old fallback
        except Exception:
            pass
    return LPList, TempWarp, TopPWarp, TopKWarp, RepPen, NoRep, EncNoRep, MinLen, FBos, FEos, Suppress, SuppressAtBeg, PrefixProc


def _try_import_stopping_bits():
    SCList = MaxLenCrit = MaxNewCrit = None
    try:
        from transformers.generation.stopping_criteria import (
            StoppingCriteriaList as SCList,
            MaxLengthCriteria as MaxLenCrit,
            MaxNewTokensCriteria as MaxNewCrit,
        )
    except Exception:
        try:
            from transformers.generation.stopping_criteria import (
                StoppingCriteriaList as SCList,
                MaxLengthCriteria as MaxLenCrit,
            )
        except Exception:
            pass
    return SCList, MaxLenCrit, MaxNewCrit


# ---------- patched subclass ----------

class ParlerTTSForConditionalGenerationPatched(ParlerTTSForConditionalGeneration):
    """
    A subclass that provides stable private helpers across newer transformers,
    and normalizes inputs so Parler's generate() works offline.
    """

    # ---- minimal validations ----
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return None

    # ---- inputs prep (varargs; robust to API diffs) ----
    def _prepare_model_inputs(self, *args, **kwargs) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        inputs_tensor: Optional[torch.Tensor] = None
        for a in args:
            if isinstance(a, torch.Tensor):
                inputs_tensor = a
                break
        if inputs_tensor is None:
            inputs_tensor = kwargs.get("inputs") or kwargs.get("input_ids") or None
        model_kwargs: Dict[str, Any] = {}
        if len(args) > 2 and isinstance(args[2], dict):
            model_kwargs = args[2]
        else:
            model_kwargs = kwargs.get("model_kwargs") or {}
        if inputs_tensor is None and "input_ids" in model_kwargs:
            inputs_tensor = model_kwargs["input_ids"]
        if inputs_tensor is None:
            raise ValueError("Parler compat: no input_ids provided to generate().")
        model_kwargs["input_ids"] = inputs_tensor
        return inputs_tensor, "input_ids", model_kwargs

    # ---- special tokens & cached tensors on GenerationConfig ----
    def _prepare_special_tokens(
            self,
            generation_config: GenerationConfig,
            kwargs_has_attention_mask: bool,
            device=None,
            **_kwargs
    ):
        cfg = getattr(self, "config", None)
        gen_default = getattr(self, "generation_config", None)

        def _fill(name):
            if getattr(generation_config, name, None) is None:
                for src in (generation_config, gen_default, cfg):
                    if src is None:
                        continue
                    val = getattr(src, name, None)
                    if val is not None:
                        setattr(generation_config, name, val)
                        return

        for field in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
            _fill(field)
        if getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = getattr(generation_config, "eos_token_id", 0) or 0

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # Create private cached tensors Parler expects
        if not hasattr(generation_config, "_bos_token_tensor") or generation_config._bos_token_tensor is None:
            generation_config._bos_token_tensor = _as_1d_tensor(generation_config.bos_token_id, device)
        if not hasattr(generation_config, "_eos_token_tensor") or generation_config._eos_token_tensor is None:
            generation_config._eos_token_tensor = _as_1d_tensor(generation_config.eos_token_id, device)
        if not hasattr(generation_config, "_pad_token_tensor") or generation_config._pad_token_tensor is None:
            generation_config._pad_token_tensor = _as_1d_tensor(generation_config.pad_token_id, device)
        if not hasattr(generation_config, "_decoder_start_token_tensor") or generation_config._decoder_start_token_tensor is None:
            generation_config._decoder_start_token_tensor = _as_1d_tensor(generation_config.decoder_start_token_id, device)
        if not hasattr(generation_config, "_decoder_start_token_id_tensor") or generation_config._decoder_start_token_id_tensor is None:
            generation_config._decoder_start_token_id_tensor = generation_config._decoder_start_token_tensor
        return None

    # ---- attention mask from input_ids ----
    def _prepare_attention_mask_for_generation(self, *args, **kwargs) -> torch.Tensor:
        inputs_tensor: Optional[torch.Tensor] = None
        for a in args:
            if isinstance(a, torch.Tensor):
                inputs_tensor = a
                break
        if inputs_tensor is None:
            inputs_tensor = kwargs.get("inputs") or kwargs.get("input_ids")
        if inputs_tensor is None:
            inputs_tensor = (kwargs.get("model_kwargs") or {}).get("input_ids")
        if inputs_tensor is None:
            raise ValueError("Parler compat: cannot build attention_mask without input_ids.")
        gc = None
        for a in args:
            if isinstance(a, GenerationConfig):
                gc = a
                break
        if gc is None:
            gc = kwargs.get("generation_config") or getattr(self, "generation_config", None)
        pad_id = getattr(gc, "pad_token_id", None)
        if pad_id is None and hasattr(self, "config"):
            pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        return (inputs_tensor != pad_id).to(dtype=torch.long, device=inputs_tensor.device)

    # ---- generation length defaults ----
    def _prepare_generated_length(self, inputs_tensor: torch.Tensor, generation_config: GenerationConfig, *_, **__) -> GenerationConfig:
        if getattr(generation_config, "max_new_tokens", None) is None and getattr(generation_config, "max_length", None) is None:
            generation_config.max_new_tokens = 2048
        if getattr(generation_config, "do_sample", None) is None:
            generation_config.do_sample = True
        return generation_config

    # ---- avoid dynamic cache branch ----
    def _supports_default_dynamic_cache(self) -> bool:
        return False

    # ---- logits processors / stopping criteria ----
    def _get_logits_processor(
            self,
            generation_config: GenerationConfig,
            input_ids_seq_length: int,
            encoder_input_ids: Optional[torch.Tensor],
            prefix_allowed_tokens_fn=None,
            logits_processor=None,
            **kwargs
    ):
        LPList, TempWarp, TopPWarp, TopKWarp, RepPen, NoRep, EncNoRep, MinLen, FBos, FEos, Suppress, SuppressAtBeg, PrefixProc = _try_import_logits_bits()
        processors = logits_processor if logits_processor is not None else (LPList() if LPList is not None else [])
        if getattr(generation_config, "temperature", None) not in (None, 1.0) and TempWarp is not None:
            processors.append(TempWarp(generation_config.temperature))
        if getattr(generation_config, "top_p", None) not in (None, 1.0) and TopPWarp is not None:
            processors.append(TopPWarp(generation_config.top_p))
        if getattr(generation_config, "top_k", None) not in (None, 0) and TopKWarp is not None:
            processors.append(TopKWarp(generation_config.top_k))
        if getattr(generation_config, "repetition_penalty", None) not in (None, 1.0) and RepPen is not None:
            processors.append(RepPen(penalty=generation_config.repetition_penalty))
        ngram = getattr(generation_config, "no_repeat_ngram_size", 0)
        if ngram and ngram > 0:
            if encoder_input_ids is not None and EncNoRep is not None:
                processors.append(EncNoRep(ngram, input_ids=encoder_input_ids))
            elif NoRep is not None:
                processors.append(NoRep(ngram))
        if MinLen is not None and getattr(generation_config, "eos_token_id", None) is not None:
            min_len = getattr(generation_config, "min_length", None)
            if min_len is not None and min_len > 0:
                processors.append(MinLen(min_length=min_len, eos_token_id=generation_config.eos_token_id))
        if FBos is not None and getattr(generation_config, "forced_bos_token_id", None) is not None and input_ids_seq_length == 1:
            processors.append(FBos(generation_config.forced_bos_token_id))
        if FEos is not None and getattr(generation_config, "forced_eos_token_id", None) is not None:
            processors.append(FEos(generation_config.forced_eos_token_id))
        if Suppress is not None and getattr(generation_config, "suppress_tokens", None):
            processors.append(Suppress(generation_config.suppress_tokens))
        if SuppressAtBeg is not None and getattr(generation_config, "begin_suppress_tokens", None):
            processors.append(SuppressAtBeg(generation_config.begin_suppress_tokens))
        if PrefixProc is not None and prefix_allowed_tokens_fn is not None:
            processors.append(PrefixProc(prefix_allowed_tokens_fn))
        return processors

    def _get_stopping_criteria(self, generation_config: GenerationConfig, stopping_criteria=None):
        SCList, MaxLenCrit, MaxNewCrit = _try_import_stopping_bits()
        if stopping_criteria is None:
            criteria = SCList() if SCList is not None else []
        else:
            criteria = stopping_criteria
        max_new = getattr(generation_config, "max_new_tokens", None)
        max_len = getattr(generation_config, "max_length", None)
        if MaxNewCrit is not None and max_new is not None:
            criteria.append(MaxNewCrit(max_new_tokens=max_new))
        elif MaxLenCrit is not None and max_len is not None:
            criteria.append(MaxLenCrit(max_length=max_len))
        elif MaxNewCrit is not None:
            criteria.append(MaxNewCrit(max_new_tokens=2048))
        elif MaxLenCrit is not None:
            criteria.append(MaxLenCrit(max_length=4096))
        return criteria

    # ---- *** the critical one *** ----
    def _expand_inputs_for_generation(self, *args, **kwargs):
        """
        Accepts both positional and keyword 'input_ids' without raising the
        'multiple values for keyword argument' error, and expands batch.
        """
        # positional input_ids?
        input_ids = None
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
            args = ()

        # keyword input_ids (pop so it can't double-bind)
        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("Parler compat: _expand_inputs_for_generation requires input_ids.")

        expand_size = kwargs.pop("expand_size", 1)
        _ = kwargs.pop("is_encoder_decoder", False)  # ignored
        attention_mask = kwargs.pop("attention_mask", None)

        model_kwargs = kwargs  # the rest

        if not expand_size or expand_size <= 1:
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            return input_ids, model_kwargs

        expanded_input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        def _maybe_expand(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == input_ids.shape[0]:
                return t.repeat_interleave(expand_size, dim=0)
            return t

        if attention_mask is not None:
            model_kwargs["attention_mask"] = _maybe_expand(attention_mask)

        for key in (
                "decoder_input_ids",
                "decoder_attention_mask",
                "prompt_input_ids",
                "prompt_attention_mask",
                "inputs_embeds",
                "encoder_outputs",
                "past_key_values",
        ):
            if key in model_kwargs:
                model_kwargs[key] = _maybe_expand(model_kwargs[key])

        for key, val in list(model_kwargs.items()):
            if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == input_ids.shape[0]:
                model_kwargs[key] = val.repeat_interleave(expand_size, dim=0)

        return expanded_input_ids, model_kwargs

    # ---- decoder prep / cache maintenance / beams no-op ----
    def _prepare_decoder_input_ids_for_generation(
            self,
            batch_size: int,
            decoder_start_token_id: Optional[int] = None,
            bos_token_id: Optional[int] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ) -> torch.Tensor:
        tok = decoder_start_token_id
        if tok is None:
            tok = bos_token_id
        if tok is None and hasattr(self, "config"):
            tok = getattr(self.config, "decoder_start_token_id", None) or getattr(self.config, "bos_token_id", 0)
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return torch.full((batch_size, 1), int(tok or 0), dtype=torch.long, device=device)

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False) -> Dict[str, Any]:
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values
        attn = model_kwargs.get("attention_mask", None)
        if isinstance(attn, torch.Tensor):
            ones = torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)
            model_kwargs["attention_mask"] = torch.cat([attn, ones], dim=1)
        return model_kwargs

    def _reorder_cache(self, past, beam_idx):
        return past


# ---------- public loader ----------

def load_parler_model_offline_patched(model_local_dir, device):
    """
    Load the Parler model from a local directory into the subclass above,
    and make sure GenerationConfig is usable offline.
    """
    with parler_safe_repr():
        model = ParlerTTSForConditionalGenerationPatched.from_pretrained(
            str(model_local_dir),
            local_files_only=True,
        ).to(device)

    # Ensure a usable generation config (offline) + safe defaults
    if getattr(model, "generation_config", None) is None:
        try:
            model.generation_config = GenerationConfig.from_pretrained(str(model_local_dir), local_files_only=True)
        except Exception:
            model.generation_config = GenerationConfig.from_model_config(model.config)

    gc = model.generation_config
    if getattr(gc, "cache_implementation", None) is None:
        gc.cache_implementation = "static"
    if getattr(gc, "num_beams", None) is None:
        gc.num_beams = 1
    if getattr(gc, "num_return_sequences", None) is None:
        gc.num_return_sequences = 1

    # Fill token ids & private cached tensors now
    model._prepare_special_tokens(gc, kwargs_has_attention_mask=False)

    return model
