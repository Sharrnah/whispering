from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from numbers import Number
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch


def _load_module_state(module, state_dict, section_name):
    """Load one checkpoint section without silently accepting learned-weight gaps."""
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError as load_error:
        original_error = load_error
        if not state_dict:
            raise RuntimeError(
                f"Kokoro checkpoint section {section_name!r} is incompatible with the runtime."
            ) from original_error

    normalized_state = (
        {key[7:]: value for key, value in state_dict.items()}
        if all(key.startswith('module.') for key in state_dict)
        else dict(state_dict)
    )
    # torch's parametrization compatibility hook does not migrate legacy
    # weight_g/weight_v keys when the parametrized layer is nested. Normalize
    # them explicitly so the original Kokoro checkpoint stays supported.
    legacy_weight_norm_suffixes = {
        'weight_g': 'parametrizations.weight.original0',
        'weight_v': 'parametrizations.weight.original1',
    }
    for old_suffix, new_suffix in legacy_weight_norm_suffixes.items():
        for key in tuple(normalized_state):
            if key == old_suffix or key.endswith(f'.{old_suffix}'):
                prefix = key[:-len(old_suffix)]
                normalized_state[f'{prefix}{new_suffix}'] = normalized_state.pop(key)
    try:
        module.load_state_dict(normalized_state, strict=True)
        return
    except RuntimeError:
        pass

    # Published Kokoro checkpoints omit the affine parameters of these
    # InstanceNorm layers. Their initialized weight=1/bias=0 values are the
    # identity transform, so explicitly supply only those known defaults. All
    # learned weights and every unexpected key remain subject to strict load.
    model_state = module.state_dict()
    missing_keys = set(model_state).difference(normalized_state)
    optional_instance_norm_keys = {
        f"{name}.{parameter_name}" if name else parameter_name
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.InstanceNorm1d) and child.affine
        for parameter_name in ('weight', 'bias')
    }
    optional_missing = missing_keys.intersection(optional_instance_norm_keys)
    if missing_keys != optional_missing:
        raise RuntimeError(
            f"Kokoro checkpoint section {section_name!r} is incompatible with the runtime."
        ) from original_error
    for key in optional_missing:
        normalized_state[key] = model_state[key]

    try:
        module.load_state_dict(normalized_state, strict=True)
    except RuntimeError as normalized_error:
        raise RuntimeError(
            f"Kokoro checkpoint section {section_name!r} is incompatible with the runtime."
        ) from normalized_error


class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    REPO_ID = 'hexgrad/Kokoro-82M'

    def __init__(self, config: Union[Dict, str, None] = None, model: Optional[str] = None, disable_complex: bool = False):
        super().__init__()
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=KModel.REPO_ID, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=KModel.REPO_ID, filename='kokoro-v1_0.pth')
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            _load_module_state(getattr(self, key), state_dict, key)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.inference_mode()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: Number = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    @torch.inference_mode()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: Number = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: Number = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform
