"""Explicit Qwen3.5 adapter for image-conditioned on-policy distillation.

Completion positions are aligned independently of teacher prompt length.
Only completion-predicting states are projected onto the large vocabulary.
"""
from pathlib import Path
import torch
from .hf import HFBackend


class Qwen35Backend(HFBackend):
    multimodal_types = {'qwen3_5'}
    chat_options = {'enable_thinking': False}

    def __init__(self, config, training=False):
        if config.get('text_only', False):
            raise ValueError('Qwen35Backend requires the multimodal checkpoint')
        super().__init__(config, training=training)
        if training:
            for name, parameter in self.model.named_parameters():
                if '.visual.' in name:
                    parameter.requires_grad_(False)

    def identity(self):
        # Do not read multi-GB weights just to construct a diagnostic receipt.
        return {'resolved_revision': self.resolved_revision,
                'files': {kind: [{'name': p.name, 'bytes': p.stat().st_size}
                                for p in sorted(Path(value).glob('*')) if p.is_file()]
                          for kind in ('path', 'adapter')
                          if (value := self.config.get(kind)) and Path(value).is_dir()},
                'integrity_claim': 'revision and file inventory, not cryptographic verification'}

    def response_inputs(self, batch, response):
        response = response.to(self.device)
        n = batch['input_ids'].shape[1]
        if response.shape[1] < 1 or n + response.shape[1] > self.max_context:
            raise ValueError('Empty response or context overflow')
        if 'position_ids' in batch:
            raise ValueError('Static positions are not valid after appending a response')
        inp = dict(batch)
        inp['input_ids'] = torch.cat([batch['input_ids'], response], dim=1)
        inp['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(response)], dim=1)
        for name in ('token_type_ids', 'mm_token_type_ids'):
            if name in inp:
                inp[name] = torch.cat([inp[name], torch.zeros_like(response)], dim=1)
        return inp, torch.arange(n-1, n+response.shape[1]-1, device=self.device)

    def response_logits(self, batch, response):
        inp, positions = self.response_inputs(batch, response)
        return self.model(**inp, use_cache=False, logits_to_keep=positions).logits
