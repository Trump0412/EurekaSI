"""Qwen3.5-specific batched processing shared by study training and evaluation.

Separate from the legacy reference trainer: no silent change to prior experiments.
RGB multi-image protocol; depth is downloaded but not silently fed to vanilla VLM.
"""
from PIL import Image
import torch


def messages(row, instruction, answer=None):
    question = row['question']
    if row.get('choices'):
        question += '\nOptions:\n' + '\n'.join(f'{k}. {v}' for k,v in row['choices'].items())
    if instruction:
        question += '\n' + instruction
    result=[{'role':'user','content': [{'type':'image'} for _ in row['media']] + [{'type':'text','text':question}]}]
    if answer is not None:
        result.append({'role':'assistant','content':[{'type':'text','text':str(answer)}]})
    return result


class Collator:
    def __init__(self, processor, max_side=448, max_context=16384, training=True):
        self.processor=processor; self.max_side=max_side; self.max_context=max_context; self.training=training
        self.processor.tokenizer.padding_side='right' if training else 'left'

    def __call__(self, rows):
        texts=[]; prompt_texts=[]; images=[]
        for row in rows:
            inst=row.get('instruction','')
            msg=messages(row,inst)
            prefix=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_texts.append(prefix)
            if self.training:
                text=self.processor.apply_chat_template(messages(row,inst,row['answer']),tokenize=False,add_generation_prompt=False,enable_thinking=False)
                if not text.startswith(prefix):
                    raise ValueError('Training chat template is not prefix-aligned; refusing incorrect labels')
            else:text=prefix
            texts.append(text)
            for path in row['media']:
                with Image.open(path) as src:
                    im=src.convert('RGB');im.thumbnail((self.max_side,self.max_side),Image.Resampling.LANCZOS)
                    images.append(im.copy())
        batch=self.processor(text=texts,images=images or None,return_tensors='pt',padding=True)
        if batch['input_ids'].shape[1]>self.max_context:
            raise ValueError('Context exceeds locked budget; do not truncate images or labels')
        if self.training:
            prompts=self.processor(text=prompt_texts,images=images or None,return_tensors='pt',padding=True)
            lengths=prompts['attention_mask'].sum(-1)
            labels=batch['input_ids'].clone()
            labels[batch['attention_mask']==0]=-100
            for i,n in enumerate(lengths.tolist()):
                if not torch.equal(batch['input_ids'][i,:n],prompts['input_ids'][i,:n]):
                    raise ValueError('Prompt token boundary mismatch')
                labels[i,:n]=-100
            if not bool((labels!=-100).any(-1).all()):raise ValueError('Empty completion supervision')
            batch['labels']=labels
        return dict(batch)


def load_model(path, training=False, freeze_vision=True):
    from transformers import Qwen3_5ForConditionalGeneration
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa')
    if training:
        model.config.use_cache=False
        if freeze_vision:
            for name,param in model.named_parameters():
                if '.visual.' in name: param.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
    else:model.requires_grad_(False);model.eval()
    return model


def completion_loss(model, inputs, reduction='token_mean'):
    inputs=dict(inputs);labels=inputs.pop('labels')
    positions=(labels[:,1:]!=-100).any(0).nonzero().flatten()
    output=model(**inputs,use_cache=False,logits_to_keep=positions)
    targets=labels[:,positions+1]
    if reduction=='sample_mean':
        token_loss=torch.nn.functional.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),
            targets.reshape(-1),ignore_index=-100,reduction='none').reshape_as(targets)
        counts=(targets!=-100).sum(-1)
        if not bool((counts>0).all()):raise ValueError('Empty completion')
        loss=(token_loss.sum(-1)/counts).mean()
    elif reduction=='token_mean':
        loss=torch.nn.functional.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),
                                               targets.reshape(-1),ignore_index=-100)
    else:raise ValueError(reduction)
    return loss,output
