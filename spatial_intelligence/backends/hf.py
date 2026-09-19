"""Single-device HF backend; frame-explicit multi-image Qwen-VL and causal text.

Custom geometry checkpoints must use their own adapter, not AutoModel fallback.
"""
import torch
from pathlib import Path
from PIL import Image

from ..io import digest, file_digest


class HFBackend:
    multimodal_types = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "internvl", "llava", "llava_next"}
    chat_options = {}
    def __init__(self, config, training=False):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        self.config = config
        self.device = config.get("device", "cuda:0")
        self.text_only = config.get("text_only", False)
        path = config["path"]
        kwargs = {"revision": config.get("revision", "main"), "trust_remote_code": config.get("trust_remote_code", False)}
        auto_cfg = AutoConfig.from_pretrained(path, **kwargs)
        if not self.text_only and auto_cfg.model_type not in self.multimodal_types:
            raise ValueError(f"{auto_cfg.model_type}: supply a tested custom backend; no silent family fallback")
        if self.text_only:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
            cls = AutoModelForCausalLM
        else:
            self.processor = AutoProcessor.from_pretrained(path, **kwargs)
            self.tokenizer = self.processor.tokenizer
            cls = AutoModelForImageTextToText
        self.model = cls.from_pretrained(path, dtype=getattr(torch, config.get("dtype", "bfloat16")),
                                         attn_implementation=config.get("attention", "sdpa"), **kwargs).to(self.device)
        self.resolved_revision = getattr(auto_cfg, "_commit_hash", None)
        if config.get("adapter"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, config["adapter"], is_trainable=training)
        elif training and config.get("lora", {}).get("enabled", False):
            from peft import LoraConfig, get_peft_model
            lc = config["lora"]
            self.model = get_peft_model(self.model, LoraConfig(r=lc["rank"], lora_alpha=lc["alpha"],
                lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=lc["target_modules"]))
        if self.tokenizer.eos_token_id is None:
            raise ValueError("An EOS token is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_context = config.get("max_context", 8192)
        # Disable dropout for reproducible on-policy probabilities; eval() still allows gradients.
        self.model.eval()
        if not training:
            self.model.requires_grad_(False)

    def identity(self):
        files = {}
        for kind in ("path", "adapter"):
            value = self.config.get(kind)
            if value and Path(value).is_dir():
                root = Path(value)
                for path in sorted(root.iterdir()):
                    if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}:
                        files[kind+"/"+path.name] = file_digest(path)
        return {"resolved_revision":self.resolved_revision, "local_files_sha256":files,
                "tokenizer_hash":self.tokenizer_fingerprint()}

    def tokenizer_fingerprint(self):
        return digest({"vocab": self.tokenizer.get_vocab(), "special": self.tokenizer.special_tokens_map,
                       "backend": self.tokenizer.backend_tokenizer.to_str() if hasattr(self.tokenizer, "backend_tokenizer") else None})

    def encode(self, sample, protocol, privileged=None):
        if len(sample["media"]) > protocol["max_frames"]:
            raise ValueError("Frame budget exceeded; create a shared pre-sampled manifest")
        if self.text_only and sample["media"]:
            raise ValueError("Text-only backbone cannot silently discard images")
        prompt = sample["question"]
        if sample.get("choices"):
            prompt += "\n" + "\n".join(f"{k}. {v}" for k, v in sample["choices"].items())
        prompt += "\n" + protocol["instruction"]
        if privileged is not None:
            prompt += "\nTraining-only verified answer: " + str(privileged)
        images = []
        for path in sample["media"]:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((protocol["image_max_side"], protocol["image_max_side"]), Image.Resampling.LANCZOS)
                images.append(im.copy())
        if self.text_only:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.chat_options)
            batch = self.processor(text=[text], images=images or None, return_tensors="pt", padding=False)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if batch["input_ids"].shape[1] + protocol["generation"]["max_new_tokens"] > self.max_context:
            raise ValueError("Context budget exceeded; truncation is forbidden for fairness")
        return batch

    def response_ids(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[-1] != self.tokenizer.eos_token_id:
            ids.append(self.tokenizer.eos_token_id)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def response_logits(self, batch, response):
        response = response.to(self.device)
        n = batch["input_ids"].shape[1]
        if n + response.shape[1] > self.max_context:
            raise ValueError("Completion exceeds context; do not silently truncate supervision")
        inp = dict(batch)
        inp["input_ids"] = torch.cat([batch["input_ids"], response], dim=1)
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        if "token_type_ids" in inp:
            inp["token_type_ids"] = torch.cat([inp["token_type_ids"], torch.zeros_like(response)], dim=1)
        if "position_ids" in inp:
            raise ValueError("Static position_ids require a model-specific adapter")
        output = self.model(**inp, use_cache=False)
        return output.logits[:, n-1:n+response.shape[1]-1, :]

    def sample_ids(self, batch, generation, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        n = batch["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(**batch, **generation,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        return output[:, n:]

    def generate(self, sample, protocol, seed):
        batch = self.encode(sample, protocol)
        ids = self.sample_ids(batch, protocol["generation"], seed)
        return {"response": self.tokenizer.decode(ids[0], skip_special_tokens=True),
                "completion_tokens": ids.shape[1], "input_tokens": batch["input_ids"].shape[1],
                "backend_status": "hf", "resolved_revision": self.resolved_revision}

    def save(self, path):
        self.model.save_pretrained(path)
        (self.processor or self.tokenizer).save_pretrained(path)
