"""GeoWire original transport/checkpoint adapter with strict graph identity checks."""
import json
import sys
from pathlib import Path

import torch

from .hf import HFBackend
from ..io import file_digest


class GeoWireBackend(HFBackend):
    def __init__(self, config, training=False):
        options = config.get("options", {})
        allowed = {"source_root", "checkpoint", "cache_root", "blocks", "phase"}
        if set(options)-allowed:
            raise ValueError(f"Unknown GeoWire options: {set(options)-allowed}")
        if not training and not options.get('checkpoint'):
            raise ValueError('GeoWire inference requires a trained options.checkpoint')
        if options.get("source_root") == "workspace":
            from ..workspace import settings
            source=Path(settings()["external"])/"geowire"/"geowire"
        else:
            source = Path(options.get("source_root", "legacy_sources/GeoWire/geowire")).resolve()
        sys.path.insert(0, str(source))
        from geowire.models.geowire import GeoWireTransport
        from geowire.models.qwen3vl_bridge import Qwen3VLGeoWireForConditionalGeneration
        if config.get("adapter"):
            raise ValueError("GeoWire uses options.checkpoint, not HF adapter")
        base_cfg = {**config, "options": {}}
        super().__init__(base_cfg, training=True)
        self.config = config
        if self.model.config.model_type not in {"qwen3_vl"}:
            raise ValueError("This GeoWire bridge is Qwen3-VL only")
        if not config["lora"]["enabled"]:
            self.model.requires_grad_(False)
        transport = GeoWireTransport(self.model.config.text_config.hidden_size,
                                     num_blocks=options.get("blocks", 2)).to(self.device)
        self.model = Qwen3VLGeoWireForConditionalGeneration(self.model, transport, None)
        if options.get("checkpoint"):
            state = torch.load(options["checkpoint"], map_location="cpu", weights_only=True)
            if options.get("phase", 2) == 1:
                transport.load_state_dict(state.get("model", state), strict=True)
            else:
                transport.load_state_dict(state["geowire"], strict=True)
                qwen_state = state["qwen_trainable"]
                expected = {k for k in self.model.base_model.state_dict() if "lora_" in k}
                if set(qwen_state) != expected:
                    raise ValueError("LoRA state mismatch: use the original rank, alpha and target modules")
                self.model.base_model.load_state_dict(qwen_state, strict=False)
        self.model.eval()
        if not training:
            self.model.requires_grad_(False)

    def encode(self, sample, protocol, privileged=None):
        from geowire.geometry.graph_io import load_graph_npz
        batch = super().encode(sample, protocol, privileged)
        if protocol["geometry_condition"] != "normal":
            raise ValueError("GeoWire adapter currently supports normal only; ablations require explicit graph construction")
        # Namespaced paths prevent cache collisions across datasets.
        clip = Path(self.config["options"]["cache_root"]) / sample["dataset"] / sample["id"]
        contract = json.loads((clip / "graph_contract.json").read_text(encoding="utf-8"))
        expected = {"media_sha256": [file_digest(p) for p in sample["media"]],
                    "frame_indices": sample["frame_indices"], "image_max_side": protocol["image_max_side"],
                    "image_grid_thw": batch["image_grid_thw"].tolist(),
                    "graph_sha256": file_digest(clip / "graph_coo.npz")}
        if any(contract.get(k) != v for k,v in expected.items()):
            raise ValueError("Graph/cache contract mismatch; rebuild for the exact frame and processor layout")
        batch["graph"] = load_graph_npz(clip / "graph_coo.npz")
        return batch

    def save(self, path):
        from geowire.training.train_sft import save_phase2_adapters
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_phase2_adapters(path / "adapter.pt", self.model)
        self.processor.save_pretrained(path)

    def identity(self):
        result = super().identity()
        checkpoint = self.config["options"].get("checkpoint")
        result["geometry_checkpoint_sha256"] = file_digest(checkpoint) if checkpoint else None
        return result
