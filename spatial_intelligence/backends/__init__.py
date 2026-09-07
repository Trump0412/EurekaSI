from ..io import symbol


def create(config, training=False):
    backend = config.get("backend", "hf")
    if backend == "hf":
        from .hf import HFBackend
        return HFBackend(config, training)
    if backend == "mock":
        if training:
            raise ValueError("Mock backend is inference-only; never use for model results")
        return MockBackend(config)
    return symbol(backend)(config, training=training)


class MockBackend:
    def __init__(self, config):
        self.config = config

    def generate(self, sample, protocol, seed):
        # Constant response independent of gold: plumbing test only.
        return {"response": "<answer>A</answer>", "completion_tokens": 1,
                "input_tokens": 0, "backend_status": "mock_not_model_result"}
