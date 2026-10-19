"""Exercise actual xiRT CUDA training and optional local ESM-C integration."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from xirt.xirtnet import xiRTNET


def main() -> None:
    """Run small integration checks without persisting or uploading scientific data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm", action="store_true")
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This integration runner must provide CUDA-enabled PyTorch and a GPU.")
    root = Path(__file__).resolve().parents[2]
    params = yaml.safe_load((root / "docs/examples/xirt-rp.yaml").read_text())
    params["learning"]["device"] = "cuda"
    params["learning"]["deterministic"] = False
    network = xiRTNET(params, input_dim=6, embedding_dim=8)
    network.build_model(alphabet=21)
    network.compile()
    model = network.model
    if model is None:
        raise RuntimeError("xiRT failed to build its network")
    rng = np.random.default_rng(42)
    features = rng.integers(1, 21, size=(8, 6))
    labels = np.linspace(0, 1, 8, dtype=np.float32).reshape(-1, 1)
    history = model.fit(features, labels, epochs=2, batch_size=4, verbose=0)
    predictions = model.predict(features)
    metrics = model.evaluate(features, labels)
    assert next(model.parameters()).device.type == "cuda"
    assert len(history.history["loss"]) == 2
    assert np.isfinite(np.asarray(predictions)).all()
    assert np.isfinite(np.asarray(metrics)).all()

    if options.esm:
        from xirt.embeddings import ESMCEncoder, ESMCRetentionModel

        sequences = ["PEPTIDE", "ACDEFGHIK"]
        encoder = ESMCEncoder(model_name="esmc_300m", device="cuda", batch_size=2)
        embeddings = encoder.encode(sequences)
        assert embeddings.shape == (2, encoder.embedding_dim)
        assert np.isfinite(embeddings).all()
        assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
        # Reuse locally loaded weights; this smoke check does not create an embedding cache.
        adapted = ESMCRetentionModel(device="cuda", model=encoder.model)
        tokens, mask = adapted.tokenize(sequences)
        result = adapted(tokens, mask)
        result.square().mean().backward()
        assert result.shape == (2, 1) and torch.isfinite(result).all()
        assert any(
            "lokr" in name and parameter.grad is not None
            for name, parameter in adapted.named_parameters()
        )
    print("xiRT CUDA" + (" and local ESM-C/LoKr" if options.esm else "") + " smoke passed")


if __name__ == "__main__":
    main()
