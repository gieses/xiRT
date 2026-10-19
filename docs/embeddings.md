# ESM-C embeddings and LoKr

## Frozen local embeddings

Use Python 3.12 and `uv sync --python 3.12 --extra cpu --extra esm` to enable the
tested ESM SDK `>=3.2.3,<3.3` interface. Inference
is local; there is no need to send peptide sequences to a hosted inference
service. Model weights may still need downloading and access authorization
before use. For air-gapped operation, pre-cache authorized weights and set
`HF_HUB_OFFLINE=1`.

```python
from xirt.embeddings import ESMCEncoder

encoder = ESMCEncoder(
    model_name="esmc_300m",
    device="cpu",
    cache_dir="embedding-cache",
    batch_size=8,
)
features = encoder.encode(["PEPTIDE", "ACDEFGHIK"])
assert features.shape == (2, encoder.embedding_dim)
```

The encoder runs its frozen model in evaluation mode without gradients and
returns float32 NumPy features. Mean pooling includes **residue tokens only**:
BOS, EOS, and padding are excluded. Duplicate sequences are reused and optional
disk cache entries are keyed by model/SDK/pooling identity and sequence.
Do not mutate the encoder weights after construction and reuse its cache.
Cache files contain sequences and embeddings; treat them as scientific data,
not anonymous artifacts.

Supported local model names are `esmc_300m` and `esmc_600m`. Empty strings,
unsupported residues/modification notation, and sequences longer than 2046
residues are rejected rather than silently truncated.

## Input limitation

Only uppercase unmodified canonical residue strings and `X` are accepted.
Oxidation, phosphorylation, isotope labels, terminal modifications, crosslink
positions, and linker chemistry are not represented by these inputs. Encoding
two crosslink peptides separately does not tell ESM-C that a linker joins them.
ESM-C protein pretraining does not establish validity for short, modified
peptides or a specific chromatographic system.

Do not silently delete modification annotations to make inputs pass validation.
If stripping modifications is an intentional experiment, preserve the original
sequences and document the transformation, excluded samples, and limitations.

## Using features with xiRTNET

For precomputed features, use `encoder.route: esm-c` and set
`encoder.embedding_dim` to the actual feature width. The retention head expects
feature vectors rather than integer residue IDs on this route. For shared
crosslink heads, compute each peptide's vector separately and supply the pair
in the same sample order. Align targets and identifiers explicitly.

Frozen feature caching is distinct from differentiable end-to-end fine-tuning.
The CLI's default remains recurrent; `encoder.route: esm-c` enables local
feature generation from peptide sequences with configurable `model`, `device`,
`batch_size`, and `cache_dir` in the encoder section. The CLI projects processed
tokens to their unmodified residue letters before calling ESM-C: PTM and
crosslink-residue labels are deliberately absent from these features. Record
this transformation explicitly; it does not preserve chemical-input
equivalence with the recurrent route.
The resulting ESM features are **not PTM-aware**, even when the original
input table contains modification annotations.

```yaml
encoder:
  route: esm-c
  model: esmc_300m
  embedding_dim: 960
  device: cpu
  batch_size: 8
  cache_dir: embedding-cache
```

This is a section fragment for the local 300M model, not a complete YAML.
The CLI obtains the effective embedding width from the loaded encoder; when
using precomputed features through the API, set that width explicitly.

## Parameter-efficient adaptation

Native GRU full/frozen/LoKr/LoRA training is described separately in the
[recurrent fine-tuning guide](fine-tuning.md). The `fine_tuning` section is not
an ESM configuration.

`xirt.adapters.apply_lokr` replaces selected `torch.nn.Linear` modules with a
frozen base plus a trainable low-rank Kronecker-product update. The base weights
remain frozen. The update is initially zero, preserving the starting forward
function. Select targets explicitly; changing unrelated layers is not a
controlled comparison.

The differentiable `ESMCRetentionModel` combines an ESM-C backbone, LoKr
transformer updates, and a retention regressor. Use `ESMCEncoder` for a frozen
baseline without backbone adaptation. In contrast to the frozen NumPy feature
route, its `forward(token_ids, residue_mask)` keeps gradients through adapted
layers, with residue-only pooling.

Record rank, factor, alpha, selected target module names, base-model identity,
and trainable parameter count. Reconstruct the same base and adapter layout
before loading a saved state dictionary. A frozen cache cannot be reused as if
it represented fine-tuned backbone outputs.

Lion and the `experiments` extra are optional. The explicit xiRT LoKr
implementation is not an implicit import of a third-party adapter preset.
It implements linear-layer updates, not convolutional adaptation, DoRA, or
every LyCORIS preset. A benchmark that adapts only retention-head layers must
be labeled **linear-head-only LoKr**, not ESM backbone fine-tuning.
No adapter is claimed to improve retention prediction without measured
held-out results. The [API reference](api.md) documents constructors and
target-selection arguments.

## Licenses and access

xiRT's Apache-2.0 license does **not** override pretrained-model licenses.
Review the exact model card, license, acceptable-use rules, and any access
conditions attached to the weight revision you download. Terms may differ
between model families and revisions; do not assume that an SDK license grants
rights to every model's weights or to commercial use.

Record weight source/revision, terms reviewed, SDK version, and checksum or
immutable identifier. Never commit access tokens, redistribute gated weights
without permission, or upload peptide caches in CI. The manual integration
workflow uses an approved runner with authorized pre-cached weights and does
not upload model data or credentials.
