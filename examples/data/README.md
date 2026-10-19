# Tutorial crosslinks

`crosslinks.csv` contains 2,048 observed peptide pairs from
`sample_data/DSS_xisearch_fdr_CSM50percent_minimal.csv`, the E. coli example
distributed with xiRT. See the xiRT publication
(Giese et al., *Nature Communications* 12, 3237, 2021;
DOI: 10.1038/s41467-021-23441-0) for the source experiment.

The subset retains `PSMID`, both peptide sequences, their one-based crosslink
positions, the observed RP time in seconds, `fdr` and `isTT`. It includes only
target-target identifications with `fdr <= 0.01`, canonical uppercase amino
acids and peptides at most 50 residues long. Modified sequences are excluded,
not stripped.

Repeated unordered sequence pairs are collapsed to the first eligible
observation in source order, including alternative link positions. This leaves
4,890 eligible pairs. The subset takes
`default_rng(17).choice(4890, size=2048, replace=False)` in the resulting order.

The notebooks assign 1,536/256/256 pairs to training/validation/test using
NumPy's `default_rng(42).permutation(2048)`. All three partitions remain fixed
across examples and fine-tuning strategies. Individual peptides, proteins and
acquisition runs can overlap between partitions; this is a within-cohort
example, not an external generalization benchmark.

| File | SHA256 |
|---|---|
| Source CSV | `77ede04d50b75a511d1ef84710a9ee709b612aba335b06c18a48c3e591f0307b` |
| Tutorial CSV | `34711b7b018edd9f69b913444017fbe2089f5267fef30051d908d984a1ba990c` |
