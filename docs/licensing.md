# Licensing and attribution

**The xiRT project code and all five bundled pretrained model checkpoints are
licensed under Creative Commons Attribution-NonCommercial 4.0 International
(CC-BY-NC-4.0)**, subject to the preserved terms below.

The authoritative terms are the repository's
[full license (`LICENCE`)](https://github.com/gieses/xiRT/blob/master/LICENCE)
and [scope and attribution notice (`NOTICE`)](https://github.com/gieses/xiRT/blob/master/NOTICE).
Both are included in wheels and source distributions. The
[official CC BY-NC 4.0 deed](https://creativecommons.org/licenses/by-nc/4.0/) is
a summary; the [legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
governs.

## What the license covers

| Material | Current license or applicable terms |
| --- | --- |
| xiRT project code | CC-BY-NC-4.0, except separately identified third-party material and historical artifacts |
| Bundled `rp.pt`, `scx.pt`, `hsax.pt`, `multitask.pt` and `faims.pt` | CC-BY-NC-4.0; prior Apache-2.0 grants remain valid |
| Chromatography training data, Zenodo record 4720760 | CC-BY-4.0 as stated by the data source, not relicensed by xiRT |
| Upstream xiFAIMS source at the pinned revision | Apache-2.0; upstream terms and notices are preserved |
| External dependencies and pretrained weights, including ESM | Their own licenses, model cards and access conditions |
| Historical benchmark/source snapshots and previously distributed copies | Their original notices and license grants |

`Predictor.load()` loads the bundled joint model; selecting `rp`, `scx`,
`hsax` or `faims` does not change the bundled-model license.
`xirt/assets/manifest.json` records CC-BY-NC-4.0 for each checkpoint alongside
its source attribution, exact byte count and SHA-256. The licensing update
does not change the model weights.

## Noncommercial use and sharing

CC BY-NC 4.0 permits reproduction, sharing and adaptation for **noncommercial
purposes**, subject to its conditions. NonCommercial means use not primarily
intended for or directed towards commercial advantage or monetary compensation.
Do not assume that every research use is noncommercial or that organization type
alone determines whether a use qualifies. Commercial use requires separate
permission from the rights holders unless another applicable grant or legal
exception permits it.

When sharing code or model weights, including modified versions:

- Retain the supplied creator attribution, copyright notice, license notice,
  warranty disclaimer and project link where reasonably practicable.
- Include the license text or a link to it.
- Indicate your modifications and retain indications of previous modifications,
  including changes to model weights.
- Do not impose restrictions that prevent recipients from exercising the rights
  granted by the license.

The license does not grant patent or trademark rights and provides the material
without warranties. Legal exceptions and limitations are unaffected. This page
is a practical summary, not legal advice or additional license terms.

## Attribution

A suggested attribution for sharing unmodified xiRT code or bundled weights is:

> xiRT by Sven Giese and Ludwig Sinn, https://github.com/gieses/xiRT,
> licensed under CC-BY-NC-4.0.

For modified material, add a description of your changes. Preserve the
checkpoint's source attribution from its manifest as well. When describing
scientific use, cite:

Giese, S.H., Sinn, L.R., Wegner, F. et al. Retention time prediction using neural
networks increases identifications in crosslinking mass spectrometry.
*Nature Communications* **12**, 3237 (2021).
[DOI: 10.1038/s41467-021-23441-0](https://doi.org/10.1038/s41467-021-23441-0).

See the [chromatography model cards](chromatography-models.md) and
[FAIMS guide](faims.md) for training provenance and intended-use limitations.

## Prior grants and third-party terms

**This change cannot withdraw Apache-2.0 permissions already granted for
previously distributed code or model checkpoints.** Those copies remain
usable under their original terms, including the same checkpoint bytes
previously distributed under Apache-2.0.

The previous license and copyright notice are preserved in
[`LICENSES/Apache-2.0.txt`](https://github.com/gieses/xiRT/blob/master/LICENSES/Apache-2.0.txt).
That file preserves historical and upstream terms; it is **not** an alternative
license offer for new xiRT material.

This update does not relicense source datasets, dependencies, external model
weights or historical source snapshots. Check their own terms before reuse or
redistribution. Installing an optional ESM extra does not grant access or rights
to gated weights; see [external model licensing and access](embeddings.md#licenses-and-access).

!!! note "Software licensing caveat"
    Creative Commons [recommends against using CC licenses for software](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software).
    The project nevertheless applies CC BY-NC 4.0 to both its code and bundled
    models. Do not describe this noncommercial grant as a permissive software
    license or assume that it includes Apache-2.0's patent grant.
