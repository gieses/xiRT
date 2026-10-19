"""Build a reproducible census of retrieved model-ready study tables.

Example:
    >>> canonical("K.ACcmDMoxK.R")
    'ACDMK'
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "benchmarks/.cache/sources"
RESULTS = ROOT / "benchmarks/results"


def canonical(sequence: str) -> str:
    """Remove flanking residues and lowercase modification annotations."""
    body = sequence.split(".")[1] if sequence.count(".") == 2 else sequence
    return "".join(re.findall("[A-Z]", body))


def _body(value: object) -> str:
    """Normalize missing second peptides and flanking-residue syntax."""
    if pd.isna(value):
        return ""
    text = str(value)
    return text.split(".")[1] if text.count(".") == 2 else text


def _census(
    path: Path,
    study: str,
    role: str,
    peptides: tuple[str, str],
    rt: str | None,
    scale: float,
    archive: str | None,
    member: str | None,
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """Count confidence-filtered modification-aware and canonical identities."""
    frame = pd.read_csv(path, low_memory=False)
    mask = frame["isTT"].eq(True) & frame["fdr"].le(0.01)
    confident = frame.loc[mask]
    pairs = {
        tuple(sorted((_body(a), _body(b))))
        for a, b in zip(confident[peptides[0]], confident[peptides[1]], strict=True)
    }
    canonical_pairs = {tuple(sorted(map(canonical, pair))) for pair in pairs}
    data = path.read_bytes()
    result: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "study": study,
        "role": role,
        "rows": len(frame),
        "confident_rows": len(confident),
        "modification_aware_identities": len(pairs),
        "canonical_identities": len(canonical_pairs),
        "sequence_columns": list(peptides),
        "observed_rt_column": rt,
        "rt_multiplier_to_seconds": scale if rt else None,
        "sha256": hashlib.sha256(data).hexdigest(),
        "archive": archive,
        "member": member,
    }
    if rt:
        seconds = pd.to_numeric(confident[rt], errors="raise") * scale
        result["finite_rt_rows"] = int(np.isfinite(seconds).sum())
        result["rt_seconds_range"] = [float(seconds.min()), float(seconds.max())]
    if archive:
        entry = next(item for item in inventory["files"] if item["name"] == archive)
        info = next(item for item in entry["members"] if item["name"] == member)
        crc = zlib.crc32(data) & 0xFFFFFFFF
        if crc != info["crc32"] or len(data) != info["size"]:
            raise ValueError(f"Retrieved archive member failed size/CRC verification: {path}")
        result.update(
            archive_published_checksum=entry["checksum"],
            member_crc32=crc,
            member_crc32_verified=True,
            archive_checksum_verified=False,
        )
    return result, pairs


def main() -> None:
    """Write source hashes, exact identity overlaps, exclusions and attribution."""
    inventory = json.loads((RESULTS / "sources-inventory.json").read_text())
    specifications = [
        (
            "ecoli/ecoli/DSS_CSM_anpxirt.csv",
            "ecoli",
            "primary",
            ("Peptide1", "Peptide2"),
            "rp",
            1.0,
            "ecoli.zip",
            "ecoli/DSS_CSM_anpxirt.csv",
        ),
        (
            "ecoli_plink2/ecoli_plink2/plink_qval050.csv",
            "ecoli",
            "alternate_search_engine",
            ("ModSeq1", "ModSeq2"),
            "rp",
            60.0,
            "ecoli_plink2.zip",
            "ecoli_plink2/plink_qval050.csv",
        ),
        (
            "FA_complex/FA_complex/xifdr_results80p/FDR_CSM_xiFDR2.1.3.csv",
            "fa_complex",
            "crosslinked",
            ("PepSeq1", "PepSeq2"),
            "ElutionStart",
            1.0,
            "FA_complex.zip",
            "FA_complex/xifdr_results80p/FDR_CSM_xiFDR2.1.3.csv",
        ),
        (
            "FA_complex/FA_complex/xifdr_results80p/FDR_Linear_PSM_xiFDR2.1.3.csv",
            "fa_complex",
            "linear",
            ("PepSeq1", "PepSeq2"),
            "ElutionStart",
            1.0,
            "FA_complex.zip",
            "FA_complex/xifdr_results80p/FDR_Linear_PSM_xiFDR2.1.3.csv",
        ),
        (
            "paper/FigS2_source_cde_part2.csv",
            "ecoli",
            "linear",
            ("Peptide1", "Peptide2"),
            "ElutionStart",
            1.0,
            None,
            None,
        ),
        (
            "paper/Fig2def_Fig4a_psms.csv",
            "ecoli",
            "duplicate_figure_export",
            ("Peptide1", "Peptide2"),
            "rp",
            60.0,
            None,
            None,
        ),
        (
            "fa_complex_ecoli_mzidentML/ecoli_initial/ecoli_1pPPIFDR_initial_CSM_xiFDR2.1.5.csv",
            "ecoli",
            "no_observed_rt",
            ("PepSeq1", "PepSeq2"),
            None,
            1.0,
            "fa_complex_ecoli_mzidentML.zip",
            "ecoli_initial/ecoli_1pPPIFDR_initial_CSM_xiFDR2.1.5.csv",
        ),
        (
            "fa_complex_ecoli_mzidentML/fa_complex_initial/fa_complex_1pRPFDR_initial_CSM_xiFDR2.1.5.csv",
            "fa_complex",
            "no_observed_rt",
            ("PepSeq1", "PepSeq2"),
            None,
            1.0,
            "fa_complex_ecoli_mzidentML.zip",
            "fa_complex_initial/fa_complex_1pRPFDR_initial_CSM_xiFDR2.1.5.csv",
        ),
    ]
    sources, identities = [], {}
    for name, study, role, peptides, rt, scale, archive, member in specifications:
        item, pairs = _census(
            CACHE / name,
            study,
            role,
            peptides,
            rt,
            scale,
            archive,
            member,
            inventory,
        )
        sources.append(item)
        identities[name] = pairs
    for name in (
        "DSS_xisearch_fdr_CSM50percent_minimal.csv",
        "DSS_xisearch_fdr_CSM50percent_transfer_scx17to23_hsax2to9_minimal.csv",
    ):
        item, pairs = _census(
            ROOT / "sample_data" / name,
            "ecoli",
            "repository_view",
            ("Peptide1", "Peptide2"),
            "rp",
            1.0,
            None,
            None,
            inventory,
        )
        sources.append(item)
        identities[name] = pairs
    primary = identities[specifications[0][0]]
    local = identities["DSS_xisearch_fdr_CSM50percent_minimal.csv"]
    comparisons = [
        {
            "first": first,
            "second": second,
            "overlap": len(a & b),
            "first_only": len(a - b),
            "second_only": len(b - a),
            "union": len(a | b),
        }
        for first, a in identities.items()
        for second, b in identities.items()
        if first < second and a & b
    ]
    paper_zip = CACHE / "paper/source-data.zip"
    with zipfile.ZipFile(paper_zip) as stream:
        corrupt = stream.testzip()
        if corrupt:
            raise ValueError(f"Publisher archive failed CRC verification: {corrupt}")
    paper_inventory = json.loads((RESULTS / "paper-source-inventory.json").read_text())
    publisher_tables = []
    for item in paper_inventory:
        name = Path(item["name"]).name
        if name == "FigS2_source_cde_part2.csv":
            role = "included_linear"
        elif name == "Fig2def_Fig4a_psms.csv":
            role = "duplicate_primary"
        elif name in {"Fig4a_jointplot_source.csv", "Fig4a_source.csv", "Fig4b_CSMs.csv"}:
            role = "post_xirt_rescored_identifications"
        elif {"PepSeq1", "Peptide1"} & set(item["columns"]):
            role = "identification_without_observed_rt"
        else:
            role = "derived_statistics_predictions_or_interactions"
        publisher_tables.append({"name": item["name"], "role": role, "crc32": item["crc32"]})
    exclusions = {
        "duplicate_figure_export": "Same peptide pairs as primary data; do not count as replication.",
        "no_observed_rt": "Identification exports contain no observed retention-time target.",
        "plink2_xirt_processed_input_refined.csv": "Derived RT-error features are outputs of historical xiRT; use original pLink2 table.",
        "FigS15_source_abc.csv": "Predictions/errors only; not independent measured targets.",
        "post_xirt_rescored_identifications": "Selection after xiRT rescoring risks circular RT evaluation.",
        "spectra_fasta_mzid_ppi_models_figures": "Raw/supporting artifacts, not additional ready-to-train observed peptide/RT tables.",
    }
    report = {
        "paper_doi": "10.1038/s41467-021-23441-0",
        "zenodo_concept": 4270323,
        "zenodo_version": inventory["record"],
        "raw_accessions": {"ecoli": "PXD020407 / JPST000916", "fa_complex": "PXD014282"},
        "biological_studies": 2,
        "confidence_filter": "isTT == True and fdr <= 0.01",
        "identity_definition": "unordered modification-aware peptide bodies; no link-position key",
        "primary_repository_union_identities": len(primary | local),
        "sources": sources,
        "identity_overlaps": comparisons,
        "exclusions": exclusions,
        "publisher_tables": publisher_tables,
        "publisher_archive_crc_verified": True,
        "publisher_source_archive_sha256": hashlib.sha256(paper_zip.read_bytes()).hexdigest(),
        "archive_count": sum(bool(item["members"]) for item in inventory["files"]),
        "archive_member_count": sum(len(item["members"]) for item in inventory["files"]),
        "integrity_note": "Retrieved Zenodo members checked by size and CRC32; published whole-ZIP "
        "MD5 recorded but not verified for range-only downloads.",
    }
    (RESULTS / "source-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Linked-data coverage",
        "",
        "Study: DOI `10.1038/s41467-021-23441-0`; Zenodo concept `4270323`, "
        "version `4720760`. Raw accessions: E. coli `PXD020407` / `JPST000916`; "
        "FA complex `PXD014282`.",
        "",
        f"Inventoried **{report['archive_count']} archives / "
        f"{report['archive_member_count']} members** plus the publisher's source-data ZIP. "
        "There are **two biological studies**, not ten independent datasets. Alternate search "
        "engines, repository subsets, linear peptides and figure exports reuse acquisitions.",
        "",
        "Counts below use target-target identifications at FDR <= 1%, modification-aware "
        "unordered peptide bodies (flanking residues removed). Canonical ESM identities collapse "
        "PTMs; link-site-distinct observations are not separate sequence identities.",
        "",
        "| Table | Role | Rows | Confident rows | Sequence identities | Canonical |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in sources:
        lines.append(
            f"| `{Path(item['path']).name}` | {item['study']} / {item['role']} | "
            f"{item['rows']} | {item['confident_rows']} | "
            f"{item['modification_aware_identities']} | {item['canonical_identities']} |"
        )
    lines.extend(
        [
            "",
            f"Primary/repository E. coli union: **{len(primary | local)} identities**. "
            "Exact pair overlaps and file SHA-256 hashes are preserved in `source-coverage.json`. "
            "Do not add overlapping counts to estimate sample size.",
            "",
            "## Exclusions",
            "",
            *[f"- **{key}**: {reason}" for key, reason in exclusions.items()],
            "",
            "## Integrity and scope",
            "",
            report["integrity_note"],
            "Publisher ZIP and extracted tables have local SHA-256 fingerprints. "
            "The member catalogue and publisher CSV schema catalogue are in "
            "`sources-inventory.json` and `paper-source-inventory.json`. All "
            f"{len(publisher_tables)} publisher CSVs have explicit dispositions in the JSON census. "
            "The three Fig4a/Fig4b exports include historical xiRT-derived error features and "
            "rescored selection, not independent holdout observations. Raw spectra were "
            "catalogued rather than re-searched; 'all linked data' does not imply "
            "reprocessing instrument files or training on derived model predictions.",
            "",
        ]
    )
    (RESULTS / "source-coverage.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
