"""Exercise linked-data census semantics without requiring downloaded archives."""

import zlib

import pandas as pd
import pytest

from xirt.data_pipelines.benchmarks.datasets import source_coverage as coverage


def test_canonical_removes_flanks_without_losing_modified_residues():
    assert coverage.canonical("K.ACcmDMoxK.R") == "ACDMK"
    assert coverage.canonical("ACcmDMoxK") == "ACDMK"
    assert coverage.canonical("") == ""


def test_census_confidence_unordered_ptms_units_and_integrity(tmp_path, monkeypatch):
    path = tmp_path / "table.csv"
    pd.DataFrame(
        {
            "isTT": [True, True, True, False, True],
            "fdr": [0.001, 0.001, 0.001, 0.001, 0.02],
            "Peptide1": ["K.ACcmK.R", "MoxK", "ACK", "AAAA", "AAAA"],
            "Peptide2": ["MoxK", "ACcmK", "MK", "BBBB", "BBBB"],
            "rp": [10.0, 11.0, 12.0, 15.0, 20.0],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(coverage, "ROOT", tmp_path)
    data = path.read_bytes()
    member = {"name": "table.csv", "crc32": zlib.crc32(data), "size": len(data)}
    inventory = {
        "files": [{"name": "source.zip", "checksum": "md5:unverified", "members": [member]}]
    }
    args = (
        path,
        "test",
        "primary",
        ("Peptide1", "Peptide2"),
        "rp",
        60.0,
        "source.zip",
        "table.csv",
        inventory,
    )
    item, pairs = coverage._census(*args)
    assert pairs == {("ACcmK", "MoxK"), ("ACK", "MK")}
    assert item["confident_rows"] == 3
    assert item["modification_aware_identities"] == 2
    assert item["canonical_identities"] == 1
    assert item["rt_seconds_range"] == [600.0, 720.0]
    assert item["finite_rt_rows"] == 3
    assert item["member_crc32_verified"]
    assert not item["archive_checksum_verified"]
    member["crc32"] += 1
    with pytest.raises(ValueError, match="CRC verification"):
        coverage._census(*args)
