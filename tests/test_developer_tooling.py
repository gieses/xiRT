"""Curated release notes and publication safeguards; no remote writes."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

import xirt

ROOT = Path(__file__).resolve().parents[1]
release_notes = runpy.run_path(str(ROOT / ".github/scripts/validate_release.py"))["release_notes"]
validate_contents = runpy.run_path(str(ROOT / ".github/scripts/validate_release.py"))[
    "validate_contents"
]


@pytest.mark.parametrize(
    "name",
    [
        "xirt/data_pipelines/benchmarks/common.py",
        "benchmark-tf-torch/cohort/results/archive.json",
        "benchmarks/reference/legacy_tf.py",
        "xirt/model.pt",
        "xirt/assets/extra.pt",
        "xirt/assets/rp.h5",
        "examples/train_predict.py",
    ],
)
def test_production_wheel_rejects_benchmark_and_checkpoint_content(name):
    with pytest.raises(SystemExit, match="content in distribution"):
        validate_contents([name], "Name: xirt-torch\n", wheel=True)


@pytest.mark.parametrize("wheel", [False, True])
def test_distributions_allow_only_named_default_checkpoints(wheel):
    from xirt.assets import MODEL_NAMES

    names = [f"xirt/assets/{name}.pt" for name in MODEL_NAMES]
    if not wheel:
        names = [f"xirt_torch-2.0.1/{name}" for name in names]
    validate_contents(names, "Name: xirt-torch\n", wheel=wheel)
    extra = "xirt/assets/unpublished.pt"
    if not wheel:
        extra = "xirt_torch-2.0.1/" + extra
    with pytest.raises(SystemExit, match="content in distribution"):
        validate_contents([extra], "", wheel=wheel)


@pytest.mark.parametrize(
    "dependency", ["tensorflow", "tensorflow-cpu", "tensorflow-addons", "keras", "snakemake"]
)
def test_distributions_reject_retired_dependencies(dependency):
    with pytest.raises(SystemExit, match="Retired dependency"):
        validate_contents(["xirt/__init__.py"], f"Requires-Dist: {dependency}>=1\n", wheel=True)


def test_only_sdist_can_contain_benchmark_scripts():
    validate_contents(
        ["xirt_torch-2.0.0/xirt/data_pipelines/benchmarks/common.py"], "", wheel=False
    )
    with pytest.raises(SystemExit, match="content in distribution"):
        validate_contents(["xirt_torch-2.0.0/benchmark-tf-torch/source.tar.gz"], "", wheel=False)


def test_sdist_contains_example_sources_but_not_generated_results():
    validate_contents(
        [
            "xirt_torch-2.0.0/examples/train_predict.py",
            "xirt_torch-2.0.0/examples/data/crosslinks.csv",
        ],
        "",
        wheel=False,
    )
    with pytest.raises(SystemExit, match="content in distribution"):
        validate_contents(["xirt_torch-2.0.0/examples/results/predictions.csv"], "", wheel=False)


@pytest.mark.parametrize("wheel", [False, True])
def test_distributions_reject_generated_notebook_state(wheel):
    path = "examples/__marimo__/session/faims.py.json"
    if not wheel:
        path = "xirt_torch-2.0.1/" + path
    with pytest.raises(SystemExit, match="content in distribution"):
        validate_contents([path], "", wheel=wheel)


def test_distribution_name_does_not_change_import_or_cli():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "xirt-torch"
    assert project["scripts"]["xirt"] == "xirt.__main__:main"
    assert version("xirt-torch") == project["version"] == xirt.__version__


def test_published_release_uses_pypi_token_without_a_duplicate_tag_trigger():
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert workflow["on"]["release"]["types"] == ["published"]
    assert "push" not in workflow["on"]
    assert workflow["jobs"]["checks"]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    jobs = workflow["jobs"]
    publishing = jobs["pypi"]
    assert publishing["if"] == "github.event_name == 'release' || inputs.destination == 'pypi'"
    assert set(publishing["needs"]) == {"build", "provenance", "github-release"}
    assert publishing["environment"]["url"] == "https://pypi.org/p/xirt-torch"
    assert "id-token" not in publishing["permissions"]
    required_token = publishing["steps"][0]
    assert required_token["env"]["PYPI_TOKEN"] == "${{ secrets.PYPI_TOKEN }}"
    assert 'if [ -z "$PYPI_TOKEN" ]' in required_token["run"]
    assert "exit 1" in required_token["run"]
    publisher = publishing["steps"][-1]
    assert publisher["with"] == {
        "password": "${{ secrets.PYPI_TOKEN }}",
        "attestations": "false",
    }
    assert publisher["uses"].startswith("pypa/gh-action-pypi-publish@")
    for name in ("checks", "build", "provenance", "github-release", "testpypi"):
        assert "PYPI_TOKEN" not in str(jobs[name])
    assert jobs["testpypi"]["permissions"]["id-token"] == "write"
    assert "password" not in jobs["testpypi"]["steps"][-1]["with"]
    release_steps = str(jobs["github-release"]["steps"])
    assert "gh release upload" in release_steps
    assert "gh release create" not in release_steps
    assert "gh release edit" not in release_steps


@pytest.mark.skipif(shutil.which("bash") is None, reason="Requires Bash")
@pytest.mark.parametrize("token", ["", "test-token-not-a-real-credential"])
def test_publishing_requires_a_token_without_printing_it(token):
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    guard = workflow["jobs"]["pypi"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", guard],
        capture_output=True,
        text=True,
        env={**os.environ, "PYPI_TOKEN": token},
    )
    assert result.returncode == (0 if token else 1)
    if token:
        assert token not in result.stdout + result.stderr
    else:
        assert "Set the PYPI_TOKEN secret" in result.stdout


def test_release_notes_extract_only_the_requested_version(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [2.0.0] - 2026-10-12\n\n### Added\n\n- PyTorch.\n\n"
        "## [1.0.0] - 2021-01-01\n\n- TensorFlow.\n"
    )
    assert release_notes(tmp_path, "2.0.0", require_date=True) == (
        "## [2.0.0] - 2026-10-12\n\n### Added\n\n- PyTorch.\n"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# Changelog\n", "exactly one section"),
        ("## [2.0.0] - Unreleased\n\n- New.\n", "Date the 2.0.0"),
        ("## [2.0.0] - 2026-99-99\n\n- New.\n", "YYYY-MM-DD"),
        ("## [2.0.0] - 20261012\n\n- New.\n", "YYYY-MM-DD"),
        ("## [2.0.0] - 2026-10-12\n", "Empty changelog"),
        ("## [2.0.0] - 2026-10-12\n- New.\n## [2.0.0] - 2026-10-12\n- Duplicate.", "exactly one"),
    ],
)
def test_release_notes_require_a_unique_nonempty_dated_section(tmp_path, text, message):
    (tmp_path / "CHANGELOG.md").write_text(text)
    with pytest.raises(SystemExit, match=message):
        release_notes(tmp_path, "2.0.0", require_date=True)


def test_unreleased_notes_can_be_previewed_without_publishing(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text("## [2.0.0] - Unreleased\n\n- Preview only.\n")
    assert release_notes(tmp_path, "2.0.0").endswith("- Preview only.\n")


@pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None, reason="Requires Make/Bash"
)
def test_make_help_and_release_never_push_a_dirty_tree(tmp_path):
    help_output = subprocess.check_output(["make", "-f", str(ROOT / "Makefile"), "help"], text=True)
    assert all(
        target in help_output
        for target in ("build", "serve-docs", "pre-commit", "release", "notebook")
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nprintf '%s\\n' '2.0.0'\n")
    uv.chmod(0o755)
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = status ]; then printf "%s\\n" " M pending.py"; exit 0; fi\n'
        'printf "%s\\n" "$*" >> "$GIT_ACTION_LOG"\nexit 77\n'
    )
    git.chmod(0o755)
    log = tmp_path / "git-actions.log"
    result = subprocess.run(
        ["make", "-f", str(ROOT / "Makefile"), "release", f"UV={uv}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GIT_ACTION_LOG": str(log),
        },
    )
    assert result.returncode != 0
    assert "commit all intended changes first" in result.stderr
    assert not log.exists()


@pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None, reason="Requires Make/Bash"
)
def test_make_release_creates_github_release_after_pushing_tag(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '  *--notes-only*) printf "%s\\n" "# Release notes" ;;\n'
        '  *) printf "%s\\n" "2.0.0" ;;\nesac\n'
    )
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/sh\ncase "$1" in\n'
        "  status) exit 0 ;;\n"
        "  show-ref) exit 1 ;;\n"
        '  remote) printf "%s\\n" "https://github.com/gieses/xiRT" ;;\n'
        '  *) printf "%s\\n" "git $*" >> "$RELEASE_LOG" ;;\nesac\n'
    )
    gh = bin_dir / "gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "gh $*" >> "$RELEASE_LOG"\ncat >> "$RELEASE_LOG"\n')
    checks = bin_dir / "checks"
    checks.write_text('#!/bin/sh\nprintf "%s\\n" "checks $*" >> "$RELEASE_LOG"\n')
    for executable in (uv, git, gh, checks):
        executable.chmod(0o755)
    log = tmp_path / "release.log"
    subprocess.run(
        [
            "make",
            "-f",
            str(ROOT / "Makefile"),
            "release",
            f"UV={uv}",
            f"GH={gh}",
            f"MAKE={checks}",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "RELEASE_LOG": str(log),
        },
    )
    actions = log.read_text()
    assert actions.index("checks check build") < actions.index("git tag --annotate v2.0.0")
    assert actions.index("git push origin refs/tags/v2.0.0") < actions.index(
        "gh release create v2.0.0"
    )
    assert "--verify-tag" in actions
    assert "--repo https://github.com/gieses/xiRT --notes-file -" in actions
    assert actions.endswith("# Release notes\n")
