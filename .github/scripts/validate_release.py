"""Validate static source and distribution versions before a release."""

import argparse
import ast
import os
import re
import tarfile
import tomllib
import zipfile
from datetime import date
from email.parser import Parser
from pathlib import Path

BUNDLED_CHECKPOINTS = {
    f"xirt/assets/{name}.pt" for name in ("rp", "scx", "hsax", "multitask", "faims")
}


def validate_contents(names: list[str], metadata: str, *, wheel: bool) -> None:
    """Keep archives out, allowing only the five intentional bundled checkpoints."""
    for name in names:
        parts = Path(name).parts
        relative = parts if wheel else parts[1:]
        benchmark_code = relative[:3] == ("xirt", "data_pipelines", "benchmarks")
        tutorial_output = relative[:2] == ("examples", "results")
        notebook_state = "__marimo__" in relative
        tutorial_in_wheel = wheel and relative[:1] == ("examples",)
        if (
            (
                relative
                and relative[0]
                in {
                    "benchmark-tf-torch",
                    "benchmark-faims",
                    "benchmarks",
                    "documentation",
                    "models",
                    "sample_data",
                }
            )
            or (benchmark_code and (wheel or not name.endswith(".py")))
            or tutorial_output
            or notebook_state
            or tutorial_in_wheel
            or (
                name.endswith((".pt", ".h5", ".hdf5", ".bin"))
                and "/".join(relative) not in BUNDLED_CHECKPOINTS
            )
        ):
            raise SystemExit(f"Benchmark/archive content in distribution: {name}")
    parsed = Parser().parsestr(metadata)
    for requirement in parsed.get_all("Requires-Dist", []):
        package = re.split(r"[\s(<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        if package in {
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tensorflow-addons",
            "keras",
            "snakemake",
        }:
            raise SystemExit(f"Retired dependency in distribution: {package}")


def release_notes(root: Path, version: str, require_date: bool = False) -> str:
    """Extract exactly one version's curated notes; releases need an ISO date."""
    lines = (root / "CHANGELOG.md").read_text().splitlines()
    pattern = re.compile(rf"^## \[{re.escape(version)}\] - (.+)$")
    matches = [(index, match) for index, line in enumerate(lines) if (match := pattern.match(line))]
    if len(matches) != 1:
        raise SystemExit(f"CHANGELOG.md must contain exactly one section for {version}")
    start, match = matches[0]
    stamp = match.group(1)
    if require_date and stamp == "Unreleased":
        raise SystemExit(f"Date the {version} changelog section before publishing (YYYY-MM-DD)")
    if stamp != "Unreleased":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
            raise SystemExit("Changelog release date must be YYYY-MM-DD")
        try:
            date.fromisoformat(stamp)
        except ValueError as error:
            raise SystemExit("Changelog release date must be YYYY-MM-DD") from error
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## [")),
        len(lines),
    )
    if not "\n".join(lines[start + 1 : end]).strip():
        raise SystemExit(f"Empty changelog section for {version}")
    return "\n".join(lines[start:end]).strip() + "\n"


def validate() -> None:
    """Reject mismatched source, tag, wheel, or sdist metadata."""
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    tree = ast.parse((root / "xirt" / "__init__.py").read_text())
    source_versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    if source_versions != [version]:
        raise SystemExit(f"Static xirt.__version__ must equal package version {version!r}")
    tag = os.environ.get("RELEASE_TAG")
    if tag is not None and tag != f"v{version}":
        raise SystemExit(f"Release tag {tag!r} must be v{version}")
    release_notes(root, version, require_date=tag is not None)

    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise SystemExit("Expected one wheel metadata file")
        wheel_metadata = archive.read(metadata_files[0]).decode()
        validate_contents(archive.namelist(), wheel_metadata, wheel=True)
    with tarfile.open(sdists[0]) as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if len(Path(member.name).parts) == 2 and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise SystemExit("Expected one top-level sdist metadata file")
        metadata_stream = archive.extractfile(metadata_files[0])
        if metadata_stream is None:
            raise SystemExit("Source distribution metadata is not a regular file")
        sdist_metadata = metadata_stream.read().decode()
        validate_contents(
            [member.name for member in archive.getmembers() if member.isfile()],
            sdist_metadata,
            wheel=False,
        )
    for metadata in (wheel_metadata, sdist_metadata):
        parsed = Parser().parsestr(metadata)
        if parsed["Version"] != version or parsed["Name"].lower() != project["name"].lower():
            raise SystemExit("Distribution name/version does not match pyproject.toml")
    print(f"Validated {project['name']} {version}" + (f" / {tag}" if tag else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notes-only", action="store_true", help="Print this version's curated notes"
    )
    args = parser.parse_args()
    if args.notes_only:
        root = Path(__file__).resolve().parents[2]
        version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        tag = os.environ.get("RELEASE_TAG")
        if tag is not None and tag != f"v{version}":
            raise SystemExit(f"Release tag {tag!r} must be v{version}")
        print(release_notes(root, version, require_date=tag is not None), end="")
    else:
        validate()
