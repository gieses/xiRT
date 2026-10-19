"""Regression coverage for published documentation and package-description links."""

import runpy
from pathlib import Path

import pytest

check_links = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / ".github/scripts/check_docs_links.py")
)["check_links"]
SITE_URL = "https://gieses.github.io/xiRT/"


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text('<h1 id="intro">xiRT</h1>')
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide/index.html").write_text(
        '<h1 id="fine-tuning">Fine-tuning</h1><a href="../#intro">Home</a>'
    )
    return tmp_path


def check(site, readme="", urls=None):
    return check_links(
        site, SITE_URL, readme, {"Documentation": SITE_URL} if urls is None else urls
    )


def test_absolute_readme_links_resolve_to_generated_site(site):
    assert check(site, f'<a href="{SITE_URL}guide/#fine-tuning">Guide</a>') == []


@pytest.mark.parametrize("attribute", ["href", "src"])
def test_relative_readme_links_are_rejected_for_pypi(site, attribute):
    errors = check(site, f'<a {attribute}="docs/guide.md">Guide</a>')
    assert len(errors) == 1
    assert "absolute HTTPS URL" in errors[0]


def test_missing_readme_destination_is_rejected(site):
    assert "missing destination" in check(site, f'<a href="{SITE_URL}missing/">Bad</a>')[0]


def test_missing_fragment_is_rejected(site):
    assert "missing anchor" in check(site, f'<a href="{SITE_URL}guide/#missing">Bad</a>')[0]


def test_nested_page_links_are_checked(site):
    (site / "guide/index.html").write_text('<a href="../missing/">Bad</a>')
    assert "missing destination" in check(site)[0]


def test_pypi_documentation_url_matches_site(site):
    assert "must match" in check(site, urls={"Documentation": "https://example.org/"})[0]


def test_unbuilt_site_is_rejected(tmp_path):
    assert "No built HTML" in check(tmp_path)[0]
