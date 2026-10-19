"""Check generated site links and README URLs used by GitHub and PyPI."""

import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Document(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if anchor := attributes.get("id"):
            self.anchors.add(anchor)
        for key in ("href", "src"):
            if reference := attributes.get(key):
                self.links.append(reference)


def check_links(
    site: Path, site_url: str, readme_html: str, project_urls: dict[str, str]
) -> list[str]:
    """Resolve documentation destinations locally, including absolute README links."""
    pages = {path.resolve(): Document(path.read_text()) for path in site.rglob("*.html")}
    if not pages:
        return [f"No built HTML pages in {site}; run mkdocs build --strict first"]
    errors: list[str] = []
    base = urlsplit(site_url)
    if base.scheme != "https" or not base.netloc or not base.path.endswith("/"):
        return ["site_url must be an absolute HTTPS URL ending in /"]

    def check(source: str, origin: str, reference: str, *, absolute: bool = False) -> None:
        parsed = urlsplit(reference)
        if absolute and (parsed.scheme != "https" or not parsed.netloc):
            errors.append(f"{source}: expected an absolute HTTPS URL: {reference}")
            return
        target_url = urlsplit(urljoin(origin, reference))
        if target_url.scheme not in {"http", "https"}:
            return
        if target_url.netloc != base.netloc or not target_url.path.startswith(base.path):
            return
        target = (site / unquote(target_url.path.removeprefix(base.path))).resolve()
        if not target.is_relative_to(site.resolve()):
            errors.append(f"{source}: link escapes the site: {reference}")
            return
        if target.is_dir():
            target /= "index.html"
        if not target.is_file():
            errors.append(f"{source}: missing destination: {reference}")
        elif target_url.fragment and target in pages:
            if unquote(target_url.fragment) not in pages[target].anchors:
                errors.append(f"{source}: missing anchor: {reference}")

    for path, document in pages.items():
        source = path.relative_to(site.resolve()).as_posix()
        origin = urljoin(site_url, source)
        for reference in document.links:
            check(source, origin, reference)
    for reference in Document(readme_html).links:
        check("README / PyPI description", site_url, reference, absolute=True)
    for label, reference in project_urls.items():
        check(f"PyPI project URL ({label})", site_url, reference, absolute=True)
    if project_urls.get("Documentation") != site_url:
        errors.append("PyPI Documentation URL must match mkdocs.yml site_url")
    return errors


def main() -> None:
    import markdown
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "mkdocs.yml").read_text())
    site = root / config.get("site_dir", "site")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    readme_html = markdown.markdown((root / project["readme"]).read_text(), extensions=["tables"])
    errors = check_links(site, config["site_url"], readme_html, project["urls"])
    if errors:
        raise SystemExit("\n".join(errors))
    print("Validated built-site links and anchors, README/PyPI URLs, and project metadata")


if __name__ == "__main__":
    main()
