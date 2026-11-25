"""Check preferred canonical metadata in a completed GitHub Pages channel."""

from __future__ import annotations

import argparse
import os
from html.parser import HTMLParser
from pathlib import Path


class Metadata(HTMLParser):
    """Collect canonical and Open Graph URLs without executing page scripts."""

    def __init__(self) -> None:
        super().__init__()
        self.canonicals: list[str | None] = []
        self.social_urls: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Retain only the URL metadata relevant to channel consolidation."""
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href"))
        if tag == "meta" and attributes.get("property") == "og:url":
            self.social_urls.append(attributes.get("content"))


def check(directory: Path, base_url: str) -> int:
    """Reject duplicate or wrong-channel metadata while excluding TypeDoc."""
    if base_url not in ("/Skulk/", "/Skulk/next/"):
        raise ValueError("Expected a stable or next Pages base URL")
    preferred = "https://docs.foxlight.ai" + base_url.replace("/Skulk/", "/skulk/", 1)
    count = 0
    for page in directory.rglob("*.html"):
        relative = page.relative_to(directory)
        if "typedoc" in relative.parts or relative.name == "404.html":
            continue
        route = (relative.parent if relative.name == "index.html" else relative.with_suffix("")).as_posix()
        if route == ".":
            route = ""
        expected = preferred + (route + "/" if route else "")
        metadata = Metadata()
        metadata.feed(page.read_text())
        if metadata.canonicals != [expected] or metadata.social_urls != [expected]:
            raise ValueError(f"Incorrect preferred metadata: {relative}")
        count += 1
    if not count:
        raise ValueError("No rendered documentation pages found")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("website/build"))
    parser.add_argument("--base-url", default=os.environ.get("DOCS_BASE_URL", "/Skulk/"))
    arguments = parser.parse_args()
    print(f"Preferred canonical metadata passed for {check(arguments.directory, arguments.base_url)} pages")
