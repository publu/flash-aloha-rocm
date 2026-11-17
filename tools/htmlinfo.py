#!/usr/bin/env python3
"""Summarise a saved HTML page as data: title, meta tags, links, media, visible text.

Usage: python3 -I htmlinfo.py <page.html> [--text]

The page is parsed with html.parser only; no script is run and nothing is fetched.
"""
import sys
from html.parser import HTMLParser


class Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = []
        self.in_title = False
        self.skip = 0
        self.meta = []
        self.links = []
        self.media = []
        self.scripts = []
        self.text = []
        self.heading = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag in ("script", "style"):
            self.skip += 1
            if tag == "script" and attrs.get("src"):
                self.scripts.append(attrs["src"])
        if tag == "meta":
            key = attrs.get("name") or attrs.get("property") or attrs.get("http-equiv")
            if key:
                self.meta.append((key, attrs.get("content", "")))
        if tag == "a" and attrs.get("href"):
            self.links.append(["a", attrs["href"], ""])
            self.heading = self.links[-1]
        if tag == "link" and attrs.get("href"):
            self.links.append(["link rel=" + str(attrs.get("rel")), attrs["href"], ""])
        if tag in ("img", "video", "source", "iframe", "audio", "embed", "object"):
            src = attrs.get("src") or attrs.get("data") or attrs.get("poster")
            if src:
                self.media.append((tag, src))
        if tag in ("h1", "h2", "h3", "h4", "p", "li", "div", "br", "section", "tr"):
            self.text.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.text.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag == "a":
            self.heading = None

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.skip:
            return
        if self.heading is not None:
            self.heading[2] += data.strip() + " "
        self.text.append(data)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1], "r", encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    parser = Collector()
    parser.feed(source)
    print(f"bytes: {len(source.encode('utf-8'))}")
    print(f"title: {''.join(parser.title).strip()!r}")
    print("meta:")
    for key, value in parser.meta:
        print(f"  {key}: {value[:300]!r}")
    print(f"links ({len(parser.links)}):")
    for kind, href, label in parser.links:
        print(f"  [{kind}] {href}   {label.strip()[:80]!r}")
    print(f"media ({len(parser.media)}):")
    for tag, src in parser.media:
        print(f"  <{tag}> {src}")
    print(f"external scripts ({len(parser.scripts)}): {parser.scripts}")
    if "--text" in argv:
        text = "".join(parser.text)
        lines = [" ".join(line.split()) for line in text.splitlines()]
        lines = [line for line in lines if line]
        print("visible text:")
        for line in lines:
            print("  | " + line[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
