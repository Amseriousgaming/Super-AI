"""Retrieval of real, checkable references.

Design rules
  * A `Source` object exists ONLY if a remote API returned it. Nothing is generated or guessed here.
  * Every result says what it is (encyclopedia, preprint, journal record) so the model and the
    user can tell established knowledge from preliminary work.
  * Sources use fixed, trusted hostnames, so user text can never make the server fetch an arbitrary URL.

Add a source by subclassing `KnowledgeSource`, implementing `search`, and listing it in
`SOURCE_CLASSES`. Enable it with RESEARCH_SOURCES=wikipedia,arxiv,crossref,yoursource.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from ai.base import Source
from cache import TTLCache

log = logging.getLogger("psai.knowledge")
_TAG = re.compile(r"<[^>]+>")


def _clean(text: str, limit: int = 500) -> str:
    text = html.unescape(_TAG.sub("", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


class KnowledgeSource(ABC):
    name: str

    @abstractmethod
    async def search(self, client: httpx.AsyncClient, query: str, k: int) -> list[Source]:
        """Return up to k sources (index is assigned later)."""


class WikipediaSource(KnowledgeSource):
    name = "wikipedia"

    async def search(self, client, query, k):
        resp = await client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": k,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 3, "inprop": "url", "redirects": 1})
        resp.raise_for_status()
        pages = (resp.json().get("query") or {}).get("pages") or {}
        out = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            if page.get("fullurl") and page.get("extract"):
                out.append(Source(0, page["title"], page["fullurl"], _clean(page["extract"]), "encyclopedia",
                                  "Wikipedia is a useful overview but is not itself a primary or peer-reviewed source."))
        return out


class ArxivSource(KnowledgeSource):
    name = "arxiv"
    NS = {"a": "http://www.w3.org/2005/Atom"}

    async def search(self, client, query, k):
        resp = await client.get("https://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}", "start": 0, "max_results": k, "sortBy": "relevance"})
        resp.raise_for_status()
        return self.parse(resp.text)

    @classmethod
    def parse(cls, xml_text: str) -> list[Source]:
        out = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return out
        for entry in root.findall("a:entry", cls.NS):
            title = _clean(entry.findtext("a:title", "", cls.NS), 300)
            url = (entry.findtext("a:id", "", cls.NS) or "").strip()
            if not title or not url:
                continue
            authors = [a.findtext("a:name", "", cls.NS) for a in entry.findall("a:author", cls.NS)][:4]
            year = (entry.findtext("a:published", "", cls.NS) or "")[:4]
            byline = ", ".join(a for a in authors if a) + (f" ({year})" if year else "")
            out.append(Source(0, title, url, (byline + ". " if byline else "") + _clean(entry.findtext("a:summary", "", cls.NS), 400),
                              "preprint", "arXiv preprint: not peer reviewed."))
        return out


class CrossrefSource(KnowledgeSource):
    name = "crossref"

    def __init__(self, contact_email: Optional[str] = None):
        self.mailto = contact_email

    async def search(self, client, query, k):
        params = {"query": query, "rows": k, "select": "DOI,title,author,issued,container-title,type"}
        if self.mailto:
            params["mailto"] = self.mailto
        resp = await client.get("https://api.crossref.org/works", params=params)
        resp.raise_for_status()
        return self.parse(resp.json())

    @staticmethod
    def parse(data: dict) -> list[Source]:
        out = []
        for item in (data.get("message") or {}).get("items") or []:
            doi, titles = item.get("DOI"), item.get("title") or []
            if not doi or not titles:
                continue
            authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author") or []][:3]
            year = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
            venue = (item.get("container-title") or [""])[0]
            line = "; ".join(filter(None, [", ".join(a for a in authors if a), venue, str(year) if year else ""]))
            kind = "journal" if item.get("type") == "journal-article" else item.get("type", "record")
            out.append(Source(0, _clean(titles[0], 300), f"https://doi.org/{doi}", line or "Bibliographic record", kind,
                              "Crossref bibliographic record (metadata only; the abstract was not retrieved)."))
        return out


SOURCE_CLASSES = {"wikipedia": WikipediaSource, "arxiv": ArxivSource, "crossref": CrossrefSource}


class KnowledgeService:
    def __init__(self, settings):
        self.settings = settings
        self.sources: dict[str, KnowledgeSource] = {}
        for name in settings.research_source_list:
            if name == "crossref":
                self.sources[name] = CrossrefSource(settings.contact_email)
            elif name in SOURCE_CLASSES:
                self.sources[name] = SOURCE_CLASSES[name]()
            else:
                log.warning("Unknown research source %r ignored", name)
        contact = f" (mailto:{settings.contact_email})" if settings.contact_email else ""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.knowledge_timeout_seconds, connect=5.0),
            headers={"User-Agent": f"PersonalSuperAI/1.0{contact}", "Accept": "application/json, application/atom+xml"},
            follow_redirects=False,
        )
        self._cache = TTLCache(maxsize=256, ttl=settings.knowledge_cache_ttl_seconds or None)

    @property
    def available(self) -> bool:
        return bool(self.sources)

    @property
    def source_names(self) -> list[str]:
        return list(self.sources)

    async def search(self, query: str, source: str = "auto", per_source: int = 3, start_index: int = 1) -> list[Source]:
        query = re.sub(r"\s+", " ", query).strip()[:200]
        if not query:
            return []
        names = self.source_names if source in {"", "auto"} else [s for s in self.source_names if s == source.lower()]
        key = (query.lower(), tuple(names))
        results = self._cache.get(key)
        if results is None:
            batches = await asyncio.gather(*(self._one(n, query, per_source) for n in names))
            results = [s for batch in batches for s in batch]
            if results:
                self._cache.set(key, results)
        seen, out = set(), []
        for s in results:  # de-duplicate and number from start_index; copy so cached objects stay unnumbered
            if s.url in seen:
                continue
            seen.add(s.url)
            out.append(Source(start_index + len(out), s.title, s.url, s.snippet, s.kind, s.note, s.retrieved_at))
        return out

    async def _one(self, name: str, query: str, k: int) -> list[Source]:
        try:
            items = await self.sources[name].search(self._client, query, k)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("Research source %s failed: %s", name, type(exc).__name__)
            return []
        now = time.time()
        for s in items:
            s.retrieved_at = now
        return items

    async def aclose(self) -> None:
        await self._client.aclose()
