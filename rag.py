"""
Live retrieval-augmented generation.

The debate topic is arbitrary and user-supplied (not a fixed domain), and
debates often reference current events, so we don't pre-build a static
knowledge base. Instead, for each topic we:
  1. Search the live web (DuckDuckGo, falling back to Tavily if it fails
     or returns nothing) for the topic.
  2. Chunk the returned snippets/pages.
  3. Embed the chunks into a throwaway in-memory Chroma collection.
  4. Retrieve the top-k chunks most relevant to what the agent is arguing.

This keeps the app correct on live/current-event topics without ever going
stale, since nothing is indexed ahead of time.
"""

import logging
import os
import uuid

import chromadb
import requests
from ddgs import DDGS

logger = logging.getLogger(__name__)

_client = chromadb.Client()  # in-memory, per-process


def _chunk(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Primary search: reliable, real-time, generous free tier. No-op
    (returns []) if TAVILY_API_KEY isn't configured, so callers can try it
    unconditionally and fall through to DuckDuckGo."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # Normalize to the same shape ddgs returns (title/href/body) so
        # callers don't need to know which provider actually answered.
        return [
            {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
            for r in results
        ]
    except Exception:
        # Covers Tavily's free-tier quota being exhausted (HTTP 429/432)
        # as well as any other transient failure — either way we fall
        # through to DuckDuckGo below.
        logger.exception("Tavily search failed for query=%r; falling back to DuckDuckGo", query)
        return []


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        return list(DDGS().text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed for query=%r", query)
        return []


def _live_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily first (reliable, real-time, needs TAVILY_API_KEY); if it's
    not configured, fails, or its free-tier quota is exhausted, fall back
    to DuckDuckGo (free, unauthenticated, no key needed)."""
    results = _tavily_search(query, max_results)
    if results:
        return results
    return _ddg_search(query, max_results)


def build_context(topic: str, angle: str = "", k: int = 4) -> dict:
    """
    Live-search `topic` (optionally narrowed by `angle`, e.g. "risks of"),
    embed the results, and return the top-k most relevant chunks plus their
    source URLs so they can be shown in the UI and cited in the PDF.

    Returns {"context": str, "sources": list[{"title","url"}]}.
    On search/embedding failure, returns an empty context (caller falls
    back to the LLM's own knowledge rather than crashing the round).
    """
    query = f"{angle} {topic}".strip()
    results = _live_search(query)
    if not results:
        return {"context": "", "sources": []}

    collection = _client.create_collection(name=f"debate-{uuid.uuid4().hex}")

    docs, metadatas, ids = [], [], []
    for i, r in enumerate(results):
        body = r.get("body") or ""
        title = r.get("title") or ""
        url = r.get("href") or ""
        for j, chunk in enumerate(_chunk(f"{title}. {body}")):
            docs.append(chunk)
            metadatas.append({"title": title, "url": url})
            ids.append(f"{i}-{j}")

    if not docs:
        _client.delete_collection(collection.name)
        return {"context": "", "sources": []}

    collection.add(documents=docs, metadatas=metadatas, ids=ids)

    try:
        found = collection.query(query_texts=[query], n_results=min(k, len(docs)))
    except Exception:
        logger.exception("Chroma retrieval failed for query=%r", query)
        _client.delete_collection(collection.name)
        return {"context": "", "sources": []}

    _client.delete_collection(collection.name)

    chunks = found["documents"][0]
    metas = found["metadatas"][0]

    context = "\n".join(f"- {c}" for c in chunks)
    seen_urls = set()
    sources = []
    for m in metas:
        if m["url"] and m["url"] not in seen_urls:
            seen_urls.add(m["url"])
            sources.append({"title": m["title"], "url": m["url"]})

    return {"context": context, "sources": sources}
