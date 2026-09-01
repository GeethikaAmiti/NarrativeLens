"""
NarrativeLens news retrieval — v2 patch.

This patch changes ONLY retrieval.

Why v1 still failed:
- GDELT timed out on the local machine.
- Google News RSS returned plenty of items, but its links remained
  news.google.com wrapper URLs, so article extraction still returned 0 bodies.

Fix in v2:
- Google News RSS is the primary provider.
- Wrapper URLs are decoded to the publisher's real article URL with the
  lightweight `googlenewsdecoder` package.
- Publisher identity is used for early dedupe, never news.google.com.
- GDELT remains an optional fallback with a longer timeout.
- Diagnostics show decoded URLs and extracted body counts.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse

import requests

REQUEST_TIMEOUT = 12
GDELT_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 NarrativeLens/1.0"
)

TARGET_EXTRACTED = 12
MIN_TEXT_CHARS = 250


def _make_id(url: str) -> str:
    return hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:10]


def _domain(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", urlparse(url).netloc.lower())
    except Exception:
        return ""


def _source_key(source: str) -> str:
    return re.sub(r"\s+", " ", (source or "").strip().lower())


def _parse_rss_time(entry) -> str:
    import calendar
    try:
        if getattr(entry, "published_parsed", None):
            ts = calendar.timegm(entry.published_parsed)
            return datetime.utcfromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        pass

    from .utils import now_iso
    return now_iso()


def _parse_gdelt_date(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).isoformat(timespec="seconds")
        except Exception:
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
            tzinfo=None
        ).isoformat(timespec="seconds")
    except Exception:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


class NewsProvider:
    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GoogleNewsRSSProvider(NewsProvider):
    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        import feedparser

        feed_url = (
            f"https://news.google.com/rss/search?q={quote_plus(query)}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )

        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            print(f"[retrieval] Google News RSS request failed: {exc}")
            return []

        results = []
        for entry in feed.entries[:max_results]:
            try:
                src_field = entry.get("source")
                source = (
                    src_field.get("title")
                    if isinstance(src_field, dict)
                    else None
                )
                source = (source or "").strip()
                wrapper_url = entry.get("link", "")

                title = entry.get("title", "").strip()
                if source and title.endswith(" - " + source):
                    title = title[: -(len(source) + 3)].strip()
                elif " - " in title:
                    title = title.rsplit(" - ", 1)[0].strip()

                results.append(
                    {
                        "id": _make_id(wrapper_url or entry.get("id", "")),
                        "title": title,
                        "url": wrapper_url,
                        "source": source or "Unknown source",
                        "published_at": _parse_rss_time(entry),
                    }
                )
            except Exception:
                continue

        return results


class GDELTProvider(NewsProvider):
    ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        # Keep this deliberately small. GDELT is fallback only.
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": min(max(max_results, 20), 50),
        }

        try:
            resp = requests.get(
                self.ENDPOINT,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=GDELT_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"[retrieval] GDELT request failed: {exc}")
            return []

        results = []
        for item in payload.get("articles", []):
            url = (item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                {
                    "id": _make_id(url),
                    "title": (item.get("title") or "").strip(),
                    "url": url,
                    "source": (item.get("domain") or _domain(url) or "Unknown source").strip(),
                    "published_at": _parse_gdelt_date(
                        item.get("seendate") or item.get("date") or ""
                    ),
                }
            )
            if len(results) >= max_results:
                break
        return results


def _decode_google_news_url(url: str) -> str:
    """Decode a news.google.com RSS/read wrapper into the publisher URL."""
    if not url or "news.google.com" not in _domain(url):
        return url

    try:
        from googlenewsdecoder import gnewsdecoder

        decoded = gnewsdecoder(url, interval=0.4)
        if isinstance(decoded, dict) and decoded.get("status"):
            real = decoded.get("decoded_url")
            if real and real.startswith(("http://", "https://")):
                return real
    except Exception as exc:
        print(f"[retrieval] Google URL decode failed: {type(exc).__name__}: {exc}")

    return url


def _extract_html(html: str, url: str) -> Dict[str, Optional[str]]:
    text = ""
    image_url = None

    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            url=url,
        ) or ""
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        tag = (
            soup.find("meta", property="og:image")
            or soup.find("meta", attrs={"name": "og:image"})
            or soup.find("meta", attrs={"name": "twitter:image"})
        )
        if tag and tag.get("content"):
            image_url = tag["content"].strip()
    except Exception:
        pass

    return {"text": text, "image_url": image_url}


def extract_article(url: str) -> Dict[str, Optional[str]]:
    """Fetch a publisher URL and extract article text + OpenGraph image."""
    if not url:
        return {"text": "", "image_url": None, "final_url": url}

    final_url = _decode_google_news_url(url)

    # If decoding failed, do not pretend the Google wrapper is a publisher URL.
    if "news.google.com" in _domain(final_url):
        return {"text": "", "image_url": None, "final_url": final_url}

    text = ""
    image_url = None

    try:
        resp = requests.get(
            final_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        final_url = resp.url or final_url
        extracted = _extract_html(resp.text, final_url)
        text = extracted["text"] or ""
        image_url = extracted["image_url"]
    except Exception:
        pass

    # Trafilatura's own downloader sometimes succeeds when requests is blocked.
    if len(text) < MIN_TEXT_CHARS:
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(final_url)
            if downloaded:
                extracted_text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    favor_recall=True,
                    url=final_url,
                ) or ""
                if len(extracted_text) > len(text):
                    text = extracted_text

                if not image_url:
                    meta = trafilatura.extract_metadata(downloaded)
                    if meta and getattr(meta, "image", None):
                        image_url = meta.image
        except Exception:
            pass

    return {
        "text": text or "",
        "image_url": image_url,
        "final_url": final_url,
    }


def download_image(url: str, dest_path: str) -> bool:
    if not url:
        return False

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        content_type = (resp.headers.get("content-type") or "").lower()
        if content_type and "image" not in content_type:
            return False

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def retrieve_candidates(
    query: str,
    max_per_domain: int = 1,
    max_results: int = 30,
) -> List[Dict[str, Any]]:
    """Retrieve enough direct, extractable publisher articles for alignment."""
    providers: List[NewsProvider] = [
        GoogleNewsRSSProvider(),
        GDELTProvider(),
    ]

    candidates: List[Dict[str, Any]] = []
    seen_sources: Dict[str, int] = {}
    seen_final_urls = set()

    total_raw = 0
    decoded_successes = 0
    attempted_extractions = 0

    for provider in providers:
        raw = provider.search(query, max_results=max_results)
        total_raw += len(raw)
        print(f"[retrieval] {provider.__class__.__name__}: {len(raw)} raw result(s)")

        for item in raw:
            # Important: dedupe Google wrappers by publisher/source identity,
            # not news.google.com domain.
            source_key = _source_key(item.get("source", ""))
            if not source_key:
                source_key = item.get("id", "")

            if seen_sources.get(source_key, 0) >= max_per_domain:
                continue
            seen_sources[source_key] = seen_sources.get(source_key, 0) + 1

            original_url = item.get("url", "")
            extracted = extract_article(original_url)
            final_url = extracted.get("final_url") or original_url

            if (
                "news.google.com" not in _domain(final_url)
                and final_url != original_url
            ):
                decoded_successes += 1

            attempted_extractions += 1

            if not final_url or final_url in seen_final_urls:
                continue
            seen_final_urls.add(final_url)

            text = extracted.get("text") or ""
            if len(text) < MIN_TEXT_CHARS:
                continue

            candidates.append(
                {
                    **item,
                    "id": _make_id(final_url),
                    "url": final_url,
                    "text": text,
                    "image_url": extracted.get("image_url"),
                    "image_path": None,
                }
            )

            if len(candidates) >= TARGET_EXTRACTED:
                break

        if len(candidates) >= TARGET_EXTRACTED:
            break

    print(f"[retrieval] total raw provider results: {total_raw}")
    print(f"[retrieval] unique publisher identities attempted: {len(seen_sources)}")
    print(f"[retrieval] Google wrapper URLs decoded: {decoded_successes}")
    print(f"[retrieval] article extractions attempted: {attempted_extractions}")
    print(f"[retrieval] successfully extracted article bodies: {len(candidates)}")
    print(
        f"[retrieval] unique extracted publishers: "
        f"{len({_source_key(c.get('source', '')) for c in candidates})}"
    )

    return candidates
