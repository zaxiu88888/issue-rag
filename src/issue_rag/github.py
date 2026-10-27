"""Minimal GitHub REST client built for issue-corpus harvesting.

Three constraints shaped this module:

1. **TLS.** On this machine curl/requests TLS fails with a broken schannel
   certificate chain while Python's bundled OpenSSL works, so all HTTP goes
   through ``urllib``.
2. **Quota.** Unauthenticated GitHub allows 60 core requests per hour. The
   harvester therefore uses *list* endpoints that return 100 objects per
   request instead of per-issue endpoints, and caches every response on disk so
   a rerun costs nothing.
3. **Resumability.** Long harvests get interrupted. Responses are cached by
   request identity and replayed transparently.

Set ``GITHUB_TOKEN`` (or ``GH_TOKEN``) to lift the core limit to 5000/hour.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

API_ROOT = "https://api.github.com"
DEFAULT_USER_AGENT = "issue-rag-harvester/0.1"


class GitHubError(RuntimeError):
    """Non-retryable API failure."""


class QuotaExhausted(GitHubError):
    """The API budget for this hour is spent and we are not allowed to wait."""

    def __init__(self, resource: str, reset_at: dt.datetime | None) -> None:
        self.resource = resource
        self.reset_at = reset_at
        when = reset_at.strftime("%H:%M:%S UTC") if reset_at else "unknown"
        super().__init__(
            f"{resource} rate limit exhausted; resets at {when}. "
            "Set GITHUB_TOKEN to raise the budget to 5000 requests/hour, "
            "or rerun later -- every response so far is cached on disk."
        )


@dataclasses.dataclass
class FetchStats:
    """Counters surfaced at the end of a harvest."""

    requests: int = 0
    cache_hits: int = 0
    not_modified: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def _parse_retry_at(headers: Any) -> dt.datetime | None:
    reset = headers.get("x-ratelimit-reset")
    if reset:
        try:
            return dt.datetime.fromtimestamp(int(reset), tz=dt.timezone.utc)
        except (TypeError, ValueError):
            return None
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(
                seconds=int(retry_after)
            )
        except (TypeError, ValueError):
            return None
    return None


def _parse_next_link(link_header: str | None) -> str | None:
    """Extract the rel="next" URL from a GitHub Link header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url_part = segments[0].strip()
        rel_part = segments[1].strip()
        if rel_part in ('rel="next"', "rel=next") and url_part.startswith("<"):
            return url_part[1:-1]
    return None


class GitHubClient:
    """Caching, rate-limit aware wrapper around the GitHub REST API."""

    def __init__(
        self,
        token: str | None = None,
        cache_dir: pathlib.Path | None = None,
        *,
        allow_wait: bool = False,
        max_wait_s: int = 900,
        verbose: bool = True,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_wait = allow_wait
        self.max_wait_s = max_wait_s
        self.verbose = verbose
        self.stats = FetchStats()

    # ------------------------------------------------------------------ cache

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> pathlib.Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: pathlib.Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, path: pathlib.Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ http

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            # Some networks truncate chunked responses at a fixed byte count
            # (~750 KB here). Compressed JSON is 4-5x smaller, which keeps a
            # 100-comment page inside the limit.
            "Accept-Encoding": "gzip",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _build_url(self, path_or_url: str, params: dict[str, Any] | None) -> str:
        url = path_or_url if path_or_url.startswith("http") else API_ROOT + path_or_url
        if params:
            cleaned = {k: v for k, v in params.items() if v is not None}
            if cleaned:
                sep = "&" if "?" in url else "?"
                url = url + sep + urllib.parse.urlencode(cleaned)
        return url

    # Truncated chunked bodies and connection resets are routine on long
    # harvests; both are worth retrying because every successful page is cached.
    TRANSIENT_ERRORS = (
        urllib.error.URLError,
        http.client.HTTPException,
        ConnectionError,
        TimeoutError,
    )

    def _fetch_raw(self, url: str, *, attempts: int = 4) -> tuple[bytes, Any]:
        last: Exception | None = None
        for attempt in range(attempts):
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                        import gzip

                        raw = gzip.decompress(raw)
                    return raw, resp.headers
            except urllib.error.HTTPError:
                raise  # policy/quota errors are handled by the caller
            except self.TRANSIENT_ERRORS as exc:
                last = exc
                if attempt == attempts - 1:
                    break
                wait_s = 2**attempt
                if self.verbose:
                    print(
                        f"  transient {type(exc).__name__} on "
                        f"{url.rsplit('/', 1)[-1][:40]}; "
                        f"retry {attempt + 2}/{attempts} in {wait_s}s"
                    )
                time.sleep(wait_s)
        raise GitHubError(f"network failure for {url} after {attempts} attempts: {last}")

    def request(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Return ``{"body": ..., "link": ...}`` for one API call."""
        url = self._build_url(path_or_url, params)
        cache_path = self._cache_path(url, None) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        try:
            raw, headers = self._fetch_raw(url)
            link = headers.get("Link")
            body = json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            reset_at = _parse_retry_at(exc.headers or {})
            remaining = (exc.headers or {}).get("x-ratelimit-remaining")
            resource = (exc.headers or {}).get("x-ratelimit-resource", "core")
            if exc.code in (403, 429) and remaining == "0":
                self._handle_quota(resource, reset_at)
                return self.request(path_or_url, params, use_cache=use_cache)
            if exc.code in (500, 502, 503, 504):
                raise GitHubError(f"server error {exc.code} for {url}") from exc
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001 - best effort diagnostics
                pass
            raise GitHubError(f"HTTP {exc.code} for {url}: {detail}") from exc

        self.stats.requests += 1
        payload = {"body": body, "link": link, "url": url}
        self._write_cache(cache_path, payload)
        return payload

    def _handle_quota(self, resource: str, reset_at: dt.datetime | None) -> None:
        if not self.allow_wait or reset_at is None:
            raise QuotaExhausted(resource, reset_at)
        wait_s = (reset_at - dt.datetime.now(tz=dt.timezone.utc)).total_seconds()
        if wait_s <= 0:
            return
        if wait_s > self.max_wait_s:
            raise QuotaExhausted(resource, reset_at)
        if self.verbose:
            print(f"  rate limit hit on {resource}; sleeping {wait_s:.0f}s until reset")
        time.sleep(wait_s + 2)

    # ------------------------------------------------------------ pagination

    def paginate(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        max_items: int | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield items across pages, following Link headers."""
        params = dict(params or {})
        params.setdefault("per_page", per_page)
        url: str | None = self._build_url(path_or_url, params)
        emitted = 0
        pages = 0
        while url:
            payload = self.request(url, None)
            body = payload.get("body")
            if isinstance(body, dict) and "items" in body:
                items = body["items"]
            elif isinstance(body, list):
                items = body
            else:
                items = []
            for item in items:
                yield item
                emitted += 1
                if max_items is not None and emitted >= max_items:
                    return
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
            url = _parse_next_link(payload.get("link"))

    # --------------------------------------------------------------- helpers

    def describe_quota(self) -> str:
        """Human readable quota line; never raises."""
        try:
            payload = self.request("/rate_limit", use_cache=False)
        except GitHubError as exc:
            return f"quota unavailable ({exc})"
        resources = (payload.get("body") or {}).get("resources") or {}
        parts = []
        for name in ("core", "search"):
            info = resources.get(name) or {}
            if info:
                parts.append(f"{name} {info.get('remaining')}/{info.get('limit')}")
        auth = "authenticated" if self.token else "anonymous"
        return f"{auth}: " + ", ".join(parts)
