"""X (Twitter) poster (SPEC section c.22).

``XClient`` is dry-run unless ``FLY_X=post`` AND all four OAuth1 user-context credentials (``X_API_KEY``,
``X_API_SECRET``, ``X_ACCESS_TOKEN``, ``X_ACCESS_TOKEN_SECRET``) are set in the environment. Dry-run never touches the
network and never imports ``tweepy``; it logs ``[X DRY RUN] <text> (+image N bytes)``.

Live path (RESEARCH section 10, tweepy 4.17 surface): image via ``POST https://api.x.com/2/media/upload`` signed with
``requests_oauthlib.OAuth1`` -> v1.1 ``API.media_upload`` fallback -> text-only; then
``tweepy.Client(...).create_tweet(text=..., media_ids=...)``. 429 disables posting until the reset time (or 15 min),
403/401 disable it for the session (``disabled_reason`` shows up in /api/health), anything else is an error string.
Errors never propagate: the orchestrator always gets a ``PostResult``.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.config import Settings

__all__ = ["PostResult", "XClient", "CRED_KEYS", "MEDIA_UPLOAD_URL"]

log = logging.getLogger("flybrain.agent.x")

CRED_KEYS: tuple[str, ...] = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")
MEDIA_UPLOAD_URL = "https://api.x.com/2/media/upload"
_RATE_LIMIT_HOLD_S = 900.0


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


@dataclass(slots=True)
class PostResult:
    posted: bool
    dry_run: bool
    id: str | None
    url: str | None
    error: str | None
    media_id: str | None


class XClient:
    """Posts tweets (optionally with a PNG) or logs them in dry-run mode."""

    def __init__(self, settings: "Settings") -> None:
        """tweepy imported lazily. dry_run = settings.x_mode != 'post'. Missing credentials with x_mode='post' ->
        log ERROR and behave as dry-run (never crash).

        The four OAuth1 secrets are read from ``os.environ`` here and never from ``Settings`` (SPEC b keeps secrets out
        of the settings object so ``GET /api/config`` cannot leak them)."""
        self.settings = settings
        self._creds: dict[str, str] = {k: str(os.environ.get(k, "") or "") for k in CRED_KEYS}
        self.mode = str(getattr(settings, "x_mode", "dryrun") or "dryrun").lower()
        self.dry_run = self.mode != "post"
        self._disabled_reason: str | None = None
        self._disabled_until: float | None = None
        self.posts = 0
        self.last_error: str | None = None
        self._client: Any = None
        if not self.dry_run:
            missing = [k for k in CRED_KEYS if not self._creds[k]]
            if missing:
                log.error("agent.x: FLY_X=post but credentials missing (%s); behaving as dry-run", ", ".join(missing))
                self.dry_run = True
                self._disabled_reason = "missing credentials: " + ", ".join(missing)

    # -- health
    @property
    def disabled_reason(self) -> str | None:
        if self._disabled_until is not None:
            if time.time() < self._disabled_until:
                return f"rate limited until {time.strftime('%H:%M:%SZ', time.gmtime(self._disabled_until))}"
            self._disabled_until = None
        return self._disabled_reason

    @property
    def live(self) -> bool:
        return not self.dry_run and self.disabled_reason is None

    def status(self) -> dict:
        return {"mode": "post" if not self.dry_run else "dryrun", "dry_run": self.dry_run,
                "disabled_reason": self.disabled_reason, "posts": self.posts, "last_error": self.last_error}

    # -- posting
    def post(self, text: str, png: bytes | None = None) -> PostResult:
        """dry-run: return PostResult(posted=False, dry_run=True) after logging '[X DRY RUN] <text> (+image N bytes)'.
        live: media upload (v2 -> v1.1 -> none) then create_tweet; never raises."""
        text = (text or "")[:280]
        n_img = len(png) if png else 0
        if self.dry_run:
            log.info("[X DRY RUN] %s (+image %d bytes)", _ascii(text), n_img)
            return PostResult(posted=False, dry_run=True, id=None, url=None, error=None, media_id=None)
        reason = self.disabled_reason
        if reason is not None:
            log.warning("agent.x: posting disabled (%s); tweet not sent", reason)
            return PostResult(posted=False, dry_run=False, id=None, url=None, error=f"disabled: {reason}", media_id=None)
        try:
            import tweepy  # optional dependency (FLY_X=post only)
        except ImportError as exc:
            self._disabled_reason = f"tweepy not installed ({exc})"
            log.error("agent.x: %s", self._disabled_reason)
            return PostResult(posted=False, dry_run=False, id=None, url=None, error=self._disabled_reason, media_id=None)

        media_id: str | None = None
        if png:
            media_id = self.upload_media_v2(png)
            if media_id is None:
                media_id = self.upload_media_v1(png)
            if media_id is None:
                log.warning("agent.x: image upload failed on both paths; posting text only")
        try:
            client = self._tweepy_client(tweepy)
            resp = client.create_tweet(text=text, media_ids=[media_id] if media_id else None)
            data = getattr(resp, "data", None)
            if data is None and isinstance(resp, dict):
                data = resp.get("data")
            tid = str(data["id"]) if isinstance(data, dict) and "id" in data else str(getattr(data, "id", ""))
            if not tid:
                raise ValueError(f"create_tweet returned no id: {resp!r}"[:200])
            url = f"https://x.com/i/web/status/{tid}"
            self.posts += 1
            log.info("agent.x: posted %s", url)
            return PostResult(posted=True, dry_run=False, id=tid, url=url, error=None, media_id=media_id)
        except tweepy.TooManyRequests as exc:
            reset = self._reset_time(exc)
            self._disabled_until = reset if reset else time.time() + _RATE_LIMIT_HOLD_S
            self.last_error = f"TooManyRequests: {exc}"[:200]
            log.error("agent.x: rate limited until %.0f (%s)", self._disabled_until, _ascii(self.last_error))
            return PostResult(posted=False, dry_run=False, id=None, url=None, error=self.last_error, media_id=media_id)
        except (tweepy.Forbidden, tweepy.Unauthorized) as exc:
            self._disabled_reason = f"{type(exc).__name__}: {exc}"[:200]
            self.last_error = self._disabled_reason
            log.error("agent.x: disabled for the session: %s", _ascii(self._disabled_reason))
            return PostResult(posted=False, dry_run=False, id=None, url=None, error=self._disabled_reason,
                              media_id=media_id)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a result, never an exception
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            log.error("agent.x: post failed: %s", _ascii(self.last_error))
            return PostResult(posted=False, dry_run=False, id=None, url=None, error=self.last_error, media_id=media_id)

    def _tweepy_client(self, tweepy: Any) -> Any:
        if self._client is None:
            c = self._creds
            self._client = tweepy.Client(consumer_key=c["X_API_KEY"], consumer_secret=c["X_API_SECRET"],
                                         access_token=c["X_ACCESS_TOKEN"],
                                         access_token_secret=c["X_ACCESS_TOKEN_SECRET"], wait_on_rate_limit=False)
        return self._client

    @staticmethod
    def _reset_time(exc: Any) -> float | None:
        """tweepy 4.17 TooManyRequests may carry ``reset_time`` (datetime) or an ``x-rate-limit-reset`` header."""
        try:
            rt = getattr(exc, "reset_time", None)
            if rt is not None:
                ts = rt.timestamp() if hasattr(rt, "timestamp") else float(rt)
                return ts if ts > time.time() else None
            hdrs = getattr(getattr(exc, "response", None), "headers", None)
            if hdrs is not None and hdrs.get("x-rate-limit-reset"):
                ts = float(hdrs.get("x-rate-limit-reset"))
                return ts if ts > time.time() else None
        except Exception:
            return None
        return None

    # -- media
    def upload_media_v2(self, png: bytes) -> str | None:
        """POST https://api.x.com/2/media/upload multipart files={'media': png}, data={'media_category':'tweet_image'},
        auth=requests_oauthlib.OAuth1(...). Returns data['id'] or None."""
        try:
            import requests  # tweepy dependency
            from requests_oauthlib import OAuth1  # tweepy dependency
        except ImportError as exc:
            log.warning("agent.x: v2 media upload unavailable (%s)", exc)
            return None
        c = self._creds
        try:
            auth = OAuth1(c["X_API_KEY"], c["X_API_SECRET"], c["X_ACCESS_TOKEN"], c["X_ACCESS_TOKEN_SECRET"])
            r = requests.post(MEDIA_UPLOAD_URL, files={"media": ("fly.png", png, "image/png")},
                              data={"media_category": "tweet_image"}, auth=auth, timeout=20)
            if r.status_code >= 400:
                log.warning("agent.x: v2 media upload HTTP %s: %s", r.status_code, _ascii(r.text[:120]))
                return None
            body = r.json()
            data = body.get("data", body) if isinstance(body, dict) else {}
            mid = data.get("id") or data.get("media_id_string") or data.get("media_id")
            return str(mid) if mid else None
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.x: v2 media upload failed: %s", _ascii(f"{type(exc).__name__}: {exc}"[:160]))
            return None

    def upload_media_v1(self, png: bytes) -> str | None:
        """tweepy.API(tweepy.OAuth1UserHandler(...)).media_upload(filename='fly.png', file=io.BytesIO(png)).media_id_string"""
        try:
            import tweepy
        except ImportError:
            return None
        c = self._creds
        try:
            auth = tweepy.OAuth1UserHandler(c["X_API_KEY"], c["X_API_SECRET"], c["X_ACCESS_TOKEN"],
                                            c["X_ACCESS_TOKEN_SECRET"])
            media = tweepy.API(auth).media_upload(filename="fly.png", file=io.BytesIO(png))
            mid = getattr(media, "media_id_string", None) or getattr(media, "media_id", None)
            return str(mid) if mid else None
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.x: v1.1 media upload failed: %s", _ascii(f"{type(exc).__name__}: {exc}"[:160]))
            return None
